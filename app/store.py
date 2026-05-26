"""LanceDB 连接、表生命周期、索引管理。

每张表对应一个 ``policy_id``，物理路径 ``STORE_DIR/{safe_policy_id}.lance``。

为了防止 ``policy_id`` 含中文 / 路径分隔符等不安全字符破坏文件系统，统一通过
:func:`_safe_policy_dir` 做 urlsafe-base64 编码后作为目录名；HTTP 接口里仍接受
原始 ``policy_id``。

LanceDB 操作大多是同步阻塞的；外层路由用 :func:`anyio.to_thread.run_sync` 调用本模块函数。
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from typing import Any

import pyarrow as pa

from .config import get_settings
from .schema import (
    ChunkRow,
    RelationKey,
    SearchHit,
    build_arrow_schema,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# 同进程内共享：lancedb.connect 是廉价的（懒打开），但 Table 句柄保留可减少重复 open。
_db_lock = threading.Lock()
_db: Any | None = None
_table_cache: dict[str, Any] = {}
_table_lock = threading.Lock()

# 每张表独立的写锁，防止并发写同一张表导致 EBUSY
_write_locks: dict[str, threading.Lock] = {}
_write_locks_lock = threading.Lock()


def _get_write_lock(policy_id: str) -> threading.Lock:
    with _write_locks_lock:
        if policy_id not in _write_locks:
            _write_locks[policy_id] = threading.Lock()
        return _write_locks[policy_id]


def _evict_and_release(policy_id: str) -> None:
    """从进程内缓存弹出 Table 句柄并强制 GC，让底层 Rust 释放 .lance 文件描述符。

    LanceDB 的 ``Table`` Python 对象包了一个 Rust ``Dataset``，后者持有 OS 级 file handles。
    仅 ``dict.pop`` 弹引用不够：必须把所有 Python 引用一并清掉，再触发一次 ``gc.collect``，
    底层文件锁才会被释放，否则 ``db.drop_table`` 在 Linux mount 卷 / Windows NTFS 上会
    抛 ``Device or resource busy (os error 16)``。
    """

    import gc

    with _table_lock:
        tbl = _table_cache.pop(policy_id, None)
    del tbl  # 局部引用也丢掉
    gc.collect()


def _drop_table_with_retry(db, name: str, *, attempts: int = 6) -> None:
    """碰到 ``Device or resource busy`` 时退避重试（最大 ~6s）。"""

    import gc as _gc
    import time as _time

    last_exc: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            db.drop_table(name)
            return
        except Exception as e:  # noqa: BLE001
            last_exc = e
            msg = str(e)
            transient = (
                "Device or resource busy" in msg
                or "os error 16" in msg
                or "PermissionError" in msg
                or "being used by another process" in msg
            )
            if not transient or i == attempts:
                raise
            backoff = 0.3 * i
            logger.warning(
                "[Store] drop_table %s 第 %d/%d 次忙，%.1fs 后重试: %s",
                name, i, attempts, backoff, e,
            )
            _gc.collect()
            _time.sleep(backoff)
    if last_exc:
        raise last_exc


def _truncate_and_add(tbl, batch: pa.RecordBatch) -> None:
    """同 schema 的 overwrite：删全部行 + add，避开任何文件系统级 rm。"""

    # LanceDB SQL 不一定接受 ``true``；用一个永真表达式做兜底。
    for predicate in ("true", "1=1", "chunk_id IS NOT NULL OR chunk_id IS NULL"):
        try:
            tbl.delete(predicate)
            break
        except Exception as e:  # noqa: BLE001
            logger.debug("[Store] delete predicate %r 失败，尝试下一个: %s", predicate, e)
    tbl.add(batch)


def _overwrite_or_create(db, policy_id: str, name: str, batch: pa.RecordBatch, dim: int, existing_dim: int):
    """overwrite 路径的 EBUSY-safe 实现，按从轻到重三步降级：

    1. 表已存在且 dim 一致 → ``tbl.delete('true') + tbl.add(batch)``
       ─ 完全不删目录，根本不会触发 ``Device or resource busy``。
    2. 表不存在 / dim 变化 → ``db.create_table(..., mode='overwrite')``
       ─ LanceDB 原生原子覆盖，写新 manifest，不 rm -rf 老文件夹。
    3. ``mode='overwrite'`` 在老版本不被支持时 → 退化到 ``drop_table + create_table``
       ─ 此时才会触发 EBUSY，已有 GC + 6 次退避兜底。
    """

    table_existed = name in _all_table_names(db)

    if table_existed and existing_dim and dim and existing_dim == dim:
        try:
            with _table_lock:
                tbl = _table_cache.get(policy_id)
            if tbl is None:
                tbl = db.open_table(name)
            _truncate_and_add(tbl, batch)
            return tbl
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[Store] truncate+add 失败，退化到 create(mode=overwrite): %s", e
            )
            _evict_and_release(policy_id)

    try:
        return db.create_table(name, data=batch, schema=batch.schema, mode="overwrite")
    except TypeError:
        # 旧版 lancedb 不接受 mode 参数 → drop + create
        pass
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[Store] create(mode=overwrite) 失败，退化到 drop+create: %s", e
        )

    if table_existed:
        _evict_and_release(policy_id)
        _drop_table_with_retry(db, name)
    return db.create_table(name, data=batch, schema=batch.schema)


def _safe_policy_dir(policy_id: str) -> str:
    """把任意 policy_id 编码为安全的目录名。"""

    raw = (policy_id or "").encode("utf-8")
    enc = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"p_{enc}"


def _decode_policy_id(safe_name: str) -> str:
    if not safe_name.startswith("p_"):
        return ""
    body = safe_name[2:]
    pad = "=" * (-len(body) % 4)
    try:
        return base64.urlsafe_b64decode(body + pad).decode("utf-8")
    except Exception:
        return ""


def _connect():
    import lancedb  # type: ignore

    settings = get_settings()
    os.makedirs(settings.store_dir, exist_ok=True)

    global _db
    with _db_lock:
        if _db is None:
            _db = lancedb.connect(settings.store_dir)
        return _db


def _all_table_names(db) -> list[str]:
    """LanceDB ``Connection.table_names`` 默认 ``limit=10``，超过 10 张表的实例
    后续表会被静默截断 —— 进而让 :func:`table_exists` 永远返 False、新表查不到、
    HTTP 上返 0 hits。这里翻页 / 大 limit 兜底，把所有表名一次性拿全。

    LanceDB 历史版本 API 不完全一致：
    - 0.13+：``table_names(page_token=None, limit=10)`` 支持翻页；
    - 老版本可能不接受 ``page_token``，但接受 ``limit``；
    - 更老的可能 ``table_names()`` 完全无参数（直接 listdir，没有截断）。
    依次降级，保证总能拿全。
    """

    PAGE = 1000
    try:
        names: list[str] = []
        page_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"limit": PAGE}
            if page_token is not None:
                kwargs["page_token"] = page_token
            page = db.table_names(**kwargs)
            if not page:
                break
            names.extend(page)
            if len(page) < PAGE:
                break
            # 翻页：以最后一个 name 为下一页 token（lancedb 0.13+ 行为）
            page_token = page[-1]
        return names
    except TypeError:
        # 不支持 page_token / limit 关键字，直接给大 limit
        try:
            return list(db.table_names(limit=1_000_000))
        except TypeError:
            return list(db.table_names())


def _table_path(policy_id: str) -> str:
    settings = get_settings()
    return os.path.join(settings.store_dir, f"{_safe_policy_dir(policy_id)}.lance")


def table_exists(policy_id: str) -> bool:
    db = _connect()
    return _safe_policy_dir(policy_id) in _all_table_names(db)


def open_table(policy_id: str):
    """打开已有表；不存在抛 ``KeyError``。"""

    if not table_exists(policy_id):
        raise KeyError(policy_id)
    name = _safe_policy_dir(policy_id)
    with _table_lock:
        cached = _table_cache.get(policy_id)
        if cached is not None:
            return cached
        db = _connect()
        tbl = db.open_table(name)
        _table_cache[policy_id] = tbl
        return tbl


def drop_table(policy_id: str) -> bool:
    write_lock = _get_write_lock(policy_id)
    with write_lock:
        db = _connect()
        name = _safe_policy_dir(policy_id)
        if name not in _all_table_names(db):
            return False
        _evict_and_release(policy_id)
        _drop_table_with_retry(db, name)
        return True


def list_policies() -> list[tuple[str, int, int]]:
    """返回 ``[(policy_id, n_chunks, dim), ...]``。"""

    db = _connect()
    out: list[tuple[str, int, int]] = []
    for safe_name in _all_table_names(db):
        pid = _decode_policy_id(safe_name)
        if not pid:
            continue
        try:
            tbl = db.open_table(safe_name)
            n = tbl.count_rows()
            dim = _detect_dim(tbl)
        except Exception as e:
            logger.warning("[Store] 读取表 %s 失败: %s", safe_name, e)
            continue
        out.append((pid, int(n), int(dim)))
    return out


# ---------------------------------------------------------------- 数据序列化


def _row_to_arrow_dict(row: ChunkRow) -> dict[str, Any]:
    base = {
        "chunk_id": int(row.chunk_id),
        "content": row.content or "",
        "content_tokenized": row.content_tokenized or "",
        "vector": list(row.vector or []),
        "heading_paths": [list(seg) for seg in (row.heading_paths or [])],
        "directories": list(row.directories or []),
        "kind": row.kind or "original",
        "parent_chunk_index": int(row.parent_chunk_index),
        "derived_seq": int(row.derived_seq),
        "relation_keys": [
            {"policy_id": k.policy_id or "", "clause_id": k.clause_id or ""}
            for k in (row.relation_keys or [])
        ],
        "hop_depth": int(row.hop_depth),
        "source": row.source or "",
        "clause_id": row.clause_id or "",
        "metadata_json": row.metadata_json or "{}",
        "built_at": int(row.built_at) or int(time.time() * 1000),
    }
    for col, value in (row.metadata_scalars or {}).items():
        if isinstance(col, str) and col.startswith("md_"):
            base[col] = value
    return base


def _metadata_fields_from_schema(schema: pa.Schema) -> dict[str, pa.DataType]:
    out: dict[str, pa.DataType] = {}
    for field in schema:
        if field.name.startswith("md_"):
            out[field.name] = field.type
    return out


def _infer_metadata_dtype(values: list[Any]) -> pa.DataType:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return pa.string()
    if all(isinstance(v, bool) for v in non_null):
        return pa.bool_()
    if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
        return pa.int64()
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return pa.float64()
    if all(isinstance(v, str) for v in non_null):
        return pa.string()
    return pa.string()


def _metadata_fields_from_rows(rows: list[ChunkRow]) -> dict[str, pa.DataType]:
    samples: dict[str, list[Any]] = {}
    for row in rows:
        for col, value in (row.metadata_scalars or {}).items():
            if not isinstance(col, str) or not col.startswith("md_"):
                continue
            samples.setdefault(col, []).append(value)
    return {col: _infer_metadata_dtype(vals) for col, vals in samples.items()}


def _merge_metadata_dtype(a: pa.DataType, b: pa.DataType) -> pa.DataType:
    if a == b:
        return a
    if pa.types.is_string(a) or pa.types.is_string(b):
        return pa.string()
    if pa.types.is_floating(a) and pa.types.is_integer(b):
        return a
    if pa.types.is_integer(a) and pa.types.is_floating(b):
        return b
    if pa.types.is_integer(a) and pa.types.is_integer(b):
        return pa.int64()
    if pa.types.is_floating(a) and pa.types.is_floating(b):
        return pa.float64()
    return pa.string()


def _coerce_metadata_value(value: Any, dtype: pa.DataType) -> Any:
    if value is None:
        return None
    try:
        if pa.types.is_string(dtype):
            return str(value)
        if pa.types.is_boolean(dtype):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lv = value.strip().lower()
                if lv in ("true", "1", "yes", "y"):
                    return True
                if lv in ("false", "0", "no", "n"):
                    return False
                return None
            if isinstance(value, (int, float)):
                return bool(value)
            return None
        if pa.types.is_integer(dtype):
            if isinstance(value, bool):
                return int(value)
            return int(value)
        if pa.types.is_floating(dtype):
            if isinstance(value, bool):
                return float(int(value))
            return float(value)
    except Exception:
        return None
    return value


def _normalize_row_for_schema(row: dict[str, Any], schema: pa.Schema, dim: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in schema:
        name = field.name
        value = row.get(name)
        if name == "vector":
            vec = list(value or [])
            if dim > 0:
                if not vec:
                    vec = [0.0] * dim
                elif len(vec) != dim:
                    raise ValueError(
                        f"vector dim mismatch: chunk_id={row.get('chunk_id')} got={len(vec)} expect={dim}"
                    )
            out[name] = vec
            continue
        if name == "metadata_json":
            out[name] = value if isinstance(value, str) else "{}"
            continue
        if name.startswith("md_"):
            out[name] = _coerce_metadata_value(value, field.type)
            continue
        if value is not None:
            out[name] = value
            continue
        if name == "chunk_id":
            out[name] = int(row.get("chunk_id", 0))
        elif name in ("content", "content_tokenized", "source", "clause_id"):
            out[name] = str(row.get(name, "") or "")
        elif name == "kind":
            out[name] = str(row.get(name, "original") or "original")
        elif name == "parent_chunk_index":
            out[name] = int(row.get(name, -1) or -1)
        elif name in ("derived_seq", "hop_depth"):
            out[name] = int(row.get(name, 0) or 0)
        elif name == "relation_keys":
            out[name] = row.get(name) or []
        elif name == "directories":
            out[name] = row.get(name) or []
        elif name == "heading_paths":
            out[name] = row.get(name) or []
        elif name == "built_at":
            out[name] = int(row.get(name) or int(time.time() * 1000))
        else:
            out[name] = value
    return out


def _build_record_batch(
    rows: list[ChunkRow],
    dim: int,
    *,
    metadata_fields: dict[str, pa.DataType] | None = None,
    include_metadata_json: bool = True,
) -> pa.RecordBatch:
    schema = build_arrow_schema(
        dim,
        metadata_fields=metadata_fields or _metadata_fields_from_rows(rows),
        include_metadata_json=include_metadata_json,
    )
    normalized = [_normalize_row_for_schema(_row_to_arrow_dict(r), schema, dim) for r in rows]
    return pa.RecordBatch.from_pylist(normalized, schema=schema)


# ---------------------------------------------------------------- 索引


def _detect_dim(tbl) -> int:
    """从表 schema 反推 vector 列的 fixed_size_list 长度，未建/可变长返回 0。"""

    try:
        schema: pa.Schema = tbl.schema
    except Exception:
        return 0
    field = schema.field("vector") if "vector" in schema.names else None
    if field is None:
        return 0
    t = field.type
    if pa.types.is_fixed_size_list(t):
        return int(t.list_size)
    return 0


def _has_index_on(tbl, column: str) -> bool:
    try:
        for idx in tbl.list_indices():
            cols = getattr(idx, "columns", None) or getattr(idx, "fields", None) or []
            if isinstance(cols, str):
                cols = [cols]
            if column in (cols or []):
                return True
    except Exception:
        return False
    return False


def ensure_indexes(tbl) -> dict[str, bool]:
    """在表上建好 FTS / 向量 / 标量索引（已存在则跳过）。"""

    settings = get_settings()
    n = tbl.count_rows()
    fts_ok = vec_ok = scalar_ok = False

    if n == 0:
        return {"fts": False, "vector": False, "scalar": False}

    # FTS 索引：whitespace tokenizer，因为 content_tokenized 已经是客户端 jieba 分词后的空格串
    if not _has_index_on(tbl, "content_tokenized"):
        try:
            tbl.create_fts_index(
                "content_tokenized",
                base_tokenizer="whitespace",
                with_position=False,
                replace=True,
            )
            fts_ok = True
        except Exception as e:
            logger.warning("[Store] 建 FTS 索引失败: %s", e)
    else:
        fts_ok = True

    # 向量索引：仅在 dim>0 且行数足够时建（小表全量扫足够）
    dim = _detect_dim(tbl)
    if dim > 0 and n >= 256 and not _has_index_on(tbl, "vector"):
        try:
            tbl.create_index(metric="cosine", vector_column_name="vector", replace=True)
            vec_ok = True
        except Exception as e:
            logger.warning("[Store] 建向量索引失败: %s", e)
    else:
        vec_ok = dim > 0

    # 标量索引（可关）
    if settings.enable_scalar_index:
        try:
            schema: pa.Schema = tbl.schema
            scalar_cols = ["kind", "parent_chunk_index"]
            for field in schema:
                if field.name.startswith("md_") and (
                    pa.types.is_string(field.type)
                    or pa.types.is_boolean(field.type)
                    or pa.types.is_integer(field.type)
                    or pa.types.is_floating(field.type)
                ):
                    scalar_cols.append(field.name)
        except Exception:
            scalar_cols = ["kind", "parent_chunk_index"]
        for col in scalar_cols:
            try:
                tbl.create_scalar_index(col, replace=True)
            except Exception as e:
                logger.debug("[Store] 建标量索引 %s 失败（忽略）: %s", col, e)
        scalar_ok = True

    return {"fts": fts_ok, "vector": vec_ok, "scalar": scalar_ok}


# ---------------------------------------------------------------- 写入


def _infer_dim(rows: list[ChunkRow], expected: int | None) -> int:
    if expected:
        return int(expected)
    for r in rows:
        if r.vector:
            return len(r.vector)
    return 0


def _all_rows(tbl) -> list[dict[str, Any]]:
    n = int(tbl.count_rows())
    if n <= 0:
        return []
    schema: pa.Schema = tbl.schema
    return tbl.search().select(list(schema.names)).limit(n).to_list()


def _build_batch_from_raw_rows(rows: list[dict[str, Any]], schema: pa.Schema, dim: int) -> pa.RecordBatch:
    normalized = [_normalize_row_for_schema(row, schema, dim) for row in rows]
    return pa.RecordBatch.from_pylist(normalized, schema=schema)


def upsert(policy_id: str, rows: list[ChunkRow], mode: str, expected_dim: int | None) -> dict:
    """单表 upsert。返回 ``{"written", "table_size", "dim"}``。"""

    if not rows:
        return {"written": 0, "table_size": _row_count(policy_id), "dim": _existing_dim(policy_id)}

    write_lock = _get_write_lock(policy_id)
    with write_lock:
        db = _connect()
        name = _safe_policy_dir(policy_id)
        table_names = _all_table_names(db)
        table_existed = name in table_names

        incoming_dim = _infer_dim(rows, expected_dim)
        existing_dim = _existing_dim(policy_id)
        if existing_dim and incoming_dim and existing_dim != incoming_dim:
            raise ValueError(
                f"dim mismatch for policy={policy_id}: existing={existing_dim} incoming={incoming_dim}"
            )
        dim = incoming_dim or existing_dim

        incoming_md_fields = _metadata_fields_from_rows(rows)
        existing_md_fields: dict[str, pa.DataType] = {}
        include_metadata_json = True
        existing_schema: pa.Schema | None = None
        if table_existed:
            with _table_lock:
                tbl_cached = _table_cache.get(policy_id)
            tbl_for_schema = tbl_cached
            if tbl_for_schema is None:
                tbl_for_schema = db.open_table(name)
            existing_schema = tbl_for_schema.schema
            existing_md_fields = _metadata_fields_from_schema(existing_schema)
            include_metadata_json = "metadata_json" in existing_schema.names

        merged_md_fields: dict[str, pa.DataType] = dict(existing_md_fields)
        for col, dtype in incoming_md_fields.items():
            if col in merged_md_fields:
                merged_md_fields[col] = _merge_metadata_dtype(merged_md_fields[col], dtype)
            else:
                merged_md_fields[col] = dtype

        # overwrite / 新建时直接用 merged schema；append/merge 若 schema 变化则走全量重写
        schema_changed = False
        if table_existed and mode in ("append", "merge_by_chunk_id") and existing_schema is not None:
            if not include_metadata_json:
                schema_changed = True
            elif set(merged_md_fields.keys()) != set(existing_md_fields.keys()):
                schema_changed = True
            else:
                for col, old_dtype in existing_md_fields.items():
                    if merged_md_fields.get(col) != old_dtype:
                        schema_changed = True
                        break

        incoming_dict_rows = [_row_to_arrow_dict(r) for r in rows]

        if mode == "overwrite" or not table_existed:
            batch = _build_record_batch(
                rows,
                dim,
                metadata_fields=merged_md_fields,
                include_metadata_json=True,
            )
            tbl = _overwrite_or_create(db, policy_id, name, batch, dim, existing_dim)
        else:
            # 始终复用缓存句柄，避免产生多个持有同一文件的 Table 对象
            with _table_lock:
                tbl = _table_cache.get(policy_id)
            if tbl is None:
                tbl = db.open_table(name)
            if mode not in ("append", "merge_by_chunk_id"):
                raise ValueError(f"unknown upsert mode: {mode}")

            if schema_changed:
                target_schema = build_arrow_schema(
                    dim,
                    metadata_fields=merged_md_fields,
                    include_metadata_json=True,
                )
                existing_rows = _all_rows(tbl)
                normalized_incoming = [
                    _normalize_row_for_schema(r, target_schema, dim) for r in incoming_dict_rows
                ]
                if mode == "append":
                    merged_rows = [
                        _normalize_row_for_schema(r, target_schema, dim) for r in existing_rows
                    ] + normalized_incoming
                else:
                    by_id: dict[int, dict[str, Any]] = {
                        int(r.get("chunk_id", 0)): _normalize_row_for_schema(r, target_schema, dim)
                        for r in existing_rows
                    }
                    for row in normalized_incoming:
                        by_id[int(row.get("chunk_id", 0))] = row
                    merged_rows = list(by_id.values())
                batch = _build_batch_from_raw_rows(merged_rows, target_schema, dim)
                tbl = _overwrite_or_create(db, policy_id, name, batch, dim, existing_dim)
            else:
                batch = _build_record_batch(
                    rows,
                    dim,
                    metadata_fields=existing_md_fields,
                    include_metadata_json=include_metadata_json,
                )
                if mode == "append":
                    tbl.add(batch)
                else:
                    try:
                        (
                            tbl.merge_insert("chunk_id")
                            .when_matched_update_all()
                            .when_not_matched_insert_all()
                            .execute(batch)
                        )
                    except Exception as e:
                        # LanceDB 老版本可能不支持 merge_insert，退化为 delete + add
                        logger.info("[Store] merge_insert 不可用，退化为 delete+add: %s", e)
                        ids = [str(r.chunk_id) for r in rows]
                        tbl.delete(f"chunk_id IN ({','.join(ids)})")
                        tbl.add(batch)

        with _table_lock:
            _table_cache[policy_id] = tbl

        ensure_indexes(tbl)
        return {
            "written": len(rows),
            "table_size": int(tbl.count_rows()),
            "dim": int(dim),
        }


# ---------------------------------------------------------------- 读取助手


def _row_count(policy_id: str) -> int:
    if not table_exists(policy_id):
        return 0
    return int(open_table(policy_id).count_rows())


def _existing_dim(policy_id: str) -> int:
    if not table_exists(policy_id):
        return 0
    return _detect_dim(open_table(policy_id))


# ---------------------------------------------------------------- 检索


def _row_to_hit(row: dict, *, include_content: bool) -> SearchHit:
    score_raw = row.get("_score")
    if score_raw is None:
        # 某些 LanceDB 版本在非检索读取路径下不会给 _score，或返回 None。
        score_raw = row.get("_distance")
        if score_raw is not None:
            try:
                score_raw = -float(score_raw)
            except Exception:
                score_raw = 0.0
    try:
        score = float(score_raw if score_raw is not None else 0.0)
    except Exception:
        score = 0.0

    rks = row.get("relation_keys") or []
    return SearchHit(
        chunk_id=int(row["chunk_id"]),
        score=score,
        content=row.get("content") if include_content else None,
        heading_paths=[list(p) for p in (row.get("heading_paths") or [])],
        directories=list(row.get("directories") or []),
        kind=row.get("kind") or "original",
        parent_chunk_index=int(row.get("parent_chunk_index", -1)),
        derived_seq=int(row.get("derived_seq", 0)),
        relation_keys=[
            RelationKey(policy_id=rk.get("policy_id", ""), clause_id=rk.get("clause_id", ""))
            for rk in rks
        ],
        hop_depth=int(row.get("hop_depth", 0)),
        source=row.get("source") or "",
        clause_id=row.get("clause_id") or "",
        metadata_json=row.get("metadata_json"),
    )


def _table_columns(tbl) -> set[str]:
    try:
        schema: pa.Schema = tbl.schema
        return set(schema.names)
    except Exception:
        return set()


def _select_columns(include_content: bool, *, available_columns: set[str] | None = None) -> list[str]:
    cols = [
        "chunk_id",
        "heading_paths",
        "directories",
        "kind",
        "parent_chunk_index",
        "derived_seq",
        "relation_keys",
        "hop_depth",
        "source",
        "clause_id",
    ]
    if include_content:
        cols.append("content")
    if available_columns is None or "metadata_json" in available_columns:
        cols.append("metadata_json")
    return cols


def _safe_search_to_list(query, top_k: int) -> list[dict]:
    if top_k <= 0:
        return []
    try:
        return query.limit(top_k).to_list()
    except Exception as e:
        logger.warning("[Store] 检索失败: %s", e)
        return []


def hybrid_search(
    policy_id: str,
    *,
    query_tokenized: str,
    query_vector: list[float],
    top_n: int,
    top_m: int,
    rrf_k: int,
    where: str | None,
    include_content: bool,
    include_derived: bool,
) -> list[SearchHit]:
    """BM25 + 向量并行召回，本地 RRF 融合，与主项目 ``inference/retrieval/rrf.py`` 同公式。

    表不存在时显式抛 :class:`KeyError`，由路由层转成 HTTP 404。历史上这里返 ``[]``
    会让"未建索引"和"索引存在但 0 命中"两种情况无法区分（HTTP 都是 200 hits=[]），
    导致客户端排障极难——曾经有一例 ``table_names()`` 默认 limit=10 截断让新建表
    永远 ``table_exists=False``，被 0 hits 静默蒙了很久。
    """

    if not table_exists(policy_id):
        logger.warning("[search] 表不存在 policy=%s（表名未在 db.table_names 中）", policy_id)
        raise KeyError(policy_id)
    tbl = open_table(policy_id)
    n_total = tbl.count_rows()
    if n_total == 0:
        return []

    cols = _select_columns(include_content, available_columns=_table_columns(tbl))
    final_where = where
    if not include_derived:
        cond = "kind = 'original'"
        final_where = f"({where}) AND ({cond})" if where else cond

    # FTS 路径
    fts_pairs: list[tuple[int, float]] = []
    if query_tokenized.strip() and top_m > 0:
        q = (
            tbl.search(query_tokenized, query_type="fts", fts_columns="content_tokenized")
            .select(cols)
        )
        if final_where:
            q = q.where(final_where, prefilter=False)
        for row in _safe_search_to_list(q, top_m):
            fts_pairs.append((int(row["chunk_id"]), float(row.get("_score", 0.0))))

    # 向量路径
    vec_pairs: list[tuple[int, float]] = []
    if query_vector and top_n > 0:
        q = tbl.search(query_vector, vector_column_name="vector").select(cols)
        if final_where:
            q = q.where(final_where, prefilter=True)
        for row in _safe_search_to_list(q, top_n):
            # LanceDB 向量搜索返回 _distance（越小越相似）；转成相似度分数仅用于排序展示
            d = float(row.get("_distance", row.get("_score", 0.0)))
            vec_pairs.append((int(row["chunk_id"]), -d))

    if not fts_pairs and not vec_pairs:
        return []

    # RRF 融合：rrf = sum(1 / (k + rank))，rank 从 1 起
    fused: dict[int, float] = {}
    for rank_list in (fts_pairs, vec_pairs):
        for rank, (cid, _s) in enumerate(rank_list, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    fused_sorted = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    keep_ids = [cid for cid, _ in fused_sorted]

    # 一次性把 chunk 内容拉回（避免逐行 select）
    if not keep_ids:
        return []
    id_filter = "chunk_id IN (" + ",".join(str(i) for i in keep_ids) + ")"
    rows = tbl.search().select(cols).where(id_filter).limit(len(keep_ids)).to_list()
    by_id = {int(r["chunk_id"]): r for r in rows}

    hits: list[SearchHit] = []
    for cid, score in fused_sorted:
        row = by_id.get(cid)
        if row is None:
            continue
        row["_score"] = score
        hits.append(_row_to_hit(row, include_content=include_content))
    return hits


def list_chunks(
    policy_id: str,
    *,
    where: str | None,
    limit: int,
    include_content: bool,
) -> list[SearchHit]:
    if not table_exists(policy_id):
        return []
    tbl = open_table(policy_id)
    cols = _select_columns(include_content, available_columns=_table_columns(tbl))
    q = tbl.search().select(cols)
    if where:
        q = q.where(where)
    rows = q.limit(max(limit, 1)).to_list()
    return [_row_to_hit(r, include_content=include_content) for r in rows]


def expand_relations(policy_id: str, chunk_id: int, *, include_content: bool) -> list[SearchHit]:
    """返回某父 chunk 的派生 chunks（``parent_chunk_index = chunk_id and kind='derived'``）。"""

    if not table_exists(policy_id):
        return []
    tbl = open_table(policy_id)
    cols = _select_columns(include_content, available_columns=_table_columns(tbl))
    rows = (
        tbl.search()
        .select(cols)
        .where(f"parent_chunk_index = {int(chunk_id)} AND kind = 'derived'")
        .limit(10000)
        .to_list()
    )
    return [_row_to_hit(r, include_content=include_content) for r in rows]


def lookup_relations(
    policy_id: str,
    *,
    target_policy_id: str,
    target_clause_id: str | None,
    include_content: bool,
) -> list[SearchHit]:
    """单表内反查：列出 ``relation_keys`` 中含 ``target_*`` 的派生 chunks。"""

    if not table_exists(policy_id):
        return []
    tbl = open_table(policy_id)
    cols = _select_columns(include_content, available_columns=_table_columns(tbl))
    # 用 list_has_struct 的能力：LanceDB SQL 支持 ``array_has(relation_keys, struct_value)``，
    # 但跨版本不稳定。回退到 Python 侧过滤，性能可接受（派生 chunks 一般 < 1k）。
    rows = (
        tbl.search()
        .select(cols + ["relation_keys"] if "relation_keys" not in cols else cols)
        .where("kind = 'derived'")
        .limit(100000)
        .to_list()
    )
    out: list[SearchHit] = []
    for r in rows:
        rks = r.get("relation_keys") or []
        for rk in rks:
            if rk.get("policy_id") != target_policy_id:
                continue
            if target_clause_id and rk.get("clause_id") != target_clause_id:
                continue
            out.append(_row_to_hit(r, include_content=include_content))
            break
    return out


def lookup_dependents(target_policy_id: str, target_clause_id: str | None) -> list[tuple[str, int]]:
    """全局反查：返回 ``[(source_policy_id, n_hits), ...]``，用于 cascade 触发。"""

    out: list[tuple[str, int]] = []
    for pid, _n, _dim in list_policies():
        if pid == target_policy_id:
            continue  # 自反不计
        hits = lookup_relations(
            pid,
            target_policy_id=target_policy_id,
            target_clause_id=target_clause_id,
            include_content=False,
        )
        if hits:
            out.append((pid, len(hits)))
    return out


# ---------------------------------------------------------------- meta


def table_meta(policy_id: str) -> dict:
    if not table_exists(policy_id):
        return {}
    tbl = open_table(policy_id)
    n = int(tbl.count_rows())
    dim = _detect_dim(tbl)
    n_orig = 0
    n_derv = 0
    built_at = 0
    try:
        for col_row in (
            tbl.search().select(["kind", "built_at"]).limit(n or 1).to_list()
        ):
            if col_row.get("kind") == "derived":
                n_derv += 1
            else:
                n_orig += 1
            ba = int(col_row.get("built_at") or 0)
            if ba > built_at:
                built_at = ba
    except Exception as e:
        logger.warning("[Store] 统计 meta 失败: %s", e)
    return {
        "policy_id": policy_id,
        "n_chunks": n,
        "n_original": n_orig,
        "n_derived": n_derv,
        "dim": dim,
        "has_vector_index": _has_index_on(tbl, "vector"),
        "has_fts_index": _has_index_on(tbl, "content_tokenized"),
        "built_at": built_at,
        "schema_version": SCHEMA_VERSION,
    }
