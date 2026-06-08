"""派生关系反向索引（target_policy -> 依赖它的 source policies）。

``lookup_dependents`` 原本要遍历所有 policy 表、逐表扫描 ``relation_keys`` 才能反查出
“哪些 source policy 引用了某个 target”，规模大时是全库扫描、可能耗时数十秒。

这里用一张 SQLite 反向索引表把它降为“点查”：

- 每次 ``store.upsert`` 成功后，按 source policy 粒度重算它的依赖足迹并写入索引；
- ``store.drop_table`` 时清除该 source 的条目；
- ``lookup_dependents`` 直接按 target 点查。

索引是“可重建的派生数据”，任何异常都不应影响主流程：store 层在索引不可用 / 未覆盖
全部 source 时会回退到全表扫描，并在后台补建。

为什么用 SQLite：纯标准库、ACID、跨进程文件锁可靠（多 worker / 多副本场景下比手写
JSON 文件安全），点查 + 索引天然合适。索引文件与 LanceDB 数据同放在 ``STORE_DIR`` 下。

计数口径与 :func:`store._count_dependents_in_table` 完全一致（每个 derived 行命中一次）：
- ``is_any=1``：clause 无关，统计引用某 target（任意 clause）的 derived 行数；
- ``is_any=0``：统计引用 ``(target, clause)`` 的 derived 行数。
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading

from .config import get_settings

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "_relation_index.sqlite3"

# 单连接 + 模块级锁：SQLite 连接默认不可跨线程共享，这里用 check_same_thread=False
# 并以锁串行化所有访问；跨进程并发由 busy_timeout + 短事务兜底。
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_conn_path: str | None = None


def enabled() -> bool:
    return bool(get_settings().enable_relation_index)


def _index_path() -> str:
    store_dir = get_settings().store_dir
    os.makedirs(store_dir, exist_ok=True)
    return os.path.join(store_dir, _INDEX_FILENAME)


def _get_conn() -> sqlite3.Connection:
    """返回指向当前 ``STORE_DIR`` 的连接；store_dir 变化（如测试切换临时目录）时自动重连。"""

    global _conn, _conn_path
    path = _index_path()
    if _conn is not None and _conn_path == path:
        return _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:  # noqa: BLE001
            pass
        _conn = None
        _conn_path = None
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        # 网络挂载卷上 WAL 可能不可用，退回默认 journal 即可。
        pass
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)
    _conn = conn
    _conn_path = path
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS rel_index (
            source_policy_id TEXT NOT NULL,
            target_policy_id TEXT NOT NULL,
            clause_id        TEXT NOT NULL,
            is_any           INTEGER NOT NULL,
            n_rows           INTEGER NOT NULL,
            PRIMARY KEY (source_policy_id, target_policy_id, clause_id, is_any)
        );
        CREATE INDEX IF NOT EXISTS ix_rel_target
            ON rel_index (target_policy_id, is_any, clause_id);
        -- 已被索引覆盖的 source 清单，用于判断索引是否完整（增量/历史数据迁移）。
        CREATE TABLE IF NOT EXISTS rel_index_sources (
            source_policy_id TEXT PRIMARY KEY
        );
        """
    )
    conn.commit()


def update_source(
    source_policy_id: str, entries: list[tuple[str, str, int, int]]
) -> None:
    """用 ``entries`` 全量替换某 source 的索引条目，并标记该 source 已被覆盖。

    ``entries``：``[(target_policy_id, clause_id, is_any, n_rows), ...]``。即使为空，也会
    把 source 记入 ``rel_index_sources``（表示“已索引、但无依赖”）。
    """

    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "DELETE FROM rel_index WHERE source_policy_id = ?", (source_policy_id,)
            )
            if entries:
                conn.executemany(
                    "INSERT OR REPLACE INTO rel_index "
                    "(source_policy_id, target_policy_id, clause_id, is_any, n_rows) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (source_policy_id, t, c, int(is_any), int(n))
                        for (t, c, is_any, n) in entries
                    ],
                )
            conn.execute(
                "INSERT OR IGNORE INTO rel_index_sources (source_policy_id) VALUES (?)",
                (source_policy_id,),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise


def remove_source(source_policy_id: str) -> None:
    """删除某 source 的全部索引条目（drop policy 时调用）。

    只清除它作为 *source* 的记录；它作为 *target* 被别的 source 引用的条目保持不变，
    以与历史“扫描源表 relation_keys”的语义一致。
    """

    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "DELETE FROM rel_index WHERE source_policy_id = ?", (source_policy_id,)
            )
            conn.execute(
                "DELETE FROM rel_index_sources WHERE source_policy_id = ?",
                (source_policy_id,),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise


def lookup(
    target_policy_id: str, target_clause_id: str | None
) -> list[tuple[str, int]]:
    """点查依赖 ``(target_policy_id[, target_clause_id])`` 的 source 列表。

    返回 ``[(source_policy_id, n_hits), ...]``，已排除 source == target 的自反项，并按
    source 排序保证输出稳定。
    """

    with _lock:
        conn = _get_conn()
        if target_clause_id is None:
            cur = conn.execute(
                "SELECT source_policy_id, n_rows FROM rel_index "
                "WHERE target_policy_id = ? AND is_any = 1 "
                "AND source_policy_id <> ? ORDER BY source_policy_id",
                (target_policy_id, target_policy_id),
            )
        else:
            cur = conn.execute(
                "SELECT source_policy_id, n_rows FROM rel_index "
                "WHERE target_policy_id = ? AND is_any = 0 AND clause_id = ? "
                "AND source_policy_id <> ? ORDER BY source_policy_id",
                (target_policy_id, target_clause_id, target_policy_id),
            )
        return [(row[0], int(row[1])) for row in cur.fetchall()]


def indexed_sources() -> set[str]:
    """返回已被索引覆盖的 source 集合，用于判断索引是否完整。"""

    with _lock:
        conn = _get_conn()
        cur = conn.execute("SELECT source_policy_id FROM rel_index_sources")
        return {row[0] for row in cur.fetchall()}


def clear() -> None:
    """清空整个反向索引（全量重建前调用）。"""

    with _lock:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM rel_index")
            conn.execute("DELETE FROM rel_index_sources")
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
