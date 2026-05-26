"""v1 领域模型与通用 v2 文档模型之间的映射。"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from ..schema import ChunkRow, RelationKey, SearchHit
from ..schema_generic import GenericDocumentInput, GenericDocumentRecord, SchemaField

_TOKEN_SPLIT_RE = re.compile(r"[\s\W_]+")
_METADATA_KEY_RE = re.compile(r"[^0-9a-zA-Z_]+")


def policy_to_collection(policy_id: str) -> str:
    """当前阶段采用同名映射，便于旧数据直接复用。"""

    return policy_id


def collection_to_policy(collection_id: str) -> str:
    """当前阶段采用同名映射，便于 v1/v2 双栈并存。"""

    return collection_id


def _fallback_tokenize(text: str) -> str:
    if not text:
        return ""
    return " ".join(tok for tok in _TOKEN_SPLIT_RE.split(text.lower()) if tok)


def _normalize_heading_paths(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    out: list[list[str]] = []
    for seg in value:
        if isinstance(seg, list):
            out.append([str(x) for x in seg])
    return out


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value]


def _normalize_relation_keys(value: Any) -> list[RelationKey]:
    if not isinstance(value, list):
        return []
    out: list[RelationKey] = []
    for item in value:
        if isinstance(item, dict):
            out.append(
                RelationKey(
                    policy_id=str(item.get("policy_id", "")),
                    clause_id=str(item.get("clause_id", "")),
                )
            )
    return out


def _to_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_kind(value: Any) -> str:
    if isinstance(value, str) and value.lower() == "derived":
        return "derived"
    return "original"


def _serialize_metadata(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _sanitize_metadata_segment(seg: str) -> str:
    s = _METADATA_KEY_RE.sub("_", seg.lower().strip())
    s = s.strip("_")
    return s or "field"


def _metadata_path_to_column(path: tuple[str, ...]) -> str:
    readable = "__".join(_sanitize_metadata_segment(seg) for seg in path) if path else "root"
    raw = ".".join(path) if path else "$"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"md_{readable}_{digest}"


def _flatten_metadata_leaves(value: Any, *, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        out: dict[tuple[str, ...], Any] = {}
        for k, v in value.items():
            out.update(_flatten_metadata_leaves(v, path=path + (str(k),)))
        return out
    if isinstance(value, list):
        out: dict[tuple[str, ...], Any] = {}
        for idx, item in enumerate(value):
            out.update(_flatten_metadata_leaves(item, path=path + (str(idx),)))
        return out
    if path:
        return {path: value}
    return {}


def _extract_metadata_scalars(value: Any) -> dict[str, Any]:
    root_path = () if isinstance(value, dict) else ("root",)
    leaves = _flatten_metadata_leaves(value, path=root_path)
    out: dict[str, Any] = {}
    for path, leaf in leaves.items():
        col = _metadata_path_to_column(path)
        if isinstance(leaf, (bool, int, float, str)) or leaf is None:
            out[col] = leaf
            continue
        # 复杂对象已经被递归拆开；仅兜底无法拆分的自定义对象为字符串
        out[col] = str(leaf)
    return out


def document_to_chunk_row(doc: GenericDocumentInput) -> ChunkRow:
    """把通用文档模型映射到当前 LanceDB 固定表结构。"""

    raw_md = doc.metadata
    md = raw_md if isinstance(raw_md, dict) else {}
    metadata_scalars = _extract_metadata_scalars(raw_md)
    now_ms = int(time.time() * 1000)
    tokenized = (doc.content_tokenized or "").strip() or _fallback_tokenize(doc.content)
    return ChunkRow(
        chunk_id=int(doc.document_id),
        content=doc.content,
        content_tokenized=tokenized,
        vector=list(doc.vector or []),
        heading_paths=_normalize_heading_paths(md.get("heading_paths")),
        directories=_normalize_str_list(md.get("directories")),
        kind=_normalize_kind(md.get("kind")),
        parent_chunk_index=_to_int(md.get("parent_chunk_index"), -1),
        derived_seq=_to_int(md.get("derived_seq"), 0),
        relation_keys=_normalize_relation_keys(md.get("relation_keys")),
        hop_depth=_to_int(md.get("hop_depth"), 0),
        source=str(md.get("source", "") or ""),
        clause_id=str(md.get("clause_id", "") or ""),
        metadata_json=_serialize_metadata(raw_md),
        metadata_scalars=metadata_scalars,
        built_at=_to_int(md.get("built_at"), now_ms),
    )


def hit_to_generic_document(hit: SearchHit) -> GenericDocumentRecord:
    fallback_metadata = {
        "heading_paths": hit.heading_paths,
        "directories": hit.directories,
        "kind": hit.kind,
        "parent_chunk_index": hit.parent_chunk_index,
        "derived_seq": hit.derived_seq,
        "relation_keys": [rk.model_dump() for rk in hit.relation_keys],
        "hop_depth": hit.hop_depth,
        "source": hit.source,
        "clause_id": hit.clause_id,
    }
    metadata: Any = fallback_metadata
    if hit.metadata_json:
        try:
            metadata = json.loads(hit.metadata_json)
        except (TypeError, ValueError):
            metadata = fallback_metadata

    return GenericDocumentRecord(
        document_id=hit.chunk_id,
        score=float(hit.score),
        content=hit.content,
        metadata=metadata,
    )


def field_to_schema_field(field: Any) -> SchemaField:
    return SchemaField(
        name=str(getattr(field, "name", "")),
        type=str(getattr(field, "type", "")),
        nullable=bool(getattr(field, "nullable", True)),
    )
