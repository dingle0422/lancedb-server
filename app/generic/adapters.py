"""v1 领域模型与通用 v2 文档模型之间的映射。"""

from __future__ import annotations

import re
import time
from typing import Any

from ..schema import ChunkRow, RelationKey, SearchHit
from ..schema_generic import GenericDocumentInput, GenericDocumentRecord, SchemaField

_TOKEN_SPLIT_RE = re.compile(r"[\s\W_]+")


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


def document_to_chunk_row(doc: GenericDocumentInput) -> ChunkRow:
    """把通用文档模型映射到当前 LanceDB 固定表结构。"""

    md = doc.metadata or {}
    now_ms = int(time.time() * 1000)
    tokenized = (doc.content_tokenized or "").strip() or _fallback_tokenize(doc.content)
    return ChunkRow(
        chunk_id=int(doc.document_id),
        content=doc.content,
        content_tokenized=tokenized,
        vector=list(doc.vector or []),
        heading_paths=_normalize_heading_paths(md.get("heading_paths")),
        directories=_normalize_str_list(md.get("directories")),
        kind=str(md.get("kind", "original") or "original"),
        parent_chunk_index=int(md.get("parent_chunk_index", -1) or -1),
        derived_seq=int(md.get("derived_seq", 0) or 0),
        relation_keys=_normalize_relation_keys(md.get("relation_keys")),
        hop_depth=int(md.get("hop_depth", 0) or 0),
        source=str(md.get("source", "") or ""),
        clause_id=str(md.get("clause_id", "") or ""),
        built_at=int(md.get("built_at", now_ms) or now_ms),
    )


def hit_to_generic_document(hit: SearchHit) -> GenericDocumentRecord:
    metadata = {
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
