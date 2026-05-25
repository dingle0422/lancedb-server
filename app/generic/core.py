"""通用向量库领域层实现（当前以 LanceDB 既有实现为内核）。"""

from __future__ import annotations

from functools import lru_cache

import pyarrow as pa

from .. import store
from ..config import get_settings
from ..schema import ChunkRow, SearchHit
from ..schema_generic import GenericDocumentInput, GenericDocumentRecord

from .adapters import (
    collection_to_policy,
    document_to_chunk_row,
    field_to_schema_field,
    hit_to_generic_document,
    policy_to_collection,
)


class CollectionStore:
    """集合生命周期与元数据查询。"""

    def list_collections(self) -> list[tuple[str, int, int]]:
        raw = store.list_policies()
        return [(policy_to_collection(pid), n, dim) for pid, n, dim in raw]

    def collection_exists(self, collection_id: str) -> bool:
        return store.table_exists(collection_to_policy(collection_id))

    def drop_collection(self, collection_id: str) -> bool:
        return store.drop_table(collection_to_policy(collection_id))

    def collection_meta(self, collection_id: str) -> dict:
        policy_id = collection_to_policy(collection_id)
        base = store.table_meta(policy_id)
        if not base:
            return {}

        schema_fields = []
        filterable_fields: list[str] = []
        searchable_fields: list[str] = []
        try:
            tbl = store.open_table(policy_id)
            schema: pa.Schema = tbl.schema
            schema_fields = [field_to_schema_field(f).model_dump() for f in schema]
            for f in schema:
                t = f.type
                if pa.types.is_string(t) or pa.types.is_integer(t) or pa.types.is_boolean(t):
                    filterable_fields.append(f.name)
            searchable_fields = [name for name in ("content_tokenized", "content", "vector") if name in schema.names]
        except Exception:
            # 元数据增强失败不应影响主流程
            schema_fields = []
            filterable_fields = []
            searchable_fields = ["content_tokenized", "vector"]

        return {
            "collection_id": collection_id,
            "n_documents": int(base.get("n_chunks", 0)),
            "n_original": int(base.get("n_original", 0)),
            "n_derived": int(base.get("n_derived", 0)),
            "dim": int(base.get("dim", 0)),
            "has_vector_index": bool(base.get("has_vector_index", False)),
            "has_fts_index": bool(base.get("has_fts_index", False)),
            "built_at": int(base.get("built_at", 0)),
            "schema_version": int(base.get("schema_version", 1)),
            "schema_fields": schema_fields,
            "filterable_fields": filterable_fields,
            "searchable_fields": searchable_fields,
        }


class DocumentStore:
    """通用文档读写。"""

    def upsert_legacy(self, policy_id: str, rows: list[ChunkRow], mode: str, expected_dim: int | None) -> dict:
        return store.upsert(policy_id, rows, mode, expected_dim)

    def upsert_documents(
        self,
        collection_id: str,
        documents: list[GenericDocumentInput],
        mode: str,
        expected_dim: int | None,
    ) -> dict:
        policy_id = collection_to_policy(collection_id)
        rows = [document_to_chunk_row(doc) for doc in documents]
        return store.upsert(policy_id, rows, mode, expected_dim)

    def list_legacy(
        self,
        policy_id: str,
        *,
        where: str | None,
        limit: int,
        include_content: bool,
    ) -> list[SearchHit]:
        return store.list_chunks(
            policy_id,
            where=where,
            limit=limit,
            include_content=include_content,
        )

    def list_documents(
        self,
        collection_id: str,
        *,
        where: str | None,
        limit: int,
        include_content: bool,
    ) -> list[GenericDocumentRecord]:
        policy_id = collection_to_policy(collection_id)
        hits = store.list_chunks(
            policy_id,
            where=where,
            limit=limit,
            include_content=include_content,
        )
        return [hit_to_generic_document(h) for h in hits]


class RetrievalEngine:
    """检索策略入口。当前默认 strategy=legacy_hybrid。"""

    SUPPORTED_STRATEGIES = ("legacy_hybrid",)

    def search_legacy(
        self,
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
        return store.hybrid_search(
            policy_id,
            query_tokenized=query_tokenized,
            query_vector=query_vector,
            top_n=top_n,
            top_m=top_m,
            rrf_k=rrf_k,
            where=where,
            include_content=include_content,
            include_derived=include_derived,
        )

    def search_documents(
        self,
        collection_id: str,
        *,
        query_tokenized: str,
        query_vector: list[float],
        top_n: int,
        top_m: int,
        rrf_k: int,
        where: str | None,
        include_content: bool,
        include_derived: bool,
        strategy: str = "legacy_hybrid",
    ) -> list[GenericDocumentRecord]:
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f"unsupported strategy: {strategy}")
        policy_id = collection_to_policy(collection_id)
        hits = self.search_legacy(
            policy_id,
            query_tokenized=query_tokenized,
            query_vector=query_vector,
            top_n=top_n,
            top_m=top_m,
            rrf_k=rrf_k,
            where=where,
            include_content=include_content,
            include_derived=include_derived,
        )
        return [hit_to_generic_document(h) for h in hits]


class GenericVectorService:
    """通用向量库服务编排器。"""

    def __init__(self) -> None:
        self.collections = CollectionStore()
        self.documents = DocumentStore()
        self.retrieval = RetrievalEngine()

    def capabilities(self) -> dict:
        settings = get_settings()
        return {
            "api_version": "0.2.0",
            "generic_api": bool(settings.enable_generic_api),
            "legacy_relations": bool(settings.enable_legacy_relations),
            "legacy_ui": bool(settings.enable_legacy_ui),
            "retrieval_modes": list(self.retrieval.SUPPORTED_STRATEGIES),
            "schema_version": int(store.SCHEMA_VERSION),
            "features": {
                "relations": bool(settings.enable_legacy_relations),
                "hybrid": True,
                "fts": True,
                "vector_index": True,
                "scalar_index": bool(settings.enable_scalar_index),
            },
        }


@lru_cache(maxsize=1)
def get_vector_service() -> GenericVectorService:
    return GenericVectorService()
