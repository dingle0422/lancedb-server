"""通用向量库 API（v2）的数据模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .schema import UpsertMode


class SchemaField(BaseModel):
    name: str
    type: str
    nullable: bool


class CollectionListItem(BaseModel):
    collection_id: str
    n_documents: int
    dim: int


class CollectionListResponse(BaseModel):
    collections: list[CollectionListItem]


class CollectionMeta(BaseModel):
    collection_id: str
    n_documents: int
    n_original: int
    n_derived: int
    dim: int
    has_vector_index: bool
    has_fts_index: bool
    built_at: int
    schema_version: int
    schema_fields: list[SchemaField] = Field(default_factory=list)
    filterable_fields: list[str] = Field(default_factory=list)
    searchable_fields: list[str] = Field(default_factory=list)


class GenericDocumentInput(BaseModel):
    document_id: int
    content: str
    content_tokenized: str = ""
    vector: list[float] = Field(default_factory=list)
    metadata: Any = Field(default_factory=dict)


class GenericDocumentRecord(BaseModel):
    document_id: int
    score: float = 0.0
    cosine_similarity: float | None = None
    bm25_score: float | None = None
    content: str | None = None
    metadata: Any = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    documents: list[GenericDocumentRecord]


class DocumentUpsertRequest(BaseModel):
    documents: list[GenericDocumentInput] = Field(..., min_length=1)
    mode: UpsertMode = "overwrite"
    expected_dim: int | None = None


class DocumentUpsertResponse(BaseModel):
    written: int
    table_size: int
    dim: int


class SearchRequestV2(BaseModel):
    query_tokenized: str = ""
    query_vector: list[float] = Field(default_factory=list)
    top_n: int = 20
    top_m: int = 20
    rrf_k: int | None = None
    where: str | None = None
    include_content: bool = True
    include_derived: bool = True
    strategy: Literal["legacy_hybrid"] = "legacy_hybrid"


class SearchRequestV2WithCollection(SearchRequestV2):
    collection_id: str


class SearchResponseV2(BaseModel):
    hits: list[GenericDocumentRecord]


class DocumentUpsertRequestWithCollection(DocumentUpsertRequest):
    collection_id: str


class CollectionOverwriteByPrefixRequest(BaseModel):
    collection_id: str
    documents: list[GenericDocumentInput] = Field(..., min_length=1)
    expected_dim: int | None = None


class CollectionOverwriteByPrefixResponse(DocumentUpsertResponse):
    dropped_collections: list[str] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    api_version: str
    generic_api: bool
    legacy_relations: bool
    legacy_ui: bool
    retrieval_modes: list[str]
    schema_version: int
    features: dict[str, bool]
