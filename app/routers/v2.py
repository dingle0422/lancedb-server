"""通用向量库 API（v2）。"""

from __future__ import annotations

import logging
from functools import partial

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..config import get_settings
from ..deps import require_api_key
from ..generic import get_vector_service
from ..schema_generic import (
    CapabilitiesResponse,
    CollectionOverwriteByPrefixRequest,
    CollectionOverwriteByPrefixResponse,
    CollectionListItem,
    CollectionListResponse,
    CollectionMeta,
    DocumentListResponse,
    DocumentUpsertRequest,
    DocumentUpsertRequestWithCollection,
    DocumentUpsertResponse,
    SearchRequestV2,
    SearchRequestV2WithCollection,
    SearchResponseV2,
)

logger = logging.getLogger("app.routers.v2")

router = APIRouter(
    prefix="/v2",
    tags=["generic-v2"],
    dependencies=[Depends(require_api_key)],
)


def _log_upsert_diag(request: Request | None, collection_id: str, body: DocumentUpsertRequest) -> None:
    """临时诊断日志：记录是谁、向哪个 collection、upsert 了哪些文档。

    用于排查“客户端未显式 upsert，但日志里仍出现 upsert”的来源。定位完成后可整体删除。
    """

    try:
        docs = body.documents
        doc_ids = [getattr(d, "document_id", None) for d in docs]
        no_vector = sum(1 for d in docs if not getattr(d, "vector", None))
        meta_keys: set[str] = set()
        for d in docs:
            meta = getattr(d, "metadata", None)
            if isinstance(meta, dict):
                meta_keys.update(map(str, meta.keys()))

        client_host = request.client.host if (request and request.client) else "?"
        headers = request.headers if request else {}
        logger.info(
            "[UpsertDiag] collection=%s mode=%s n=%d no_vector=%d doc_ids=%s "
            "meta_keys=%s client=%s xff=%s xri=%s ua=%s referer=%s",
            collection_id,
            body.mode,
            len(docs),
            no_vector,
            doc_ids[:50],
            sorted(meta_keys),
            client_host,
            headers.get("x-forwarded-for"),
            headers.get("x-real-ip"),
            headers.get("user-agent"),
            headers.get("referer"),
        )
    except Exception:  # noqa: BLE001 - 诊断日志绝不能影响正常写入
        logger.exception("[UpsertDiag] 记录诊断日志失败（忽略，不影响 upsert）")


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities_v2() -> CapabilitiesResponse:
    svc = get_vector_service()
    data = await anyio.to_thread.run_sync(svc.capabilities)
    return CapabilitiesResponse(**data)


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections() -> CollectionListResponse:
    svc = get_vector_service()
    rows = await anyio.to_thread.run_sync(svc.collections.list_collections)
    return CollectionListResponse(
        collections=[
            CollectionListItem(collection_id=cid, n_documents=n_docs, dim=dim)
            for cid, n_docs, dim in rows
        ]
    )


@router.get("/collections/{collection_id}/meta", response_model=CollectionMeta)
async def get_collection_meta(collection_id: str) -> CollectionMeta:
    svc = get_vector_service()
    data = await anyio.to_thread.run_sync(svc.collections.collection_meta, collection_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"collection not found: {collection_id}")
    return CollectionMeta(**data)


@router.delete("/collections/{collection_id}")
async def drop_collection(collection_id: str) -> dict:
    svc = get_vector_service()
    ok = await anyio.to_thread.run_sync(svc.collections.drop_collection, collection_id)
    return {"ok": bool(ok)}


@router.get("/collections/{collection_id}/documents", response_model=DocumentListResponse)
async def list_documents(
    collection_id: str,
    where: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=100000),
    offset: int = Query(default=0, ge=0),
    include_content: bool = Query(default=False),
) -> DocumentListResponse:
    svc = get_vector_service()
    fn = partial(
        svc.documents.list_documents,
        collection_id,
        where=where,
        limit=limit,
        include_content=include_content,
        offset=offset,
    )
    docs = await anyio.to_thread.run_sync(fn)
    return DocumentListResponse(documents=docs)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents_alias(
    collection_id: str = Query(...),
    where: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=100000),
    offset: int = Query(default=0, ge=0),
    include_content: bool = Query(default=False),
) -> DocumentListResponse:
    return await list_documents(
        collection_id=collection_id,
        where=where,
        limit=limit,
        offset=offset,
        include_content=include_content,
    )


@router.post(
    "/collections/{collection_id}/documents:upsert",
    response_model=DocumentUpsertResponse,
)
async def upsert_documents(
    collection_id: str,
    body: DocumentUpsertRequest,
    request: Request,
) -> DocumentUpsertResponse:
    svc = get_vector_service()
    _log_upsert_diag(request, collection_id, body)
    try:
        result = await anyio.to_thread.run_sync(
            svc.documents.upsert_documents,
            collection_id,
            body.documents,
            body.mode,
            body.expected_dim,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upsert failed: {e}")
    return DocumentUpsertResponse(**result)


@router.post("/documents:upsert", response_model=DocumentUpsertResponse)
async def upsert_documents_alias(
    body: DocumentUpsertRequestWithCollection,
    request: Request,
) -> DocumentUpsertResponse:
    return await upsert_documents(
        collection_id=body.collection_id,
        body=DocumentUpsertRequest(
            documents=body.documents,
            mode=body.mode,
            expected_dim=body.expected_dim,
        ),
        request=request,
    )


@router.post(
    "/collectionOverwriteByPrefix",
    response_model=CollectionOverwriteByPrefixResponse,
)
async def collection_overwrite_by_prefix(
    body: CollectionOverwriteByPrefixRequest,
) -> CollectionOverwriteByPrefixResponse:
    svc = get_vector_service()

    def _run() -> dict:
        dropped = svc.collections.drop_collections_with_same_prefix(body.collection_id)
        result = svc.documents.upsert_documents(
            body.collection_id,
            body.documents,
            "overwrite",
            body.expected_dim,
        )
        return {**result, "dropped_collections": dropped}

    try:
        result = await anyio.to_thread.run_sync(_run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"collection overwrite by prefix failed: {e}")
    return CollectionOverwriteByPrefixResponse(**result)


@router.post("/collections/{collection_id}/search", response_model=SearchResponseV2)
async def search_documents(collection_id: str, body: SearchRequestV2) -> SearchResponseV2:
    svc = get_vector_service()
    settings = get_settings()
    rrf_k = body.rrf_k if body.rrf_k is not None else settings.rrf_k
    fn = partial(
        svc.retrieval.search_documents,
        collection_id,
        query_tokenized=body.query_tokenized,
        query_vector=body.query_vector,
        top_n=body.top_n,
        top_m=body.top_m,
        rrf_k=rrf_k,
        where=body.where,
        include_content=body.include_content,
        include_derived=body.include_derived,
        strategy=body.strategy,
    )
    try:
        hits = await anyio.to_thread.run_sync(fn)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"collection not indexed: {collection_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return SearchResponseV2(hits=hits)


@router.post("/search", response_model=SearchResponseV2)
async def search_documents_alias(body: SearchRequestV2WithCollection) -> SearchResponseV2:
    return await search_documents(
        collection_id=body.collection_id,
        body=SearchRequestV2(
            query_tokenized=body.query_tokenized,
            query_vector=body.query_vector,
            top_n=body.top_n,
            top_m=body.top_m,
            rrf_k=body.rrf_k,
            where=body.where,
            include_content=body.include_content,
            include_derived=body.include_derived,
            strategy=body.strategy,
        ),
    )
