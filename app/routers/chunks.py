"""chunks 写入 / 列表 / 删除。"""

from __future__ import annotations

import logging
from functools import partial

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import require_api_key
from ..generic import get_vector_service
from ..schema import (
    SearchHit,
    UpsertRequest,
    UpsertResponse,
)

router = APIRouter(
    prefix="/v1/policies/{policy_id}",
    tags=["chunks"],
    dependencies=[Depends(require_api_key)],
)

logger = logging.getLogger(__name__)


@router.post("/chunks:upsert", response_model=UpsertResponse)
async def upsert_chunks(policy_id: str, body: UpsertRequest) -> UpsertResponse:
    svc = get_vector_service()
    try:
        result = await anyio.to_thread.run_sync(
            svc.documents.upsert_legacy,
            policy_id,
            body.chunks,
            body.mode,
            body.expected_dim,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("[chunks] upsert 失败 policy=%s", policy_id)
        raise HTTPException(status_code=500, detail=f"upsert failed: {e}")
    return UpsertResponse(**result)


@router.get("/chunks", response_model=list[SearchHit])
async def list_chunks(
    policy_id: str,
    where: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=100000),
    include_content: bool = Query(default=False),
) -> list[SearchHit]:
    svc = get_vector_service()
    try:
        fn = partial(
            svc.documents.list_legacy,
            policy_id,
            where=where,
            limit=limit,
            include_content=include_content,
        )
        hits = await anyio.to_thread.run_sync(
            fn,
        )
    except Exception as e:
        logger.exception("[chunks] list 失败 policy=%s", policy_id)
        raise HTTPException(status_code=500, detail=str(e))
    return hits


@router.delete("")
async def drop_policy(policy_id: str) -> dict:
    svc = get_vector_service()
    ok = await anyio.to_thread.run_sync(svc.collections.drop_collection, policy_id)
    return {"ok": bool(ok)}
