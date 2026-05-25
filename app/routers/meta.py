"""表 meta / 列出所有 policy。"""

from __future__ import annotations

import logging

import anyio
from fastapi import APIRouter, Depends, HTTPException

from ..generic import get_vector_service
from ..deps import require_api_key
from ..schema import (
    CapabilitiesResponse,
    PolicyListItem,
    PolicyListResponse,
    TableMeta,
)

router = APIRouter(tags=["meta"], dependencies=[Depends(require_api_key)])

logger = logging.getLogger(__name__)


@router.get("/v1/policies", response_model=PolicyListResponse)
async def list_policies() -> PolicyListResponse:
    svc = get_vector_service()
    raw = await anyio.to_thread.run_sync(svc.collections.list_collections)
    return PolicyListResponse(
        policies=[PolicyListItem(policy_id=pid, n_chunks=n, dim=dim) for pid, n, dim in raw]
    )


@router.get("/v1/policies/{policy_id}/meta", response_model=TableMeta)
async def get_meta(policy_id: str) -> TableMeta:
    svc = get_vector_service()
    data = await anyio.to_thread.run_sync(svc.collections.collection_meta, policy_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"policy not found: {policy_id}")
    return TableMeta(
        policy_id=policy_id,
        n_chunks=int(data.get("n_documents", 0)),
        n_original=int(data.get("n_original", 0)),
        n_derived=int(data.get("n_derived", 0)),
        dim=int(data.get("dim", 0)),
        has_vector_index=bool(data.get("has_vector_index", False)),
        has_fts_index=bool(data.get("has_fts_index", False)),
        built_at=int(data.get("built_at", 0)),
        schema_version=int(data.get("schema_version", 1)),
        schema_fields=list(data.get("schema_fields", [])),
        filterable_fields=list(data.get("filterable_fields", [])),
        searchable_fields=list(data.get("searchable_fields", [])),
    )


@router.get("/v1/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities() -> CapabilitiesResponse:
    svc = get_vector_service()
    data = await anyio.to_thread.run_sync(svc.capabilities)
    return CapabilitiesResponse(**data)
