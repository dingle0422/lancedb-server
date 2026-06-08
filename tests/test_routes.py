"""端到端回归：覆盖 v1 兼容、v2 通用 API、能力协商与性能基线。"""

from __future__ import annotations

import tempfile
import time

import pytest
from fastapi.testclient import TestClient


def _build_client(monkeypatch, **env_overrides) -> TestClient:
    tmp = tempfile.mkdtemp(prefix="retrieval_test_")
    monkeypatch.setenv("STORE_DIR", tmp)
    monkeypatch.setenv("API_KEY", "")  # 关闭鉴权方便测试
    monkeypatch.setenv("ENABLE_SCALAR_INDEX", "0")  # 极小表跳过标量索引省时间
    monkeypatch.setenv("ENABLE_ASYNC_INDEXING", "0")  # 测试同步建索引，保证 upsert 后立即可检索
    monkeypatch.setenv("ENABLE_GENERIC_API", "1")
    monkeypatch.setenv("ENABLE_LEGACY_RELATIONS", "1")
    monkeypatch.setenv("ENABLE_LEGACY_UI", "1")
    monkeypatch.setenv("ENABLE_GENERIC_GRADIO", "1")
    monkeypatch.setenv("PROXY_ROOT_PATH", "")
    for k, v in env_overrides.items():
        monkeypatch.setenv(k, str(v))

    # config 和 store 都用 lru_cache / 模块级单例缓存，需要在 monkeypatch 后重新导入
    from importlib import reload

    from app import config as cfg_mod
    from app.generic import core as generic_core_mod
    from app import store as store_mod
    from app import main as main_mod

    reload(cfg_mod)
    reload(store_mod)
    reload(generic_core_mod)
    reload(main_mod)

    return TestClient(main_mod.app)


@pytest.fixture()
def client(monkeypatch):
    with _build_client(monkeypatch) as c:
        yield c


def _make_chunks() -> list[dict]:
    """三个 chunk：1 原始 + 2 派生。dim=4 简化测试。"""
    return [
        {
            "chunk_id": 1,
            "content": "农产品自产自销 免税 增值税",
            "content_tokenized": "农产品 自产自销 免税 增值税",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "heading_paths": [["2_涉税处理", "2.1_增值税"]],
            "directories": ["/k/2_涉税处理/2.1_增值税"],
            "kind": "original",
            "parent_chunk_index": -1,
            "derived_seq": 0,
            "relation_keys": [],
            "hop_depth": 0,
            "source": "",
            "clause_id": "",
            "built_at": 1700000000000,
        },
        {
            "chunk_id": 2,
            "content": "蔬菜主要品种目录 萝卜 胡萝卜 茄子",
            "content_tokenized": "蔬菜 主要 品种 目录 萝卜 胡萝卜 茄子",
            "vector": [0.0, 1.0, 0.0, 0.0],
            "heading_paths": [["附件", "蔬菜主要品种目录"]],
            "directories": ["/k/附件/蔬菜主要品种目录"],
            "kind": "derived",
            "parent_chunk_index": 1,
            "derived_seq": 1,
            "relation_keys": [{"policy_id": "OTHER_POL", "clause_id": "C-001"}],
            "hop_depth": 1,
            "source": "local",
            "clause_id": "C-001",
            "built_at": 1700000000000,
        },
        {
            "chunk_id": 3,
            "content": "鲜活肉蛋 流通环节 免税",
            "content_tokenized": "鲜活 肉蛋 流通 环节 免税",
            "vector": [0.0, 0.0, 1.0, 0.0],
            "heading_paths": [["附件", "鲜活肉蛋"]],
            "directories": ["/k/附件/鲜活肉蛋"],
            "kind": "derived",
            "parent_chunk_index": 1,
            "derived_seq": 2,
            "relation_keys": [{"policy_id": "OTHER_POL", "clause_id": "C-002"}],
            "hop_depth": 1,
            "source": "local",
            "clause_id": "C-002",
            "built_at": 1700000000000,
        },
    ]


def test_full_lifecycle(client: TestClient):
    pid = "test_pol_v1"

    # 1) upsert overwrite
    resp = client.post(
        f"/v1/policies/{pid}/chunks:upsert",
        json={"chunks": _make_chunks(), "mode": "overwrite", "expected_dim": 4},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["written"] == 3
    assert body["table_size"] == 3
    assert body["dim"] == 4

    # 2) meta
    meta = client.get(f"/v1/policies/{pid}/meta").json()
    assert meta["n_chunks"] == 3
    assert meta["n_original"] == 1
    assert meta["n_derived"] == 2

    # 3) search：BM25 命中"萝卜"应该返回 chunk_id=2
    resp = client.post(
        f"/v1/policies/{pid}/search",
        json={
            "query_tokenized": "萝卜",
            "query_vector": [0.0, 1.0, 0.0, 0.0],
            "top_n": 5,
            "top_m": 5,
            "include_content": True,
        },
    )
    assert resp.status_code == 200, resp.text
    hits = resp.json()["hits"]
    assert any(h["chunk_id"] == 2 for h in hits)

    # 4) expand：父=1 应该有 2 个派生
    resp = client.post(
        f"/v1/policies/{pid}/relations:expand",
        json={"chunk_id": 1, "include_content": False},
    )
    assert resp.status_code == 200
    children = resp.json()["chunks"]
    assert {c["chunk_id"] for c in children} == {2, 3}

    # 5) lookup-in-policy：找 OTHER_POL/C-001 的引用应只命中 chunk 2
    resp = client.get(
        f"/v1/policies/{pid}/relations:lookup",
        params={"target_policy_id": "OTHER_POL", "target_clause_id": "C-001"},
    )
    assert resp.status_code == 200
    chunks = resp.json()["chunks"]
    assert len(chunks) == 1 and chunks[0]["chunk_id"] == 2

    # 6) global dependents：OTHER_POL 应被 test_pol_v1 引用
    resp = client.get(
        "/v1/relations:lookup-dependents",
        params={"target_policy_id": "OTHER_POL"},
    )
    assert resp.status_code == 200
    deps = resp.json()["dependents"]
    assert any(d["source_policy_id"] == pid for d in deps)

    # 7) drop
    resp = client.delete(f"/v1/policies/{pid}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # 8) drop 后 meta 404
    assert client.get(f"/v1/policies/{pid}/meta").status_code == 404


def test_v1_contract_shape(client: TestClient):
    pid = "contract_pid"
    upsert_resp = client.post(
        f"/v1/policies/{pid}/chunks:upsert",
        json={"chunks": _make_chunks(), "mode": "overwrite", "expected_dim": 4},
    )
    assert upsert_resp.status_code == 200
    assert set(upsert_resp.json().keys()) == {"written", "table_size", "dim"}

    meta_resp = client.get(f"/v1/policies/{pid}/meta")
    assert meta_resp.status_code == 200
    meta = meta_resp.json()
    assert {"policy_id", "n_chunks", "n_original", "n_derived", "dim", "schema_version"} <= set(meta.keys())
    assert isinstance(meta.get("schema_fields", []), list)
    assert isinstance(meta.get("filterable_fields", []), list)
    assert isinstance(meta.get("searchable_fields", []), list)

    search_resp = client.post(
        f"/v1/policies/{pid}/search",
        json={
            "query_tokenized": "萝卜",
            "query_vector": [0.0, 1.0, 0.0, 0.0],
            "top_n": 5,
            "top_m": 5,
            "include_content": True,
        },
    )
    assert search_resp.status_code == 200
    body = search_resp.json()
    assert set(body.keys()) == {"hits"}
    if body["hits"]:
        must_have = {
            "chunk_id",
            "score",
            "content",
            "heading_paths",
            "directories",
            "kind",
            "parent_chunk_index",
            "derived_seq",
            "relation_keys",
            "hop_depth",
            "source",
            "clause_id",
        }
        assert must_have <= set(body["hits"][0].keys())


def test_capabilities_endpoint(client: TestClient):
    resp = client.get("/v1/capabilities")
    assert resp.status_code == 200
    caps = resp.json()
    assert caps["generic_api"] is True
    assert caps["legacy_relations"] is True
    assert "legacy_hybrid" in caps["retrieval_modes"]
    assert isinstance(caps["features"], dict)

    resp_v2 = client.get("/v2/capabilities")
    assert resp_v2.status_code == 200
    assert resp_v2.json()["schema_version"] == caps["schema_version"]


def test_v2_collection_flow(client: TestClient):
    cid = "generic_collection_1"
    docs = [
        {
            "document_id": 101,
            "content": "vector database general platform",
            "content_tokenized": "vector database general platform",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"kind": "original", "directories": ["docs"], "hop_depth": 0},
        },
        {
            "document_id": 102,
            "content": "hybrid retrieval with rrf",
            "content_tokenized": "hybrid retrieval with rrf",
            "vector": [0.0, 1.0, 0.0, 0.0],
            "metadata": {"kind": "derived", "parent_chunk_index": 101, "hop_depth": 1},
        },
    ]
    upsert = client.post(
        f"/v2/collections/{cid}/documents:upsert",
        json={"documents": docs, "mode": "overwrite", "expected_dim": 4},
    )
    assert upsert.status_code == 200, upsert.text
    assert upsert.json()["written"] == 2

    upsert_alias = client.post(
        "/v2/documents:upsert",
        json={"collection_id": cid, "documents": docs, "mode": "merge_by_chunk_id", "expected_dim": 4},
    )
    assert upsert_alias.status_code == 200

    lst = client.get("/v2/collections")
    assert lst.status_code == 200
    assert any(x["collection_id"] == cid for x in lst.json()["collections"])

    meta = client.get(f"/v2/collections/{cid}/meta")
    assert meta.status_code == 200
    assert meta.json()["n_documents"] == 2
    assert isinstance(meta.json()["schema_fields"], list)

    docs_resp = client.get(f"/v2/collections/{cid}/documents", params={"include_content": "true"})
    assert docs_resp.status_code == 200
    returned_docs = docs_resp.json()["documents"]
    assert any(d["document_id"] == 101 for d in returned_docs)

    docs_alias = client.get("/v2/documents", params={"collection_id": cid, "include_content": "true"})
    assert docs_alias.status_code == 200
    assert any(d["document_id"] == 101 for d in docs_alias.json()["documents"])

    search = client.post(
        f"/v2/collections/{cid}/search",
        json={
            "query_tokenized": "hybrid retrieval",
            "query_vector": [0.0, 1.0, 0.0, 0.0],
            "top_n": 5,
            "top_m": 5,
            "include_content": True,
            "strategy": "legacy_hybrid",
        },
    )
    assert search.status_code == 200, search.text
    assert any(h["document_id"] == 102 for h in search.json()["hits"])
    hit_102 = next(h for h in search.json()["hits"] if h["document_id"] == 102)
    assert "cosine_similarity" in hit_102
    assert "bm25_score" in hit_102

    search_alias = client.post(
        "/v2/search",
        json={
            "collection_id": cid,
            "query_tokenized": "hybrid retrieval",
            "query_vector": [0.0, 1.0, 0.0, 0.0],
            "top_n": 5,
            "top_m": 5,
            "include_content": True,
            "strategy": "legacy_hybrid",
        },
    )
    assert search_alias.status_code == 200
    assert any(h["document_id"] == 102 for h in search_alias.json()["hits"])


def test_v2_upsert_auto_embedding_when_vector_missing(client: TestClient, monkeypatch):
    import app.store as store_mod

    monkeypatch.setattr(
        store_mod,
        "embed_texts",
        lambda texts: ([[0.11, 0.22, 0.33, 0.44] for _ in texts], ""),
    )

    cid = "generic_collection_auto_embed_upsert"
    docs = [
        {
            "document_id": 501,
            "content": "auto embedding for missing vector",
            "content_tokenized": "auto embedding for missing vector",
            "metadata": {"kind": "original"},
        }
    ]
    upsert = client.post(
        f"/v2/collections/{cid}/documents:upsert",
        json={"documents": docs, "mode": "overwrite"},
    )
    assert upsert.status_code == 200, upsert.text
    assert upsert.json()["written"] == 1
    assert upsert.json()["dim"] == 4

    meta = client.get(f"/v2/collections/{cid}/meta")
    assert meta.status_code == 200, meta.text
    assert meta.json()["dim"] == 4


def test_v2_search_auto_embedding_when_query_vector_missing(client: TestClient, monkeypatch):
    import app.store as store_mod

    cid = "generic_collection_auto_embed_search"
    docs = [
        {
            "document_id": 601,
            "content": "alpha content",
            "content_tokenized": "alpha content",
            "vector": [0.9, 0.0, 0.0, 0.0],
            "metadata": {"kind": "original"},
        },
        {
            "document_id": 602,
            "content": "beta content",
            "content_tokenized": "beta content",
            "vector": [0.0, 1.0, 0.0, 0.0],
            "metadata": {"kind": "original"},
        },
    ]
    upsert = client.post(
        f"/v2/collections/{cid}/documents:upsert",
        json={"documents": docs, "mode": "overwrite", "expected_dim": 4},
    )
    assert upsert.status_code == 200, upsert.text

    monkeypatch.setattr(store_mod, "embed_query", lambda text: ([0.0, 1.0, 0.0, 0.0], ""))

    search = client.post(
        f"/v2/collections/{cid}/search",
        json={
            "query_tokenized": "semantic probe",
            "query_vector": [],
            "top_n": 5,
            "top_m": 0,
            "include_content": True,
            "strategy": "legacy_hybrid",
        },
    )
    assert search.status_code == 200, search.text
    assert any(h["document_id"] == 602 for h in search.json()["hits"])


def test_v2_search_returns_independent_bm25_and_cosine_scores(client: TestClient):
    cid = "generic_collection_dual_scores"
    docs = [
        {
            "document_id": 701,
            "content": "apple banana",
            "content_tokenized": "apple banana",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "metadata": {"kind": "original"},
        },
        {
            "document_id": 702,
            "content": "carrot radish",
            "content_tokenized": "carrot radish",
            "vector": [0.0, 1.0, 0.0, 0.0],
            "metadata": {"kind": "original"},
        },
    ]
    upsert = client.post(
        f"/v2/collections/{cid}/documents:upsert",
        json={"documents": docs, "mode": "overwrite", "expected_dim": 4},
    )
    assert upsert.status_code == 200, upsert.text

    pure_vec = client.post(
        f"/v2/collections/{cid}/search",
        json={
            "query_tokenized": "",
            "query_vector": [0.0, 1.0, 0.0, 0.0],
            "top_n": 5,
            "top_m": 0,
            "include_content": True,
            "strategy": "legacy_hybrid",
        },
    )
    assert pure_vec.status_code == 200, pure_vec.text
    vec_hit = next(h for h in pure_vec.json()["hits"] if h["document_id"] == 702)
    assert vec_hit["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert vec_hit["bm25_score"] is None

    bm25_only = client.post(
        f"/v2/collections/{cid}/search",
        json={
            "query_tokenized": "radish",
            "query_vector": [],
            "top_n": 0,
            "top_m": 5,
            "include_content": True,
            "strategy": "legacy_hybrid",
        },
    )
    assert bm25_only.status_code == 200, bm25_only.text
    bm25_hit = next(h for h in bm25_only.json()["hits"] if h["document_id"] == 702)
    assert bm25_hit["bm25_score"] is not None
    assert bm25_hit["cosine_similarity"] is None


def test_v2_merge_by_chunk_id_partial_patch_keeps_content_and_vector(client: TestClient):
    cid = "generic_collection_safe_partial_merge"
    base_doc = {
        "document_id": 801,
        "content": "preserve original content",
        "content_tokenized": "preserve original content",
        "vector": [0.0, 0.0, 1.0, 0.0],
        "metadata": {"kind": "original", "attempts": 1},
    }
    upsert = client.post(
        f"/v2/collections/{cid}/documents:upsert",
        json={"documents": [base_doc], "mode": "overwrite", "expected_dim": 4},
    )
    assert upsert.status_code == 200, upsert.text

    patch_doc = {
        "document_id": 801,
        "content": "",
        "content_tokenized": "",
        "vector": [],
        "metadata": {"attempts": 2, "tombstone": True},
    }
    patch_resp = client.post(
        f"/v2/collections/{cid}/documents:upsert",
        json={"documents": [patch_doc], "mode": "merge_by_chunk_id", "expected_dim": 4},
    )
    assert patch_resp.status_code == 200, patch_resp.text

    docs_resp = client.get(f"/v2/collections/{cid}/documents", params={"include_content": "true"})
    assert docs_resp.status_code == 200, docs_resp.text
    row = next(d for d in docs_resp.json()["documents"] if d["document_id"] == 801)
    assert row["content"] == base_doc["content"]
    assert row["metadata"]["attempts"] == 2
    assert row["metadata"]["tombstone"] is True

    vec_search = client.post(
        f"/v2/collections/{cid}/search",
        json={
            "query_tokenized": "",
            "query_vector": [0.0, 0.0, 1.0, 0.0],
            "top_n": 5,
            "top_m": 0,
            "include_content": True,
            "strategy": "legacy_hybrid",
        },
    )
    assert vec_search.status_code == 200, vec_search.text
    assert any(h["document_id"] == 801 for h in vec_search.json()["hits"])


def test_v2_metadata_roundtrip_accepts_arbitrary_shapes(client: TestClient):
    cid = "generic_collection_metadata_shape"
    docs = [
        {
            "document_id": 201,
            "content": "metadata arbitrary fields",
            "content_tokenized": "metadata arbitrary fields",
            "vector": [0.2, 0.1, 0.3, 0.4],
            "metadata": {
                "kind": "original",
                "directories": ["docs"],
                "hop_depth": "not-a-number",
                "nested": {"a": [1, {"b": True}]},
                "custom_null": None,
                "custom_float": 3.14,
            },
        },
        {
            "document_id": 202,
            "content": "metadata list payload",
            "content_tokenized": "metadata list payload",
            "vector": [0.0, 0.1, 0.0, 0.2],
            "metadata": ["free", {"shape": "list"}],
        },
    ]
    upsert = client.post(
        f"/v2/collections/{cid}/documents:upsert",
        json={"documents": docs, "mode": "overwrite", "expected_dim": 4},
    )
    assert upsert.status_code == 200, upsert.text
    assert upsert.json()["written"] == 2

    docs_resp = client.get(f"/v2/collections/{cid}/documents", params={"include_content": "true"})
    assert docs_resp.status_code == 200, docs_resp.text
    by_id = {d["document_id"]: d for d in docs_resp.json()["documents"]}
    assert by_id[201]["metadata"] == docs[0]["metadata"]
    assert by_id[202]["metadata"] == docs[1]["metadata"]

    search_resp = client.post(
        f"/v2/collections/{cid}/search",
        json={
            "query_tokenized": "metadata",
            "query_vector": [0.2, 0.1, 0.3, 0.4],
            "top_n": 5,
            "top_m": 5,
            "include_content": True,
            "strategy": "legacy_hybrid",
        },
    )
    assert search_resp.status_code == 200, search_resp.text
    search_by_id = {d["document_id"]: d for d in search_resp.json()["hits"]}
    assert search_by_id[201]["metadata"] == docs[0]["metadata"]


def test_v2_metadata_flattened_columns_support_where_filter(client: TestClient):
    cid = "generic_collection_metadata_filter"
    docs = [
        {
            "document_id": 301,
            "content": "document for acme",
            "content_tokenized": "document for acme",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"tenant": "acme", "risk": {"level": 2}, "active": True},
        },
        {
            "document_id": 302,
            "content": "document for beta",
            "content_tokenized": "document for beta",
            "vector": [0.4, 0.3, 0.2, 0.1],
            "metadata": {"tenant": "beta", "risk": {"level": 1}, "active": False},
        },
    ]
    upsert = client.post(
        f"/v2/collections/{cid}/documents:upsert",
        json={"documents": docs, "mode": "overwrite", "expected_dim": 4},
    )
    assert upsert.status_code == 200, upsert.text

    meta = client.get(f"/v2/collections/{cid}/meta")
    assert meta.status_code == 200, meta.text
    filterable = meta.json()["filterable_fields"]

    tenant_col = next((x for x in filterable if x.startswith("md_tenant_")), None)
    level_col = next((x for x in filterable if x.startswith("md_risk__level_")), None)
    active_col = next((x for x in filterable if x.startswith("md_active_")), None)
    assert tenant_col is not None
    assert level_col is not None
    assert active_col is not None

    docs_resp = client.get(
        f"/v2/collections/{cid}/documents",
        params={"include_content": "true", "where": f"{tenant_col} = 'acme' AND {level_col} = 2"},
    )
    assert docs_resp.status_code == 200, docs_resp.text
    returned = docs_resp.json()["documents"]
    assert {x["document_id"] for x in returned} == {301}

    search_resp = client.post(
        f"/v2/collections/{cid}/search",
        json={
            "query_tokenized": "document",
            "query_vector": [0.1, 0.2, 0.3, 0.4],
            "top_n": 5,
            "top_m": 5,
            "where": f"{active_col} = true",
            "include_content": True,
            "strategy": "legacy_hybrid",
        },
    )
    assert search_resp.status_code == 200, search_resp.text
    assert {x["document_id"] for x in search_resp.json()["hits"]} == {301}

    append_resp = client.post(
        f"/v2/collections/{cid}/documents:upsert",
        json={
            "documents": [
                {
                    "document_id": 303,
                    "content": "document for acme region",
                    "content_tokenized": "document for acme region",
                    "vector": [0.2, 0.3, 0.4, 0.5],
                    "metadata": {"tenant": "acme", "region": "cn"},
                }
            ],
            "mode": "append",
            "expected_dim": 4,
        },
    )
    assert append_resp.status_code == 200, append_resp.text

    meta_after_append = client.get(f"/v2/collections/{cid}/meta")
    assert meta_after_append.status_code == 200, meta_after_append.text
    filterable_after = meta_after_append.json()["filterable_fields"]
    region_col = next((x for x in filterable_after if x.startswith("md_region_")), None)
    assert region_col is not None

    region_docs = client.get(
        f"/v2/collections/{cid}/documents",
        params={"include_content": "true", "where": f"{region_col} = 'cn'"},
    )
    assert region_docs.status_code == 200, region_docs.text
    assert {x["document_id"] for x in region_docs.json()["documents"]} == {303}


def test_v2_collection_overwrite_by_prefix(client: TestClient):
    def _upsert(collection_id: str, document_id: int):
        payload = {
            "collection_id": collection_id,
            "documents": [
                {
                    "document_id": document_id,
                    "content": f"doc-{document_id}",
                    "content_tokenized": f"doc {document_id}",
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "metadata": {"kind": "original"},
                }
            ],
            "mode": "overwrite",
            "expected_dim": 4,
        }
        resp = client.post("/v2/documents:upsert", json=payload)
        assert resp.status_code == 200, resp.text

    _upsert("invoice_202405", 1)
    _upsert("invoice_202406", 2)
    _upsert("notice_202406", 3)

    overwrite_resp = client.post(
        "/v2/collectionOverwriteByPrefix",
        json={
            "collection_id": "invoice_202407",
            "documents": [
                {
                    "document_id": 4,
                    "content": "latest invoice",
                    "content_tokenized": "latest invoice",
                    "vector": [0.9, 0.1, 0.2, 0.3],
                    "metadata": {"kind": "original"},
                }
            ],
            "expected_dim": 4,
        },
    )
    assert overwrite_resp.status_code == 200, overwrite_resp.text
    body = overwrite_resp.json()
    assert body["written"] == 1
    assert set(body["dropped_collections"]) == {"invoice_202405", "invoice_202406"}

    lst = client.get("/v2/collections")
    assert lst.status_code == 200, lst.text
    remaining = {x["collection_id"] for x in lst.json()["collections"]}
    assert "invoice_202407" in remaining
    assert "invoice_202405" not in remaining
    assert "invoice_202406" not in remaining
    assert "notice_202406" in remaining


def test_disable_relations_flag(monkeypatch):
    with _build_client(monkeypatch, ENABLE_LEGACY_RELATIONS=0) as c:
        resp = c.get(
            "/v1/relations:lookup-dependents",
            params={"target_policy_id": "OTHER_POL"},
        )
        assert resp.status_code == 404


def test_search_performance_baseline(client: TestClient):
    pid = "perf_pid"
    client.post(
        f"/v1/policies/{pid}/chunks:upsert",
        json={"chunks": _make_chunks(), "mode": "overwrite", "expected_dim": 4},
    )

    elapsed = []
    for _ in range(8):
        t0 = time.perf_counter()
        resp = client.post(
            f"/v1/policies/{pid}/search",
            json={
                "query_tokenized": "萝卜 免税",
                "query_vector": [0.0, 1.0, 0.0, 0.0],
                "top_n": 5,
                "top_m": 5,
                "include_content": True,
            },
        )
        assert resp.status_code == 200
        elapsed.append(time.perf_counter() - t0)

    p95_like = sorted(elapsed)[int(len(elapsed) * 0.95) - 1]
    assert p95_like < 2.0


def test_async_indexing_eventually_builds_fts(monkeypatch):
    """开启异步建索引：upsert 立即返回，后台最终把 FTS 索引建好且检索可命中。"""

    with _build_client(monkeypatch, ENABLE_ASYNC_INDEXING=1) as c:
        pid = "async_idx_pid"
        resp = c.post(
            f"/v1/policies/{pid}/chunks:upsert",
            json={"chunks": _make_chunks(), "mode": "overwrite", "expected_dim": 4},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["written"] == 3

        # 后台线程建索引：轮询 meta 等到 FTS 就绪（最多 ~5s）。
        has_fts = False
        for _ in range(50):
            meta = c.get(f"/v1/policies/{pid}/meta").json()
            if meta.get("has_fts_index"):
                has_fts = True
                break
            time.sleep(0.1)
        assert has_fts, "后台建索引未在预期时间内完成"

        search = c.post(
            f"/v1/policies/{pid}/search",
            json={
                "query_tokenized": "萝卜",
                "query_vector": [0.0, 1.0, 0.0, 0.0],
                "top_n": 5,
                "top_m": 5,
                "include_content": True,
            },
        )
        assert search.status_code == 200, search.text
        assert any(h["chunk_id"] == 2 for h in search.json()["hits"])


def test_append_is_idempotent(client: TestClient):
    """append 模式按 chunk_id 幂等：同一文档重复 upsert 不产生重复行，且内容被更新。"""

    cid = "idem_collection"

    def _upsert(content: str) -> dict:
        resp = client.post(
            f"/v2/collections/{cid}/documents:upsert",
            json={
                "documents": [
                    {
                        "document_id": 7,
                        "content": content,
                        "content_tokenized": content,
                        "vector": [0.1, 0.2, 0.3, 0.4],
                        "metadata": {"kind": "original"},
                    }
                ],
                "mode": "append",
                "expected_dim": 4,
            },
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    first = _upsert("第一版内容")
    assert first["written"] == 1
    assert first["table_size"] == 1

    # 模拟客户端超时重试：同一 document_id 再发两次
    second = _upsert("第二版内容")
    third = _upsert("第二版内容")
    # 关键断言：表里始终只有 1 行，没有因重试而重复
    assert second["table_size"] == 1
    assert third["table_size"] == 1

    docs = client.get(
        f"/v2/collections/{cid}/documents",
        params={"include_content": "true"},
    )
    assert docs.status_code == 200, docs.text
    body = docs.json()["documents"]
    assert len(body) == 1
    assert body[0]["document_id"] == 7
    assert body[0]["content"] == "第二版内容"  # 内容被更新为最新版


def test_upsert_dedupes_within_request(client: TestClient):
    """同一请求内出现重复 chunk_id 时去重，保留最后一条。"""

    pid = "dedupe_pid"
    dup_chunk = {
        "chunk_id": 1,
        "content": "旧",
        "content_tokenized": "旧",
        "vector": [1.0, 0.0, 0.0, 0.0],
        "kind": "original",
        "parent_chunk_index": -1,
        "derived_seq": 0,
        "relation_keys": [],
        "hop_depth": 0,
        "source": "",
        "clause_id": "",
        "built_at": 1700000000000,
    }
    dup_chunk_new = {**dup_chunk, "content": "新", "content_tokenized": "新"}

    resp = client.post(
        f"/v1/policies/{pid}/chunks:upsert",
        json={"chunks": [dup_chunk, dup_chunk_new], "mode": "overwrite", "expected_dim": 4},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["written"] == 1
    assert body["table_size"] == 1


def test_frontend_routes(client: TestClient):
    # 旧版静态 UI 已移除
    assert client.get("/ui").status_code == 404

    # 保留 policy gradio
    legacy = client.get("/gradio")
    assert legacy.status_code in (200, 307)

    # 新增 generic gradio
    generic = client.get("/gradio-generic")
    assert generic.status_code in (200, 307)


def _deps_map(resp) -> dict[str, int]:
    assert resp.status_code == 200, resp.text
    return {d["source_policy_id"]: d["n_hits"] for d in resp.json()["dependents"]}


def test_lookup_dependents_reverse_index(client: TestClient):
    """反向索引点查：target-only / clause-specific / 不存在的 clause 都要正确。"""

    pid = "src_pol_idx"
    client.post(
        f"/v1/policies/{pid}/chunks:upsert",
        json={"chunks": _make_chunks(), "mode": "overwrite", "expected_dim": 4},
    )

    # chunk2 -> OTHER_POL/C-001, chunk3 -> OTHER_POL/C-002：任意 clause 命中 2 行
    any_deps = _deps_map(
        client.get("/v1/relations:lookup-dependents", params={"target_policy_id": "OTHER_POL"})
    )
    assert any_deps.get(pid) == 2

    c1 = _deps_map(
        client.get(
            "/v1/relations:lookup-dependents",
            params={"target_policy_id": "OTHER_POL", "target_clause_id": "C-001"},
        )
    )
    assert c1.get(pid) == 1

    c2 = _deps_map(
        client.get(
            "/v1/relations:lookup-dependents",
            params={"target_policy_id": "OTHER_POL", "target_clause_id": "C-002"},
        )
    )
    assert c2.get(pid) == 1

    nope = _deps_map(
        client.get(
            "/v1/relations:lookup-dependents",
            params={"target_policy_id": "OTHER_POL", "target_clause_id": "NOPE"},
        )
    )
    assert nope == {}

    # 自反不计：以 source 自身为 target 应为空
    self_ref = _deps_map(
        client.get("/v1/relations:lookup-dependents", params={"target_policy_id": pid})
    )
    assert pid not in self_ref


def test_lookup_dependents_index_matches_scan(monkeypatch):
    """索引路径与全表扫描兜底路径结果一致。"""

    def _collect(client: TestClient, pid: str) -> dict[str, int]:
        client.post(
            f"/v1/policies/{pid}/chunks:upsert",
            json={"chunks": _make_chunks(), "mode": "overwrite", "expected_dim": 4},
        )
        out = {}
        for clause in (None, "C-001", "C-002", "NOPE"):
            params = {"target_policy_id": "OTHER_POL"}
            if clause is not None:
                params["target_clause_id"] = clause
            out[str(clause)] = _deps_map(
                client.get("/v1/relations:lookup-dependents", params=params)
            ).get(pid, 0)
        return out

    with _build_client(monkeypatch, ENABLE_RELATION_INDEX=1) as c:
        indexed = _collect(c, "p_idx")
    with _build_client(monkeypatch, ENABLE_RELATION_INDEX=0) as c:
        scanned = _collect(c, "p_idx")

    assert indexed == scanned
    assert indexed == {"None": 2, "C-001": 1, "C-002": 1, "NOPE": 0}


def test_drop_removes_source_from_index(client: TestClient):
    """drop policy 后，它作为 source 的依赖应从反向索引消失。"""

    pid = "drop_src_idx"
    client.post(
        f"/v1/policies/{pid}/chunks:upsert",
        json={"chunks": _make_chunks(), "mode": "overwrite", "expected_dim": 4},
    )
    before = _deps_map(
        client.get("/v1/relations:lookup-dependents", params={"target_policy_id": "OTHER_POL"})
    )
    assert before.get(pid) == 2

    assert client.delete(f"/v1/policies/{pid}").status_code == 200

    after = _deps_map(
        client.get("/v1/relations:lookup-dependents", params={"target_policy_id": "OTHER_POL"})
    )
    assert pid not in after


def test_reindex_endpoint_rebuilds(client: TestClient):
    """reindex 端点全量重建后仍能正确点查。"""

    pid = "reindex_src"
    client.post(
        f"/v1/policies/{pid}/chunks:upsert",
        json={"chunks": _make_chunks(), "mode": "overwrite", "expected_dim": 4},
    )

    resp = client.post("/v1/relations:reindex")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["sources"] >= 1

    deps = _deps_map(
        client.get("/v1/relations:lookup-dependents", params={"target_policy_id": "OTHER_POL"})
    )
    assert deps.get(pid) == 2
