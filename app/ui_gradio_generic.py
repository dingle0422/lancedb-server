"""通用知识（v2 collection/document）Gradio 前端。

同进程挂载到 FastAPI:
    /gradio-generic
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import gradio as gr

from .generic import get_vector_service

logger = logging.getLogger(__name__)
_SVC = get_vector_service()

_TOKEN_SPLIT_RE = re.compile(r"[\s\W_]+")
_BARS = "▁▂▃▄▅▆▇█"
_PREVIEW_MAX = 64

_DEFAULT_EMBEDDING_BASE_URL = ""
_DEFAULT_EMBEDDING_MODEL = ""


def _tokenize_query(text: str) -> str:
    if not text or not text.strip():
        return ""
    return " ".join(tok for tok in _TOKEN_SPLIT_RE.split(text.lower()) if tok)


def _embed_query(text: str) -> tuple[list[float], str]:
    if not text or not text.strip():
        return [], "查询文本为空"
    base_url = os.environ.get("EMBEDDING_BASE_URL", _DEFAULT_EMBEDDING_BASE_URL).rstrip("/")
    model = os.environ.get("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
    if not base_url or not model:
        return [], "未配置 EMBEDDING_BASE_URL / EMBEDDING_MODEL"

    api_key = os.environ.get("EMBEDDING_API_KEY") or ""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        import httpx

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{base_url}/embeddings",
                headers=headers,
                json={"model": model, "input": text},
            )
        if resp.status_code != 200:
            return [], f"embedding 服务返回 {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        vec = data["data"][0]["embedding"]
        return [float(x) for x in vec], ""
    except Exception as e:  # noqa: BLE001
        return [], f"调用 embedding 服务失败: {e}"


def _vector_sparkline(vec: list[float] | None) -> str:
    if not vec:
        return ""
    n = len(vec)
    if n > _PREVIEW_MAX:
        step = n / _PREVIEW_MAX
        sampled = [vec[int(i * step)] for i in range(_PREVIEW_MAX)]
    else:
        sampled = list(vec)
    lo = min(sampled)
    hi = max(sampled)
    rng = hi - lo or 1.0
    chars = []
    for v in sampled:
        idx = int((v - lo) / rng * (len(_BARS) - 1))
        chars.append(_BARS[max(0, min(len(_BARS) - 1, idx))])
    return f"[{n}d] " + "".join(chars)


def _dump(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _list_collections() -> list[tuple[str, int, int]]:
    return _SVC.collections.list_collections()


def _refresh_collection_list():
    items = _list_collections()
    rows = [[cid, n_docs, dim] for cid, n_docs, dim in items]
    choices = [cid for cid, _n, _dim in items]
    selected = choices[0] if choices else None
    return rows, gr.update(choices=choices, value=selected)


def _load_meta(collection_id: str | None):
    if not collection_id:
        return "_(未选择 collection)_", []
    meta = _SVC.collections.collection_meta(collection_id)
    if not meta:
        return f"_(找不到 collection `{collection_id}`)_", []

    md = (
        f"### `{meta['collection_id']}`\n\n"
        f"- **文档数**: {meta['n_documents']}（original={meta['n_original']} · derived={meta['n_derived']}）\n"
        f"- **向量维度**: {meta['dim']}\n"
        f"- **向量索引**: {'✅' if meta['has_vector_index'] else '—'}\n"
        f"- **FTS 索引**: {'✅' if meta['has_fts_index'] else '—'}\n"
        f"- **built_at(ms)**: {meta['built_at']}\n"
        f"- **schema_version**: {meta['schema_version']}\n"
        f"- **filterable_fields**: {', '.join(meta.get('filterable_fields', [])) or '-'}\n"
        f"- **searchable_fields**: {', '.join(meta.get('searchable_fields', [])) or '-'}\n"
    )
    schema_rows = [
        [field.get("name", ""), field.get("type", ""), "Y" if field.get("nullable", True) else "N"]
        for field in meta.get("schema_fields", [])
    ]
    return md, schema_rows


def _browse_documents(
    collection_id: str | None,
    where: str,
    page: int,
    page_size: int,
    include_content: bool,
    show_vector: bool,
):
    if not collection_id:
        return gr.update(value=[], headers=[]), "未选择 collection", ""

    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 50), 1000))
    offset = (page - 1) * page_size

    try:
        docs = _SVC.documents.list_documents(
            collection_id,
            where=where.strip() or None,
            limit=offset + page_size,
            include_content=include_content,
        )
    except Exception as e:  # noqa: BLE001
        return gr.update(value=[], headers=[]), f"❌ 查询失败: {e}", ""

    if not docs:
        return gr.update(value=[], headers=[]), "无数据", "[]"

    page_docs = docs[offset: offset + page_size]
    raw_docs = [_dump(d) for d in page_docs]

    headers = ["document_id", "score"]
    if include_content:
        headers.append("content")
    headers.extend(["metadata", "vector_preview"])

    table = []
    for item in raw_docs:
        metadata = item.get("metadata", {}) or {}
        row = [item.get("document_id"), float(item.get("score", 0.0))]
        if include_content:
            row.append(item.get("content") or "")
        row.append(json.dumps(metadata, ensure_ascii=False))
        row.append(_vector_sparkline(metadata.get("vector") if show_vector else []))
        table.append(row)

    info = (
        f"page {page} · 本页 {len(page_docs)} 行 · 已扫描 {len(docs)} 行"
        + (f" · where: `{where.strip()}`" if where.strip() else "")
    )
    raw_json = json.dumps(raw_docs, ensure_ascii=False, indent=2, default=str)
    return gr.update(value=table, headers=headers), info, raw_json


def _search_documents(
    collection_id: str | None,
    query_text: str,
    top_n: int,
    top_m: int,
    rrf_k: int,
    where: str,
    include_derived: bool,
    skip_embedding: bool,
):
    if not collection_id:
        return [], "未选择 collection", ""
    if not (query_text or "").strip():
        return [], "请输入查询文本", ""

    tokenized = _tokenize_query(query_text)
    notes: list[str] = []
    qv: list[float] = []

    meta = _SVC.collections.collection_meta(collection_id)
    expected_dim = int(meta.get("dim", 0)) if meta else 0

    if skip_embedding:
        notes.append("⏭ 跳过 embedding，仅 BM25")
    else:
        qv, err = _embed_query(query_text)
        if err:
            notes.append(f"⚠️ {err} → 降级为 BM25-only")
            qv = []
        elif expected_dim and len(qv) != expected_dim:
            notes.append(
                f"⚠️ embedding 维度 {len(qv)} 与表维度 {expected_dim} 不一致 → 跳过向量召回"
            )
            qv = []
        else:
            notes.append(f"✅ embedding ok（dim={len(qv)}）")

    try:
        hits = _SVC.retrieval.search_documents(
            collection_id,
            query_tokenized=tokenized,
            query_vector=qv,
            top_n=int(top_n or 0),
            top_m=int(top_m or 0),
            rrf_k=int(rrf_k or 60),
            where=where.strip() or None,
            include_content=True,
            include_derived=bool(include_derived),
            strategy="legacy_hybrid",
        )
    except Exception as e:  # noqa: BLE001
        return [], f"❌ search 失败: {e}", ""

    raw_hits = [_dump(h) for h in hits]
    rows = []
    for item in raw_hits:
        rows.append([
            item.get("document_id", ""),
            round(float(item.get("score", 0.0)), 6),
            item.get("content", "") or "",
            json.dumps(item.get("metadata", {}), ensure_ascii=False),
        ])

    info_md = (
        f"**hits={len(raw_hits)}** · tokenized=`{tokenized[:120]}{'…' if len(tokenized)>120 else ''}`\n\n"
        + "\n".join(f"- {n}" for n in notes)
    )
    raw_json = json.dumps(raw_hits, ensure_ascii=False, indent=2, default=str)
    return rows, info_md, raw_json


def _drop_collection(collection_id: str | None, confirm: str):
    if not collection_id:
        return "❌ 未选择 collection", gr.update()
    if confirm.strip() != collection_id:
        return (
            f"⚠️ 取消：请在确认框输入完整 collection 名 `{collection_id}` 才会删除",
            gr.update(),
        )
    ok = _SVC.collections.drop_collection(collection_id)
    msg = f"✅ 已删除 `{collection_id}`" if ok else f"❌ `{collection_id}` 不存在"
    items = _list_collections()
    choices = [cid for cid, _n, _dim in items]
    return msg, gr.update(choices=choices, value=(choices[0] if choices else None))


def build_generic_demo() -> gr.Blocks:
    initial = _list_collections()
    initial_choices = [cid for cid, _n, _dim in initial]
    initial_rows = [[cid, n, dim] for cid, n, dim in initial]

    with gr.Blocks(
        title="Generic Knowledge Viewer (Gradio)",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# Generic Knowledge Viewer\n"
            "_collection/document 通用知识视图 · 基于 v2 语义_"
        )
        caps = _SVC.capabilities()
        gr.Markdown(
            f"- generic_api: **{caps.get('generic_api', False)}**\n"
            f"- retrieval_modes: `{', '.join(caps.get('retrieval_modes', [])) or '-'}`\n"
            f"- schema_version: `{caps.get('schema_version', '-')}`"
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Collections")
                refresh_btn = gr.Button("🔄 刷新", size="sm")
                collection_table = gr.Dataframe(
                    headers=["collection_id", "documents", "dim"],
                    value=initial_rows,
                    interactive=False,
                    wrap=True,
                    row_count=(0, "dynamic"),
                )
                collection_dd = gr.Dropdown(
                    label="当前 collection",
                    choices=initial_choices,
                    value=initial_choices[0] if initial_choices else None,
                    interactive=True,
                )

            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Schema"):
                        meta_md = gr.Markdown("_(选择左侧 collection 查看)_")
                        schema_df = gr.Dataframe(
                            headers=["column", "type", "nullable"],
                            interactive=False,
                            wrap=True,
                            row_count=(0, "dynamic"),
                        )

                    with gr.Tab("Browse"):
                        with gr.Row():
                            where_in = gr.Textbox(
                                label="where (LanceDB SQL 表达式，可空)",
                                placeholder="e.g. document_id > 10",
                                scale=4,
                            )
                            page_in = gr.Number(label="page", value=1, precision=0, scale=1)
                            page_size_in = gr.Number(label="page size", value=50, precision=0, scale=1)
                        with gr.Row():
                            include_content_cb = gr.Checkbox(label="include content", value=True)
                            show_vector_cb = gr.Checkbox(label="show vector preview", value=True)
                            browse_btn = gr.Button("🔍 浏览", variant="primary")
                        browse_info = gr.Markdown("")
                        browse_df = gr.Dataframe(
                            value=[],
                            interactive=False,
                            wrap=True,
                            row_count=(0, "dynamic"),
                            line_breaks=True,
                        )
                        browse_raw = gr.Code(
                            value="",
                            language="json",
                            label="raw documents JSON",
                            interactive=False,
                            lines=12,
                        )

                    with gr.Tab("Hybrid Search"):
                        gr.Markdown(
                            "BM25 + 向量混合检索（RRF 融合）。输入原文，自动分词并按配置调用 embedding；"
                            "未配置时自动降级为 BM25-only。"
                        )
                        query_in = gr.Textbox(
                            label="查询原文",
                            placeholder="例如：这个知识库如何做增量更新？",
                            lines=2,
                        )
                        with gr.Row():
                            topn_in = gr.Number(label="top_n (向量)", value=20, precision=0)
                            topm_in = gr.Number(label="top_m (BM25)", value=20, precision=0)
                            rrfk_in = gr.Number(label="rrf_k", value=60, precision=0)
                            include_derived_cb = gr.Checkbox(label="include derived", value=True)
                            skip_emb_cb = gr.Checkbox(label="跳过 embedding（仅 BM25）", value=False)
                        search_where_in = gr.Textbox(label="where (可空)", placeholder="可空")
                        search_btn = gr.Button("🚀 搜索", variant="primary")
                        search_info = gr.Markdown("")
                        search_df = gr.Dataframe(
                            headers=["document_id", "score", "content", "metadata"],
                            interactive=False,
                            wrap=True,
                            row_count=(0, "dynamic"),
                            line_breaks=True,
                        )
                        search_raw = gr.Code(
                            value="",
                            language="json",
                            label="raw hits JSON",
                            interactive=False,
                            lines=12,
                        )

                    with gr.Tab("⚠️ Danger"):
                        gr.Markdown(
                            "**删除 collection 是不可逆操作**。请在下方输入完整 `collection_id` 二次确认。"
                        )
                        confirm_in = gr.Textbox(label="二次确认（输入完整 collection_id）")
                        drop_btn = gr.Button("🗑️ 删除当前 collection", variant="stop")
                        drop_msg = gr.Markdown("")

        refresh_btn.click(_refresh_collection_list, outputs=[collection_table, collection_dd])
        collection_dd.change(_load_meta, inputs=[collection_dd], outputs=[meta_md, schema_df])
        if initial_choices:
            demo.load(_load_meta, inputs=[collection_dd], outputs=[meta_md, schema_df])

        browse_btn.click(
            _browse_documents,
            inputs=[collection_dd, where_in, page_in, page_size_in, include_content_cb, show_vector_cb],
            outputs=[browse_df, browse_info, browse_raw],
        )
        search_btn.click(
            _search_documents,
            inputs=[collection_dd, query_in, topn_in, topm_in, rrfk_in, search_where_in, include_derived_cb, skip_emb_cb],
            outputs=[search_df, search_info, search_raw],
        )
        drop_btn.click(
            _drop_collection,
            inputs=[collection_dd, confirm_in],
            outputs=[drop_msg, collection_dd],
        )

    return demo
