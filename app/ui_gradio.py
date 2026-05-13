"""Lance Data Viewer 风格的 Gradio 前端。

同进程挂载到 FastAPI 上（``/gradio``），直接调用 ``app.store`` 的函数，
不走 HTTP，因此也不需要 API_Key 中转。

启动后访问：
    http://127.0.0.1:5000/gradio
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import gradio as gr

from . import store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 查询侧：分词 + embedding


def _load_bm25_tokenize():
    """单文件加载 ``retrieval/bm25.py`` 拿 ``tokenize``，避开 ``retrieval/__init__.py``
    对主项目 ``reasoner`` 的依赖（这里只是个轻量服务，不应被传递依赖污染）。
    """

    import importlib.util
    from pathlib import Path

    bm25_path = Path(__file__).resolve().parent.parent / "retrieval" / "bm25.py"
    if not bm25_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_retrieval_bm25_local", str(bm25_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "tokenize", None)


_BM25_TOKENIZE = _load_bm25_tokenize()


def _tokenize_query(text: str) -> str:
    """复用 ``retrieval/bm25.py::tokenize``（jieba 分词），返回空格连接的字符串。
    服务端 FTS 索引用 whitespace tokenizer，必须给出与 client 同源的分词。
    缺 jieba / 加载失败时回退为简单非词切分，至少不阻塞调试。
    """

    if not text or not text.strip():
        return ""
    if _BM25_TOKENIZE is not None:
        try:
            return " ".join(_BM25_TOKENIZE(text))
        except Exception as e:  # noqa: BLE001
            logger.warning("retrieval/bm25.py::tokenize 调用失败: %s", e)
    import re

    return " ".join(t for t in re.split(r"[\s\W_]+", text.lower()) if t)


_DEFAULT_EMBEDDING_BASE_URL = "http://mlp.paas.dc.servyou-it.com/qwen3-embedding/v1"
_DEFAULT_EMBEDDING_MODEL = "qwen3-embedding"


def _embed_query(text: str) -> tuple[list[float], str]:
    """OpenAI 兼容 embedding（`{EMBEDDING_BASE_URL}/embeddings`）。

    默认指向公司内网 Qwen3-Embedding (1024 维)；可通过环境变量覆盖：
        EMBEDDING_BASE_URL  默认 http://mlp.paas.dc.servyou-it.com/qwen3-embedding/v1
        EMBEDDING_MODEL     默认 qwen3-embedding
        EMBEDDING_API_KEY   可空
    任何一项显式设为空字符串都视作"未配置"，自动降级 BM25-only。
    """

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
        import httpx  # FastAPI 已带

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


_BARS = "▁▂▃▄▅▆▇█"
_PREVIEW_MAX = 64  # vector 预览最多采样多少维度


def _vector_sparkline(vec: list[float] | None) -> str:
    """把 vector 渲染成一个紧凑 unicode sparkline，用于 Dataframe 单元格直接展示。"""

    if not vec:
        return ""
    n = len(vec)
    # 等距采样到 _PREVIEW_MAX
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


# ---------------------------------------------------------------- 数据获取


def _list_policies_choices() -> list[tuple[str, str]]:
    items = store.list_policies()
    return [(f"{pid}  ({n} rows · dim={dim})", pid) for pid, n, dim in items]


def _refresh_dataset_list():
    items = store.list_policies()
    rows = [[pid, n, dim] for pid, n, dim in items]
    choices = [pid for pid, _n, _dim in items]
    selected = choices[0] if choices else None
    return (
        rows,
        gr.update(choices=choices, value=selected),
    )


def _load_meta(policy_id: str | None):
    if not policy_id:
        return "_(未选择 dataset)_", []
    meta = store.table_meta(policy_id)
    if not meta:
        return f"_(找不到 dataset `{policy_id}`)_", []

    md = (
        f"### `{meta['policy_id']}`\n\n"
        f"- **行数**: {meta['n_chunks']}（original={meta['n_original']} · derived={meta['n_derived']}）\n"
        f"- **向量维度**: {meta['dim']}\n"
        f"- **向量索引**: {'✅' if meta['has_vector_index'] else '—'}\n"
        f"- **FTS 索引**: {'✅' if meta['has_fts_index'] else '—'}\n"
        f"- **built_at(ms)**: {meta['built_at']}\n"
        f"- **schema_version**: {meta['schema_version']}\n"
    )

    # schema 行：列名 / pyarrow 类型 / 是否可空
    try:
        tbl = store.open_table(policy_id)
        schema = tbl.schema
        schema_rows = [
            [f.name, str(f.type), "Y" if f.nullable else "N"] for f in schema
        ]
    except Exception as e:  # noqa: BLE001
        schema_rows = [["<error>", str(e), ""]]

    return md, schema_rows


# ---------------------------------------------------------------- 浏览


def _browse_rows(
    policy_id: str | None,
    where: str,
    page: int,
    page_size: int,
    include_content: bool,
    show_vector: bool,
):
    """返回 (table_update, info_md, raw_json_str)。

    - 表格里 content / heading_paths / directories / relation_keys 都不再截断，
      Dataframe 的 wrap=True 会让长文本自动换行展示，单元格点击可整段选中。
    - 同时把当前页的完整原始字典序列化为 JSON 字符串返回，
      由 gr.Code 组件渲染（右上角自带复制按钮，鼠标悬停即可一键复制）。
    """

    if not policy_id:
        return gr.update(value=[], headers=[]), "未选择 dataset", ""

    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 50), 1000))
    offset = (page - 1) * page_size

    try:
        if not store.table_exists(policy_id):
            return gr.update(value=[], headers=[]), f"dataset `{policy_id}` 不存在", ""
        tbl = store.open_table(policy_id)
        cols = [
            "chunk_id",
            "kind",
            "parent_chunk_index",
            "derived_seq",
            "source",
            "clause_id",
            "hop_depth",
            "heading_paths",
            "directories",
            "relation_keys",
            "built_at",
        ]
        if include_content:
            cols.append("content")
        if show_vector:
            cols.append("vector")

        q = tbl.search().select(cols)
        if where.strip():
            q = q.where(where.strip())
        rows_raw = q.limit(offset + page_size).to_list()
    except Exception as e:  # noqa: BLE001
        return gr.update(value=[], headers=[]), f"❌ 查询失败: {e}", ""

    page_rows = rows_raw[offset : offset + page_size]

    headers = [
        "chunk_id",
        "kind",
        "parent",
        "derived_seq",
        "source",
        "clause_id",
        "hop",
        "heading_paths",
        "directories",
        "relation_keys",
    ]
    if include_content:
        headers.append("content")
    if show_vector:
        headers.append("vector")

    def _stringify(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    table: list[list[Any]] = []
    for r in page_rows:
        row = [
            r.get("chunk_id"),
            r.get("kind"),
            r.get("parent_chunk_index"),
            r.get("derived_seq"),
            r.get("source"),
            r.get("clause_id"),
            r.get("hop_depth"),
            _stringify(r.get("heading_paths")),
            _stringify(r.get("directories")),
            _stringify(r.get("relation_keys")),
        ]
        if include_content:
            row.append(_stringify(r.get("content")))
        if show_vector:
            # vector 列展示 sparkline，完整向量太长不放表里（可在下方 JSON 区看到）
            row.append(_vector_sparkline(r.get("vector")))
        table.append(row)

    # 完整原始字典（含 vector 全量）丢给 gr.Code 复制
    raw_json = json.dumps(page_rows, ensure_ascii=False, indent=2, default=str)

    info = (
        f"page {page} · 本页 {len(page_rows)} 行 · 已扫描 {len(rows_raw)} 行"
        + (f" · where: `{where.strip()}`" if where.strip() else "")
    )
    return gr.update(value=table, headers=headers), info, raw_json


# ---------------------------------------------------------------- search


def _do_search(
    policy_id: str | None,
    query_text: str,
    top_n: int,
    top_m: int,
    rrf_k: int,
    where: str,
    include_derived: bool,
    skip_embedding: bool,
):
    """单一查询入口：UI 给原文，内部完成 jieba 分词 + 取 query embedding，
    再调 ``store.hybrid_search``。embedding 失败 / 跳过时自动降级为纯 BM25。
    返回 (table_rows, info_md, raw_json)。
    """

    if not policy_id:
        return [], "未选择 dataset", ""
    if not (query_text or "").strip():
        return [], "请输入查询文本", ""

    # 1) 分词（FTS 路径）
    tokenized = _tokenize_query(query_text)

    # 2) embedding（向量路径），失败/跳过都允许
    notes: list[str] = []
    qv: list[float] = []
    if skip_embedding:
        notes.append("⏭ 跳过 embedding，仅 BM25")
    else:
        # 维度预检：拿表上的 dim，向 embedding 之后做一次校验
        expected_dim = store._existing_dim(policy_id) if store.table_exists(policy_id) else 0  # noqa: SLF001
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

    # 3) 调存储层混合检索
    try:
        hits = store.hybrid_search(
            policy_id,
            query_tokenized=tokenized,
            query_vector=qv,
            top_n=int(top_n or 0),
            top_m=int(top_m or 0),
            rrf_k=int(rrf_k or 60),
            where=where.strip() or None,
            include_content=True,
            include_derived=bool(include_derived),
        )
    except Exception as e:  # noqa: BLE001
        return [], f"❌ search 失败: {e}", ""

    rows = []
    for h in hits:
        rows.append([
            h.chunk_id,
            round(h.score, 6),
            h.kind,
            h.parent_chunk_index,
            h.source,
            h.clause_id,
            h.content or "",  # 不截断
        ])

    info_md = (
        f"**hits={len(hits)}** · tokenized=`{tokenized[:120]}{'…' if len(tokenized)>120 else ''}`\n\n"
        + "\n".join(f"- {n}" for n in notes)
    )
    raw_json = json.dumps(
        [h.model_dump() if hasattr(h, "model_dump") else dict(h) for h in hits],
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return rows, info_md, raw_json


# ---------------------------------------------------------------- danger zone


def _drop_policy(policy_id: str | None, confirm: str):
    if not policy_id:
        return "❌ 未选择 dataset", gr.update()
    if confirm.strip() != policy_id:
        return (
            f"⚠️ 取消：请在确认框输入完整 dataset 名 `{policy_id}` 才会删除",
            gr.update(),
        )
    ok = store.drop_table(policy_id)
    msg = f"✅ 已删除 `{policy_id}`" if ok else f"❌ `{policy_id}` 不存在"
    items = store.list_policies()
    choices = [pid for pid, _n, _dim in items]
    return msg, gr.update(choices=choices, value=(choices[0] if choices else None))


# ---------------------------------------------------------------- UI


def build_demo() -> gr.Blocks:
    initial = store.list_policies()
    initial_choices = [pid for pid, _n, _dim in initial]
    initial_rows = [[pid, n, dim] for pid, n, dim in initial]

    with gr.Blocks(
        title="Lance Data Viewer (Gradio)",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# Lance Data Viewer\n"
            "_最低成本前端 · Gradio 同进程挂载 · 直接读 LanceDB_"
        )

        with gr.Row():
            # 左侧：dataset 列表
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Datasets")
                refresh_btn = gr.Button("🔄 刷新", size="sm")
                dataset_table = gr.Dataframe(
                    headers=["policy_id", "rows", "dim"],
                    value=initial_rows,
                    interactive=False,
                    wrap=True,
                    row_count=(0, "dynamic"),
                )
                policy_dd = gr.Dropdown(
                    label="当前 dataset",
                    choices=initial_choices,
                    value=initial_choices[0] if initial_choices else None,
                    interactive=True,
                )

            # 右侧：tabs
            with gr.Column(scale=4):
                with gr.Tabs():
                    # ---------------- Schema / Meta ----------------
                    with gr.Tab("Schema"):
                        meta_md = gr.Markdown("_(选择左侧 dataset 查看)_")
                        schema_df = gr.Dataframe(
                            headers=["column", "type", "nullable"],
                            interactive=False,
                            wrap=True,
                            row_count=(0, "dynamic"),
                        )

                    # ---------------- Browse ----------------
                    with gr.Tab("Browse"):
                        with gr.Row():
                            where_in = gr.Textbox(
                                label="where (LanceDB SQL 表达式，可空)",
                                placeholder="e.g. kind = 'original' AND hop_depth = 0",
                                scale=4,
                            )
                            page_in = gr.Number(label="page", value=1, precision=0, scale=1)
                            page_size_in = gr.Number(
                                label="page size", value=50, precision=0, scale=1
                            )
                        with gr.Row():
                            include_content_cb = gr.Checkbox(label="include content", value=True)
                            show_vector_cb = gr.Checkbox(label="show vector sparkline", value=True)
                            browse_btn = gr.Button("🔍 浏览", variant="primary")
                        browse_info = gr.Markdown("")
                        gr.Markdown("**表格区**：长文本不截断，鼠标点击单元格即可拖选复制。")
                        browse_df = gr.Dataframe(
                            value=[],
                            interactive=False,
                            wrap=True,
                            row_count=(0, "dynamic"),
                            line_breaks=True,
                        )
                        gr.Markdown("**完整原始数据**（鼠标悬停在右上角点 📋 一键复制全文）")
                        browse_raw = gr.Code(
                            value="",
                            language="json",
                            label="raw rows JSON",
                            interactive=False,
                            lines=12,
                        )

                    # ---------------- Search ----------------
                    with gr.Tab("Hybrid Search"):
                        gr.Markdown(
                            "BM25 + 向量混合检索（RRF 融合）。**直接输入原文**，UI 内部用 jieba 分词、"
                            "并通过环境变量 `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` "
                            "调用 OpenAI 兼容的 `/embeddings` 接口取向量；未配置则自动降级为 **BM25-only**。"
                        )
                        query_in = gr.Textbox(
                            label="查询原文",
                            placeholder="例如：理赔范围内是否包含猝死？",
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
                            headers=[
                                "chunk_id",
                                "score",
                                "kind",
                                "parent",
                                "source",
                                "clause_id",
                                "content",
                            ],
                            interactive=False,
                            wrap=True,
                            row_count=(0, "dynamic"),
                            line_breaks=True,
                        )
                        gr.Markdown("**完整命中明细**（含全部字段，右上角 📋 复制）")
                        search_raw = gr.Code(
                            value="",
                            language="json",
                            label="raw hits JSON",
                            interactive=False,
                            lines=12,
                        )

                    # ---------------- Danger ----------------
                    with gr.Tab("⚠️ Danger"):
                        gr.Markdown(
                            "**删除 dataset 是不可逆操作**。请在下方输入完整 `policy_id` 二次确认。"
                        )
                        confirm_in = gr.Textbox(
                            label="二次确认（输入完整 policy_id）"
                        )
                        drop_btn = gr.Button("🗑️ 删除当前 dataset", variant="stop")
                        drop_msg = gr.Markdown("")

        # ---------------- 事件 ----------------

        refresh_btn.click(_refresh_dataset_list, outputs=[dataset_table, policy_dd])

        # 选中 dataset → 自动加载 meta
        policy_dd.change(_load_meta, inputs=[policy_dd], outputs=[meta_md, schema_df])

        # 初始 meta 加载
        if initial_choices:
            demo.load(_load_meta, inputs=[policy_dd], outputs=[meta_md, schema_df])

        browse_btn.click(
            _browse_rows,
            inputs=[
                policy_dd,
                where_in,
                page_in,
                page_size_in,
                include_content_cb,
                show_vector_cb,
            ],
            outputs=[browse_df, browse_info, browse_raw],
        )

        search_btn.click(
            _do_search,
            inputs=[
                policy_dd,
                query_in,
                topn_in,
                topm_in,
                rrfk_in,
                search_where_in,
                include_derived_cb,
                skip_emb_cb,
            ],
            outputs=[search_df, search_info, search_raw],
        )

        drop_btn.click(
            _drop_policy,
            inputs=[policy_dd, confirm_in],
            outputs=[drop_msg, policy_dd],
        )

    return demo
