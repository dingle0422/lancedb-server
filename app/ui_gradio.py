"""Lance Data Viewer 风格的 Gradio 前端。

同进程挂载到 FastAPI 上（``/gradio``），直接调用 ``app.store`` 的函数，
不走 HTTP，因此也不需要 API_Key 中转。

启动后访问：
    http://127.0.0.1:5000/gradio
"""

from __future__ import annotations

import json
import logging
from typing import Any

import gradio as gr

from . import store

logger = logging.getLogger(__name__)


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


def _truncate(s: Any, n: int = 80) -> str:
    if s is None:
        return ""
    text = str(s)
    return text if len(text) <= n else text[: n - 1] + "…"


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
    if not policy_id:
        return [], [], "未选择 dataset"

    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 50), 1000))
    offset = (page - 1) * page_size

    # store.list_chunks 不支持 offset；这里取 offset+page_size 然后切片
    try:
        if not store.table_exists(policy_id):
            return [], [], f"dataset `{policy_id}` 不存在"
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
        return [], [], f"❌ 查询失败: {e}"

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
            _truncate(r.get("heading_paths"), 60),
            _truncate(r.get("directories"), 60),
            _truncate(r.get("relation_keys"), 60),
        ]
        if include_content:
            row.append(_truncate(r.get("content"), 200))
        if show_vector:
            row.append(_vector_sparkline(r.get("vector")))
        table.append(row)

    info = (
        f"page {page} · 本页 {len(page_rows)} 行 · 已扫描 {len(rows_raw)} 行"
        + (f" · where: `{where.strip()}`" if where.strip() else "")
    )
    return [headers], table, info


def _browse_with_headers(*args):
    headers_wrap, table, info = _browse_rows(*args)
    headers = headers_wrap[0] if headers_wrap else []
    return gr.update(value=table, headers=headers), info


# ---------------------------------------------------------------- search


def _do_search(
    policy_id: str | None,
    query_tokenized: str,
    query_vector_json: str,
    top_n: int,
    top_m: int,
    rrf_k: int,
    where: str,
    include_derived: bool,
):
    if not policy_id:
        return [], "未选择 dataset"
    qv: list[float] = []
    if query_vector_json.strip():
        try:
            parsed = json.loads(query_vector_json)
            if not isinstance(parsed, list):
                raise ValueError("query_vector 必须是 JSON 数组")
            qv = [float(x) for x in parsed]
        except Exception as e:  # noqa: BLE001
            return [], f"❌ query_vector JSON 解析失败: {e}"

    try:
        hits = store.hybrid_search(
            policy_id,
            query_tokenized=query_tokenized or "",
            query_vector=qv,
            top_n=int(top_n or 0),
            top_m=int(top_m or 0),
            rrf_k=int(rrf_k or 60),
            where=where.strip() or None,
            include_content=True,
            include_derived=bool(include_derived),
        )
    except Exception as e:  # noqa: BLE001
        return [], f"❌ search 失败: {e}"

    rows = []
    for h in hits:
        rows.append([
            h.chunk_id,
            round(h.score, 6),
            h.kind,
            h.parent_chunk_index,
            h.source,
            h.clause_id,
            _truncate(h.content, 200),
        ])
    return rows, f"hits={len(hits)}"


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
                        browse_df = gr.Dataframe(
                            value=[],
                            interactive=False,
                            wrap=True,
                            row_count=(0, "dynamic"),
                        )

                    # ---------------- Search ----------------
                    with gr.Tab("Hybrid Search"):
                        gr.Markdown(
                            "调试 BM25 + 向量混合检索（RRF 融合）。"
                            "`query_tokenized` 用空格分隔的 token 串，`query_vector` 是 JSON 数组。"
                        )
                        with gr.Row():
                            qt_in = gr.Textbox(
                                label="query_tokenized",
                                placeholder="例如：保险 理赔 范围",
                                scale=2,
                            )
                            qv_in = gr.Textbox(
                                label="query_vector (JSON)",
                                placeholder="[0.1, 0.2, ...]",
                                scale=2,
                            )
                        with gr.Row():
                            topn_in = gr.Number(label="top_n (向量)", value=20, precision=0)
                            topm_in = gr.Number(label="top_m (BM25)", value=20, precision=0)
                            rrfk_in = gr.Number(label="rrf_k", value=60, precision=0)
                            include_derived_cb = gr.Checkbox(label="include derived", value=True)
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
            _browse_with_headers,
            inputs=[
                policy_dd,
                where_in,
                page_in,
                page_size_in,
                include_content_cb,
                show_vector_cb,
            ],
            outputs=[browse_df, browse_info],
        )

        search_btn.click(
            _do_search,
            inputs=[
                policy_dd,
                qt_in,
                qv_in,
                topn_in,
                topm_in,
                rrfk_in,
                search_where_in,
                include_derived_cb,
            ],
            outputs=[search_df, search_info],
        )

        drop_btn.click(
            _drop_policy,
            inputs=[policy_dd, confirm_in],
            outputs=[drop_msg, policy_dd],
        )

    return demo
