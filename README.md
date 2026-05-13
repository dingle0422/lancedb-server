# retrieval_service

基于 LanceDB 的 BM25 + 向量混合检索 HTTP 微服务，是 page-know-how 主项目的存储/检索后端。

## 设计要点

- **零外部依赖部署**：纯 Python wheel（lancedb + pyarrow），单进程 uvicorn 即可启动；无需 Docker 即可裸跑。
- **每 policy 一张表**：物理上 `STORE_DIR/{safe_policy_id}.lance/`，互不干扰；表删除等价于 `rm -rf`。
- **服务端不依赖 jieba**：分词在客户端做（主项目 `inference/retrieval/bm25.py::tokenize`），服务端只接收已分词字符串，FTS 索引 base_tokenizer 用 `whitespace`，分词逻辑 100% 与现有 BM25 同源。
- **本地 RRF 融合**：跟主项目 `inference/retrieval/rrf.py` 同公式 `1/(k+rank)`，避开 LanceDB 内置 hybrid+reranker 的版本差异。
- **关联结构原生入索**：`kind`/`parent_chunk_index`/`derived_seq`/`relation_keys`/`hop_depth`/`source`/`clause_id` 都是表列，推理期可按高亮链路展开 / 反查；跨 policy cascade 走全局接口 `/v1/relations:lookup-dependents`。

## 表 schema（pyarrow）

| 列 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | int64 | 主键 = 原 KnowledgeChunk.index |
| `content` | string | 给 LLM 用 |
| `content_tokenized` | string | **客户端 jieba 分词后空格连接** |
| `vector` | fixed_size_list&lt;float32&gt;[dim] | 客户端 embed 后送入 |
| `heading_paths` | list&lt;list&lt;string&gt;&gt; | KnowledgeChunk 元数据 |
| `directories` | list&lt;string&gt; | KnowledgeChunk 元数据 |
| `kind` | string | original \| derived |
| `parent_chunk_index` | int64 | 派生 chunk 父 chunk_id；原始 -1 |
| `derived_seq` | int32 | 同父下的序号 |
| `relation_keys` | list&lt;struct&lt;policy_id, clause_id&gt;&gt; | 派生命中的外部条款 |
| `hop_depth` | int32 | 派生跳数 |
| `source` | string | local/remote/missing |
| `clause_id` | string | 派生 chunk 的目标条款 id |
| `built_at` | int64 | 毫秒时间戳 |

## 快速开始

### 本地裸跑

```bash
cd retrieval_service
python -m venv .venv && .venv\Scripts\activate          # Windows
# source .venv/bin/activate                              # Linux/Mac
pip install -e .
cp .env.example .env                                     # Windows: copy .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 5000 --reload
# 或者直接使用项目入口脚本
python main.py
```

健康检查：

```bash
curl http://127.0.0.1:5000/healthz
```

### WebUI（最轻量管理页）

服务启动后可直接访问：

```bash
http://127.0.0.1:5000/ui          # 原始 HTML + JS（带 X-API-Key 调试）
http://127.0.0.1:5000/gradio/     # Lance Data Viewer 风格的 Gradio UI（推荐）
```

说明：

- 两个 UI 都由 FastAPI 同进程托管，**起一个后端就完事**，无需 Node / 前端构建链。
- `/ui`：原 `app/static/ui.html`，页面顶部填写 `X-API-Key`（保存在 `localStorage`）后即可调用 HTTP 接口。
- `/gradio`：参考 [Lance Data Viewer](https://www.lancedb.com/blog/lance-data-viewer) 的设计，用 Gradio 写的 dataset 浏览器。**直接调用 `app.store` 的函数（同进程，零 HTTP 开销，也无需鉴权中转）**，覆盖：
  - 左侧 dataset（policy）列表，显示行数 / 向量维度
  - **Schema** tab：表 meta（行数、维度、索引状态）+ 完整 pyarrow schema
  - **Browse** tab：where 过滤 + 分页 + 列选择，**vector 列以 unicode sparkline 紧凑展示**（如 `[768d] ▃▅▂▇▆▄▁█…`），免去看一长串浮点数
  - **Hybrid Search** tab：**直接输查询原文**，UI 内部自动用 jieba 分词（复用 `retrieval/bm25.py::tokenize`，与建索引时同源），并调用 OpenAI 兼容 `/embeddings` 接口取 query 向量。默认硬编码指向公司内网 Qwen3-Embedding（1024 维），无需任何配置开箱可用：

    | 环境变量 | 默认值 | 说明 |
    |---|---|---|
    | `EMBEDDING_BASE_URL` | `http://mlp.paas.dc.servyou-it.com/qwen3-embedding/v1` | `/embeddings` 路径会自动拼上 |
    | `EMBEDDING_MODEL` | `qwen3-embedding` | 必须与建索引时模型同维（这里 1024） |
    | `EMBEDDING_API_KEY` | 空 | 内网无需鉴权 |

    Tab 内还提供了「跳过 embedding（仅 BM25）」复选框，方便快速跑纯 BM25 调试；维度不匹配时也会自动降级到 BM25-only 并在结果区显示告警。
  - **Danger** tab：二次确认后删除 dataset
- **反向代理前缀**：通过环境变量 `PROXY_ROOT_PATH` 配置（默认 `/kh-lancedb`，匹配当前线上代理），它会被传给 uvicorn 作为 ASGI `root_path`。Gradio 6 会自动从 ASGI scope 读取并把这个前缀加到 HTML `<config>` 的 `root` 字段里，所以浏览器后续的 API 请求会带上 `/kh-lancedb` 前缀，正好让代理剥掉后变成内部的 `/gradio/...` 路由。FastAPI 的 `/v1/...` 接口在反向代理下生成的 OpenAPI URL 也会自动带前缀。

  | 部署形态 | `PROXY_ROOT_PATH` | 浏览器 URL |
  |---|---|---|
  | 反向代理（默认） | `/kh-lancedb` | `https://your.domain/kh-lancedb/gradio/` |
  | 本机直连 | 空字符串 | `http://127.0.0.1:5000/gradio/` |

  本机直连示例（绕开代理前缀）：

  ```powershell
  $env:PROXY_ROOT_PATH=""
  python main.py
  ```

  > 前提：代理把 `/kh-lancedb/*` 剥前缀转发给后端（即转 `/gradio/...`、`/v1/...` 这种内部路径）。如果你们的代理是**不剥前缀**透传整个 `/kh-lancedb/...`，那后端反而不要设 `PROXY_ROOT_PATH`，并且要在 main.py 中给所有 router 加 `prefix="/kh-lancedb"`，告诉我，我再改。

- 已覆盖原 `/ui` 常用管理与调试能力：
  - policy 列表、meta 查看
  - chunks 列表查询（where / limit / include_content）
  - hybrid search 调试（`query_tokenized` + `query_vector`）
  - upsert chunks、删除 policy
  - relations 的 expand / lookup / lookup-dependents

### Docker

```bash
cd retrieval_service
cp .env.example .env
docker compose up -d --build
```

数据持久化在 `./data/` 目录（容器内 `/app/data`）。

## API 概览

所有接口需带 `X-API-Key: <API_KEY>` 或 `Authorization: Bearer <API_KEY>` 头；当 `API_KEY` 为空时关闭鉴权（仅本地开发）。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/healthz` | 健康检查 |
| POST | `/v1/policies/{policy_id}/chunks:upsert` | 整批写入（mode=overwrite\|append\|merge_by_chunk_id） |
| GET | `/v1/policies/{policy_id}/chunks` | 列出 chunks，支持 where/limit |
| DELETE | `/v1/policies/{policy_id}` | 删表 |
| POST | `/v1/policies/{policy_id}/search` | 混合检索（query_tokenized + query_vector） |
| POST | `/v1/policies/{policy_id}/relations:expand` | 取某父 chunk 的派生 chunks |
| GET | `/v1/policies/{policy_id}/relations:lookup` | 单 policy 内反查依赖某 (target_policy_id, target_clause_id) 的 chunks |
| GET | `/v1/relations:lookup-dependents` | 全局反查：哪些源 policy 引用了 target |
| GET | `/v1/policies` | 列出所有 policy |
| GET | `/v1/policies/{policy_id}/meta` | 表行数 / dim / 索引状态 |

## 与主项目的对接

主项目 `inference/retrieval/client.py` 是这个服务的官方 SDK（基于 httpx async）。`hybrid_search` / `indexer.build_for_root` / `app.py._cascade_dependent_rebuilds` 三处会调用本服务。

主项目 `.env` 需要新增：

```env
RETRIEVAL_SERVICE_URL=http://127.0.0.1:8088
RETRIEVAL_SERVICE_API_KEY=changeme
```

## 测试

```bash
pip install -e ".[dev]"
pytest -q
```

`tests/test_routes.py` 跑一整套 upsert / search / expand / lookup / drop 的端到端冒烟。
