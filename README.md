# retrieval_service

基于 LanceDB 的 BM25 + 向量混合检索 HTTP 微服务，支持 legacy `v1` policy API 与通用 `v2` collection API 并行。

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

### Gradio 前端

服务启动后可直接访问：

```bash
http://127.0.0.1:5000/gradio/           # policy 兼容界面（legacy）
http://127.0.0.1:5000/gradio-generic/   # 通用 collection/document 界面（v2）
```

说明：

- Gradio 由 FastAPI 同进程托管，**起一个后端即可使用**，无需 Node / 前端构建链。
- `/ui` 静态页已移除，避免维护两套前端。
- `/gradio`：policy 兼容界面，保留 legacy 业务体验与字段。
- `/gradio-generic`：通用 collection/document 界面，走 v2 语义，适配通用知识形态。
- `/gradio` 覆盖：
  - 左侧 dataset（policy）列表，显示行数 / 向量维度
  - **Schema** tab：表 meta（行数、维度、索引状态）+ 完整 pyarrow schema
  - **Browse** tab：where 过滤 + 分页 + 列选择，**vector 列以 unicode sparkline 紧凑展示**（如 `[768d] ▃▅▂▇▆▄▁█…`），免去看一长串浮点数
- **Hybrid Search** tab：**直接输查询原文**，UI 内部自动用 jieba 分词（复用 `retrieval/bm25.py::tokenize`，与建索引时同源），并调用 OpenAI 兼容 `/embeddings` 接口取 query 向量。默认不启用 embedding，避免绑定内网环境；配置后即可启用：

    | 环境变量 | 默认值 | 说明 |
    |---|---|---|
    | `EMBEDDING_BASE_URL` | 空 | 例如 `https://api.openai.com/v1` |
    | `EMBEDDING_MODEL` | 空 | 必须与索引时向量维度一致 |
    | `EMBEDDING_API_KEY` | 空 | 可选 |

    Tab 内还提供了「跳过 embedding（仅 BM25）」复选框，方便快速跑纯 BM25 调试；维度不匹配时也会自动降级到 BM25-only 并在结果区显示告警。
  - **Danger** tab：二次确认后删除 dataset
- **反向代理前缀**：通过环境变量 `PROXY_ROOT_PATH` 配置（默认 `/kh-lancedb`，匹配当前线上代理），它会被传给 uvicorn 作为 ASGI `root_path`。Gradio 6 会自动从 ASGI scope 读取并把这个前缀加到 HTML `<config>` 的 `root` 字段里，所以浏览器后续的 API 请求会带上 `/kh-lancedb` 前缀。

  | 部署形态 | `PROXY_ROOT_PATH` | 浏览器 URL |
  |---|---|---|
  | 反向代理（默认） | `/kh-lancedb` | `https://your.domain/kh-lancedb/gradio/` 或 `.../gradio-generic/` |
  | 本机直连 | 空字符串 | `http://127.0.0.1:5000/gradio/` 或 `.../gradio-generic/` |

  本机直连示例（绕开代理前缀）：

  ```powershell
  $env:PROXY_ROOT_PATH=""
  python main.py
  ```

  > 前提：代理把 `/kh-lancedb/*` 剥前缀转发给后端（即转 `/gradio/...`、`/gradio-generic/...`、`/v1/...`、`/v2/...` 这种内部路径）。

## Feature Flags

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `ENABLE_GENERIC_API` | `true` | 是否启用 `/v2/*` 通用接口 |
| `ENABLE_LEGACY_RELATIONS` | `true` | 是否启用 legacy relations 路由 |
| `ENABLE_LEGACY_UI` | `true` | 是否启用 policy 兼容 Gradio(`/gradio`) |
| `ENABLE_GENERIC_GRADIO` | `true` | 是否启用通用 Gradio(`/gradio-generic`) |
| `GRADIO_USE_HTTP` | `false` | Gradio 是否通过 HTTP 调后端（否则 direct store） |
| `GRADIO_API_BASE_URL` | 空 | Gradio HTTP 模式下的服务地址 |
| `GRADIO_API_KEY` | 空 | Gradio HTTP 模式下鉴权 key |

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
| GET | `/v1/capabilities` | 能力协商（UI/客户端可按能力显隐功能） |
| POST | `/v1/policies/{policy_id}/chunks:upsert` | 整批写入（mode=overwrite\|append\|merge_by_chunk_id） |
| GET | `/v1/policies/{policy_id}/chunks` | 列出 chunks，支持 where/limit |
| DELETE | `/v1/policies/{policy_id}` | 删表 |
| POST | `/v1/policies/{policy_id}/search` | 混合检索（query_tokenized + query_vector） |
| POST | `/v1/policies/{policy_id}/relations:expand` | 取某父 chunk 的派生 chunks |
| GET | `/v1/policies/{policy_id}/relations:lookup` | 单 policy 内反查依赖某 (target_policy_id, target_clause_id) 的 chunks |
| GET | `/v1/relations:lookup-dependents` | 全局反查：哪些源 policy 引用了 target |
| GET | `/v1/policies` | 列出所有 policy |
| GET | `/v1/policies/{policy_id}/meta` | 表行数 / dim / 索引状态 |
| GET | `/v2/capabilities` | 通用 API 能力协商 |
| GET | `/v2/collections` | 列出所有 collection |
| GET | `/v2/collections/{collection_id}/meta` | collection 元信息与 schema |
| GET | `/v2/collections/{collection_id}/documents` | 通用文档列表 |
| POST | `/v2/collections/{collection_id}/documents:upsert` | 通用文档 upsert |
| POST | `/v2/collections/{collection_id}/search` | 通用检索 |

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

## 迁移与回滚

平滑迁移建议见 [docs/migration_runbook.md](docs/migration_runbook.md)，并可用 `scripts/shadow_compare.py` 做新旧服务 shadow 对比。
