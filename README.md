# retrieval_service

基于 LanceDB 的 BM25 + 向量混合检索 HTTP 微服务，支持 legacy `v1` policy API 与通用 `v2` collection API 并行。

## 设计要点

- **零外部依赖部署**：纯 Python wheel（lancedb + pyarrow），单进程 uvicorn 即可启动；无需 Docker 即可裸跑。
- **每 policy 一张表**：物理上 `STORE_DIR/{safe_policy_id}.lance/`，互不干扰；表删除等价于 `rm -rf`。
- **服务端不依赖 jieba**：分词在客户端做（主项目 `inference/retrieval/bm25.py::tokenize`），服务端只接收已分词字符串，FTS 索引 base_tokenizer 用 `whitespace`，分词逻辑 100% 与现有 BM25 同源。
- **API 自动 embedding 兜底（可开关）**：当 upsert 未传 `vector` 或 search 未传 `query_vector` 时，服务端会调用 OpenAI 兼容 `/embeddings` 自动补齐；失败或维度不匹配时自动降级到 BM25/FTS-only。
- **本地 RRF 融合**：跟主项目 `inference/retrieval/rrf.py` 同公式 `1/(k+rank)`，避开 LanceDB 内置 hybrid+reranker 的版本差异。
- **关联结构原生入索**：`kind`/`parent_chunk_index`/`derived_seq`/`relation_keys`/`hop_depth`/`source`/`clause_id` 都是表列，推理期可按高亮链路展开 / 反查；跨 policy cascade 走全局接口 `/v1/relations:lookup-dependents`。

## 表 schema（pyarrow）

| 列 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | int64 | 主键 = 原 KnowledgeChunk.index |
| `content` | string | 给 LLM 用 |
| `content_tokenized` | string | **客户端 jieba 分词后空格连接** |
| `vector` | fixed_size_list&lt;float32&gt;[dim] | 客户端可直接传；缺失时服务端可自动 embedding 兜底 |
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
- **Hybrid Search** tab：**直接输查询原文**，UI 内部自动用 jieba 分词（复用 `retrieval/bm25.py::tokenize`，与建索引时同源），并调用 OpenAI 兼容 `/embeddings` 接口取 query 向量。默认已写死内网 qwen3-embedding，启动后直接可用；想换服务再用环境变量覆盖：

    | 环境变量 | 默认值 | 说明 |
    |---|---|---|
    | `EMBEDDING_BASE_URL` | `http://mlp.paas.dc.servyou-it.com/qwen3-embedding/v1` | 代码里硬编码；显式置空则降级 BM25-only |
    | `EMBEDDING_MODEL` | `qwen3-embedding` | 必须与索引时向量维度一致 |
    | `EMBEDDING_API_KEY` | 空 | 可选 |

    上述环境变量可以直接在 shell 里 `export` / `$env:` 设置，也可以写到仓库根目录的 `.env` 文件（参考 `.env.example`）。服务启动时 `app/main.py` 会 `load_dotenv()` 把 `.env` 注入 `os.environ`，所以 Gradio 与裸 `uvicorn app.main:app` 两种启动方式都生效。

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
| `ENABLE_SERVER_EMBEDDING_FALLBACK` | `true` | 是否启用 API 自动 embedding 兜底 |
| `EMBEDDING_BASE_URL` | `http://mlp.paas.dc.servyou-it.com/qwen3-embedding/v1` | OpenAI 兼容 embedding 服务地址 |
| `EMBEDDING_MODEL` | `qwen3-embedding` | embedding 模型名 |
| `EMBEDDING_API_KEY` | 空 | embedding 服务鉴权 key（可选） |
| `EMBEDDING_TIMEOUT_SEC` | `10` | embedding 请求超时秒数 |
| `ENABLE_RELATION_INDEX` | `true` | 是否启用 relation 反向索引（SQLite），把 `lookup-dependents` 降为点查；`0` 退回全表扫描 |
| `LOOKUP_DEPENDENTS_MAX_WORKERS` | `8` | `lookup-dependents` 扫描兜底路径的并行线程上限；`1` 表示串行（仅在索引关闭/未覆盖时生效） |
| `GRADIO_USE_HTTP` | `false` | Gradio 是否通过 HTTP 调后端（否则 direct store） |
| `GRADIO_API_BASE_URL` | 空 | Gradio HTTP 模式下的服务地址 |
| `GRADIO_API_KEY` | 空 | Gradio HTTP 模式下鉴权 key |

### lookup-dependents 性能与反向索引

`/v1/relations:lookup-dependents` 供主项目 cascade 触发器反查“哪些 source policy 引用了某 target”。早期实现为全库扫描，policy 多时可能耗时数十秒。现按两层优化：

- **反向索引（默认开启，`ENABLE_RELATION_INDEX=1`）**：用一张 SQLite 表（`STORE_DIR/_relation_index.sqlite3`）维护 `target -> 依赖它的 source` 映射，`upsert` 成功后按 source 粒度增量更新、`drop` 时清除，查询变为点查。首次启用或迁移历史数据时索引尚未覆盖全部 source，本次请求会回退全表扫描并在后台自动补建，逐步收敛；也可手动预热：

```bash
curl -X POST -H "X-API-Key: <API_KEY>" "<BASE_URL>/v1/relations:reindex"
```

- **扫描兜底（`ENABLE_RELATION_INDEX=0` 或索引未覆盖时）**：按 `LOOKUP_DEPENDENTS_MAX_WORKERS` 并行扫描各源表。调参建议先 `4~8` 灰度，观测延迟与 I/O 负载后再上调；出现磁盘抖动或其他接口尾延迟升高时回调或设为 `1` 串行。

> 反向索引是可重建的派生数据：任何异常都不影响主流程，store 层会自动回退扫描；怀疑漂移时用 `:reindex` 一键重建。

### Docker

```bash
cd retrieval_service
cp .env.example .env
docker compose up -d --build
```

数据持久化在 `./data/` 目录（容器内 `/app/data`）。

## API 概览

所有接口需带 `X-API-Key: <API_KEY>` 或 `Authorization: Bearer <API_KEY>` 头；当 `API_KEY` 为空时关闭鉴权（仅本地开发）。

对接 v2 通用能力（collection/document/search）建议直接参考：`docs/lancedb_v2_api.md`。

API 自动 embedding 兜底说明：

- `documents:upsert` 未传 `vector`：服务端会基于 `content` 自动生成向量
- `search` 未传 `query_vector`：服务端会基于 `query_tokenized` 自动生成查询向量
- 若 embedding 失败或维度不匹配：自动降级到 BM25/FTS-only（不阻断请求）

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
| POST | `/v1/relations:reindex` | 全量重建 relation 反向索引（运维 / 预热用） |
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
