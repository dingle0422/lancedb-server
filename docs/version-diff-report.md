# 版本对比报告（Understand Diff）

> 生成时间：2026-05-28 03:17 UTC  
> 知识图谱基准：`ba57eea`（`2026-05-27T07:20:33.724275+00:00`）  
> 对比工具：`/understand-diff` + `.understand-anything/knowledge-graph.json`

---

## 1. 版本概览

| 版本 | Commit | 说明 | 时间 |
|------|--------|------|------|
| **当前版本** | `33371dc` | understand-anything | 2026-05-27 |
| **上一版本** | `ba57eea` | fallback of embedding serve | 2026-05-27 |

当前 `HEAD` 相对 `HEAD^` 的 **git 变更**仅包含 Understand 分析产物（4 个文件，+10996 行）。  
**上一版**（`ba57eea`）相对其父提交的核心业务变更为 **API 服务端 embedding 自动兜底**（7 个文件，+291 / -9 行）。

下文分别给出两次对比的知识图谱影响分析；**业务评审建议以第 3 节（embedding 兜底）为主**。

---

## 2. 当前版本 vs 上一版本（`ba57eea` → `33371dc`）

### 2.1 变更文件

| 文件 | 变更量 |
|------|--------|
| `.understand-anything/.understandignore` | +35 |
| `.understand-anything/fingerprints.json` | +3683 |
| `.understand-anything/knowledge-graph.json` | +7272 |
| `.understand-anything/meta.json` | +6 |

### 2.2 知识图谱影响

- **直接变更节点**：1（`file:.understand-anything/.understandignore`）
- **间接影响节点**：0
- **触及架构层**：Operations & Utilities

**说明**：本次提交为代码理解/可视化基础设施，不改变运行时 API 行为。

---

## 3. 上一版本业务变更（`ba57eea` 相对其父提交）

> 提交信息：`fallback of embedding serve`  
> 统计：**7 files changed, 291 insertions(+), 9 deletions(-)**

### 3.1 变更文件清单

| 文件 | 类型 | 复杂度 | 摘要 |
|------|------|--------|------|
| `app/store.py` | file | complex | Core LanceDB persistence layer: policy-scoped tables, upsert with schema normalization, hybrid FTS+v… |
| `app/config.py` | file | simple | Pydantic Settings loaded from environment and .env; exposes cached get_settings() for host, store pa… |
| `app/embedding_fallback.py` | file | moderate | OpenAI-compatible /embeddings HTTP client used when vectors are missing; batch embed_texts and singl… |
| `.env.example` | config | simple | Template environment variables for embedding service URLs, proxy root path, API key, store directory… |
| `README.md` | document | moderate | Primary project documentation covering LanceDB hybrid retrieval architecture, PyArrow table schema, … |
| `docs/lancedb_v2_api.md` | document | complex | Comprehensive v2 API reference covering capabilities negotiation, collection/document models, CRUD e… |
| `tests/test_routes.py` | file | complex | End-to-end pytest suite covering v1 lifecycle, v2 collection flows, auto-embedding, metadata roundtr… |

### 3.2 功能变更摘要

1. **新增** `app/embedding_fallback.py`  
   - OpenAI 兼容 `POST {EMBEDDING_BASE_URL}/embeddings`  
   - 提供 `embed_texts()` / `embed_query()`，失败返回错误字符串、不抛异常

2. **扩展** `app/config.py`  
   - `ENABLE_SERVER_EMBEDDING_FALLBACK`（默认 `true`）  
   - `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` / `EMBEDDING_TIMEOUT_SEC`

3. **改造** `app/store.py`  
   - `upsert`：缺 `vector` 时调用 `_auto_embed_missing_vectors()` 按 `content` 补向量  
   - `hybrid_search`：缺 `query_vector` 时基于 `query_tokenized` 自动 embedding  
   - 失败或维度不匹配时降级（零向量 / BM25-only）

4. **测试** `tests/test_routes.py`  
   - 新增 v2 upsert/search 自动 embedding 用例

5. **文档** `README.md`、`docs/lancedb_v2_api.md`、`.env.example`  
   - 明确 API 与 Gradio 共用 embedding 配置；说明兜底与降级行为

### 3.3 知识图谱：直接变更

- **变更节点数**：69
- **涉及文件级节点**：7

**核心函数（新增/改造）**：

- `function:app/embedding_fallback.py:embed_texts`
- `function:app/embedding_fallback.py:embed_query`
- `function:app/store.py:_auto_embed_missing_vectors`
- `function:app/store.py:hybrid_search`
- `function:tests/test_routes.py:test_v2_upsert_auto_embedding_when_vector_missing`
- `function:tests/test_routes.py:test_v2_search_auto_embedding_when_query_vector_missing`

### 3.4 知识图谱：间接影响（1-hop）

- **受影响节点数**：41

**上游调用方（可能受 store/config 行为变化影响）**：

- `file:app/routers/chunks.py`
- `file:app/routers/search.py`
- `file:app/routers/v2.py`
- `file:app/routers/relations.py`
- `file:app/routers/meta.py`
- `file:app/generic/core.py`
- `file:app/ui_gradio.py`
- `file:app/main.py`
- `file:main.py`
- `file:app/deps.py`

**下游依赖**：

- `file:app/schema.py`
- `class:app/schema.py:SearchHit`
- `class:app/schema.py:RelationKey`
- `file:app/main.py`
- `file:main.py`
- `config:pyproject.toml`

### 3.5 触及架构层

| 架构层 | 直接变更 | 间接影响 |
|--------|----------|----------|
| API Layer | — | `app/routers/meta.py`, `app/routers/relations.py`, `app/routers/search.py`, `app/routers/v2.py`, `app/routers/chunks.py`, `app/routers/health.py` |
| Retrieval Service Layer | `app/store.py`, `app/embedding_fallback.py` | `app/generic/core.py`, `app/deps.py`, `retrieval/bm25.py`, `retrieval/rrf.py` |
| Schema & Types Layer | — | `app/schema.py`, `app/schema_generic.py` |
| UI Layer | — | `app/ui_gradio.py`, `app/ui_gradio_generic.py`, `app/static/ui.html` |
| Application Bootstrap | `app/config.py` | `app/main.py`, `main.py` |
| Documentation | `README.md`, `docs/lancedb_v2_api.md` | `docs/migration_runbook.md`, `requirements.txt` |
| Configuration | `.env.example` | `pyproject.toml` |
| Test Layer | `tests/test_routes.py` | — |
| Operations & Utilities | — | `scripts/shadow_compare.py` |

### 3.6 风险评估

**等级：中偏高**

| 维度 | 说明 |
|------|------|
| 爆炸半径 | `store.py`（complex）是 v1/v2 写入与检索唯一内核，影响全部 API 路由 |
| 外部依赖 | 新增 embedding HTTP 调用；不可用时降级为 BM25/FTS，行为与旧版不同 |
| 跨层耦合 | Service → API → UI 三层联动；Gradio 仍走独立 `_embed_query`，与 API 兜底双轨 |
| 维度风险 | auto-embed 维度 mismatch 会跳过向量召回或补零向量，需回归 hybrid 质量 |
| 延迟 | `EMBEDDING_TIMEOUT_SEC=10` 可能拉高 upsert/search P99 |

### 3.7 建议 Review 清单

- [ ] `store._auto_embed_missing_vectors` 与 `_infer_dim` / 零向量补齐的交互是否正确
- [ ] `hybrid_search` 在 `query_tokenized` 为空时是否仍误触发 embedding
- [ ] v2 全路径（`generic/core` → `store`）是否一致受益；relations 直连 store 是否需单独验证
- [ ] embedding 服务故障时的降级日志与监控是否足够
- [ ] 文档与 `.env.example` 默认值是否与生产环境 embedding 服务一致

---

## 4. Dashboard 可视化

已写入 diff overlay：`.understand-anything/diff-overlay.json`

- `changedNodeIds`：70
- `affectedNodeIds`：41

运行 `/understand-dashboard` 并刷新页面，可在图谱中高亮上述变更与影响范围。

---

## 5. 附录：Git 命令

```bash
# 当前 vs 上一 commit
git diff ba57eea..33371dc --stat

# 上一版业务功能 commit
git diff ba57eea^..ba57eea --stat
git show ba57eea
```
