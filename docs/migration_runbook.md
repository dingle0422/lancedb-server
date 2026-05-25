# 通用化改造迁移运行手册

## 目标

在不影响既有 `v1` 调用方的前提下，逐步把服务迁移到通用 `v2` 能力，并保留快速回滚路径。

## 阶段化执行

### 阶段1：暗发布（不切流量）

- 部署新版本服务，先验证：
  - `GET /healthz`
  - `GET /v1/capabilities`
  - `GET /v2/capabilities`
- 对照旧服务执行同批 upsert，确保目标 `policy_id/collection_id` 的 `meta.n_chunks` 一致。

### 阶段2：Shadow 对比（双读不影响用户）

- 使用 `scripts/shadow_compare.py` 同时请求旧/新服务：
  - 比较 top-k 命中集合（默认 `chunk_id/document_id`）
  - 记录平均延迟差异
- 差异收敛后再进入灰度。

示例：

```bash
python scripts/shadow_compare.py \
  --legacy-base-url http://legacy:5000 \
  --candidate-base-url http://candidate:5000 \
  --policy-id test_pol_v1 \
  --queries-file queries.jsonl
```

`queries.jsonl` 每行一个请求体（与 `/v1/policies/{id}/search` 一致）：

```json
{"query_tokenized":"萝卜 免税","query_vector":[0,1,0,0],"top_n":10,"top_m":10}
```

### 阶段3：灰度切换

- 按 policy/collection 分批切流（建议先只读低风险集合）。
- 监控：
  - 检索 4xx/5xx 比例
  - 空结果比例
  - 检索 P95 延迟

### 阶段4：收口

- 默认前端切换到通用视图。
- 保留 legacy 路由与 relations 插件一段观察期。
- 稳定后仅在 legacy 项目开启 `ENABLE_LEGACY_RELATIONS=1`。

## 回滚策略

1. 入口回滚：将调用方 URL 指回旧服务（首选，影响最小）。
2. 功能回滚：设置 `ENABLE_GENERIC_API=0`，仅暴露既有 `v1`。
3. 领域回滚：设置 `ENABLE_LEGACY_RELATIONS=1`，恢复旧 relations 路由。
4. UI 回滚：设置 `ENABLE_LEGACY_UI=1`，继续使用 `/ui` 和 `/gradio`。

## 发布前检查清单

- [ ] `pytest -q` 全部通过。
- [ ] v1 契约接口字段兼容（包括 `/v1/policies/*` 与 relations）。
- [ ] Shadow 对比误差在可接受范围。
- [ ] 回滚演练通过（URL 回切 + 功能开关）。
