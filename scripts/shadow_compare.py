"""对比两套检索服务的 shadow 结果。

输入文件为 JSONL，每行一个 `/v1/policies/{policy_id}/search` 请求体。
脚本会比较：
- top-k 命中 ID（chunk_id/document_id）
- 请求延迟
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def _load_queries(path: Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        queries.append(json.loads(s))
    return queries


def _call_search(
    client: httpx.Client,
    base_url: str,
    policy_id: str,
    body: dict[str, Any],
    api_key: str,
) -> tuple[list[dict[str, Any]], float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    url = f"{base_url.rstrip('/')}/v1/policies/{policy_id}/search"
    t0 = time.perf_counter()
    resp = client.post(url, headers=headers, json=body)
    elapsed = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    payload = resp.json()
    return list(payload.get("hits", [])), elapsed


def _extract_ids(hits: list[dict[str, Any]], top_k: int) -> list[str]:
    out: list[str] = []
    for item in hits[:top_k]:
        if "chunk_id" in item:
            out.append(str(item["chunk_id"]))
        elif "document_id" in item:
            out.append(str(item["document_id"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow compare retrieval results")
    parser.add_argument("--legacy-base-url", required=True)
    parser.add_argument("--candidate-base-url", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--queries-file", required=True, type=Path)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--top-k", default=10, type=int)
    args = parser.parse_args()

    queries = _load_queries(args.queries_file)
    if not queries:
        raise SystemExit("queries-file is empty")

    legacy_lat_ms: list[float] = []
    cand_lat_ms: list[float] = []
    mismatch = 0

    with httpx.Client(timeout=20.0) as client:
        for idx, body in enumerate(queries, start=1):
            legacy_hits, legacy_ms = _call_search(
                client,
                args.legacy_base_url,
                args.policy_id,
                body,
                args.api_key,
            )
            cand_hits, cand_ms = _call_search(
                client,
                args.candidate_base_url,
                args.policy_id,
                body,
                args.api_key,
            )
            legacy_lat_ms.append(legacy_ms)
            cand_lat_ms.append(cand_ms)
            legacy_ids = _extract_ids(legacy_hits, args.top_k)
            cand_ids = _extract_ids(cand_hits, args.top_k)
            if legacy_ids != cand_ids:
                mismatch += 1
                print(
                    f"[DIFF] query#{idx}: legacy={legacy_ids} candidate={cand_ids}",
                )

    total = len(queries)
    print(f"queries={total}")
    print(f"mismatch={mismatch} ({(mismatch / total) * 100:.2f}%)")
    print(f"legacy_avg_ms={statistics.mean(legacy_lat_ms):.2f}")
    print(f"candidate_avg_ms={statistics.mean(cand_lat_ms):.2f}")


if __name__ == "__main__":
    main()
