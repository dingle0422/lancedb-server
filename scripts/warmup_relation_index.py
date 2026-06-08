"""一次性预热 / 重建 relation 反向索引。

部署更新后运行一次即可：调用 ``POST /v1/relations:reindex`` 让服务端清空并按所有
policy 表重建反向索引（``STORE_DIR/_relation_index.sqlite3``）。重建完成后，
``/v1/relations:lookup-dependents`` 即为毫秒级点查，主项目 cascade 触发不再踩慢扫描。

幂等：可重复运行；不传 ``--api-key`` 时按服务端 ``API_KEY`` 为空（关闭鉴权）处理。

示例：

    python scripts/warmup_relation_index.py \\
        --base-url http://mlp.paas.dc.servyou-it.com/kh-lancedb \\
        --api-key "$RETRIEVAL_SERVICE_API_KEY"

可选用 ``--verify-target`` 在重建后做一次反查抽样，确认链路通畅并打印耗时。
"""

from __future__ import annotations

import argparse
import time

import httpx


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _reindex(client: httpx.Client, base_url: str, api_key: str) -> dict:
    url = f"{base_url.rstrip('/')}/v1/relations:reindex"
    t0 = time.perf_counter()
    resp = client.post(url, headers=_headers(api_key))
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    payload = resp.json()
    payload["_elapsed_sec"] = round(elapsed, 3)
    return payload


def _verify(
    client: httpx.Client, base_url: str, api_key: str, target_policy_id: str
) -> dict:
    url = f"{base_url.rstrip('/')}/v1/relations:lookup-dependents"
    t0 = time.perf_counter()
    resp = client.get(
        url, headers=_headers(api_key), params={"target_policy_id": target_policy_id}
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    deps = resp.json().get("dependents", [])
    return {"n_dependents": len(deps), "_elapsed_sec": round(elapsed, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up / rebuild relation reverse index")
    parser.add_argument(
        "--base-url",
        required=True,
        help="服务地址，含反向代理前缀，例如 http://host/kh-lancedb",
    )
    parser.add_argument("--api-key", default="", help="X-API-Key；服务端未启用鉴权时留空")
    parser.add_argument(
        "--timeout",
        default=300.0,
        type=float,
        help="HTTP 超时秒数；首次全量重建可能较久，默认 300s",
    )
    parser.add_argument(
        "--verify-target",
        default="",
        help="可选：重建后用该 target_policy_id 做一次反查抽样验证",
    )
    args = parser.parse_args()

    with httpx.Client(timeout=args.timeout) as client:
        print(f"[warmup] 触发重建 -> {args.base_url.rstrip('/')}/v1/relations:reindex")
        try:
            result = _reindex(client, args.base_url, args.api_key)
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response is not None else ""
            raise SystemExit(f"[warmup] 重建失败：HTTP {e.response.status_code} {body}")
        except httpx.HTTPError as e:
            raise SystemExit(f"[warmup] 重建请求异常：{e}")

        sources = result.get("sources")
        print(
            f"[warmup] 重建完成：sources={sources} "
            f"耗时={result.get('_elapsed_sec')}s"
        )

        if args.verify_target:
            v = _verify(client, args.base_url, args.api_key, args.verify_target)
            print(
                f"[warmup] 抽样反查 target={args.verify_target}: "
                f"dependents={v['n_dependents']} 耗时={v['_elapsed_sec']}s"
            )

    print("[warmup] 完成。后续 lookup-dependents 将走点查。")


if __name__ == "__main__":
    main()
