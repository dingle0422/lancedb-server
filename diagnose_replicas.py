"""lancedb-server 诊断升级版。

证据已经表明：磁盘上 ``data/p_S0gx....lance/`` 这张表存在、_indices 建好了。
那 `检索到 0 个 chunk` 只可能是这几种情况之一：

    A. /gradio 列不出 → 实际访问的是另一个 lancedb-server 实例（多副本）
    B. 客户端 search 时 query_tokenized 空 + query_vector 空（或维度不对）
       → 服务端 hybrid_search 双路径都不进，直接返回 hits=[]，HTTP 200
    C. policy_id 名字客户端读写两侧不一致（多打了下划线 / cs500 后缀漏了）
    D. STORE_DIR 进程跑起来是另一条路径（容器漂移）

这个脚本一次性把这 4 条都验掉：
    - 多次调用 /v1/policies/{id}/meta，确认是否稳定 200，并打印 n_chunks/dim/has_*_index
    - 多次调用 /v1/policies，看返回的总 policies 数 + 是否包含目标 id
    - 调一次 /chunks?limit=3&include_content=true，看真实的 content_tokenized
    - 用表里的真实 token 做一次纯 BM25 search（query_vector=[]），看能不能命中
    - 全部带响应头打印（暴露 X-Pod-Name / Server / Set-Cookie 之类的副本指纹）

用法：
    1. 改下面 BASE_URL / API_KEY，POLICY_ID 已经填好了
    2. python diagnose_replicas.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from urllib.parse import quote

import httpx

BASE_URL = "https://your.domain/kh-lancedb"   # ← 改这里
API_KEY = "changeme"                           # ← 改这里
POLICY_ID = "KH1498713693824016384_20260515094327__cs500"

N_META = 10
N_LIST = 10


def headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY, "Accept": "application/json"}


def _fingerprint(resp: httpx.Response) -> str:
    interesting = ("server", "x-upstream", "x-served-by", "x-pod", "x-pod-name",
                   "set-cookie", "x-real-ip", "via")
    return " | ".join(
        f"{k}={v}" for k, v in resp.headers.items() if k.lower() in interesting
    ) or "<no upstream hint>"


def probe_meta(client: httpx.Client) -> bool:
    print(f"\n[1] 连续 {N_META} 次 GET /v1/policies/{POLICY_ID}/meta")
    url = f"{BASE_URL}/v1/policies/{quote(POLICY_ID, safe='')}/meta"
    codes: Counter[int] = Counter()
    fps: Counter[str] = Counter()
    last_ok_body = None
    last_404_body = None
    for i in range(N_META):
        try:
            r = client.get(url, headers=headers(), timeout=10)
        except Exception as e:
            print(f"  #{i:02d} 请求失败: {e}")
            codes[-1] += 1
            continue
        codes[r.status_code] += 1
        fps[_fingerprint(r)] += 1
        if r.status_code == 200 and last_ok_body is None:
            last_ok_body = r.text[:500]
        if r.status_code == 404 and last_404_body is None:
            last_404_body = r.text[:200]
    print(f"  status 分布: {dict(codes)}")
    print(f"  上游/cookie 指纹分布: {dict(fps)}")
    if last_ok_body:
        print(f"  200 样本: {last_ok_body}")
    if last_404_body:
        print(f"  404 样本: {last_404_body}")
    return codes.get(200, 0) > 0


def probe_list(client: httpx.Client) -> None:
    print(f"\n[2] 连续 {N_LIST} 次 GET /v1/policies，看列表里是否包含目标")
    url = f"{BASE_URL}/v1/policies"
    contains_counter: Counter[bool] = Counter()
    sizes: Counter[int] = Counter()
    fps: Counter[str] = Counter()
    for i in range(N_LIST):
        try:
            r = client.get(url, headers=headers(), timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  #{i:02d} 请求失败: {e}")
            continue
        fps[_fingerprint(r)] += 1
        policies = [p["policy_id"] for p in data.get("policies", [])]
        contains_counter[POLICY_ID in policies] += 1
        sizes[len(policies)] += 1
    print(f"  '列表包含目标 policy_id?': {dict(contains_counter)}")
    print(f"  总 policies 数分布: {dict(sizes)}  ← 出现 2 个不同总数 = 多副本")
    print(f"  上游指纹: {dict(fps)}")


def probe_chunks(client: httpx.Client) -> list[str]:
    """返回前 3 行的 content_tokenized 拆出来的真实 token，用于后续 BM25 自我验证。"""

    print(f"\n[3] GET /v1/policies/{POLICY_ID}/chunks?limit=3&include_content=true")
    url = (
        f"{BASE_URL}/v1/policies/{quote(POLICY_ID, safe='')}/chunks"
        "?limit=3&include_content=true"
    )
    try:
        r = client.get(url, headers=headers(), timeout=15)
    except Exception as e:
        print(f"  失败: {e}")
        return []
    print(f"  status={r.status_code} 指纹={_fingerprint(r)}")
    if r.status_code != 200:
        print(f"  body: {r.text[:500]}")
        return []
    data = r.json()
    if not isinstance(data, list):
        print(f"  奇怪的返回结构: {r.text[:300]}")
        return []
    print(f"  返回 {len(data)} 行")
    sample_tokens: list[str] = []
    for i, row in enumerate(data):
        cid = row.get("chunk_id")
        content = (row.get("content") or "")[:80]
        print(f"  row#{i}: chunk_id={cid} content_preview={content!r}")
        # 注意 chunks 列表接口没返回 content_tokenized 列（_select_columns 没含它）；
        # 这里用 content 头几字（中文）作为 BM25 探针不一定命中（FTS 用 whitespace tokenizer
        # 且建索引时是 jieba 分词后的空格串）。仍可作为可读性参考。
        if content:
            sample_tokens.append(content[:4])
    return sample_tokens


def probe_search(client: httpx.Client, probes: list[str]) -> None:
    """
    用几种不同输入对同一张表打 search，看是哪条路径有/没有命中。

    case 1: query_tokenized="" + query_vector=[]  → 期望 hits=0 （服务端 hybrid 短路）
    case 2: query_tokenized=<probe>(取自表的真实内容) + query_vector=[] → 走 BM25 only
    case 3: query_tokenized="" + query_vector=[0.1]*dim → 走纯向量（dim 错就 0 hits）
    """

    print(f"\n[4] POST /v1/policies/{POLICY_ID}/search 多个变体")
    url = f"{BASE_URL}/v1/policies/{quote(POLICY_ID, safe='')}/search"

    def _hit(body: dict, label: str) -> None:
        try:
            r = client.post(url, headers=headers(), json=body, timeout=20)
        except Exception as e:
            print(f"  [{label}] 失败: {e}")
            return
        if r.status_code != 200:
            print(f"  [{label}] status={r.status_code} body={r.text[:200]}")
            return
        try:
            hits = r.json().get("hits", [])
        except Exception:
            hits = []
        print(f"  [{label}] status=200 hits={len(hits)} 指纹={_fingerprint(r)}")

    _hit(
        {"query_tokenized": "", "query_vector": [], "top_n": 5, "top_m": 5, "include_content": False},
        "case1: 全空（期望 0 hits）",
    )

    for probe in (probes[:1] or [""]):
        if not probe:
            continue
        _hit(
            {
                "query_tokenized": probe,
                "query_vector": [],
                "top_n": 0,
                "top_m": 10,
                "include_content": False,
            },
            f"case2: BM25 only, token={probe!r}",
        )

    _hit(
        {
            "query_tokenized": "",
            "query_vector": [0.01] * 1024,  # 与建索引时的 dim 一致（log 显示 dim=1024）
            "top_n": 5,
            "top_m": 0,
            "include_content": False,
        },
        "case3: 向量 only (dim=1024 全 0.01)",
    )


def main() -> int:
    if BASE_URL.startswith("https://your.domain"):
        print("先在脚本顶部改 BASE_URL / API_KEY 再跑")
        return 2
    print(f"目标: {BASE_URL}  policy_id={POLICY_ID}")
    with httpx.Client(verify=False) as client:
        has_meta = probe_meta(client)
        probe_list(client)
        if not has_meta:
            print("\n所有 meta 都 404 了，先确认 BASE_URL 真的指向 upsert 命中的那个 lancedb-server")
            return 1
        probes = probe_chunks(client)
        probe_search(client, probes)
    print(
        "\n判读小抄："
        "\n  - [1] meta 抖动 / 全 404 → 多副本路由问题（缩 1 副本 / 挂共享盘）"
        "\n  - [2] 总 policies 数分布有 2 个值 → 多副本铁证"
        "\n  - [3] chunks 接口能拿出 3 行 → 数据完整，问题在 search 输入"
        "\n  - [4] case2 BM25 还能命中说明索引 OK → 客户端真正发的 query_tokenized/query_vector 有问题"
        "\n  - [4] case1/2/3 全 0 命中 → 索引建坏（罕见），或者 search 命中了空副本"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
