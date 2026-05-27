"""服务端 embedding 兜底（OpenAI 兼容 /embeddings）。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import get_settings


def _parse_embeddings(payload: Any, expected_size: int) -> list[list[float]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise ValueError("invalid embeddings payload: missing data[]")

    # OpenAI 兼容格式通常带 index，按 index 排序保证顺序稳定。
    data_sorted = sorted(
        data,
        key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
    )
    vectors: list[list[float]] = []
    for item in data_sorted:
        emb = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(emb, list):
            raise ValueError("invalid embeddings payload: missing embedding")
        vectors.append([float(x) for x in emb])

    if len(vectors) != expected_size:
        raise ValueError(
            f"embeddings size mismatch: got={len(vectors)} expect={expected_size}"
        )
    return vectors


def embed_texts(texts: list[str]) -> tuple[list[list[float]], str]:
    """批量 embedding（失败返回错误字符串，不抛异常）。"""

    clean_texts = [t for t in (texts or []) if (t or "").strip()]
    if not clean_texts:
        return [], "empty input texts"

    settings = get_settings()
    if not settings.enable_server_embedding_fallback:
        return [], "server embedding fallback disabled"

    base_url = (settings.embedding_base_url or "").rstrip("/")
    model = (settings.embedding_model or "").strip()
    if not base_url or not model:
        return [], "embedding config missing (EMBEDDING_BASE_URL / EMBEDDING_MODEL)"

    headers = {"Content-Type": "application/json"}
    if settings.embedding_api_key:
        headers["Authorization"] = f"Bearer {settings.embedding_api_key}"

    body = json.dumps({"model": model, "input": clean_texts}, ensure_ascii=False).encode("utf-8")
    request = Request(f"{base_url}/embeddings", data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=float(settings.embedding_timeout_sec)) as resp:  # noqa: S310
            raw = resp.read()
            payload = json.loads(raw.decode("utf-8"))
        return _parse_embeddings(payload, len(clean_texts)), ""
    except HTTPError as e:
        try:
            detail = e.read().decode("utf-8")[:200]
        except Exception:
            detail = str(e)
        return [], f"embedding http {e.code}: {detail}"
    except URLError as e:
        return [], f"embedding network error: {e}"
    except Exception as e:  # noqa: BLE001
        return [], f"embedding request failed: {e}"


def embed_query(text: str) -> tuple[list[float], str]:
    if not (text or "").strip():
        return [], "empty query text"
    vecs, err = embed_texts([text])
    if err:
        return [], err
    return (vecs[0] if vecs else []), ""
