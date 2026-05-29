"""服务端配置，全部通过环境变量 / .env 文件注入。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（lancedb-server/），用于把默认数据目录锚定到与本仓库“同级”的位置，
# 而不是依赖进程启动时的当前工作目录。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 默认 LanceDB 数据目录：与 lancedb-server 文件夹同级的 ../resources。
_DEFAULT_STORE_DIR = str(_PROJECT_ROOT.parent / "resources")


class Settings(BaseSettings):
    """运行时配置。

    - ``STORE_DIR``：LanceDB 数据目录，每个 policy 一张子目录 ``{policy_id}.lance``。
      默认指向与 lancedb-server 文件夹同级的 ``../resources``（绝对路径，不随 cwd 变化）。
    - ``API_KEY``：客户端 X-API-Key / Bearer Token 校验值；为空时关闭鉴权（仅供本地开发）。
    - ``ENABLE_SCALAR_INDEX``：是否在 ``kind`` / ``parent_chunk_index`` 上建标量索引。
      极小表（< 1k chunks）可关掉以省构建时间。
    - ``ENABLE_ASYNC_INDEXING``：是否把 upsert 后的建索引放到后台异步执行。默认开启，
      可显著降低 upsert 响应时延、避免客户端读超时；关掉则恢复同步建索引（旧行为）。
    - ``IDEMPOTENT_APPEND``：是否让 ``append`` 模式按 ``chunk_id`` 幂等 upsert。默认开启，
      客户端重试 / 重复发送同一文档不会产生重复行；关掉则恢复旧的「无脑追加」语义。
    - ``RRF_K``：RRF 融合常量，与主项目 ``inference/config.py::RRF_K`` 默认对齐。
    - ``ENABLE_LEGACY_UI``：是否启用 policy 专用 Gradio（/gradio）。
    - ``ENABLE_GENERIC_GRADIO``：是否启用通用 Gradio（/gradio-generic）。
    """

    store_dir: str = _DEFAULT_STORE_DIR
    api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 5000
    log_level: str = "INFO"
    enable_scalar_index: bool = True
    # 是否把 upsert 后的建索引动作放到后台线程异步执行（写完即返回，避免客户端 read timeout）。
    enable_async_indexing: bool = True
    # 是否让 append 模式按 chunk_id 幂等 upsert（同 id 更新、新 id 插入），避免重试产生重复行。
    # 关掉则恢复旧的「无脑追加」语义（允许同 chunk_id 多行）。
    idempotent_append: bool = True
    rrf_k: int = 60
    enable_generic_api: bool = True
    enable_legacy_relations: bool = True
    enable_legacy_ui: bool = True
    enable_generic_gradio: bool = True
    gradio_use_http: bool = False
    gradio_api_base_url: str = ""
    gradio_api_key: str = ""
    enable_server_embedding_fallback: bool = True
    embedding_base_url: str = "http://mlp.paas.dc.servyou-it.com/qwen3-embedding/v1"
    embedding_model: str = "qwen3-embedding"
    embedding_api_key: str = ""
    embedding_timeout_sec: float = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
