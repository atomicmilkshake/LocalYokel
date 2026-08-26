import os
import re
from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


def _get_addr(prefix: str, offset: int, unique_suffix: str) -> str:
    if os.name == "nt":
        match = re.search(r"\d+", unique_suffix)
        pid_val = int(match.group(0)) if match else 20000
        port = 20000 + ((pid_val % 1000) * 10) + offset
        return f"tcp://127.0.0.1:{port}"
    return prefix + unique_suffix


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return _get_addr("ipc:///tmp/freetoken_0", 0, self._unique_suffix)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return _get_addr("ipc:///tmp/freetoken_1", 1, self._unique_suffix)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return _get_addr("ipc:///tmp/freetoken_2", 2, self._unique_suffix)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
