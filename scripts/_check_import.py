import inspect
import os
import freetoken
import freetoken.server.args as a

print("freetoken file:", freetoken.__file__)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
src = inspect.getsource(a.parse_args)
print("kv_quant in parse_args:", "kv_quant" in src or "--kv-quant" in src)
print("EngineConfig kv_quant:", hasattr(__import__("freetoken.engine.config", fromlist=["EngineConfig"]).EngineConfig, "__dataclass_fields__"))
from freetoken.engine.config import EngineConfig
print("fields", "kv_quant" in EngineConfig.__dataclass_fields__)
