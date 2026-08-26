from pathlib import Path
import freetoken_kernel_cache

d = Path(freetoken_kernel_cache.get_jit_cache_dir())
print("cache", d)
n = 0
for p in d.iterdir():
    if n < 8:
        print(p.name, [k.name for k in list(p.iterdir())[:10]])
        n += 1
print("count", sum(1 for _ in d.iterdir()))
