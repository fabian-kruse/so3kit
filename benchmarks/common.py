import csv
import math
import time
from pathlib import Path

import torch

RESULTS = Path(__file__).parent / "results"


def gpu_time(fn, reps=10, warmup=2, repeats=5, sync=torch.cuda.synchronize):
    """
    Wall time per call of fn() in seconds: `reps` calls are queued back to back and synchronised
    once at the end (no host sync between calls); the best mean of `repeats` such loops is returned.
    `sync(last_result)` waits for the device; the default synchronises CUDA (torch), bench_sofft
    passes JAX's block_until_ready.
    """
    out = None
    for _ in range(warmup):
        out = fn()
    _sync(sync, out)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn()
        _sync(sync, out)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def _sync(sync, out):
    if sync is torch.cuda.synchronize:
        torch.cuda.synchronize()
    else:
        sync(out)


def cpu_time(fn, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def random_angles(n, device="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)
    alpha = torch.rand(n, generator=g) * 2 * math.pi
    beta = torch.rand(n, generator=g) * math.pi
    gamma = torch.rand(n, generator=g) * 2 * math.pi
    return alpha.to(device), beta.to(device), gamma.to(device)


def write_csv(rows, name):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name  # an absolute name replaces RESULTS
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def run_backend(name, fn):
    """Run one baseline; a missing package or API change skips it instead of aborting the suite."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
        return None
