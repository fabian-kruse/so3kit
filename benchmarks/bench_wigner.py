"""
Wigner-D matrices for all degrees ell <= ell_max and a batch of rotations.

  so3kit     GPU, complex basis
  e3nn       GPU, real basis, o3.wigner_D: three matrix exponentials per degree
  J-matrix   GPU, real basis, Z(a) J Z(b) J Z(g) with precomputed J matrices
             (fairchem's EquiformerV2 code, vendored in jmatrix_wigner.py)
  spherical  CPU reference (float64); also the ground truth for the error column
Every GPU method gets the same n rotations per call, in single precision. Setup (so3kit tables,
J matrices on the GPU) is timed separately and excluded from the timing.
"""
import argparse
import time

import numpy as np
import quaternionic
import spherical
import torch

from common import cpu_time, gpu_time, random_angles, run_backend, write_csv
from so3kit import WignerDMatrices


def so3kit_time(ell_max, n, precision="float64"):
    a, b, g = random_angles(n)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wig = WignerDMatrices(ell_max, batchsize=n, num_candidates=1, device=torch.device("cuda:0"),
                          precision=precision)
    torch.cuda.synchronize()
    setup = time.perf_counter() - t0
    return gpu_time(lambda: wig.D(alpha=a, beta=b, gamma=g)), setup


def so3kit_error(ell_max, n=200, precision="float64"):
    a, b, g = random_angles(n)
    wig = WignerDMatrices(ell_max, batchsize=n, num_candidates=1, device=torch.device("cuda:0"),
                          precision=precision)
    D = wig.D(alpha=a, beta=b, gamma=g).reshape(n, -1).cpu().numpy()
    R = quaternionic.array.from_euler_angles(*(t.double().cpu().numpy() for t in (a, b, g)))
    w = spherical.Wigner(ell_max)
    return float(np.max(np.abs(D - np.stack([w.D(r) for r in R]))))


# so3kit's default computes in float64 and returns complex64; "float32" computes in float32 like the baselines
SO3KIT_PRECISIONS = {"so3kit": "float64", "so3kit (float32)": "float32"}


def e3nn_time(ell_max, n):
    from e3nn import o3

    a, b, g = random_angles(n)

    def run():
        with torch.device("cuda"):  # e3nn builds its generators on the default device
            for ell in range(ell_max + 1):
                o3.wigner_D(ell, a, b, g)

    return gpu_time(run, reps=3), 0.0


def jmatrix_time(ell_max, n):
    import jmatrix_wigner

    a, b, g = random_angles(n)
    t0 = time.perf_counter()
    jmatrix_wigner.to_device("cuda")
    torch.cuda.synchronize()
    setup = time.perf_counter() - t0

    def run():
        for ell in range(ell_max + 1):
            jmatrix_wigner.wigner_D(ell, a, b, g)

    return gpu_time(run, reps=3), setup


def spherical_time(ell_max, n):
    a, b, g = (t.double().cpu().numpy() for t in random_angles(n))
    R = quaternionic.array.from_euler_angles(a, b, g)
    t0 = time.perf_counter()
    w = spherical.Wigner(ell_max)
    setup = time.perf_counter() - t0
    return cpu_time(lambda: [w.D(r) for r in R], reps=1), setup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000, help="rotations per call on the GPU")
    p.add_argument("--n-cpu", type=int, default=500, help="rotations timed on the CPU")
    p.add_argument("--ell", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--out", default="wigner.csv", help="file name in benchmarks/results, or a path")
    p.add_argument("--backends", nargs="+", default=None, help="run only backends whose name contains one of these")
    args = p.parse_args()
    selected = lambda name: args.backends is None or any(b in name for b in args.backends)

    rows = []
    for ell_max in args.ell:
        err = {name: run_backend(f"{name} error", lambda: so3kit_error(ell_max, precision=prec))
               for name, prec in SO3KIT_PRECISIONS.items() if selected(name)}
        for backend, fn, n in [
            *((name, lambda e, k, p=prec: so3kit_time(e, k, p), args.n) for name, prec in SO3KIT_PRECISIONS.items()),
            ("e3nn", e3nn_time, args.n),
            ("J-matrix", jmatrix_time, args.n),
            ("spherical (CPU)", spherical_time, args.n_cpu),
        ]:
            if not selected(backend):
                continue
            res = run_backend(backend, lambda: fn(ell_max, n))
            if res is None:
                continue
            t, setup = res
            row = dict(backend=backend, ell_max=ell_max, n=n, seconds=t, rotations_per_s=n / t, setup_seconds=setup)
            if backend in err:
                row["max_abs_err_vs_spherical"] = err[backend]
            rows.append(row)
            print(f"ell_max={ell_max:3d}  {backend:16s} {n / t:12.3g} rotations/s", flush=True)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
