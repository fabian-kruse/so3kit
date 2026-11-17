"""
Inverse SO(3) FFT (coefficients -> grid) for degrees ell < L, a batch of B functions.

  so3kit   GPU, PyTorch, 2L x 2L x 2L equiangular grid, real output (Re sum c D);
           batches natively (SOFFT batchsize = B)
  s2fft    GPU, JAX, McEwen-Wiaux sampling with (2L-1) x L x (2L-1) points, N = L;
           batches with jax.vmap. Four transforms:
             recursive (setup in call)  wigner.inverse_jax without precomps: the recursion
                                        coefficients are rebuilt inside every call
             recursive                  wigner.inverse_jax with precomps built once
             precomputed kernel         precompute_transforms.wigner, O(N L^3) Wigner-d kernel
             Fourier-Wigner             precompute_transforms.fourier_wigner
           each for complex signals and for real signals (reality=True). so3kit returns a real
           function, so "real" is the comparable task.

Setup (so3kit tables, s2fft precomputes and kernels) is timed separately and excluded from the
transform time; so is JIT compilation. Everything runs in single precision (complex64; JAX
without x64). Kernels are built by s2fft's NumPy constructors in float64 and used in float32.
"""
import argparse
import os
import time

import numpy as np
import torch

from common import gpu_time, run_backend, write_csv
from so3kit import SOFFT

# JAX grabs most of the GPU memory on import by default, which starves PyTorch in the same process.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

S2FFT_METHODS = ["recursive (setup in call)", "recursive", "precomputed kernel", "Fourier-Wigner"]


def so3kit_time(L, B):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sofft = SOFFT(L=L, device=torch.device("cuda:0"), batchsize=B)
    torch.cuda.synchronize()
    setup = time.perf_counter() - t0
    g = torch.Generator().manual_seed(0)
    coeffs = torch.randn(B, sofft.Dsize, dtype=torch.complex64, generator=g).cuda()
    points = sofft.num_alphas * sofft.num_betas * sofft.num_gammas
    return gpu_time(lambda: sofft.eval(coeffs)), setup, points


def s2fft_transform(method, L, N, reality):
    """(setup data, fn(flmn, setup) -> signal) for one s2fft transform; setup is built here."""
    import jax.numpy as jnp
    import s2fft
    from s2fft.precompute_transforms import construct, fourier_wigner
    from s2fft.precompute_transforms import wigner as pwigner

    if method == "recursive (setup in call)":
        return None, lambda flmn, _: s2fft.wigner.inverse_jax(flmn, L=L, N=N, reality=reality)
    if method == "recursive":
        pre = s2fft.generate_precomputes_wigner_jax(L, N, "mw", None, False, reality)
        return pre, lambda flmn, p: s2fft.wigner.inverse_jax(flmn, L=L, N=N, reality=reality, precomps=p)
    if method == "precomputed kernel":
        kernel = jnp.asarray(construct.wigner_kernel(L, N, reality, "mw", forward=False))
        return kernel, lambda flmn, k: pwigner.inverse_transform_jax(flmn, k, L, N, "mw", reality, None)
    if method == "Fourier-Wigner":
        pre = tuple(jnp.asarray(x) for x in construct.fourier_wigner_kernel(L))
        return pre, lambda flmn, p: fourier_wigner.inverse_transform_jax(flmn, L, N, precomps=p, reality=reality, sampling="mw")
    raise ValueError(method)


def s2fft_time(method, L, B, reality, reps=10):
    import jax
    from s2fft.utils.signal_generator import generate_flmn

    N = L
    rng = np.random.default_rng(0)
    flmn = np.stack([generate_flmn(rng, L, N, reality=reality) for _ in range(B)]).astype(np.complex64)
    flmn = jax.device_put(flmn)
    t0 = time.perf_counter()
    setup_data, fn = s2fft_transform(method, L, N, reality)
    jax.block_until_ready(setup_data)
    setup = time.perf_counter() - t0
    if B == 1:
        call = jax.jit(lambda f, s: fn(f[0], s)[None])
    else:
        call = jax.jit(jax.vmap(fn, in_axes=(0, None)))
    out = call(flmn, setup_data).block_until_ready()  # compile
    t = gpu_time(lambda: call(flmn, setup_data), reps=reps, sync=jax.block_until_ready)
    return t, setup, int(np.prod(out.shape[1:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--batch", type=int, nargs="+", default=[1], help="functions per call")
    p.add_argument("--methods", nargs="+", default=S2FFT_METHODS, choices=S2FFT_METHODS)
    p.add_argument("--out", default="sofft.csv", help="file name in benchmarks/results, or a path")
    args = p.parse_args()

    rows = []
    for L in args.L:
        for B in args.batch:
            runs = [("so3kit", "real", lambda: so3kit_time(L, B))]
            for method in args.methods:
                for reality in (False, True):
                    runs.append((f"s2fft {method}", "real" if reality else "complex",
                                 lambda m=method, r=reality: s2fft_time(m, L, B, r)))
            for backend, signal, fn in runs:
                res = run_backend(f"{backend} ({signal}) L={L} B={B}", fn)
                if res is None:
                    continue
                t, setup, points = res
                rows.append(dict(backend=backend, signal=signal, L=L, batch=B, seconds=t,
                                 seconds_per_function=t / B, setup_seconds=setup, grid_points=points))
                print(f"L={L:3d} B={B:3d}  {backend:36s} {signal:7s} {t * 1e3:10.3f} ms/call "
                      f"{t / B * 1e3:10.4f} ms/function  setup {setup:8.2f} s  ({points} grid points)", flush=True)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
