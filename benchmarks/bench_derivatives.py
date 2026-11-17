"""
Gradient and Hessian of a bandlimited function on SO(3) at a batch of rotations.

  so3kit     analytic (one Wigner-D evaluation, then linear combinations)
  autodiff   torch.func / torch.autograd through a Wigner-D implementation (real basis):
               e3nn       o3.wigner_D (three matrix exponentials per degree)
               J-matrix   Z(a) J Z(b) J Z(g) with precomputed J (fairchem's EquiformerV2 code,
                          vendored in jmatrix_wigner.py)
             in three autodiff styles:
               2 passes              vmap(grad) and vmap(hessian) separately (the original baseline)
               forward-over-reverse  vmap(jacfwd(grad)), gradient returned as aux of the same pass
               batched reverse       one backward with create_graph, then 3 double-backwards;
                                     points are independent, so d(sum f)/dx gives per-point values
Every method gets the same n points per call, in float32. Setup (so3kit tables, J matrices on
the GPU) is excluded from the timing.
"""
import argparse
import time

import torch

from common import gpu_time, random_angles, run_backend, write_csv
from so3kit import SO3Function, WignerDsize


def so3kit_time(ell_max, n, precision="float64"):
    coeffs = torch.randn(1, WignerDsize(0, ell_max, ell_max), dtype=torch.complex64, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    f = SO3Function(coeffs, ell_max, num_points=n, precision=precision)
    torch.cuda.synchronize()
    setup = time.perf_counter() - t0
    a, b, g = (t[None] for t in random_angles(n))
    return gpu_time(lambda: f.grad_hess(a, b, g)), setup


# ---- autodiff baselines (also used by check_baselines.py) ---------------------------------------

def wigner_impl(name):
    """Wigner-D function D(l, alpha, beta, gamma) -> (..., 2l+1, 2l+1), real basis."""
    if name == "e3nn":
        from e3nn import o3
        return o3.wigner_D
    if name == "J-matrix":
        import jmatrix_wigner
        jmatrix_wigner.to_device("cuda")
        return jmatrix_wigner.wigner_D
    raise ValueError(name)


def make_f(D, C, ell_max):
    """f(angles) = sum_l <C_l, D_l(angles)>; angles (..., 3) -> (...)."""
    def f(angles):
        a, b, g = angles.unbind(-1)
        return sum((C[l] * D(l, a, b, g)).sum((-1, -2)) for l in range(ell_max + 1))
    return f


def ad_two_pass(f):
    grad = torch.func.vmap(torch.func.grad(f))
    hess = torch.func.vmap(torch.func.hessian(f))
    return lambda x: (grad(x), hess(x))


def ad_forward_over_reverse(f):
    def grad_aux(x):
        g = torch.func.grad(f)(x)
        return g, g
    jac = torch.func.vmap(torch.func.jacfwd(grad_aux, has_aux=True))

    def run(x):
        H, G = jac(x)
        return G, H
    return run


def ad_batched_reverse(f):
    def run(x):
        x = x.detach().requires_grad_(True)
        (G,) = torch.autograd.grad(f(x).sum(), x, create_graph=True)
        H = torch.stack([torch.autograd.grad(G[:, i].sum(), x, retain_graph=i < 2)[0] for i in range(3)], dim=1)
        return G.detach(), H
    return run


AD_STYLES = {"2 passes": ad_two_pass, "forward-over-reverse": ad_forward_over_reverse,
             "batched reverse": ad_batched_reverse}


def autodiff_time(impl, style, ell_max, n):
    D = wigner_impl(impl)
    g = torch.Generator().manual_seed(0)
    C = [torch.randn(2 * l + 1, 2 * l + 1, generator=g).cuda() for l in range(ell_max + 1)]
    run = AD_STYLES[style](make_f(D, C, ell_max))
    x = torch.stack(random_angles(n), dim=1)

    def call():
        with torch.device("cuda"):  # e3nn builds its generators on the default device
            run(x)

    return gpu_time(call, reps=3, warmup=1), 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_000)
    p.add_argument("--ell", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--out", default="derivatives.csv", help="file name in benchmarks/results, or a path")
    p.add_argument("--backends", nargs="+", default=None, help="run only backends whose name contains one of these")
    args = p.parse_args()
    selected = lambda name: args.backends is None or any(b in name for b in args.backends)

    rows = []
    for ell_max in args.ell:
        # so3kit's default computes in float64; "float32" computes in float32 like the baselines
        runs = [("so3kit", lambda: so3kit_time(ell_max, args.n)),
                ("so3kit (float32)", lambda: so3kit_time(ell_max, args.n, "float32"))]
        for impl in ("e3nn", "J-matrix"):
            for style in AD_STYLES:
                runs.append((f"autodiff {impl} ({style})",
                             lambda i=impl, s=style: autodiff_time(i, s, ell_max, args.n)))
        for backend, fn in runs:
            if not selected(backend):
                continue
            res = run_backend(f"{backend} ell_max={ell_max}", fn)
            if res is None:
                continue
            t, setup = res
            rows.append(dict(backend=backend, ell_max=ell_max, n=args.n, seconds=t,
                             points_per_s=args.n / t, setup_seconds=setup))
            print(f"ell_max={ell_max:3d}  {backend:42s} {args.n / t:12.3g} points/s", flush=True)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
