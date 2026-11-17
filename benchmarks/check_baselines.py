"""
Correctness checks for the benchmark baselines (not for so3kit), at the precision they are timed in.

  wigner       e3nn and J-matrix Wigner-D in float32. Basis-independent checks against float64
               rotations: orthogonality D D^T = I, homomorphism D(R1) D(R2) = D(R1 R2), and the
               character tr D_l(R) = sin((2l+1) w / 2) / sin(w / 2) for rotation angle w.
               Also e3nn vs J-matrix entry by entry.
  s2fft        every inverse transform used in bench_sofft.py (complex and real signals) against a
               float64 direct sum with spherical's Wigner-D at random grid points. The D convention
               is searched over conjugation, transposition and a (-1)^(m+n) sign; the best match is
               reported with its name.
  derivatives  every autodiff gradient/Hessian in bench_derivatives.py against float64 central
               finite differences of the same function (e3nn's D in float64).

Errors are relative to the largest reference value (max |x - ref| / max |ref|).
Usage: python check_baselines.py [--out FILE]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from common import random_angles, run_backend, write_csv

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROWS = []


def report(check, impl, case, err, tol, note=""):
    ok = bool(err <= tol)
    ROWS.append(dict(check=check, impl=impl, case=case, rel_err=float(err), tol=tol, ok=ok, note=note))
    print(f"{check:11s} {impl:42s} {case:26s} {err:10.2e} (tol {tol:.0e}) {'ok' if ok else 'FAIL'} {note}", flush=True)


def rel(x, ref):
    return float((x - ref).abs().max() / ref.abs().max()) if torch.is_tensor(x) else float(np.abs(x - ref).max() / np.abs(ref).max())


# ---- Wigner-D ----------------------------------------------------------------------------------

def check_wigner(ells=(1, 2, 4, 8, 16, 32), n=256, tol=2e-4):
    from e3nn import o3

    import jmatrix_wigner
    from bench_derivatives import wigner_impl

    # J_l built from e3nn (jmatrix_wigner.py) vs the J_l fairchem ships for l <= 11
    Jf = torch.load(Path(__file__).parent / "fairchem_Jd.pt", weights_only=True)
    Jm = jmatrix_wigner.build_Jd(len(Jf) - 1, "cpu", torch.float64)
    for l, J in enumerate(Jf):
        J = J.double()
        report("wigner", "J_l from e3nn vs fairchem Jd.pt", f"l={l} (up to sign)",
               min((Jm[l] - J).abs().max(), (Jm[l] + J).abs().max()), 1e-6)

    A1 = [t.double() for t in random_angles(n, seed=1)]
    A2 = [t.double() for t in random_angles(n, seed=2)]
    R1, R2 = o3.angles_to_matrix(*A1), o3.angles_to_matrix(*A2)
    A12 = list(o3.matrix_to_angles(R1 @ R2))
    w = torch.arccos(((R1.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    impls = {name: wigner_impl(name) for name in ("e3nn", "J-matrix")}
    with torch.device("cuda"):
        for l in ells:
            chi = torch.sin((2 * l + 1) * w / 2) / torch.sin(w / 2)
            eye = torch.eye(2 * l + 1, dtype=torch.float64)
            D1s = {}
            for name, D in impls.items():
                D1, D2, D12 = (D(l, *[t.float() for t in A]).double() for A in (A1, A2, A12))
                D1s[name] = D1
                report("wigner", name, f"l={l} orthogonality", (D1 @ D1.mT - eye).abs().max(), tol)
                report("wigner", name, f"l={l} homomorphism", (D1 @ D2 - D12).abs().max(), tol)
                report("wigner", name, f"l={l} character", ((D1.diagonal(dim1=-2, dim2=-1).sum(-1) - chi).abs() / (2 * l + 1)).max(), tol)
            report("wigner", "e3nn vs J-matrix", f"l={l} entries", (D1s["e3nn"] - D1s["J-matrix"]).abs().max(), tol)


# ---- s2fft -------------------------------------------------------------------------------------

def check_s2fft(Ls=(8, 16, 32), npts=200, tol=1e-4):
    import jax
    import jax.numpy as jnp
    import quaternionic
    import spherical
    from s2fft.sampling import s2_samples
    from s2fft.utils.signal_generator import generate_flmn

    from bench_sofft import S2FFT_METHODS, s2fft_transform

    for L in Ls:
        N = L
        rng = np.random.default_rng(L)
        alphas = s2_samples.phis_equiang(L, "mw")
        betas = s2_samples.thetas(L, "mw")
        gammas = 2 * np.pi * np.arange(2 * N - 1) / (2 * N - 1)
        ig, ib, ia = (rng.integers(len(x), size=npts) for x in (gammas, betas, alphas))
        R = quaternionic.array.from_euler_angles(alphas[ia], betas[ib], gammas[ig])
        w = spherical.Wigner(L - 1)
        Dsph = np.stack([w.D(r) for r in R])  # (npts, Dsize), spherical's D(l, mp, m)
        lmn = [(l, m, n) for l in range(L) for m in range(-l, l + 1) for n in range(-l, l + 1)]
        l_, m_, n_ = (np.array(v) for v in zip(*lmn))
        i_mn = np.array([w.Dindex(l, m, n) for l, m, n in lmn])
        i_nm = np.array([w.Dindex(l, n, m) for l, m, n in lmn])
        weight = (2 * l_ + 1) / (8 * np.pi ** 2)
        sign = (-1.0) ** (m_ + n_)
        variants = {"conj D[m,n]": np.conj(Dsph[:, i_mn]), "D[m,n]": Dsph[:, i_mn],
                    "conj D[n,m]": np.conj(Dsph[:, i_nm]), "D[n,m]": Dsph[:, i_nm]}
        for reality in (False, True):
            flmn = generate_flmn(rng, L, N, reality=reality)
            c = weight * flmn[n_ + N - 1, l_, m_ + L - 1]
            truths = {}
            for vname, X in variants.items():
                truths[vname] = X @ c
                truths[vname + " (-1)^(m+n)"] = X @ (c * sign)
            for method in S2FFT_METHODS:
                setup, fn = s2fft_transform(method, L, N, reality)
                out = np.asarray(jax.jit(fn)(jnp.asarray(flmn.astype(np.complex64)), setup))
                vals = out[ig, ib, ia]
                errs = {k: rel(vals, t) for k, t in truths.items()}
                best = min(errs, key=errs.get)
                report("s2fft", f"{method} ({'real' if reality else 'complex'})", f"L={L} vs float64 sum",
                       errs[best], tol, note=f"convention: {best}")


# ---- autodiff derivatives ------------------------------------------------------------------------

def check_derivatives(ells=(4, 8), n=64, h=1e-4, tol=1e-3):
    from e3nn import o3

    from bench_derivatives import AD_STYLES, make_f, wigner_impl

    for ell_max in ells:
        g = torch.Generator().manual_seed(0)
        C = [torch.randn(2 * l + 1, 2 * l + 1, generator=g).cuda() for l in range(ell_max + 1)]
        x = torch.stack(random_angles(n, seed=3), dim=1)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)  # e3nn builds its generators in the default dtype
        try:
            with torch.device("cuda"):
                f = make_f(o3.wigner_D, [c.double() for c in C], ell_max)
                X, E = x.double(), torch.eye(3)
                G_ref = torch.stack([(f(X + h * E[i]) - f(X - h * E[i])) / (2 * h) for i in range(3)], -1)
                H_ref = torch.stack([torch.stack([
                    (f(X + h * E[i] + h * E[j]) - f(X + h * E[i] - h * E[j])
                     - f(X - h * E[i] + h * E[j]) + f(X - h * E[i] - h * E[j])) / (4 * h * h)
                    for j in range(3)], -1) for i in range(3)], -2)
        finally:
            torch.set_default_dtype(prev)
        for impl in ("e3nn", "J-matrix"):
            for style, make in AD_STYLES.items():
                def run():
                    with torch.device("cuda"):
                        return make(make_f(wigner_impl(impl), C, ell_max))(x)
                res = run_backend(f"autodiff {impl} ({style})", run)
                if res is None:
                    continue
                G, H = res
                report("derivative", f"autodiff {impl} ({style})", f"ell_max={ell_max} gradient", rel(G.double(), G_ref), tol)
                report("derivative", f"autodiff {impl} ({style})", f"ell_max={ell_max} Hessian", rel(H.double(), H_ref), tol)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="check_baselines.csv", help="file name in benchmarks/results, or a path")
    p.add_argument("--only", nargs="+", default=["wigner", "s2fft", "derivatives"])
    args = p.parse_args()
    checks = {"wigner": check_wigner, "s2fft": check_s2fft, "derivatives": check_derivatives}
    for name in args.only:
        run_backend(f"check {name}", checks[name])
    write_csv(ROWS, args.out)
    bad = [r for r in ROWS if not r["ok"]]
    print(f"{len(ROWS)} checks, {len(bad)} above tolerance")


if __name__ == "__main__":
    main()
