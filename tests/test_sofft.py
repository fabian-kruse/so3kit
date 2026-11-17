# GPU tests of the SO(3) FFT (SOFFT.eval): the fused backend against the legacy one (einsum + scatter_add +
# ifft2, kept as backend="legacy") and both against a float64 direct sum with spherical's d; launch
# configurations, oversampling, input handling, fast vs numba launches.
import numpy as np
import pytest
import spherical
import torch

from so3kit import SOFFT
from so3kit._fused_sofft import FusedSOFFT

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda:0")


def coeffs(sofft, seed=0, dtype=torch.complex64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(sofft.batch_size, sofft.Dsize, dtype=torch.complex128, generator=g).to(dtype).to(DEV)


def reference64(sofft, c, npts=1500, seed=0):
    """f(j, p, q) = Re sum_{l<L,m,n} c^l_mn d^l_mn(beta_j) e^{2 pi i (m p / A + n q / G)} / (A G) in float64 at random
    grid points (beta_j = j pi / (NB - 1), alpha_p = 2 pi p / A, gamma_q = 2 pi q / G)."""
    L, NB, A, G = sofft.L, sofft.num_betas, sofft.num_alphas, sofft.num_gammas
    w = spherical.Wigner(L)
    rng = np.random.default_rng(seed)
    j, p, q = rng.integers(0, NB, npts), rng.integers(0, A, npts), rng.integers(0, G, npts)
    ms = np.array([m for l in range(L) for m in range(-l, l + 1) for _ in range(-l, l + 1)])
    ns = np.array([n for l in range(L) for _ in range(-l, l + 1) for n in range(-l, l + 1)])
    K = ms.size
    betas = np.linspace(0, np.pi, NB, endpoint=True)
    d = np.stack([w.d(np.exp(1j * b))[:K] for b in betas]).real
    cc = c.detach().cpu().numpy().astype(np.complex128)[:, :K]
    ph = np.exp(2j * np.pi * (np.outer(p, ms) / A + np.outer(q, ns) / G))
    out = np.einsum("bk,nk->bn", cc, d[j] * ph).real / (A * G)
    return (j, p, q), out


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


@pytest.mark.parametrize("L", [1, 2, 3, 5, 8, 13, 16, 32])
@pytest.mark.parametrize("o", [1, 2, 3])
def test_fused_matches_legacy(L, o):
    for B in (1, 3):
        new = SOFFT(L, DEV, B, oversampling_factor=o)
        old = SOFFT(L, DEV, B, oversampling_factor=o, backend="legacy")
        assert new.backend == "fused" and old.backend == "legacy"
        c = coeffs(new, seed=L + 10 * B + 100 * o)
        f_new, f_old = new.eval(c), old.eval(c)
        assert f_new.shape == f_old.shape == (B, new.num_betas, new.num_alphas, new.num_gammas)
        assert f_new.dtype == torch.float32 and torch.isfinite(f_new).all()
        assert rel(f_new, f_old) < 2e-6


@pytest.mark.parametrize("L", [2, 5, 8, 16, 33])
def test_against_float64(L):
    """The fused backend is as accurate as the legacy one against the float64 direct sum (both ~3e-7)."""
    for o in (1, 2):
        new = SOFFT(L, DEV, 2, oversampling_factor=o)
        old = SOFFT(L, DEV, 2, oversampling_factor=o, backend="legacy")
        c = coeffs(new, seed=L)
        (j, p, q), r = reference64(new, c)
        s = np.abs(r).max()
        err = lambda f: np.abs(f[:, j, p, q].double().cpu().numpy() - r).max() / s
        e_new, e_old = err(new.eval(c)), err(old.eval(c))
        assert e_new < 1e-6 and e_old < 1e-6
        assert e_new < 2 * e_old + 1e-7


@pytest.mark.parametrize("L", [3, 8, 32])
def test_launch_configs(L):
    """Every tile (betas x functions per block) and both coefficient paths give the same result up to rounding;
    launches through numba and through the fast path are identical."""
    B = 5
    s = SOFFT(L, DEV, B)
    c = coeffs(s, seed=3)
    ref = SOFFT(L, DEV, B, backend="legacy").eval(c)
    for TB, BT, prep in [(1, 1, 0), (2, 1, 1), (4, 2, 0), (3, 5, 1), (8, 4, 1), (16, 1, 0)]:
        fz = FusedSOFFT(s._d_half, L, s.num_betas, s.num_alphas, s.num_gammas, B, DEV, TB=TB, BT=BT, prep=prep)
        a = fz(c)
        b = fz(c)
        fz.fast_launch = False
        slow = fz(c)
        assert torch.equal(a, b) and torch.equal(a, slow)
        assert rel(a, ref) < 2e-6


def test_inputs():
    s = SOFFT(8, DEV, 3)
    c = coeffs(s, seed=1)
    f = s.eval(c)
    # wider coefficient rows (truncated as in the legacy code), complex128 and non-contiguous input
    wide = torch.cat([c, torch.randn(3, 7, dtype=c.dtype, device=DEV)], 1)
    assert torch.equal(s.eval(wide), f)
    assert rel(s.eval(c.to(torch.complex128)), f) < 1e-6
    big = torch.zeros(3, 2 * s.Dsize, dtype=c.dtype, device=DEV)
    big[:, ::2] = c
    assert torch.equal(s.eval(big[:, ::2]), f)
    with pytest.raises(ValueError):
        s.eval(c[:2])
    with pytest.raises(ValueError):
        s.eval(c[:, :-1])
    with pytest.raises(ValueError):
        SOFFT(8, DEV, 3, backend="nope")


def test_real_part_of_complex_function():
    """eval returns Re sum c D for arbitrary complex coefficients, not only Hermitian ones: compare with the
    legacy backend for coefficients whose (m, n) and (-m, -n) entries are unrelated, and for purely imaginary ones."""
    s = SOFFT(6, DEV, 2)
    old = SOFFT(6, DEV, 2, backend="legacy")
    c = coeffs(s, seed=7)
    assert rel(s.eval(1j * c), old.eval(1j * c)) < 2e-6
    assert rel(s.eval(c.conj()), old.eval(c.conj())) < 2e-6
