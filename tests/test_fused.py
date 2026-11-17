# GPU tests of the fused Wigner-D backend: float64 identical to the legacy kernels, float32 close to them,
# both close to `spherical`; quaternion input, truncate, launch configurations, table structure, options.
# Fused contractions (SO3Function values, gradients, Hessians): against float64 `spherical` with central
# differences, against the legacy coefficient bank, across launch configurations.
import numpy as np
import pytest
import quaternionic
import spherical
import torch

from so3kit import SO3Function, WignerDMatrices, WignerDsize
from so3kit._fused import MODE_Y, FusedWignerD, _fits, _Launch

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda:0")


def angles(n, seed=0):
    """Random angles over several periods (beta over [0, pi]), plus poles, multiples of pi/2 and near-misses."""
    g = torch.Generator().manual_seed(seed)
    alpha = (torch.rand(n, generator=g) - 0.5) * 8 * np.pi
    beta = torch.rand(n, generator=g) * np.pi
    gamma = (torch.rand(n, generator=g) - 0.5) * 8 * np.pi
    edge = torch.tensor([0.0, np.pi / 2, np.pi, 2 * np.pi, -np.pi / 2, 1e-4, np.pi / 2 + 1e-4, np.pi - 1e-4])
    k = min(n, edge.numel())
    alpha[:k], beta[:k], gamma[:k] = edge[:k], edge[:k].flip(0).clamp(0, np.pi), edge[:k].roll(3)
    return alpha, beta, gamma


def spherical_D(ell_max, alpha, beta, gamma):
    R = quaternionic.array.from_euler_angles(alpha.double().numpy(), beta.double().numpy(), gamma.double().numpy())
    w = spherical.Wigner(ell_max)
    return np.stack([w.D(r) for r in R])


def flat(D):
    return D.reshape(D.shape[0] * D.shape[1], -1)


@pytest.mark.parametrize("ell_max", [1, 2, 3, 5, 8, 16, 32])
def test_fused_tables(ell_max):
    # every wedge entry written once; every read is of an entry written in an earlier phase
    assert WignerDMatrices(ell_max, batchsize=1, device=DEV)._fused.tables.check_tables()


@pytest.mark.parametrize("ell_max", [2, 5, 8, 16, 32])
def test_fused_float64_identical_to_legacy(ell_max):
    batch, cand = 43, 7  # 301 rotations: not a multiple of any rotations-per-block
    a, b, c = angles(batch * cand, seed=ell_max)
    legacy = flat(WignerDMatrices(ell_max, batch, cand, device=DEV, backend="legacy").D(alpha=a, beta=b, gamma=c))
    wig = WignerDMatrices(ell_max, batch, cand, device=DEV)
    assert wig.backend == "fused" and wig.precision == "float64"
    for warps, rot in [(None, None), (1, 1), (4, 3), (2, 2), (2, 16)]:
        if warps is not None:
            if not _fits(ell_max, warps, rot):  # shared memory
                continue
            wig._fused = FusedWignerD(wig, "float64", warps=warps, rotations_per_block=rot)
        D1 = wig.D(alpha=a.to(DEV), beta=b.to(DEV), gamma=c.to(DEV))
        D2 = wig.D(alpha=a, beta=b, gamma=c)  # CPU input
        assert D1.shape == (batch, cand, legacy.shape[1]) and D1.dtype == torch.complex64
        assert torch.equal(flat(D1), legacy), f"W={warps} R={rot}: max diff {(flat(D1) - legacy).abs().max():.3g}"
        assert torch.equal(D1, D2)


@pytest.mark.parametrize("ell_max", [2, 8, 16, 32])
def test_fused_float32_close_to_legacy(ell_max):
    n = 301
    a, b, c = angles(n, seed=10 + ell_max)
    legacy = flat(WignerDMatrices(ell_max, n, device=DEV, backend="legacy").D(alpha=a, beta=b, gamma=c))
    D = flat(WignerDMatrices(ell_max, n, device=DEV, precision="float32").D(alpha=a, beta=b, gamma=c))
    assert torch.isfinite(torch.view_as_real(D)).all()
    assert (D - legacy).abs().max() < 1e-5


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("ell_max", [1, 4, 32])
def test_fused_matches_spherical(ell_max, precision):
    n = 200
    a, b, c = angles(n, seed=20 + ell_max)
    D = flat(WignerDMatrices(ell_max, n, device=DEV, precision=precision).D(alpha=a, beta=b, gamma=c))
    np.testing.assert_allclose(D.cpu().numpy(), spherical_D(ell_max, a, b, c), atol=1e-5)


@pytest.mark.parametrize("ell_max", [2, 8, 16])
def test_fused_quaternions(ell_max):
    n = 150
    a, b, c = angles(n, seed=30 + ell_max)
    q64 = np.ascontiguousarray(
        quaternionic.array.from_euler_angles(a.double().numpy(), b.double().numpy(), c.double().numpy()).ndarray)
    ref = spherical_D(ell_max, a, b, c)
    for q in (q64, torch.tensor(q64, dtype=torch.float32, device=DEV)):
        legacy = flat(WignerDMatrices(ell_max, n, device=DEV, backend="legacy").D(q=q))
        assert torch.equal(flat(WignerDMatrices(ell_max, n, device=DEV).D(q=q)), legacy)
        D32 = flat(WignerDMatrices(ell_max, n, device=DEV, precision="float32").D(q=q))
        assert (D32 - legacy).abs().max() < 1e-5
        np.testing.assert_allclose(D32.cpu().numpy(), ref, atol=1e-5)


def test_fused_truncate():
    ell_max, n = 8, 100
    a, b, c = angles(n, seed=40)
    for t in (0, 1, 2, 5, 7, 8, 9, None):
        legacy = flat(WignerDMatrices(ell_max, n, device=DEV, backend="legacy").D(alpha=a, beta=b, gamma=c, truncate=t))
        D = flat(WignerDMatrices(ell_max, n, device=DEV).D(alpha=a, beta=b, gamma=c, truncate=t))
        assert D.shape == legacy.shape and torch.equal(D, legacy), f"truncate={t}"
        D32 = flat(WignerDMatrices(ell_max, n, device=DEV, precision="float32").D(alpha=a, beta=b, gamma=c, truncate=t))
        assert D32.shape == legacy.shape and (D32 - legacy).abs().max() < 1e-5, f"truncate={t}"


def test_backend_options():
    assert WignerDMatrices(4, 2, device=DEV).backend == "fused"
    assert WignerDMatrices(4, 2, device=DEV, mode="eval").backend == "legacy"
    with pytest.raises(ValueError):
        WignerDMatrices(4, 2, device=DEV, backend="legacy", precision="float32")
    with pytest.raises(ValueError):
        WignerDMatrices(4, 2, device=DEV, backend="fused", mode="eval")
    with pytest.raises(ValueError):
        WignerDMatrices(4, 2, device=DEV, precision="float16")
    with pytest.raises(ValueError):
        WignerDMatrices(64, 2, device=DEV, backend="fused")
    wig = WignerDMatrices(4, 2, device=DEV)
    a = torch.zeros(3)
    with pytest.raises(ValueError):  # 3 rotations for an instance of 2
        wig.D(alpha=a, beta=a, gamma=a)


def rel_err(x, y):
    """max |x - y| per trailing component, relative to that component's max |y|; max over components"""
    x = torch.as_tensor(x).reshape(-1, x.shape[-1]).cpu().to(torch.complex128)
    y = torch.as_tensor(y).reshape(-1, y.shape[-1]).cpu().to(torch.complex128)
    return float(((x - y).abs().amax(0) / y.abs().amax(0).clamp_min(1e-6)).max())


def random_function(ell_max, batch, k, seed):
    g = torch.Generator().manual_seed(seed)
    D = WignerDsize(0, ell_max, ell_max)
    c = torch.complex(torch.randn(batch, D, generator=g), torch.randn(batch, D, generator=g)) / D**0.5
    h = torch.Generator().manual_seed(seed + 1)  # away from the poles, as in test_gpu.py
    a, b, gam = (torch.rand(batch, k, generator=h) * s + o for s, o in ((6.28, 0), (2.94, 0.1), (6.28, 0)))
    b[0, 0] = np.pi / 2
    return c, a, b, gam


def spherical_function(c, ell_max, a, b, g):
    """value, gradient and Hessian of f_i(a, b, g) = Re sum_j c_ij D_j(a, b, g) with respect to (a, b, g), from
    float64 `spherical` by central differences (steps 1e-5 for the gradient, 1e-4 for the Hessian)"""
    w = spherical.Wigner(ell_max)
    c = c.numpy().astype(np.complex128)
    a, b, g = (t.double().numpy() for t in (a, b, g))

    def f(d):
        R = quaternionic.array.from_euler_angles((a + d[0]).ravel(), (b + d[1]).ravel(), (g + d[2]).ravel())
        D = np.stack([w.D(r) for r in R]).reshape(*a.shape, -1)
        return np.einsum("bkj,bj->bk", D, c).real

    e = np.eye(3)
    h1, h2 = 1e-5, 1e-4
    f0 = f(np.zeros(3))
    grad = np.stack([(f(h1 * e[i]) - f(-h1 * e[i])) / (2 * h1) for i in range(3)], -1)
    hess = np.zeros(a.shape + (3, 3))
    for i in range(3):
        hess[..., i, i] = (f(h2 * e[i]) - 2 * f0 + f(-h2 * e[i])) / h2**2
        for j in range(i + 1, 3):
            hess[..., i, j] = hess[..., j, i] = (f(h2 * (e[i] + e[j])) - f(h2 * (e[i] - e[j]))
                                                 - f(h2 * (e[j] - e[i])) + f(-h2 * (e[i] + e[j]))) / (4 * h2**2)
    return torch.tensor(f0), torch.tensor(grad), torch.tensor(hess)


@pytest.mark.parametrize("ell_max,precision,backend,tol", [
    (1, "float64", None, 2e-5), (2, "float64", None, 2e-5), (5, "float64", None, 2e-5),
    (16, "float64", None, 2e-5), (32, "float64", None, 2e-5), (5, "float32", None, 2e-5),
    (16, "float32", None, 2e-5), (32, "float32", None, 2e-5),
    # control (D + torch einsum with the complex64 coefficient bank, less accurate at large ell_max)
    (2, "float64", "legacy", 1e-4), (16, "float64", "legacy", 1e-4)])
def test_function_matches_spherical(ell_max, precision, backend, tol):
    batch, k = 3, 5
    c, a, b, gam = random_function(ell_max, batch, k, seed=60 + ell_max)
    f = SO3Function(c.to(DEV), ell_max, num_points=k, device=DEV, precision=precision, backend=backend)
    v0, g0, h0 = spherical_function(c, ell_max, a, b, gam)
    v = f(a.to(DEV), b.to(DEV), gam.to(DEV))
    grad, hess = f.grad_hess(a.to(DEV), b.to(DEV), gam.to(DEV))
    assert v.shape == (batch, k) and grad.shape == (batch, k, 3) and hess.shape == (batch, k, 3, 3)
    assert v.dtype == grad.dtype == hess.dtype == torch.float32
    assert rel_err(v[..., None], v0[..., None]) < tol
    assert rel_err(grad, g0) < tol
    assert rel_err(hess.flatten(-2), h0.flatten(-2)) < tol
    if f.wigner.backend == "fused":
        assert torch.equal(hess, hess.transpose(-1, -2))


@pytest.mark.parametrize("ell_max", [2, 3, 8])  # the legacy D kernels do not compile for ell_max = 1
def test_fused_Y_matches_legacy_bank(ell_max):
    # the 15 columns of so3kit.derivatives: fused contraction vs legacy D times the coefficient bank
    batch, k = 4, 9
    c, a, b, gam = random_function(ell_max, batch, k, seed=70 + ell_max)
    leg = SO3Function(c.to(DEV), ell_max, num_points=k, device=DEV, backend="legacy")
    Y_leg = torch.einsum("BnC,BcC->Bnc", leg._D(a, b, gam), leg.derivatives.CoeffBank)
    for precision in ("float64", "float32"):
        f = SO3Function(c.to(DEV), ell_max, num_points=k, device=DEV, precision=precision)
        Y = f.wigner._fused.Y(f.coeffs, a, b, gam)
        assert Y.shape == Y_leg.shape == (batch, k, 15) and Y.dtype == torch.complex64
        assert rel_err(Y, Y_leg) < 2e-5, precision


@pytest.mark.parametrize("ell_max", [4, 16])
def test_fused_contraction_launch_configs(ell_max):
    # other (warps, rotations per block): other summation order only
    batch, k = 5, 7  # 35 rotations: not a multiple of any rotations-per-block
    c, a, b, gam = random_function(ell_max, batch, k, seed=80 + ell_max)
    f = SO3Function(c.to(DEV), ell_max, num_points=k, device=DEV)
    v0 = f(a, b, gam)
    g0, h0 = f.grad_hess(a, b, gam)
    for warps, rot in [(1, 1), (4, 3), (2, 16), (4, 1)]:
        if not _fits(ell_max, warps, rot, MODE_Y):
            continue
        f.wigner._fused = FusedWignerD(f.wigner, "float64", warps=warps, rotations_per_block=rot)
        v = f(a, b, gam)
        grad, hess = f.grad_hess(a, b, gam)
        assert rel_err(v[..., None], v0[..., None]) < 5e-6, (warps, rot)
        assert rel_err(grad, g0) < 5e-6 and rel_err(hess.flatten(-2), h0.flatten(-2)) < 5e-6, (warps, rot)
        assert torch.equal(v, f(a, b, gam))  # deterministic
        assert all(torch.equal(x, y) for x, y in zip((grad, hess), f.grad_hess(a, b, gam)))


def test_function_precision():
    ell_max, batch, k = 8, 4, 8
    c, a, b, gam = random_function(ell_max, batch, k, seed=0)
    a, b, gam = a.to(DEV), b.to(DEV), gam.to(DEV)
    f64 = SO3Function(c.to(DEV), ell_max, num_points=k, device=DEV)
    f32 = SO3Function(c.to(DEV), ell_max, num_points=k, device=DEV, precision="float32")
    leg = SO3Function(c.to(DEV), ell_max, num_points=k, device=DEV, backend="legacy")
    assert f32.wigner.precision == "float32" and leg.wigner.backend == "legacy"
    # fused: contracted in the kernel, legacy: einsum; other summation order, so close but not identical
    for f in (f64, f32):
        assert rel_err(f(a, b, gam)[..., None], leg(a, b, gam)[..., None]) < 1e-5
        for x, y in zip(f.grad_hess(a, b, gam), leg.grad_hess(a, b, gam)):
            assert x.shape == y.shape and x.dtype == y.dtype
            assert rel_err(x.flatten(2), y.flatten(2)) < 1e-5
    torch.testing.assert_close(f32(a, b, gam), f64(a, b, gam), rtol=0, atol=1e-5)
    for x, y in zip(f32.grad_hess(a, b, gam), f64.grad_hess(a, b, gam)):
        torch.testing.assert_close(x, y, rtol=0, atol=1e-4)
    # the coefficient bank is still available with the fused backend (built on first access)
    assert torch.equal(f64.derivatives.CoeffBank, leg.derivatives.CoeffBank)


def test_fast_launch_identical():
    # launches after the first bypass numba's argument conversion: same outputs as launches through numba, for
    # new input tensors every call
    ell, batch, k = 8, 5, 7
    c = random_function(ell, batch, k, seed=90)[0]
    fast = SO3Function(c.to(DEV), ell, num_points=k, device=DEV)
    slow = SO3Function(c.to(DEV), ell, num_points=k, device=DEV)
    slow.wigner._fused.fast_launch = False
    for seed in range(3):
        _, a, b, gam = random_function(ell, batch, k, seed=91 + seed)
        a, b, gam = a.to(DEV), b.to(DEV), gam.to(DEV)
        assert torch.equal(fast(a, b, gam), slow(a, b, gam))
        assert all(torch.equal(x, y) for x, y in zip(fast.grad_hess(a, b, gam), slow.grad_hess(a, b, gam)))
        assert torch.equal(fast.wigner._fused.Y(fast.coeffs, a, b, gam), slow.wigner._fused.Y(slow.coeffs, a, b, gam))
        for t in (None, 3):
            assert torch.equal(fast.wigner.D(alpha=a, beta=b, gamma=gam, truncate=t),
                               slow.wigner.D(alpha=a, beta=b, gamma=gam, truncate=t))
        q64 = np.ascontiguousarray(quaternionic.array.from_euler_angles(
            a.double().cpu().numpy(), b.double().cpu().numpy(), gam.double().cpu().numpy()).ndarray)
        for q in (q64, torch.tensor(q64, dtype=torch.float32, device=DEV)):
            assert torch.equal(fast.wigner.D(q=q), slow.wigner.D(q=q))
    launches = fast.wigner._fused._launches
    assert len(launches) == 8  # value, derivatives, Y, D, D truncated, D from q, phases (float64, float32)
    if _Launch.enabled:
        assert all(L.fast for L in launches.values()), "the fast launch path is not used"
