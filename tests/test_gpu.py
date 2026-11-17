# GPU tests: Wigner-D against `spherical`, derivatives against finite differences,
# and the SO(3) FFT grid against direct evaluation.
import numpy as np
import pytest
import quaternionic
import spherical
import torch

from so3kit import SO3Function, WignerDMatrices, WignerDsize

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda:0")


def random_angles(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    alpha = torch.rand(n, generator=g) * 2 * np.pi
    beta = torch.rand(n, generator=g) * (np.pi - 0.2) + 0.1  # away from the poles
    gamma = torch.rand(n, generator=g) * 2 * np.pi
    return alpha, beta, gamma


def random_function(ell_max, batch, num_points, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = WignerDsize(0, ell_max, ell_max)
    decay = torch.tensor([1.0 / (1 + ell) ** 2 for ell in range(ell_max + 1) for _ in range((2 * ell + 1) ** 2)])
    c = torch.complex(torch.randn(batch, D, generator=g), torch.randn(batch, D, generator=g)) * decay
    return SO3Function(c.to(DEV), ell_max, num_points=num_points, device=DEV), c


@pytest.mark.parametrize("ell_max", [2, 8, 16])
def test_wigner_matches_spherical(ell_max):
    n = 32
    alpha, beta, gamma = random_angles(n)
    wig = WignerDMatrices(ell_max, batchsize=n, num_candidates=1, device=DEV)
    D = wig.D(alpha=alpha.to(DEV), beta=beta.to(DEV), gamma=gamma.to(DEV)).reshape(n, -1).cpu().numpy()

    ref = spherical.Wigner(ell_max)
    R = quaternionic.array.from_euler_angles(alpha.double().numpy(), beta.double().numpy(), gamma.double().numpy())
    D_ref = np.stack([ref.D(r) for r in R])
    np.testing.assert_allclose(D, D_ref, atol=2e-4)


def test_derivatives_match_finite_differences():
    ell_max, batch, k = 8, 4, 8
    f, _ = random_function(ell_max, batch, k)
    alpha, beta, gamma = (t.reshape(batch, k).to(DEV) for t in random_angles(batch * k, seed=1))
    grad, hess = f.grad_hess(alpha, beta, gamma)

    h = 1e-2  # float32 kernels: moderate step, central differences
    angles = [alpha, beta, gamma]
    for i in range(3):
        plus = [a + h * (j == i) for j, a in enumerate(angles)]
        minus = [a - h * (j == i) for j, a in enumerate(angles)]
        fd = (f(*plus) - f(*minus)) / (2 * h)
        torch.testing.assert_close(grad[..., i], fd, rtol=2e-2, atol=2e-3)
        g_plus, _ = f.grad_hess(*plus)
        g_minus, _ = f.grad_hess(*minus)
        torch.testing.assert_close(hess[..., i, :], (g_plus - g_minus) / (2 * h), rtol=3e-2, atol=5e-3)


def test_grid_matches_direct_evaluation():
    ell_max, batch = 6, 2
    f, _ = random_function(ell_max, batch, num_points=1)
    values, (alphas, betas, gammas) = f.grid()
    rng = np.random.default_rng(0)
    for _ in range(10):
        ib, ia, ig = rng.integers(len(betas)), rng.integers(len(alphas)), rng.integers(len(gammas))
        a = alphas[ia].repeat(batch, 1)
        b = betas[ib].repeat(batch, 1)
        g = gammas[ig].repeat(batch, 1)
        torch.testing.assert_close(values[:, ib, ia, ig], f(a, b, g)[:, 0], rtol=1e-3, atol=1e-3)
