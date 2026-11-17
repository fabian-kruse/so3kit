"""
Find the maximum of a bandlimited function on SO(3): coarse grid search with the
SO(3) FFT, then a few Newton steps with analytic gradients and Hessians.
"""
import torch

from so3kit import SO3Function, WignerDsize

torch.manual_seed(0)
ell_max, batch = 16, 4
decay = torch.tensor([1.0 / (1 + l) ** 2 for l in range(ell_max + 1) for _ in range((2 * l + 1) ** 2)])
D = WignerDsize(0, ell_max, ell_max)
coeffs = (torch.randn(batch, D, dtype=torch.complex64) * decay).cuda()
f = SO3Function(coeffs, ell_max, num_points=1)

# 1) grid search
values, (alphas, betas, gammas) = f.grid()
flat = values.reshape(batch, -1).argmax(dim=1)
ib, ia, ig = torch.unravel_index(flat, values.shape[1:])
a, b, g = alphas[ia][:, None], betas[ib][:, None], gammas[ig][:, None]
print("grid  :", f(a, b, g).squeeze(1).tolist())

# 2) Newton refinement on (alpha, beta, gamma)
for _ in range(5):
    grad, hess = f.grad_hess(a, b, g)
    step = torch.linalg.solve(hess, grad)
    a, b, g = a - step[..., 0], b - step[..., 1], g - step[..., 2]
print("newton:", f(a, b, g).squeeze(1).tolist())
