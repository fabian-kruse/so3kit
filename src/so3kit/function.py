"""
High-level interface: a batch of bandlimited functions on SO(3),

    f_b(alpha, beta, gamma) = Re sum_i c_{b,i} D_i(alpha, beta, gamma),

with values, analytic gradients/Hessians and grid evaluation by the SO(3) FFT.
"""
import math
import torch
from torch.nn import functional as F

from so3kit.wigner import WignerDMatrices, WignerDsize
from so3kit.derivatives import WignerDGradients
from so3kit.sofft import SOFFT


class SO3Function:
    """
    Parameters:
    - coeffs: torch.Tensor, shape (B, WignerDsize(0, ell_max, ell_max)), complex
        Coefficients in the Wigner-D basis, ordered as [(ell, mp, m) for ell for mp for m].
    - ell_max: int
        Bandlimit.
    - num_points: int
        Number of rotations evaluated per function in each call.
    - precision, backend: passed to WignerDMatrices ("float64"/"float32"; "fused"/"legacy"/None).
        With the fused backend, values and derivatives are contracted inside the Wigner-D kernel (D is never
        stored, see so3kit._fused); with the legacy backend, D is computed and contracted with torch.
    """

    def __init__(self, coeffs: torch.Tensor, ell_max: int, num_points: int = 1, device=torch.device("cuda:0"),
                 precision: str = "float64", backend=None):
        self.ell_max = int(ell_max)
        self.device = torch.device(device)
        self.Dsize = WignerDsize(0, self.ell_max, self.ell_max)
        assert coeffs.ndim == 2 and coeffs.shape[1] >= self.Dsize, f"coeffs must have shape (B, {self.Dsize})"
        self.batch_size = coeffs.shape[0]
        self.num_points = int(num_points)

        self.wigner = WignerDMatrices(
            ell_max=self.ell_max,
            batchsize=self.batch_size,
            num_candidates=self.num_points,
            device=self.device,
            precision=precision,
            backend=backend,
        )
        # The derivative tables index one extra, always-zero entry for out-of-range shifts.
        self.coeffs = F.pad(coeffs[:, :self.Dsize].to(self.device, torch.complex64), (0, 1), value=0)
        self._fused = self.wigner.backend == "fused"
        self._derivatives = None
        if self._fused:
            self._cf = self.wigner._fused.prepare_coeffs(self.coeffs)
        else:
            self._derivatives = self._make_derivatives()

    def _make_derivatives(self):
        d = WignerDGradients(wigner=self.wigner, device=self.device)
        d.set_so3_coeffs(self.coeffs)
        return d

    @property
    def derivatives(self) -> WignerDGradients:
        """WignerDGradients with the coefficient bank of these coefficients (with the fused backend: built on
        first access; grad_hess does not use it)."""
        if self._derivatives is None:
            self._derivatives = self._make_derivatives()
        return self._derivatives

    def _D(self, alpha, beta, gamma):
        D = self.wigner.D(alpha=alpha, beta=beta, gamma=gamma)
        return F.pad(D, (0, 1), value=0)

    def __call__(self, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Values at ZYZ Euler angles of shape (B, num_points). Returns shape (B, num_points)."""
        if self._fused:
            return self.wigner._fused.value(self._cf, alpha, beta, gamma)
        return torch.einsum("nbc,nc->nb", self._D(alpha, beta, gamma), self.coeffs).real

    def grad_hess(self, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor):
        """
        Gradient (B, num_points, 3) and Hessian (B, num_points, 3, 3) with respect to (alpha, beta, gamma).
        """
        if self._fused:
            _, grad, hess = self.wigner._fused.derivatives(self._cf, alpha, beta, gamma)
            return grad, hess
        return self.derivatives.get_derivatives(self._D(alpha, beta, gamma), alpha.to(self.device))

    def grid(self, oversampling: int = 1):
        """
        Values on the equiangular grid of the SO(3) FFT.
        Returns:
        - f: torch.Tensor, shape (B, num_betas, num_alphas, num_gammas)
        - (alphas, betas, gammas): 1D tensors with the grid angles
        """
        sofft = SOFFT(L=self.ell_max + 1, device=self.device, batchsize=self.batch_size,
                      oversampling_factor=oversampling)
        coeffs = self.coeffs[:, :self.Dsize]
        coeffs = F.pad(coeffs, (0, max(0, sofft.Dsize - self.Dsize)), value=0)[:, :sofft.Dsize]
        # ifft2 normalises by 1/(num_alphas * num_gammas); undo it to get function values
        f = sofft.eval(coeffs) * (sofft.num_alphas * sofft.num_gammas)
        alphas = torch.arange(sofft.num_alphas, device=self.device) * (2 * math.pi / sofft.num_alphas)
        betas = torch.linspace(0, math.pi, sofft.num_betas, device=self.device)
        gammas = torch.arange(sofft.num_gammas, device=self.device) * (2 * math.pi / sofft.num_gammas)
        return f, (alphas, betas, gammas)
