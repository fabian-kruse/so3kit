"""
J-matrix Wigner-D in e3nn's real basis: D_l(a, b, c) = Z_l(a) J_l Z_l(b) J_l Z_l(c).

wigner_D and _z_rot_mat follow fairchem-core 2.25.0, fairchem/core/models/uma/common/rotation.py
(MIT License, Copyright (c) Meta Platforms, Inc. and affiliates), which borrowed them from e3nn 0.4.0
("In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower"). _z_rot_mat uses the
vectorised e3nn 0.4.0 form, which fairchem keeps as a comment; fairchem's own version loops over
the 2l+1 frequencies for torch.export reasons, which only adds kernel launches.

fairchem ships J_l (fairchem_Jd.pt) only for l <= 11, as did e3nn 0.4.0. J_l is the Wigner-D of the
rotation by pi about (x + y)/sqrt(2), which maps the y axis onto the x axis, so that
J_l Z_l(b) J_l is the rotation about x. It is built here from e3nn's o3.wigner_D in float64 for
every l, and stored in float32 on the GPU (setup). check_baselines.py compares it with
fairchem_Jd.pt for l <= 11.
"""
import torch

_JD = None


def build_Jd(lmax, device="cuda", dtype=torch.float32):
    from e3nn import o3

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # e3nn builds its generators in the default dtype
    try:
        R = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        a, b, c = o3.matrix_to_angles(R)
        return [o3.wigner_D(l, a, b, c).to(device=device, dtype=dtype) for l in range(lmax + 1)]
    finally:
        torch.set_default_dtype(prev)


def to_device(device="cuda", lmax=32):
    """Setup: build J_0..J_lmax once per process and keep them on the device."""
    global _JD
    if _JD is None or len(_JD) <= lmax:
        _JD = build_Jd(lmax, device)


def wigner_D(l, alpha, beta, gamma):
    if _JD is None or len(_JD) <= l:
        to_device(alpha.device, max(l, 32))
    return _wigner_D(l, alpha, beta, gamma, _JD)


# ---- from fairchem rotation.py (e3nn 0.4.0) ----------------------------------------------------

def _wigner_D(lv, alpha, beta, gamma, _Jd):
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[lv]
    Xa = _z_rot_mat(alpha, lv)
    Xb = _z_rot_mat(beta, lv)
    Xc = _z_rot_mat(gamma, lv)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, lv):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * lv + 1, 2 * lv + 1))
    inds = torch.arange(0, 2 * lv + 1, 1, device=device)
    reversed_inds = torch.arange(2 * lv, -1, -1, device=device)
    frequencies = torch.arange(lv, -lv - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M
