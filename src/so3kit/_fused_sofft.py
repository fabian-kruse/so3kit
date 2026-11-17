"""
Fused SO(3) FFT for SOFFT.eval: one Numba CUDA kernel (two for larger batches) and one complex FFT.

SOFFT.eval returns f = Re ifft2(F) on the (beta, alpha, gamma) grid, F[b, j, m mod A, n mod G] =
sum_l c^l_mn[b] d^l_mn(beta_j) (legacy backend: einsum, scatter_add over l, ifft2, real part). Here:

1. Real output. f = sum_{m,n} X[m,n] e^{2 pi i (m p / A + n q / G)} with the Hermitian
       X[m, n] = sum_{l >= max(|m|, |n|)} h^l_mn d^l_mn(beta_j) / (A G),
       h^l_mn = (c^l_mn + (-1)^(m-n) conj c^l_{-m,-n}) / 2         (d real, d^l_{-m,-n} = (-1)^(m-n) d^l_mn),
   so only n >= 0 is computed.
2. beta -> pi - beta. The grid beta_j = j pi / (NB - 1) (NB even) is symmetric and
       d^l_mn(pi - beta) = (-1)^(l+n) d^l_{-m,n}(beta),
   so the d table holds only j < NB/2, and one block handles +m and -m: two table loads per (l, beta_j) give
   X at (beta_j, +-m) and (beta_{NB-1-j}, +-m).
3. The sum over l runs in registers: one block per (|m|, tile of TB betas, tile of BT functions), lane = n.
   Table rows (m, l) of length l+1 (n = 0..l), float32, from spherical's float64 d (the values the legacy
   backend uses). For larger batches a first kernel writes h once in the table's row layout (prep); otherwise
   the main loop forms h from the packed coefficients c[b, l(4l^2-1)/3 + (m+l)(2l+1) + n+l] directly.
4. Packed complex FFT. With M = G/2, z[p, r] = f[p, 2r] + i f[p, 2r+1] is the inverse DFT over (A, M) of
       Z[m, k] = X[m, k] (1 + i w^k) + conj X[-m, M - k] (1 - i w^k),   w = e^{2 pi i / G},
   (the second term is X at the negative frequency k - M). The kernel writes Z; f is view_as_real(ifft2(Z))
   reshaped, a C2C transform that torch runs without copying its input (irfft2's C2R copies it), with the
   scale 1/(A G) applied in the kernel. For oversampling 1 (M = L) the second term of lane n comes from lane
   L - n, through shared memory one beta at a time; for oversampling >= 2 it lands on k = M - n >= L and each
   lane writes both terms.
Sums are accumulated in float32, as in the legacy backend.
"""
import numba
import numpy as np
import torch
from numba import cuda

from so3kit._fused import _Launch

_KERNELS = {}
_SHARED_BYTES = 48 * 1024


def pack_table(d_half, L):
    """d_half: (NBH, >= K) float64 d^l_mn(beta_j) in packed D order. Returns the float32 table (NBH * dstride,)
    with rows (m, l), l = |m| .. L-1, of length l + 1 (n = 0..l), the row-block starts mstart (2L-1,) per m,
    and dstride."""
    am = np.abs(np.arange(-L + 1, L))
    sizes = L * (L + 1) // 2 - am * (am + 1) // 2
    mstart = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    dstride = int(sizes.sum())
    src = np.empty(dstride, np.int64)
    for mi, m in enumerate(range(-L + 1, L)):
        p = mstart[mi]
        for l in range(abs(m), L):
            first = l * (4 * l * l - 1) // 3 + (m + l) * (2 * l + 1) + l  # (l, m, n = 0)
            src[p:p + l + 1] = first + np.arange(l + 1)
            p += l + 1
    table = np.ascontiguousarray(d_half[:, src], dtype=np.float64).astype(np.float32).reshape(-1)
    return table, mstart, dstride


def choose_config(L, B):
    """(TB, BT, prep) from the timings on an RTX 4090."""
    if B == 1:
        return (2 if L <= 32 else 4 if L < 64 else 8), 1, False
    if L <= 16:
        return 4, 2, False
    if L <= 32:
        return 8, 1, B >= 4
    return 4, 4, B >= 4


def make_prep(L):
    key = ("prep", L)
    if key in _KERNELS:
        return _KERNELS[key]
    M2 = 2 * L - 1

    @cuda.jit
    def sofft_prep(C, Hh, mstart, dstride, CS):
        """Hh[f, row(m, l) + n] = h^l_mn for n = 0..l (float32 pairs). Block per (m, f)."""
        blk = cuda.blockIdx.x
        mi = blk % M2
        f = blk // M2
        m = mi - (L - 1)
        am = abs(m)
        base = f * 2 * dstride + 2 * (mstart[mi] - am * (am + 1) // 2)
        cb = f * CS
        half = numba.float32(0.5)
        for l in range(am, L):
            off = l * (4 * l * l - 1) // 3
            w = 2 * l + 1
            for n in range(cuda.threadIdx.x, l + 1, cuda.blockDim.x):
                x = cb + 2 * (off + (m + l) * w + n + l)
                y = cb + 2 * (off + (l - m) * w + l - n)
                sg = numba.float32(1 - 2 * ((m - n) & 1))
                o = base + 2 * (l * (l + 1) // 2 + n)
                Hh[o] = half * (C[x] + sg * C[y])
                Hh[o + 1] = half * (C[x + 1] - sg * C[y + 1])

    _KERNELS[key] = sofft_prep
    return sofft_prep


def make_kernel(L, TB, BT, PREP, EXCH):
    key = (L, TB, BT, PREP, EXCH)
    if key in _KERNELS:
        return _KERNELS[key]
    NACC = TB * BT
    LP = 32 * ((L + 31) // 32)
    NSL = 8 * BT  # shared values per lane for the exchange: 4 outputs x re/im x BT functions
    if EXCH:
        assert NSL * LP * 4 <= _SHARED_BYTES

    @cuda.jit
    def sofft_z(C, D, mstart, Z, TW, NB, B, A, M, NBT, dstride, CS):
        """C: PREP=0 packed coefficients (function f at f*CS, float32 pairs), PREP=1 the h table (CS = 2*dstride);
        D: d table for j < NB/2; Z: (B, NB, A, M) complex as float32 pairs; TW: cos, sin of 2 pi k / (2M).
        Integer arguments only (fast launch path)."""
        n = cuda.threadIdx.x
        blk = cuda.blockIdx.x
        mp = blk % L
        rest = blk // L
        j0 = (rest % NBT) * TB
        f0 = (rest // NBT) * BT
        NBH = NB // 2
        scale = numba.float32(1.0 / (A * 2 * M))
        nn = min(n, L - 1)  # lanes n >= L only take part in the shared-memory exchange
        sg = numba.float32(1 - 2 * ((mp + nn) & 1))
        t0 = mp * (mp + 1) // 2
        rp = mstart[L - 1 + mp] - t0 + nn  # row (+m, l) at rp + tri(l)
        rm = mstart[L - 1 - mp] - t0 + nn  # row (-m, l)
        ppr = cuda.local.array(NACC, numba.float32)  # X(beta_j, +m)
        ppi = cuda.local.array(NACC, numba.float32)
        mmr = cuda.local.array(NACC, numba.float32)  # X(beta_j, -m)
        mmi = cuda.local.array(NACC, numba.float32)
        prr = cuda.local.array(NACC, numba.float32)  # X(pi - beta_j, +m)
        pri = cuda.local.array(NACC, numba.float32)
        mrr = cuda.local.array(NACC, numba.float32)  # X(pi - beta_j, -m)
        mri = cuda.local.array(NACC, numba.float32)
        hpr = cuda.local.array(BT, numba.float32)
        hpi = cuda.local.array(BT, numba.float32)
        hmr = cuda.local.array(BT, numba.float32)
        hmi = cuda.local.array(BT, numba.float32)
        for i in range(NACC):
            ppr[i] = 0.0
            ppi[i] = 0.0
            mmr[i] = 0.0
            mmi[i] = 0.0
            prr[i] = 0.0
            pri[i] = 0.0
            mrr[i] = 0.0
            mri[i] = 0.0
        l = max(mp, nn) if n < L else L
        off = l * (4 * l * l - 1) // 3
        tri = l * (l + 1) // 2
        s2 = numba.float32(1 - 2 * ((l + nn) & 1))  # (-1)^(l+n)
        half = numba.float32(0.5)
        while l < L:
            if PREP:
                for q in range(BT):
                    f = f0 + q
                    if f < B:
                        x = f * CS + 2 * (rp + tri)
                        y = f * CS + 2 * (rm + tri)
                        hpr[q] = C[x]
                        hpi[q] = C[x + 1]
                        hmr[q] = C[y]
                        hmi[q] = C[y + 1]
                    else:
                        hpr[q] = 0.0
                        hpi[q] = 0.0
                        hmr[q] = 0.0
                        hmi[q] = 0.0
            else:
                w = 2 * l + 1
                a = off + l + (mp + l) * w  # (l, +m, 0)
                b = off + l + (l - mp) * w  # (l, -m, 0)
                for q in range(BT):
                    f = f0 + q
                    if f < B:
                        base = f * CS
                        x = base + 2 * (a + nn)  # c(+m, n)
                        y = base + 2 * (b - nn)  # c(-m, -n)
                        u = base + 2 * (b + nn)  # c(-m, n)
                        v = base + 2 * (a - nn)  # c(+m, -n)
                        hpr[q] = half * (C[x] + sg * C[y])
                        hpi[q] = half * (C[x + 1] - sg * C[y + 1])
                        hmr[q] = half * (C[u] + sg * C[v])
                        hmi[q] = half * (C[u + 1] - sg * C[v + 1])
                    else:
                        hpr[q] = 0.0
                        hpi[q] = 0.0
                        hmr[q] = 0.0
                        hmi[q] = 0.0
                off += w * w
            for t in range(TB):
                j = j0 + t
                if j < NBH:
                    dp = D[j * dstride + rp + tri]
                    dm = D[j * dstride + rm + tri]
                    dps = s2 * dp
                    dms = s2 * dm
                    for q in range(BT):
                        i = t * BT + q
                        ppr[i] += dp * hpr[q]
                        ppi[i] += dp * hpi[q]
                        mmr[i] += dm * hmr[q]
                        mmi[i] += dm * hmi[q]
                        prr[i] += dms * hpr[q]
                        pri[i] += dms * hpi[q]
                        mrr[i] += dps * hmr[q]
                        mri[i] += dps * hmi[q]
            tri += l + 1
            l += 1
            s2 = -s2
        for i in range(NACC):
            ppr[i] *= scale
            ppi[i] *= scale
            mmr[i] *= scale
            mmi[i] *= scale
            prr[i] *= scale
            pri[i] *= scale
            mrr[i] *= scale
            mri[i] *= scale
        ap = mp % A
        am = (A - mp) % A
        c0 = TW[2 * nn]
        s0 = TW[2 * nn + 1]
        fpr = numba.float32(1.0) - s0  # 1 + i w^n
        fpi = c0
        if EXCH:  # M == L: the negative-frequency term of lane n is conj X[-+m, L - n] of lane L - n
            S = cuda.shared.array((NSL, LP), numba.float32)
            gr = numba.float32(1.0) + s0  # 1 - i w^n
            gi = -c0
            pn = L - nn
            for t in range(TB):
                j = j0 + t
                if n < L:
                    for q in range(BT):
                        i = t * BT + q
                        S[8 * q + 0, n] = ppr[i]
                        S[8 * q + 1, n] = ppi[i]
                        S[8 * q + 2, n] = mmr[i]
                        S[8 * q + 3, n] = mmi[i]
                        S[8 * q + 4, n] = prr[i]
                        S[8 * q + 5, n] = pri[i]
                        S[8 * q + 6, n] = mrr[i]
                        S[8 * q + 7, n] = mri[i]
                cuda.syncthreads()
                if n < L and j < NBH:
                    jr = NB - 1 - j
                    for q in range(BT):
                        f = f0 + q
                        if f < B:
                            i = t * BT + q
                            for kind in range(4):
                                if kind == 0:
                                    xr = ppr[i]; xi = ppi[i]; row = ap; jj = j; ps = 2  # noqa: E702
                                elif kind == 1:
                                    xr = mmr[i]; xi = mmi[i]; row = am; jj = j; ps = 0  # noqa: E702
                                elif kind == 2:
                                    xr = prr[i]; xi = pri[i]; row = ap; jj = jr; ps = 6  # noqa: E702
                                else:
                                    xr = mrr[i]; xi = mri[i]; row = am; jj = jr; ps = 4  # noqa: E702
                                zr = xr * fpr - xi * fpi
                                zi = xr * fpi + xi * fpr
                                if n >= 1:
                                    yr = S[8 * q + ps, pn]
                                    yi = -S[8 * q + ps + 1, pn]
                                    zr += yr * gr - yi * gi
                                    zi += yr * gi + yi * gr
                                o = (((f * NB + jj) * A + row) * M + n) * 2
                                Z[o] = zr
                                Z[o + 1] = zi
                cuda.syncthreads()
        elif n < L:  # M >= 2L: X[row, -n] = conj X[-row, n] lands on k = M - n >= L
            k2 = M - n
            c2 = TW[2 * (k2 % M)]
            s2b = TW[2 * (k2 % M) + 1]
            gr = numba.float32(1.0) + s2b  # 1 - i w^{M-n}
            gi = -c2
            for t in range(TB):
                j = j0 + t
                if j < NBH:
                    jr = NB - 1 - j
                    for q in range(BT):
                        f = f0 + q
                        if f < B:
                            i = t * BT + q
                            for kind in range(4):
                                if kind == 0:
                                    xr = ppr[i]; xi = ppi[i]; yr = mmr[i]; yi = -mmi[i]; row = ap; jj = j  # noqa
                                elif kind == 1:
                                    xr = mmr[i]; xi = mmi[i]; yr = ppr[i]; yi = -ppi[i]; row = am; jj = j  # noqa
                                elif kind == 2:
                                    xr = prr[i]; xi = pri[i]; yr = mrr[i]; yi = -mri[i]; row = ap; jj = jr  # noqa
                                else:
                                    xr = mrr[i]; xi = mri[i]; yr = prr[i]; yi = -pri[i]; row = am; jj = jr  # noqa
                                o = (((f * NB + jj) * A + row) * M) * 2
                                Z[o + 2 * n] = xr * fpr - xi * fpi
                                Z[o + 2 * n + 1] = xr * fpi + xi * fpr
                                if n >= 1:
                                    Z[o + 2 * k2] = yr * gr - yi * gi
                                    Z[o + 2 * k2 + 1] = yr * gi + yi * gr

    _KERNELS[key] = sofft_z
    return sofft_z


class FusedSOFFT:
    """The fused SOFFT.eval for fixed L, grid (NB, A, G) and batch size B. __call__(coeffs) takes a complex
    (B, >= K) tensor on the device (K = number of coefficients with l < L) and returns the real (B, NB, A, G)
    grid values. TB, BT, prep default to choose_config(L, B)."""

    fast_launch = True

    def __init__(self, d_half, L, NB, A, G, B, device, TB=None, BT=None, prep=None):
        assert NB % 2 == 0 and G % 2 == 0 and A >= 2 * L - 1 and G >= 2 * L
        self.L, self.NB, self.A, self.G, self.B = int(L), int(NB), int(A), int(G), int(B)
        self.M = self.G // 2
        dtb, dbt, dprep = choose_config(self.L, self.B)
        TB = dtb if TB is None else TB
        BT = dbt if BT is None else BT
        prep = dprep if prep is None else prep
        TB, BT = max(1, min(TB, NB // 2)), max(1, min(BT, B))
        self.EXCH = int(self.M == self.L)
        LP = 32 * ((self.L + 31) // 32)
        while self.EXCH and 8 * BT * LP * 4 > _SHARED_BYTES:
            BT //= 2
        self.TB, self.BT, self.prep = TB, BT, bool(prep)
        table, mstart, self.dstride = pack_table(d_half, self.L)
        self.table = torch.as_tensor(table, device=device)
        self.mstart = torch.as_tensor(mstart, device=device)
        k = np.arange(self.M)
        tw = np.stack([np.cos(np.pi * k / self.M), np.sin(np.pi * k / self.M)], 1).astype(np.float32)
        self.TW = torch.as_tensor(tw.reshape(-1), device=device)
        self.Z = torch.zeros(self.B, self.NB, self.A, self.M, 2, dtype=torch.float32, device=device)
        self.Zc = torch.view_as_complex(self.Z)
        self.NBT = -(-(self.NB // 2) // TB)
        self.grid = self.L * self.NBT * -(-self.B // BT)
        self.block = LP
        self.kernel = make_kernel(self.L, TB, BT, int(self.prep), self.EXCH)
        if self.prep:
            self.H = torch.zeros(self.B * self.dstride * 2, dtype=torch.float32, device=device)
            self.prep_kernel = make_prep(self.L)
        self._row = None  # coefficient row length the launches were built for

    def _build(self, row):
        arr = cuda.as_cuda_array
        if self.prep:
            self.prep_launch = _Launch(self.prep_kernel, (2 * self.L - 1) * self.B, 64,
                                       [None, arr(self.H), arr(self.mstart), self.dstride, row], [0])
            src, cs, dyn = arr(self.H), 2 * self.dstride, []
        else:
            src, cs, dyn = None, row, [0]
        template = [src, arr(self.table), arr(self.mstart), arr(self.Z.view(-1)), arr(self.TW),
                    self.NB, self.B, self.A, self.M, self.NBT, self.dstride, cs]
        self.launch = _Launch(self.kernel, self.grid, self.block, template, dyn)
        self._row = row

    def __call__(self, coeffs):
        # resolve lazy conjugate / negative views (e.g. coeffs.conj()), which view_as_real rejects
        c = coeffs.to(torch.complex64).resolve_conj().resolve_neg().contiguous()
        c = torch.view_as_real(c).reshape(-1)
        row = c.numel() // self.B
        if row != self._row:
            self._build(row)
        if self.prep:
            self.prep_launch([c], fast=self.fast_launch)
            self.launch([], fast=self.fast_launch)
        else:
            self.launch([c], fast=self.fast_launch)
        z = torch.fft.ifft2(self.Zc, dim=(2, 3), norm="forward")
        return torch.view_as_real(z).reshape(self.B, self.NB, self.A, self.G)
