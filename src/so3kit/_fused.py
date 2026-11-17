"""
Fused Wigner-D kernel: the whole pipeline of WignerDMatrices.D (Euler phases, complex powers, H recursion
steps 1-5, fill) in one Numba CUDA launch. The recursion is adapted from spherical by Michael Boyle
(https://github.com/moble/spherical), via so3kit.wigner.

Same recursion and, with precision="float64", the same floating-point operations as the step-by-step
("legacy") kernels in so3kit.wigner, so the output is identical to theirs; precision="float32" evaluates the
recursion coefficients in float32 (faster, differences of order 1e-6).

Layout:
  * a block handles R rotations with W warps;
  * quaternion input: the Euler phases come from the legacy kernel to_euler_phases_cuda (one extra launch),
    so that they are bitwise the legacy ones;
  * prologue (all threads): Euler phases (angle input: float32 cos/sin as in the legacy kernel), powers of
    e^{i alpha} and e^{i gamma}, and the m'=0 column of every
    degree (steps 1-2, a serial recurrence in m) into shared memory; the beta-independent constants come from
    host tables built with the same IEEE operations as the legacy kernel;
  * the rest of the H recursion decouples across degree n (step 3 for n reads only the m'=0 column of n+1):
    each warp builds the n-wedges of its degrees in shared memory; step-4 (m' > 0) and step-5 (m' <= 0) items
    are grouped into dependency-free phases separated by __syncwarp;
  * the fill writes D(n, ., .) from shared memory straight into a complex64 tensor, coalesced.
All index/coefficient tables are the legacy tables of the WignerDMatrices instance, regrouped (check_tables
verifies the regrouping).

Contractions (used by SO3Function): the same kernel up to the wedges, but instead of the fill each warp
contracts D(n, ., .) of its degrees with the coefficients of the rotation's function right away, so D is never
written to memory: the value Re sum_j c_j D_j, or the 15 sums of so3kit.derivatives (D times the coefficient
bank of WignerDGradients, here formed on the fly from the coefficients and per-row factors) assembled into
value, gradient and Hessian in the kernel. Sums are accumulated in float32 (as the einsum of the torch path).
"""
import ctypes
import math
import os
import threading
from collections import defaultdict

import numba
import numpy as np
import torch
from numba import cuda

try:
    from cuda.bindings import driver as _cudrv
except ImportError:  # pragma: no cover
    _cudrv = None

MAX_ELL = 63  # the fill table packs |m| into 6 bits
_KERNELS = {}
_SHARED_FLOATS = 48 * 1024 // 4  # static shared memory per block

# (largest ell_max, (warps, rotations per block)), tuned on an RTX 4090
_PREFERRED = {
    "float64": [(8, (2, 16)), (16, (4, 4)), (10**9, (4, 2))],
    "float32": [(4, (2, 16)), (8, (4, 4)), (16, (4, 2)), (10**9, (4, 1))],
}


NACC = 30  # contraction kernels (Y, derivatives): 15 complex accumulators per rotation and warp
MODE_D, MODE_Y, MODE_DERIV, MODE_VALUE = 0, 1, 2, 3
# register cap of the Y / derivative kernels: 0.97 instead of 1.19 ms at ell = 32 (141 registers uncapped), neutral
# at smaller ell; lower caps spill and are slower
_CONTRACT_MAXREG = 128


def _fits(L, W, R, mode=MODE_D):
    acc = 0 if mode == MODE_D else W * R * (1 if mode == MODE_VALUE else NACC)
    return W * R * (L + 1) ** 2 + R * (L + 2) ** 2 + 4 * R * (L + 1) + 2 * R + acc <= _SHARED_FLOATS


def _shrink(L, W, R, mode=MODE_D):
    """halve R, then W, until the shared memory of the kernel of this mode fits"""
    while not _fits(L, W, R, mode):
        if R > 1:
            R //= 2
        elif W > 1:
            W //= 2
        else:
            raise ValueError(f"ell_max={L} is too large for the fused kernel's shared memory")
    return W, R


def launch_config(ell_max, precision, num_rotations, num_sms):
    """(warps, rotations per block): the tuned preference, with R halved while the grid would have fewer than
    8 blocks per SM (small batches are latency-bound: more blocks beat more rotations per block), then R and W
    halved until the shared memory fits."""
    W, R = next(c for lmax, c in _PREFERRED[precision] if ell_max <= lmax)
    while R > 1 and -(-num_rotations // R) < 8 * num_sms:
        R //= 2
    return _shrink(ell_max, W, R)


def _wedge_offset(n, mp_max):
    from so3kit.wigner import WignerHsize
    return WignerHsize(mp_max, n - 1) if n > 0 else 0


class FusedTables:
    """Host-side regrouping of the legacy tables of a WignerDMatrices instance."""

    def __init__(self, wig, precision):
        from so3kit.wigner import WignerDsize, WignerHindex

        L = wig.ell_max
        assert wig.mp_max == L
        assert 0 <= L <= MAX_ELL, "the fill table packs |m| into 6 bits"
        self.L = L
        cdt = np.float64 if precision == "float64" else np.float32
        host = lambda a: a.copy_to_host() if a.size else np.zeros(0)  # noqa: E731

        # steps 1-2
        self.g = host(wig._g).astype(cdt)
        self.h = host(wig._h).astype(cdt)
        U = np.zeros(L + 2, np.float32)  # the legacy float32 chain of un-normalised H(n, 0, n)
        U[1] = np.float32(math.sqrt(3))
        for k in range(2, L + 2):
            U[k] = np.float32(math.sqrt(1.0 + 0.5 / k) * float(U[k - 1]))
        self.U = U
        self.C1 = np.array([0.0] + [1.0 / math.sqrt(4 * n + 2) for n in range(1, L + 2)], cdt)
        self.SQ = np.array([1.0] + [math.sqrt(4 * n + 2) for n in range(1, L + 2)], cdt)
        self.one = np.ones(1, cdt)

        # step 3: main (n = 1..L-1) and extra (n = L) tables, concatenated in n order
        b6 = np.concatenate([host(wig.b6s), host(wig.b6s_extra)])
        b7 = np.concatenate([host(wig.b7s), host(wig.b7s_extra)])
        a8 = np.concatenate([host(wig.a8s), host(wig.a8s_extra)])
        inds0 = np.concatenate([host(wig.inds0), host(wig.inds0_extra)]).astype(np.int64)
        assert len(b6) == L * (L + 1) // 2
        s3w = np.empty(len(b6), np.int32)
        k = 0
        for n in range(1, L + 1):
            off = _wedge_offset(n, L)
            for i in range(n):
                assert inds0[k] == WignerHindex(n, 1, 1 + i, L), "step-3 table order"
                s3w[k] = inds0[k] - off
                k += 1
        self.s3 = (b6.astype(cdt), b7.astype(cdt), a8.astype(cdt), s3w)

        # steps 4 + 5: (n, m') of every legacy table entry, in table order
        s4 = [(n, mp) for n in range(2, L + 1) for mp in range(1, min(n, L)) for _ in range(n - mp)]
        s5 = [(n, mp) for n in range(0, L + 1) for mp in range(0, -min(n, L), -1) for _ in range(n + mp)]
        t4 = [host(a) for a in (wig.inds1_4, wig.inds2_4, wig.inds3_4, wig.inds4_4, wig.d6_4, wig.d7_4, wig.d8_4)]
        t5 = [host(a) for a in (wig.inds1_5, wig.inds2_5, wig.inds3_5, wig.inds4_5, wig.d6_5, wig.d7_5, wig.d8_5)]
        assert len(s4) == len(t4[0]) and len(s5) == len(t5[0]), "step-4/5 table lengths"
        assert np.array_equal(np.array([n for n, _ in s4], dtype=np.int64), np.asarray(wig.ns_4, dtype=np.int64))
        assert np.array_equal(np.array([n for n, _ in s5], dtype=np.int64), np.asarray(wig.ns_5, dtype=np.int64))
        # phase of an item: step 4 at m' writes column m'+1 -> phase m'; step 5 at m' <= 0 writes column m'-1
        # -> phase 1 - m'. Step 5 is stored with negated d7, d8 so that both use c6*H2 - c7*H3 + c8*H4
        # (negation is exact).
        groups = defaultdict(list)
        seq = 0
        for (n, mp), w, r2, r3, r4, d6, d7, d8 in zip(s4, *t4):
            groups[n].append((mp, seq, w, r2, r3, r4, d6, d7, d8))
            seq += 1
        for (n, mp), w, r2, r3, r4, d6, d7, d8 in zip(s5, *t5):
            groups[n].append((1 - mp, seq, w, r2, r3, r4, d6, -d7, -d8))
            seq += 1
        ph = np.zeros((L + 1, L + 2), np.int32)  # ph[n, p] = first item of phase p of degree n
        cols = [[] for _ in range(7)]
        pos = 0
        for n in range(L + 1):
            off = _wedge_offset(n, L)
            mine = sorted(groups.get(n, []), key=lambda it: (it[0], it[1]))
            counts = np.bincount([it[0] for it in mine], minlength=L + 2)[:L + 2] if mine else np.zeros(L + 2, int)
            ph[n] = pos + np.concatenate([[0], np.cumsum(counts)[:-1]])
            for _, _, w, r2, r3, r4, c6, c7, c8 in mine:
                if c8 == 0.0:
                    # last item of a column: the legacy kernel multiplies 0 by an entry that, in a parallel
                    # schedule, may not be written yet; read an entry that is (0 * finite = 0)
                    r4 = r3
                for col, v in zip(cols, (w - off, r2 - off, r3 - off, r4 - off, c6, c7, c8)):
                    col.append(v)
            pos += len(mine)
        self.ph = ph
        self.s45 = tuple(np.array(c, np.int32) for c in cols[:4]) + tuple(np.array(c, cdt) for c in cols[4:])

        # fill: one packed int32 per D entry
        Dsize = WignerDsize(0, L, L)
        res, za_c, za, zy_c, zy, H_idx = (host(a).astype(np.int64) if i else host(a) for i, a in enumerate(
            (wig.res, wig.z_a_conj_id, wig.z_a_id, wig.z_y_conj_id, wig.z_y_id, wig.H_indices)))
        Dbase = np.zeros(L + 2, np.int64)
        for n in range(L + 1):
            Dbase[n + 1] = Dbase[n] + (2 * n + 1) ** 2
        assert Dbase[L + 1] == Dsize
        hloc = np.empty(Dsize, np.int64)
        for n in range(L + 1):
            sl = slice(Dbase[n], Dbase[n + 1])
            hloc[sl] = H_idx[sl] - _wedge_offset(n, L)
            assert hloc[sl].min() >= 0 and hloc[sl].max() < (n + 1) ** 2, "fill reads outside its wedge"
        assert set(np.unique(res)) <= {-1.0, 1.0}
        pack = hloc | (za << 16) | (zy << 22) | (za_c << 28) | (zy_c << 29) | ((res < 0).astype(np.int64) << 30)
        assert pack.max() < 2**31
        p = pack
        assert np.array_equal(p & 0xFFFF, hloc) and np.array_equal((p >> 16) & 63, za) and \
            np.array_equal((p >> 22) & 63, zy) and np.array_equal((p >> 28) & 1, za_c) and \
            np.array_equal((p >> 29) & 1, zy_c) and np.array_equal(np.where((p >> 30) & 1, -1.0, 1.0), res)
        self.fill = pack.astype(np.int32)
        self.Dbase = Dbase
        self.Dsize = Dsize
        self.top0 = np.array([WignerHindex(n, 0, n, L) - _wedge_offset(n, L) if n > 0 else 0
                              for n in range(L + 1)], np.int32)

    def check_tables(self):
        """Every wedge entry is written exactly once, and every read is of an entry written in an earlier phase
        (column 0: steps 1-2 = phase -2, column 1: step 3 = phase -1)."""
        from so3kit.wigner import WignerHindex
        L = self.L
        w, r2, r3, r4 = self.s45[:4]
        s3w = self.s3[3]
        for n in range(1, L + 1):
            off = _wedge_offset(n, L)
            phase = np.full((n + 1) ** 2, 10**9)
            for m in range(n + 1):
                phase[WignerHindex(n, 0, m, L) - off] = -2
            for i in range(n):
                k = n * (n - 1) // 2 + i
                assert phase[s3w[k]] == 10**9
                phase[s3w[k]] = -1
            for p in range(1, n + 1):
                for k in range(self.ph[n, p], self.ph[n, p + 1]):
                    assert phase[w[k]] == 10**9, "entry written twice"
                    for r in (r2[k], r3[k], r4[k]):
                        assert phase[r] < p, f"read of an unwritten entry (n={n}, phase {p})"
                    phase[w[k]] = p
            assert (phase < 10**9).all(), f"wedge n={n} not fully written"
            assert self.ph[n, n + 1] == self.ph[n, L + 1], "items beyond phase n"
        return True

    def warp_plan(self, W):
        """Static assignment of degrees to the W warps of a block (largest first, to the least loaded warp)."""
        cost = lambda n: (2 * n + 1) ** 2 + 2 * n * n + 8 * n  # noqa: E731  fill + recursion + serial work
        loads, plan = [0.0] * W, [[] for _ in range(W)]
        for n in sorted(range(self.L + 1), key=cost, reverse=True):
            j = min(range(W), key=lambda i: loads[i])
            plan[j].append(n)
            loads[j] += cost(n)
        out = np.full((W, max(len(p) for p in plan)), -1, np.int32)
        for j, p in enumerate(plan):
            out[j, :len(p)] = sorted(p)
        return out


def _tensor_params(t):
    """The kernel parameters numba passes for a C-contiguous array argument (numba.cuda's
    _Kernel._prepare_args): meminfo, parent, size, itemsize, data pointer, shape, strides in bytes."""
    isz = t.element_size()
    shape = tuple(t.shape)
    strides = []
    s = isz
    for d in reversed(shape):
        strides.append(s)
        s *= d
    return [0, 0, t.numel(), isz, t.data_ptr(), *shape, *reversed(strides)]


class _Launch:
    """Launches of one compiled kernel with a fixed grid, fixed (static) arguments and per-call torch tensors.

    numba converts every argument in Python at every launch (about 100 us for the 26 arguments of fused_D).
    The first launch goes through numba, which compiles. After it, the parameters are
    built once with numba's own argument marshalling into an int64 buffer; later launches write only the per-call
    tensors into it and call cuLaunchKernel. The fast path is used only if every parameter is 8 bytes at the
    offset the driver reports for the kernel, and if the parameters computed from the tensors equal numba's;
    otherwise, with SO3KIT_FAST_LAUNCH=0, or if a fast launch fails, launches go through numba.
    """

    enabled = os.environ.get("SO3KIT_FAST_LAUNCH", "1") != "0" and _cudrv is not None

    def __init__(self, dispatcher, griddim, blockdim, template, dynamic):
        """template: the kernel arguments, None at the positions `dynamic` of the per-call tensors."""
        self.dispatcher = dispatcher
        self.griddim = int(griddim)
        self.blockdim = int(blockdim)
        self.template = list(template)
        self.dynamic = list(dynamic)
        assert self.dynamic == sorted(self.dynamic), "per-call tensors in argument order"
        self.fast = None  # decided after the first launch
        self.lock = threading.Lock()

    def __call__(self, tensors, fast=True):
        if fast and self.fast:
            ok = all(t.dtype == dt and t.dim() == nd and t.is_contiguous() and t.device == dev
                     for t, (dt, nd, dev) in zip(tensors, self.kinds))
            if ok:
                with self.lock:  # the driver reads the buffer during the call (the GIL is released)
                    for (a, b), t in zip(self.slices, tensors):
                        self.buf[a:b] = _tensor_params(t)
                    err, = _cudrv.cuLaunchKernel(self.handle, self.griddim, 1, 1, self.blockdim, 1, 1, 0, 0,
                                                 self.pptr, 0)
                if err == _cudrv.CUresult.CUDA_SUCCESS:
                    return
                self.fast = False  # e.g. no current context in this thread: numba's launch handles it
        args = list(self.template)
        for i, t in zip(self.dynamic, tensors):
            args[i] = cuda.as_cuda_array(t)
        self.dispatcher[self.griddim, self.blockdim](*args)
        if self.fast is None:
            self.fast = self._build(args, tensors)

    def _build(self, args, tensors):
        if not self.enabled:
            return False
        try:
            kern = self.dispatcher.overloads[tuple(self.dispatcher.typeof_pyval(a) for a in args)]
            params, slices = [], []
            for i, (ty, val) in enumerate(zip(kern.argument_types, args)):
                lst = []
                kern._prepare_args(ty, val, 0, [], lst)
                vals = []
                for v in lst:
                    if isinstance(v, int):
                        vals.append(v)
                    elif isinstance(v, (ctypes.c_int64, ctypes.c_uint64)):
                        vals.append(v.value)
                    else:
                        return False
                if i in self.dynamic:
                    if _tensor_params(tensors[self.dynamic.index(i)]) != vals:
                        return False
                    slices.append((len(params), len(params) + len(vals)))
                params += vals
            cufunc = kern._codelibrary.get_cufunc()
            for i in range(len(params)):  # (not cuda.core's arguments_info: it counts by probing past the end)
                err, offset, size = _cudrv.cuKernelGetParamInfo(cufunc.handle, i)
                if err != _cudrv.CUresult.CUDA_SUCCESS or int(offset) != 8 * i or int(size) != 8:
                    return False
            self.buf = np.array([v & 0xFFFFFFFFFFFFFFFF for v in params], dtype=np.uint64).view(np.int64)
            self.ptrs = (ctypes.c_void_p * len(params))(*(self.buf.ctypes.data + 8 * i for i in range(len(params))))
            self.pptr = ctypes.addressof(self.ptrs)
            self.handle = cufunc.handle
            self.slices = slices
            self.kinds = [(t.dtype, t.dim(), t.device) for t in tensors]
            return True  # the static device arrays the buffer points to stay alive in self.template
        except Exception:
            return False


def _make_kernel(L, W, R, QIN, MODE=MODE_D, MAXREG=None):
    """Kernel for ell_max L, W warps and R rotations per block; QIN: phases precomputed from quaternions.

    MODE_D: fused_D, which writes D. The other modes: fused_contract, which contracts each degree's D(n, ., .)
    with the coefficients cf of the rotation's function as soon as the n-wedge is built (D is never stored):
      MODE_Y      the 15 columns Y of so3kit.derivatives (val, a, g, aa, gg, ag, b-, b+, ab-, ab+, gb-, gb+,
                  bb-2, bb0, bb+2), out (B, 15, 2);
      MODE_DERIV  value, gradient and Hessian, assembled from Y as in WignerDGradients.get_derivatives,
                  out (B, 13, 1) = [value, gradient (3), Hessian (3 x 3, row-major)];
      MODE_VALUE  the value Re sum_j D_j c_j, out (B, 1, 1).
    Rotation b belongs to function b // npts; the sums are accumulated in float32."""
    KA = 1 if MODE == MODE_VALUE else NACC  # accumulators per rotation and warp
    HW = (L + 1) ** 2  # one n-wedge
    ZL = L + 1  # powers z^0 .. z^L
    M0N = L + 2  # degrees 0..L+1 of the m' = 0 column
    NT = 32 * W  # threads per block
    NTASK = 2 * R + R * (L + 1)  # prologue tasks: z-powers of alpha and gamma, then (rotation, degree)

    if QIN:
        @cuda.jit(device=True)
        def phases(P, unused1, unused2, i):
            """Phases precomputed by so3kit.wigner.to_euler_phases_cuda: rows alpha, beta, gamma (real, imag)."""
            return P[0, i], P[1, i], P[2, i], P[3, i], P[4, i], P[5, i]
    else:
        @cuda.jit(device=True)
        def phases(alpha, beta, gamma, i):
            """so3kit.wigner._angles_to_phases_cuda for element i (float32 cos/sin)."""
            a = alpha[i]
            b = beta[i]
            g = gamma[i]
            return math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(g), math.sin(g)

    @cuda.jit(device=True)
    def z_powers(z_real, z_imag, M, zr, zi, r):
        """so3kit.wigner._complex_powers_and_conjugate_cuda, channel 0, into shared zr[r, :], zi[r, :]."""
        zr[r, 0] = 1.0
        zi[r, 0] = 0.0
        if M > 0:
            θ_real = 1.0
            θ_imag = 0.0
            while z_real < 0.0 or z_imag < 0.0:
                new_θ_real = -θ_imag
                new_θ_imag = θ_real
                θ_real = new_θ_real
                θ_imag = new_θ_imag
                new_z_real = z_imag
                new_z_imag = -z_real
                z_real = new_z_real
                z_imag = new_z_imag
            zr[r, 1] = z_real
            zi[r, 1] = z_imag
            clock_real = θ_real
            clock_imag = θ_imag
            x = numba.float64(z_real)
            y = numba.float64(z_imag)
            sqrt_real = math.sqrt(0.5 * (math.sqrt(x * x + y * y) + x))
            sqrt_imag = y / (2.0 * sqrt_real) if sqrt_real > 0.0 else 0.0
            dc = -2 * (sqrt_imag ** 2)
            t = 2 * dc
            dz_real = dc * (1 + 2 * z_real)
            dz_imag = dc * (2 * z_imag) + math.sqrt(-dc * (2 + dc))
            for m in range(2, M + 1):
                zr[r, m] = zr[r, m - 1] + dz_real
                zi[r, m] = zi[r, m - 1] + dz_imag
                dz_real += t * zr[r, m]
                dz_imag += t * zi[r, m]
                temp_real = zr[r, m - 1] * clock_real - zi[r, m - 1] * clock_imag
                temp_imag = zr[r, m - 1] * clock_imag + zi[r, m - 1] * clock_real
                zr[r, m - 1] = temp_real
                zi[r, m - 1] = temp_imag
                temp_clock_real = clock_real * θ_real - clock_imag * θ_imag
                temp_clock_imag = clock_real * θ_imag + clock_imag * θ_real
                clock_real = temp_clock_real
                clock_imag = temp_clock_imag
            temp_real = zr[r, M] * clock_real - zi[r, M] * clock_imag
            temp_imag = zr[r, M] * clock_imag + zi[r, M] * clock_real
            zr[r, M] = temp_real
            zi[r, M] = temp_imag

    @cuda.jit(device=True)
    def column_m0(M0, r, n, cosβ, sinβ, sinβ_squared, g, h, U, C1, SQ, one):
        """H(n, 0, m), m = 0..n, into M0[r, n, m]: so3kit.wigner.step_1_2_cuda for one n, same expressions."""
        nn_index = n + n * (n + 1)
        if n == 1:
            M0[r, 1, 1] = U[1]
            M0[r, 1, 0] = (g[nn_index - 1] * cosβ) * 1.0/math.sqrt(2)
        else:
            M0[r, n, n] = U[n]
            gi = g[nn_index - 1]
            M0[r, n, n - 1] = gi * cosβ * M0[r, n, n]
            for i in range(2, n):
                gi = g[nn_index - i]
                hi = h[nn_index - i]
                M0[r, n, n - i] = gi * cosβ * M0[r, n, n - i + 1] - hi * sinβ_squared * M0[r, n, n - i + 2]
            const = C1[n]
            gi = g[nn_index - n]
            hi = h[nn_index - n]
            M0[r, n, 0] = (gi * cosβ * M0[r, n, 1] - hi * sinβ_squared * M0[r, n, 2]) * const
            prefactor = const
            for i in range(1, n):
                prefactor *= sinβ
                M0[r, n, i] *= prefactor
        # normalisation of the m = n element (the legacy kernel does this for all n after its n loop)
        prefactor = one[0]
        for k in range(1, n + 1):
            prefactor *= sinβ
        M0[r, n, n] *= prefactor / SQ[n]

    @cuda.jit(device=True, forceinline=True)
    def prologue(x0, x1, x2, b0, nr, tid, M0, zar, zai, zgr, zgi, cb, sb, g, h, U, C1, SQ, one):
        """All threads: Euler phases, powers of e^{i alpha} and e^{i gamma}, cos/sin beta and the m' = 0
        columns of degrees 1..L+1 of the block's rotations, into shared memory."""
        for task in range(tid, NTASK, NT):
            if task < 2 * R:
                r = task % R
                if r < nr:
                    pa_r, pa_i, pb_r, pb_i, pg_r, pg_i = phases(x0, x1, x2, b0 + r)
                    if task < R:
                        z_powers(pa_r, pa_i, L, zar, zai, r)
                        cb[r] = pb_r
                        sb[r] = pb_i
                    else:
                        z_powers(pg_r, pg_i, L, zgr, zgi, r)
            else:
                u = task - 2 * R
                r = u % R
                n = 1 + u // R
                if r < nr:
                    pa_r, pa_i, cosβ, sinβ, pg_r, pg_i = phases(x0, x1, x2, b0 + r)
                    column_m0(M0, r, n, cosβ, sinβ, sinβ * sinβ, g, h, U, C1, SQ, one)

    @cuda.jit(device=True, forceinline=True)
    def build_wedge(Hs, w, lane, nr, n, M0, cb, sb, one, b6s, b7s, a8s, s3w, ph, ww, r2, r3, r4, c6, c7, c8, top0):
        """One warp: the n-wedge of H of every rotation of the block, into Hs[w]."""
        if n == 0:
            for r in range(lane, nr, 32):
                Hs[w, r, 0] = 1.0
        else:
            # column m' = 0 into the wedge, and step 3: H(n, 1, m) from H(n + 1, 0, .)
            t0 = top0[n] - n
            cnt = n + 1
            for k in range(lane, nr * cnt, 32):
                r = k // cnt if R > 1 else 0
                m = k - r * cnt
                Hs[w, r, t0 + m] = M0[r, n, m]
            k0 = n * (n - 1) // 2
            for k in range(lane, nr * n, 32):
                r = k // n if R > 1 else 0
                i = k - r * n
                kk = k0 + i
                cosβ = cb[r]
                one_plus_cosβ = one[0] + cosβ
                one_minus_cosβ = one[0] - cosβ
                Hs[w, r, s3w[kk]] = (b6s[kk] * one_minus_cosβ * M0[r, n + 1, i + 2]
                                     - b7s[kk] * one_plus_cosβ * M0[r, n + 1, i]
                                     - a8s[kk] * sb[r] * M0[r, n + 1, i + 1])
            cuda.syncwarp()
            # steps 4 (m' > 0) and 5 (m' <= 0), phase by phase
            for p in range(1, n + 1):
                s = ph[n, p]
                cnt = ph[n, p + 1] - s
                for k in range(lane, nr * cnt, 32):
                    r = k // cnt if R > 1 else 0
                    kk = s + k - r * cnt
                    Hs[w, r, ww[kk]] = c6[kk] * Hs[w, r, r2[kk]] - c7[kk] * Hs[w, r, r3[kk]] + c8[kk] * Hs[w, r, r4[kk]]
                cuda.syncwarp()
        cuda.syncwarp()

    @cuda.jit(device=True, forceinline=True)
    def entry(Hs, w, r, zar, zai, zgr, zgi, pk):
        """(real, imag) of the D entry of rotation r with packed fill word pk."""
        hval = Hs[w, r, pk & 0xFFFF]
        za = (pk >> 16) & 63
        zy = (pk >> 22) & 63
        a_real = zar[r, za]
        a_imag = zai[r, za]
        if (pk >> 28) & 1:
            a_imag = -a_imag
        b_real = zgr[r, zy]
        b_imag = zgi[r, zy]
        if (pk >> 29) & 1:
            b_imag = -b_imag
        res = numba.float32(1.0)
        if (pk >> 30) & 1:
            res = numba.float32(-1.0)
        real_part = (a_real * b_real - a_imag * b_imag)
        imag_part = (a_real * b_imag + a_imag * b_real)
        return res * hval * real_part, res * hval * imag_part

    if MODE == MODE_D:
        @cuda.jit
        def fused_D(x0, x1, x2, out, g, h, U, C1, SQ, one, b6s, b7s, a8s, s3w, ph, ww, r2, r3, r4,
                    c6, c7, c8, top0, fill, Dbase, plan):
            Hs = cuda.shared.array((W, R, HW), numba.float32)
            M0 = cuda.shared.array((R, M0N, M0N), numba.float32)
            zar = cuda.shared.array((R, ZL), numba.float32)
            zai = cuda.shared.array((R, ZL), numba.float32)
            zgr = cuda.shared.array((R, ZL), numba.float32)
            zgi = cuda.shared.array((R, ZL), numba.float32)
            cb = cuda.shared.array(R, numba.float32)
            sb = cuda.shared.array(R, numba.float32)

            b0 = cuda.blockIdx.x * R
            tid = cuda.threadIdx.x
            w = tid // 32
            lane = tid % 32
            nr = min(R, out.shape[0] - b0)  # rotations in this block
            prologue(x0, x1, x2, b0, nr, tid, M0, zar, zai, zgr, zgi, cb, sb, g, h, U, C1, SQ, one)
            cuda.syncthreads()
            for slot in range(plan.shape[1]):
                n = plan[w, slot]
                if n < 0:
                    break
                build_wedge(Hs, w, lane, nr, n, M0, cb, sb, one, b6s, b7s, a8s, s3w, ph, ww, r2, r3, r4,
                            c6, c7, c8, top0)
                # fill D(n, ., .)
                base = Dbase[n]
                cnt = Dbase[n + 1] - base
                for r in range(nr):
                    b = b0 + r
                    for j in range(lane, cnt, 32):
                        d_re, d_im = entry(Hs, w, r, zar, zai, zgr, zgi, fill[base + j])
                        out[b, base + j, 0] = d_re
                        out[b, base + j, 1] = d_im
                cuda.syncwarp()

        return fused_D

    @cuda.jit(device=True, forceinline=True)
    def warp_sum(v):
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, 16)
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, 8)
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, 4)
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, 2)
        v += cuda.shfl_down_sync(0xFFFFFFFF, v, 1)
        return v

    @cuda.jit(device=True, forceinline=True)
    def seg_sum(v, lane, seg_end):
        """Sum of v over lanes lane .. seg_end - 1 (valid at the first lane of a segment; seg_end <= 32)."""
        o = cuda.shfl_down_sync(0xFFFFFFFF, v, 1)
        if lane + 1 < seg_end:
            v += o
        o = cuda.shfl_down_sync(0xFFFFFFFF, v, 2)
        if lane + 2 < seg_end:
            v += o
        o = cuda.shfl_down_sync(0xFFFFFFFF, v, 4)
        if lane + 4 < seg_end:
            v += o
        o = cuda.shfl_down_sync(0xFFFFFFFF, v, 8)
        if lane + 8 < seg_end:
            v += o
        o = cuda.shfl_down_sync(0xFFFFFFFF, v, 16)
        if lane + 16 < seg_end:
            v += o
        return v

    @cuda.jit(device=True, forceinline=True)
    def push(Acc, w, r, k, v, lane, seg_end, head):
        """Add the segment sum of v to accumulator k of rotation r (by the segment's first lane)."""
        v = seg_sum(v, lane, seg_end)
        if head:
            Acc[w, r, k] += v

    @cuda.jit(**({} if MAXREG is None else {"max_registers": MAXREG}))
    def fused_contract(x0, x1, x2, out, g, h, U, C1, SQ, one, b6s, b7s, a8s, s3w, ph, ww, r2, r3, r4,
                       c6, c7, c8, top0, fill, Dbase, plan, cf, npts, FMr, FPr, TNr, TPr, T0r):
        Hs = cuda.shared.array((W, R, HW), numba.float32)
        M0 = cuda.shared.array((R, M0N, M0N), numba.float32)
        zar = cuda.shared.array((R, ZL), numba.float32)
        zai = cuda.shared.array((R, ZL), numba.float32)
        zgr = cuda.shared.array((R, ZL), numba.float32)
        zgi = cuda.shared.array((R, ZL), numba.float32)
        cb = cuda.shared.array(R, numba.float32)
        sb = cuda.shared.array(R, numba.float32)
        Acc = cuda.shared.array((W, R, KA), numba.float32)  # per-warp partial sums

        b0 = cuda.blockIdx.x * R
        tid = cuda.threadIdx.x
        w = tid // 32
        lane = tid % 32
        nr = min(R, out.shape[0] - b0)  # rotations in this block
        for t in range(tid, W * R * KA, NT):
            Acc[t // (R * KA), (t // KA) % R, t % KA] = 0.0
        prologue(x0, x1, x2, b0, nr, tid, M0, zar, zai, zgr, zgi, cb, sb, g, h, U, C1, SQ, one)
        cuda.syncthreads()

        for slot in range(plan.shape[1]):
            n = plan[w, slot]
            if n < 0:
                break
            build_wedge(Hs, w, lane, nr, n, M0, cb, sb, one, b6s, b7s, a8s, s3w, ph, ww, r2, r3, r4,
                        c6, c7, c8, top0)
            base = Dbase[n]
            if MODE == MODE_VALUE:
                # Re sum_j D_j c_j: lanes over consecutive entries, one warp reduction per rotation
                cnt = Dbase[n + 1] - base
                for r in range(nr):
                    fb = (b0 + r) // npts
                    s_re = numba.float32(0.0)
                    for j in range(lane, cnt, 32):
                        d_re, d_im = entry(Hs, w, r, zar, zai, zgr, zgi, fill[base + j])
                        s_re += d_re * cf[fb, base + j, 0] - d_im * cf[fb, base + j, 1]
                    v = warp_sum(s_re)
                    if lane == 0:
                        Acc[w, r, 0] += v
            else:
                # Lane <-> (rotation, column m). Each lane walks down the rows m' of its column, with the
                # coefficients of rows m'-2 .. m'+2 in a register window (one coefficient load per entry), and
                # accumulates the row-weighted and row-shifted products; column weights are applied at the end.
                # One segmented warp reduction per pass over the (rotation, column) pairs.
                rowlen = 2 * n + 1
                NL = nr * rowlen
                for t0 in range(0, NL, 32):
                    t = t0 + lane
                    active = t < NL
                    r = t // rowlen if active else nr - 1
                    col = t - r * rowlen if active else 0
                    fb = (b0 + r) // npts
                    cbase = base + col
                    P_re = numba.float32(0.0)  # sum D c
                    P_im = numba.float32(0.0)
                    Pr_re = numba.float32(0.0)  # sum m' D c
                    Pr_im = numba.float32(0.0)
                    Prr_re = numba.float32(0.0)  # sum m'^2 D c
                    Prr_im = numba.float32(0.0)
                    Pt_re = numba.float32(0.0)  # sum t0(m') D c
                    Pt_im = numba.float32(0.0)
                    Qm_re = numba.float32(0.0)  # sum D c(m' + 1) f_minus
                    Qm_im = numba.float32(0.0)
                    Qmr_re = numba.float32(0.0)
                    Qmr_im = numba.float32(0.0)
                    Qp_re = numba.float32(0.0)  # sum D c(m' - 1) f_plus
                    Qp_im = numba.float32(0.0)
                    Qpr_re = numba.float32(0.0)
                    Qpr_im = numba.float32(0.0)
                    Bm_re = numba.float32(0.0)  # sum D c(m' + 2) t_minus2
                    Bm_im = numba.float32(0.0)
                    Bp_re = numba.float32(0.0)  # sum D c(m' - 2) t_plus2
                    Bp_im = numba.float32(0.0)
                    mc = numba.float32(0.0)
                    if active:
                        mc = numba.float32(col - n)
                        cm2_re = numba.float32(0.0)
                        cm2_im = numba.float32(0.0)
                        cm1_re = numba.float32(0.0)
                        cm1_im = numba.float32(0.0)
                        c0_re = cf[fb, cbase, 0]
                        c0_im = cf[fb, cbase, 1]
                        cp1_re = numba.float32(0.0)
                        cp1_im = numba.float32(0.0)
                        cp2_re = numba.float32(0.0)
                        cp2_im = numba.float32(0.0)
                        if n > 0:
                            cp1_re = cf[fb, cbase + rowlen, 0]
                            cp1_im = cf[fb, cbase + rowlen, 1]
                            cp2_re = cf[fb, cbase + 2 * rowlen, 0]
                            cp2_im = cf[fb, cbase + 2 * rowlen, 1]
                        for row in range(rowlen):
                            d_re, d_im = entry(Hs, w, r, zar, zai, zgr, zgi, fill[cbase + row * rowlen])
                            pre = d_re * c0_re - d_im * c0_im
                            pim = d_re * c0_im + d_im * c0_re
                            P_re += pre
                            P_im += pim
                            mr = numba.float32(row - n)
                            Pr_re += mr * pre
                            Pr_im += mr * pim
                            Prr_re += (mr * mr) * pre
                            Prr_im += (mr * mr) * pim
                            f = T0r[n, row]
                            Pt_re += f * pre
                            Pt_im += f * pim
                            if row + 1 <= 2 * n:  # coefficient one row down, times f_minus there
                                f = FMr[n, row]
                                sre = cp1_re * f
                                sim = cp1_im * f
                                qre = d_re * sre - d_im * sim
                                qim = d_re * sim + d_im * sre
                                Qm_re += qre
                                Qm_im += qim
                                Qmr_re += mr * qre
                                Qmr_im += mr * qim
                            if row >= 1:  # one row up, times f_plus there
                                f = FPr[n, row]
                                sre = cm1_re * f
                                sim = cm1_im * f
                                qre = d_re * sre - d_im * sim
                                qim = d_re * sim + d_im * sre
                                Qp_re += qre
                                Qp_im += qim
                                Qpr_re += mr * qre
                                Qpr_im += mr * qim
                            if row + 2 <= 2 * n:  # two rows down, times t_minus2 there
                                f = TNr[n, row]
                                sre = cp2_re * f
                                sim = cp2_im * f
                                Bm_re += d_re * sre - d_im * sim
                                Bm_im += d_re * sim + d_im * sre
                            if row >= 2:  # two rows up, times t_plus2 there
                                f = TPr[n, row]
                                sre = cm2_re * f
                                sim = cm2_im * f
                                Bp_re += d_re * sre - d_im * sim
                                Bp_im += d_re * sim + d_im * sre
                            # shift the window down one row
                            cm2_re = cm1_re
                            cm2_im = cm1_im
                            cm1_re = c0_re
                            cm1_im = c0_im
                            c0_re = cp1_re
                            c0_im = cp1_im
                            cp1_re = cp2_re
                            cp1_im = cp2_im
                            if row + 3 < rowlen:
                                cp2_re = cf[fb, cbase + (row + 3) * rowlen, 0]
                                cp2_im = cf[fb, cbase + (row + 3) * rowlen, 1]
                            else:
                                cp2_re = numba.float32(0.0)
                                cp2_im = numba.float32(0.0)
                    mc2 = mc * mc
                    # this pass's lanes of rotation r: [r * rowlen, (r + 1) * rowlen) - t0, clipped to the warp
                    # (shfl_down past lane 31 returns the caller's own value)
                    seg_end = min(min((r + 1) * rowlen, NL) - t0, 32)
                    if not active:
                        seg_end = 32
                    head = active and (lane == 0 or t == r * rowlen)
                    push(Acc, w, r, 0, P_re, lane, seg_end, head)
                    push(Acc, w, r, 1, P_im, lane, seg_end, head)
                    push(Acc, w, r, 2, Pr_re, lane, seg_end, head)
                    push(Acc, w, r, 3, Pr_im, lane, seg_end, head)
                    push(Acc, w, r, 4, mc * P_re, lane, seg_end, head)
                    push(Acc, w, r, 5, mc * P_im, lane, seg_end, head)
                    push(Acc, w, r, 6, Prr_re, lane, seg_end, head)
                    push(Acc, w, r, 7, Prr_im, lane, seg_end, head)
                    push(Acc, w, r, 8, mc2 * P_re, lane, seg_end, head)
                    push(Acc, w, r, 9, mc2 * P_im, lane, seg_end, head)
                    push(Acc, w, r, 10, mc * Pr_re, lane, seg_end, head)
                    push(Acc, w, r, 11, mc * Pr_im, lane, seg_end, head)
                    push(Acc, w, r, 12, Qm_re, lane, seg_end, head)
                    push(Acc, w, r, 13, Qm_im, lane, seg_end, head)
                    push(Acc, w, r, 14, Qp_re, lane, seg_end, head)
                    push(Acc, w, r, 15, Qp_im, lane, seg_end, head)
                    push(Acc, w, r, 16, Qmr_re, lane, seg_end, head)
                    push(Acc, w, r, 17, Qmr_im, lane, seg_end, head)
                    push(Acc, w, r, 18, Qpr_re, lane, seg_end, head)
                    push(Acc, w, r, 19, Qpr_im, lane, seg_end, head)
                    push(Acc, w, r, 20, mc * Qm_re, lane, seg_end, head)
                    push(Acc, w, r, 21, mc * Qm_im, lane, seg_end, head)
                    push(Acc, w, r, 22, mc * Qp_re, lane, seg_end, head)
                    push(Acc, w, r, 23, mc * Qp_im, lane, seg_end, head)
                    push(Acc, w, r, 24, Bm_re, lane, seg_end, head)
                    push(Acc, w, r, 25, Bm_im, lane, seg_end, head)
                    push(Acc, w, r, 26, Pt_re, lane, seg_end, head)
                    push(Acc, w, r, 27, Pt_im, lane, seg_end, head)
                    push(Acc, w, r, 28, Bp_re, lane, seg_end, head)
                    push(Acc, w, r, 29, Bp_im, lane, seg_end, head)
                    cuda.syncwarp()  # Acc[w, r, :] may be updated by another lane in the next pass
            cuda.syncwarp()

        # sum the warps' partial sums (fixed order) and apply the column factors i or -1
        cuda.syncthreads()
        if MODE == MODE_VALUE:
            for r in range(tid, nr, NT):
                s_re = numba.float32(0.0)
                for wv in range(W):
                    s_re += Acc[wv, r, 0]
                out[b0 + r, 0, 0] = s_re
        elif MODE == MODE_Y:
            for t in range(tid, nr * 15, NT):
                r = t // 15
                col = t - r * 15
                s_re = numba.float32(0.0)
                s_im = numba.float32(0.0)
                for wv in range(W):
                    s_re += Acc[wv, r, 2 * col]
                    s_im += Acc[wv, r, 2 * col + 1]
                if col == 1 or col == 2 or (col >= 8 and col <= 11):  # times i
                    out[b0 + r, col, 0] = -s_im
                    out[b0 + r, col, 1] = s_re
                elif col >= 3 and col <= 5:  # times -1
                    out[b0 + r, col, 0] = -s_re
                    out[b0 + r, col, 1] = -s_im
                else:
                    out[b0 + r, col, 0] = s_re
                    out[b0 + r, col, 1] = s_im
        else:
            for r in range(tid, nr, NT):
                for k in range(NACC):
                    s_re = numba.float32(0.0)
                    for wv in range(W):
                        s_re += Acc[wv, r, k]
                    Acc[0, r, k] = s_re
                # real (and where needed imaginary) parts of the 15 columns, with their factors i or -1
                y_a = -Acc[0, r, 3]  # Re(i S)
                y_g = -Acc[0, r, 5]
                y_aa = -Acc[0, r, 6]
                y_gg = -Acc[0, r, 8]
                y_ag = -Acc[0, r, 10]
                bm_r = Acc[0, r, 12]
                bm_i = Acc[0, r, 13]
                bp_r = Acc[0, r, 14]
                bp_i = Acc[0, r, 15]
                abm_r = -Acc[0, r, 17]
                abm_i = Acc[0, r, 16]
                abp_r = -Acc[0, r, 19]
                abp_i = Acc[0, r, 18]
                gbm_r = -Acc[0, r, 21]
                gbm_i = Acc[0, r, 20]
                gbp_r = -Acc[0, r, 23]
                gbp_i = Acc[0, r, 22]
                bbm_r = Acc[0, r, 24]
                bbm_i = Acc[0, r, 25]
                bb0_r = Acc[0, r, 26]
                bbp_r = Acc[0, r, 28]
                bbp_i = Acc[0, r, 29]
                pc = numba.float32(1.0)  # e^{i alpha}
                ps = numba.float32(0.0)
                if L >= 1:
                    pc = zar[r, 1]
                    ps = zai[r, 1]
                p2c = pc * pc - ps * ps  # e^{2 i alpha}
                p2s = 2 * pc * ps
                # get_derivatives: beta = Re(P bm + N bp), gb = Re(P gbm + N gbp),
                # ab = Re(P (i bm + abm) + N (-i bp + abp)), bb = Re(bb0 + P2 bbm + N2 bbp), N = conj(P)
                beta = pc * bm_r - ps * bm_i + pc * bp_r + ps * bp_i
                gb = pc * gbm_r - ps * gbm_i + pc * gbp_r + ps * gbp_i
                u_r = -bm_i + abm_r
                u_i = bm_r + abm_i
                v_r = bp_i + abp_r
                v_i = -bp_r + abp_i
                ab = pc * u_r - ps * u_i + pc * v_r + ps * v_i
                bb = bb0_r + p2c * bbm_r - p2s * bbm_i + p2c * bbp_r + p2s * bbp_i
                b = b0 + r
                out[b, 0, 0] = Acc[0, r, 0]
                out[b, 1, 0] = y_a
                out[b, 2, 0] = beta
                out[b, 3, 0] = y_g
                out[b, 4, 0] = y_aa
                out[b, 5, 0] = ab
                out[b, 6, 0] = y_ag
                out[b, 7, 0] = ab
                out[b, 8, 0] = bb
                out[b, 9, 0] = gb
                out[b, 10, 0] = y_ag
                out[b, 11, 0] = gb
                out[b, 12, 0] = y_gg

    return fused_contract


class FusedWignerD:
    """The fused D() of one WignerDMatrices instance, and the fused contractions value(), derivatives() and Y()
    of D with the coefficients of one function per batch entry, which never store D."""

    def __init__(self, wig, precision="float64", warps=None, rotations_per_block=None):
        self.wig = wig
        self.L = wig.ell_max
        num_sms = torch.cuda.get_device_properties(torch.device(wig.device)).multi_processor_count
        W, R = launch_config(self.L, precision, wig.total_batch_size, num_sms)
        self.W = int(warps or W)
        self.R = int(rotations_per_block or R)
        assert _fits(self.L, self.W, self.R), "shared memory"
        self.tables = t = FusedTables(wig, precision)
        d = cuda.to_device
        self.args = (d(t.g), d(t.h), d(t.U), d(t.C1), d(t.SQ), d(t.one), *map(d, t.s3), d(t.ph), *map(d, t.s45),
                     d(t.top0), d(t.fill), d(t.Dbase))
        self._plans = {}
        self._contract_config = {}  # mode -> (W, R)
        self._contract_tables = None  # built on first use
        self._launches = {}
        self.fast_launch = True  # False: every launch through numba (for comparisons)

    def _launch(self, key, kernel, griddim, blockdim, template, dynamic, tensors):
        L = self._launches.get(key)
        if L is None:
            L = self._launches[key] = _Launch(kernel, griddim, blockdim, template, dynamic)
        L(tensors, self.fast_launch)

    def _plan(self, nmax, W=None):
        W = W or self.W
        if (nmax, W) not in self._plans:
            full = self.tables.warp_plan(W)
            rows = [sorted(int(n) for n in row if 0 <= n <= nmax) for row in full]
            p = np.full((W, max(1, max(len(r) for r in rows))), -1, np.int32)
            for j, r in enumerate(rows):
                p[j, :len(r)] = r
            self._plans[nmax, W] = cuda.to_device(p)
        return self._plans[nmax, W]

    def _kernel(self, qin, mode=MODE_D, W=None, R=None):
        W, R = W or self.W, R or self.R
        maxreg = _CONTRACT_MAXREG if mode in (MODE_Y, MODE_DERIV) else None
        key = (self.L, W, R, qin, mode, maxreg)
        if key not in _KERNELS:
            _KERNELS[key] = _make_kernel(self.L, W, R, qin, mode, maxreg)
        return _KERNELS[key]

    def _inputs(self, q, alpha, beta, gamma):
        """Kernel inputs (x0, x1, x2) as torch tensors, and whether they are quaternion phases."""
        wig = self.wig
        B = wig.total_batch_size
        if q is not None:
            # Euler phases from the legacy kernel itself (one extra launch): compiled separately, the same
            # expressions may contract into different FMAs and round differently near 0
            from so3kit.wigner import to_euler_phases_cuda
            x = q.reshape(-1, 4)
            if isinstance(x, torch.Tensor):
                x = x.to(wig.device).contiguous()
            else:
                x = torch.as_tensor(np.ascontiguousarray(x), device=wig.device)
            if x.shape[0] != B:
                raise ValueError(f"expected {B} rotations (batchsize * num_candidates), got {x.shape[0]}")
            P = torch.empty((6, B), dtype=torch.float32, device=wig.device)
            if B > 0:
                self._launch(("phases", x.dtype), to_euler_phases_cuda, (B + 127) // 128, 128, [None] * 7,
                             range(7), [x, P[0], P[1], P[2], P[3], P[4], P[5]])
            return (P, P, P), True
        if alpha is not None and beta is not None and gamma is not None:
            x = tuple(t.reshape(-1).to(wig.device, torch.float32).contiguous() for t in (alpha, beta, gamma))
        else:
            raise ValueError("Either q or alpha, beta, gamma must be provided")
        for t in x:
            if t.shape[0] != B:
                raise ValueError(f"expected {B} rotations (batchsize * num_candidates), got {t.shape[0]}")
        return x, False

    def D(self, q=None, alpha=None, beta=None, gamma=None, truncate=None):
        wig = self.wig
        B = wig.total_batch_size
        # truncation as in the legacy D(): everything for None or ell_max, otherwise the degrees
        # below min(truncate + 1, ell_max)
        if truncate is None or truncate == self.L:
            nmax = self.L
        else:
            nmax = min(int(truncate) + 1, self.L) - 1
        width = int(self.tables.Dbase[nmax + 1])
        x, qin = self._inputs(q, alpha, beta, gamma)
        out = torch.empty((B, width), dtype=torch.complex64, device=wig.device)
        if B > 0:
            self._launch(("D", qin, nmax), self._kernel(qin), (B + self.R - 1) // self.R, 32 * self.W,
                         [None] * 4 + [*self.args, self._plan(nmax)], range(4), [*x, torch.view_as_real(out)])
        return out.view(wig.batchsize, wig.num_candidates, -1)

    # ---- contractions with per-function coefficients ----

    def prepare_coeffs(self, coeffs):
        """Coefficients (batchsize, >= Dsize) complex, ordered as D, as the (batchsize, ., 2) float32 tensor the
        contraction kernels read. Pass the result instead of the coefficients to save the conversion."""
        c = torch.view_as_real(coeffs.to(self.wig.device, torch.complex64).contiguous())
        if c.ndim != 3 or c.shape[0] != self.wig.batchsize or c.shape[1] < self.tables.Dsize:
            raise ValueError(f"coefficients must have shape ({self.wig.batchsize}, >= {self.tables.Dsize}), "
                             f"got {tuple(coeffs.shape)}")
        return c

    def _contract_args(self):
        """Per-row factor tables (L + 1, 2L + 1) of the columns of so3kit.derivatives: f_minus, f_plus,
        t_minus2, t_plus2 at the destination row of their shift (the source is 1 or 2 rows away), and t_0;
        derived from its index tables, with checks that they depend on (degree, row) only."""
        if self._contract_tables is None:
            from so3kit.derivatives import _load_indices
            L, Ds = self.L, self.tables.Dsize
            ms, factors, t_factors, map_p1, map_n1, map_p2, map_n2 = (
                np.asarray(a) for a in _load_indices(self.wig, None))
            ell = np.array([n for n in range(L + 1) for _ in range((2 * n + 1) ** 2)])
            row = np.array([r for n in range(L + 1) for r in range(2 * n + 1) for _ in range(2 * n + 1)])
            col = np.array([c for n in range(L + 1) for _ in range(2 * n + 1) for c in range(2 * n + 1)])
            rowlen = 2 * ell + 1
            Mr = ms[0][:Ds]
            if not (np.array_equal(Mr, row - ell) and np.array_equal(ms[1][:Ds], col - ell)):
                raise RuntimeError("so3kit.derivatives: unexpected (m', m) order")

            def per_row(per_entry, name):
                t = np.zeros((L + 1, 2 * L + 1))
                t[ell, row] = per_entry
                if not np.array_equal(t[ell, row], per_entry):
                    raise RuntimeError(f"so3kit.derivatives: {name} is not constant along rows")
                return t.astype(np.float32)

            idx = np.arange(Ds)
            tabs = []
            # WignerDGradients.set_so3_coeffs scatters dest <- src; here the factor sits at the destination
            for name, mp, fac, k in (("f_minus", map_n1, factors[0], 1), ("f_plus", map_p1, factors[1], -1),
                                     ("t_minus2", map_n2, t_factors[0], 2), ("t_plus2", map_p2, t_factors[2], -2)):
                mp = mp[:Ds].astype(np.int64)
                v = mp >= 0
                if not np.array_equal(mp[v], idx[v] - k * rowlen[v]):
                    raise RuntimeError(f"so3kit.derivatives: {name} shift is not {k} rows")
                valid = np.zeros(Ds, bool)
                valid[mp[v]] = True
                if not np.array_equal(valid, np.abs(Mr + k) <= ell):  # the kernel's row test
                    raise RuntimeError(f"so3kit.derivatives: {name} validity")
                t = np.zeros(Ds)
                t[mp[v]] = fac[:Ds][v]
                tabs.append(per_row(t, name))
            tabs.append(per_row(t_factors[1][:Ds], "t_0"))
            self._contract_tables = tuple(cuda.to_device(t) for t in tabs)
        return self._contract_tables

    def _contract(self, mode, width, depth, coeffs, alpha, beta, gamma):
        wig = self.wig
        B = wig.total_batch_size
        x, qin = self._inputs(None, alpha, beta, gamma)
        cf = coeffs
        if cf.is_complex() or cf.dim() != 3:
            cf = self.prepare_coeffs(cf)
        if (cf.dtype != torch.float32 or cf.device.type != "cuda" or not cf.is_contiguous()
                or cf.shape[0] != wig.batchsize or cf.shape[1] < self.tables.Dsize or cf.shape[2] != 2):
            raise ValueError("coefficients do not match this instance (see prepare_coeffs)")
        out = torch.empty((B, width, depth), dtype=torch.float32, device=wig.device)
        if B > 0:
            if mode not in self._contract_config:
                self._contract_config[mode] = _shrink(self.L, self.W, self.R, mode)
            W, R = self._contract_config[mode]
            template = [None] * 4 + [*self.args, self._plan(self.L, W), None, wig.num_candidates,
                                     *self._contract_args()]
            self._launch(("contract", mode, qin), self._kernel(qin, mode, W, R), (B + R - 1) // R, 32 * W,
                         template, (0, 1, 2, 3, 5 + len(self.args)), [*x, out, cf])
        return out

    def value(self, coeffs, alpha, beta, gamma):
        """Re sum_j c_{b, j} D_j at rotation (b, i): shape (batchsize, num_candidates), float32."""
        out = self._contract(MODE_VALUE, 1, 1, coeffs, alpha, beta, gamma)
        return out.view(self.wig.batchsize, self.wig.num_candidates)

    def derivatives(self, coeffs, alpha, beta, gamma):
        """Value (batchsize, k), gradient (batchsize, k, 3) and Hessian (batchsize, k, 3, 3) with respect to
        (alpha, beta, gamma), float32: the assembly of WignerDGradients.get_derivatives, in the kernel."""
        out = self._contract(MODE_DERIV, 13, 1, coeffs, alpha, beta, gamma)
        Bf, k = self.wig.batchsize, self.wig.num_candidates
        return out[:, 0, 0].view(Bf, k), out[:, 1:4, 0].view(Bf, k, 3), out[:, 4:13, 0].view(Bf, k, 3, 3)

    def Y(self, coeffs, alpha, beta, gamma):
        """The 15 columns of so3kit.derivatives (sum over D times WignerDGradients.CoeffBank):
        (batchsize, k, 15) complex64."""
        out = self._contract(MODE_Y, 15, 2, coeffs, alpha, beta, gamma)
        return torch.view_as_complex(out).view(self.wig.batchsize, self.wig.num_candidates, 15)
