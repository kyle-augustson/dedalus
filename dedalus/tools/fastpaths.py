"""
Compiled fast paths for spherical problems.

Each entry point below returns True when it has done the work, and False when the caller must run its own (original)
code. The fast paths reproduce the original operations; differences are at round-off level (summation order), and the
cross product and subproblem gather/scatter are bitwise identical. They are used only for real (float64) data and
never for adjoint fields or adjoint/JVP evaluations. Switch them off with [optimization] FAST_PATHS = False in
dedalus.cfg, or at runtime with dedalus.tools.fastpaths.ENABLED = False.

  ell_operate     SphericalEllOperator / SphericalCurl (grad, div, curl, lap, spherical Convert, ...): a block table per
                  operator and layout, applied by fast_kernels.apply_ell_blocks (each small radial CSR matrix applied to
                  all m of its block), instead of a Python loop over components and ells.
  dot_operate     DotProduct of two vectors over the same coordinate system: one pass, instead of np.einsum.
  cross_operate   CrossProduct: one pass for all three components, instead of three numexpr calls.
  swsh_forward / swsh_backward
                  SWSH colatitude transforms: np.matmul straight into the strided output per m.
  md_forward / md_backward
                  Sphere/shell/ball transforms: azimuth FFT with the m permutation folded into the pack/unpack; colatitude
                  transforms with reusable workspaces; shell radial transforms with one DCT over all components.
  recombine       Regularity recombination: one pass over all ells.
  gather / scatter
                  Subproblem gather/scatter: one call per subproblem from flat index tables, instead of one np.copyto per
                  (field, subsystem) view.
  axis0_first_comm
                  Rank relabelling used by Distributor when [parallelism] MESH_AXIS0_NODE_LOCAL = True.
"""

import numpy as np
import numexpr as ne
from scipy import sparse
from ..tools.config import config

try:
    from ..libraries import fast_kernels as _kern
except ImportError:   # extension not built: everything falls back to the original code
    _kern = None

try:
    ENABLED = config['optimization'].getboolean('FAST_PATHS')
except KeyError:
    ENABLED = True
ENABLED = bool(ENABLED) and _kern is not None


# ---------------------------------------------------------------------------------------------------------
# Spherical ell operators

def _slice_bounds(s, n):
    if not isinstance(s, slice):
        return None
    start, stop, step = s.indices(n)
    if step != 1:
        return None
    return start, max(start, stop)


def _build_table(op, operand, out, curl_like):
    """Block table for dfast_kernels.apply_ell_blocks, or None if this case needs the stock code."""
    if curl_like:
        basis = op.input_basis
    else:
        basis = op.output_basis if op.input_basis is None else op.input_basis
    radial_basis = op.radial_basis
    axis = op.dist.last_axis(radial_basis)
    rank_in, rank_out = len(operand.tensorsig), len(out.tensorsig)
    if op.dist.dim != 3 or axis != 2 or operand.data.ndim != rank_in + 3 or out.data.ndim != rank_out + 3:
        return None
    tshape_in, tshape_out = operand.data.shape[:rank_in], out.data.shape[:rank_out]
    Nm_in, Nl_in, Nr_in = operand.data.shape[rank_in:]
    Nm_out, Nl_out, Nr_out = out.data.shape[rank_out:]
    if (Nm_in, Nl_in) != (Nm_out, Nl_out):
        return None
    csize_in, csize_out = Nm_in * Nl_in * Nr_in, Nm_out * Nl_out * Nr_out
    R_in = radial_basis.regularity_classes(operand.tensorsig)
    blocks, mats, IP, IX, DT = [], [], [], [], []
    p_off = d_off = 0
    mat_ids = {}
    keep = []   # hold source matrices so their ids stay unique while the table is built

    def add_matrix(M, key=None):
        nonlocal p_off, d_off
        if key is not None and key in mat_ids:
            return mat_ids[key]
        # copy=True: Dedalus caches these matrices and some are non-canonical; sum_duplicates on a shared
        # view would compact the cached arrays in place and corrupt them for every other user
        M = sparse.csr_matrix(M, copy=True)
        M.sum_duplicates()
        if M.nnz == 0:
            idx = -1
        else:
            mats.append((M.shape[0], p_off, d_off))
            IP.append(M.indptr.astype(np.int64)); IX.append(M.indices.astype(np.int64)); DT.append(M.data.astype(np.float64))
            p_off += M.shape[0] + 1; d_off += M.nnz
            idx = len(mats) - 1
        if key is not None:
            mat_ids[key] = idx
        return idx
    for regindex_in, regtotal_in in np.ndenumerate(R_in):
        for regindex_out in op.regindex_out(regindex_in):
            cin = (int(np.ravel_multi_index(regindex_in, tshape_in)) if rank_in else 0) * csize_in
            cout = (int(np.ravel_multi_index(regindex_out, tshape_out)) if rank_out else 0) * csize_out
            for ell, m_ind, ell_ind in basis.ell_maps(op.dist):
                if not (radial_basis.regularity_allowed(ell, regindex_in) and radial_basis.regularity_allowed(ell, regindex_out)):
                    continue
                mb, lb = _slice_bounds(m_ind, Nm_in), _slice_bounds(ell_ind, Nl_in)
                n_slice = radial_basis.n_slice(ell)
                nb_in, nb_out = _slice_bounds(n_slice, Nr_in), _slice_bounds(n_slice, Nr_out)
                if mb is None or lb is None or nb_in is None or nb_out is None:
                    return None
                (m0, m1), (l0, l1) = mb, lb
                if m1 <= m0 or l1 <= l0 or nb_in[1] <= nb_in[0] or nb_out[1] <= nb_out[0]:
                    continue
                A = op.radial_matrix(regindex_in, regindex_out, ell)
                keep.append(A)
                if A.shape != (nb_out[1] - nb_out[0], nb_in[1] - nb_in[0]):
                    return None
                if sparse.issparse(A):
                    Ar, Ai = A.real, A.imag
                else:
                    Ar, Ai = np.real(A), np.imag(A)
                if curl_like:
                    if (m1 - m0) % 2:
                        return None
                    ar, ai = add_matrix(Ar, (id(A), 'r')), add_matrix(Ai, (id(A), 'i'))
                    if ar < 0 and ai < 0:
                        continue
                    blocks.append((cin, cout, m0, m1, l0, l1, nb_in[0], nb_out[0], 1, ar, ai))
                else:
                    if (sparse.issparse(Ai) and Ai.nnz and np.any(Ai.data)) or (not sparse.issparse(Ai) and np.any(Ai)):
                        return None
                    ar = add_matrix(Ar, (id(A), 'r'))
                    if ar < 0:
                        continue
                    blocks.append((cin, cout, m0, m1, l0, l1, nb_in[0], nb_out[0], 0, ar, -1))
    if not blocks:
        blocks = np.zeros((0, 11), dtype=np.int64); mats_a = np.zeros((0, 3), dtype=np.int64)
        return (blocks, mats_a, np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1), Nl_in, Nr_in, Nl_out, Nr_out)
    return (np.array(blocks, dtype=np.int64), np.array(mats, dtype=np.int64),
            np.concatenate(IP), np.concatenate(IX), np.concatenate(DT), Nl_in, Nr_in, Nl_out, Nr_out)


_NOFAST = object()


def ell_operate(op, operand, out, curl_like):
    """Fast SphericalEllOperator/SphericalCurl._operate for float64 data (non-adjoint). Returns False if not handled."""
    if not ENABLED or op.dtype != np.float64 or op.input_basis is None:
        return False
    if not (operand.data.flags.c_contiguous and operand.data.dtype == np.float64):
        return False
    out.preset_layout(operand.layout)
    key = (operand.layout, operand.data.shape, out.data.shape)
    cache = op.__dict__.setdefault('_fast_tables', {})
    table = cache.get(key)
    if table is None:
        table = _build_table(op, operand, out, curl_like)
        cache[key] = _NOFAST if table is None else table
    if table is None or table is _NOFAST:
        return False
    o = out.data.reshape(-1)
    o.fill(0)
    blocks, mats, ip, ix, dt, Nl_in, Nr_in, Nl_out, Nr_out = table
    if blocks.shape[0]:
        _kern.apply_ell_blocks(operand.data.reshape(-1), o, blocks, mats, ip, ix, dt, Nl_in, Nr_in, Nl_out, Nr_out)
    return True


# ---------------------------------------------------------------------------------------------------------
# Products

_dot_exprs = {}


def dot_operate(op, out):
    """Fast DotProduct.operate for vector @ vector over the same coordinate system. Returns False if not handled."""
    if not ENABLED:
        return False
    arg0, arg1 = op.args
    if not (len(arg0.tensorsig) == 1 and len(arg1.tensorsig) == 1 and len(out.tensorsig) == 0
            and tuple(op.indices) == (0, 0) and arg0.tensorsig[0] is arg1.tensorsig[0]):
        return False
    out.preset_layout(arg0.layout)
    a = op.arg0_ghost_broadcaster.cast(arg0)
    b = op.arg1_ghost_broadcaster.cast(arg1)
    if out.data.size:
        n = arg0.tensorsig[0].dim
        o = out.data
        if (n == 3 and a.dtype == b.dtype == o.dtype == np.float64 and a.shape[1:] == b.shape[1:] == o.shape
                and a.flags.c_contiguous and b.flags.c_contiguous and o.flags.c_contiguous):
            _kern.dot3(a[0].reshape(-1), a[1].reshape(-1), a[2].reshape(-1), b[0].reshape(-1), b[1].reshape(-1), b[2].reshape(-1), o.reshape(-1))
        else:
            expr = _dot_exprs.get(n)
            if expr is None:
                expr = _dot_exprs[n] = ' + '.join('a%d*b%d' % (i, i) for i in range(n))
            local = {}
            for i in range(n):
                local['a%d' % i] = a[i]; local['b%d' % i] = b[i]
            ne.evaluate(expr, local_dict=local, out=out.data)
    return True


def cross_operate(op, out, right_handed):
    """Fast CrossProduct.operate_{right,left}_handed. Returns False if not handled (before any ghost cast)."""
    if not ENABLED:
        return False
    arg0, arg1 = op.args
    out.preset_layout(arg0.layout)
    a = op.arg0_ghost_broadcaster.cast(arg0)
    b = op.arg1_ghost_broadcaster.cast(arg1)
    o = out.data
    if (a.dtype == b.dtype == o.dtype == np.float64 and a.shape == b.shape == o.shape and o.shape[0] == 3
            and a.flags.c_contiguous and b.flags.c_contiguous and o.flags.c_contiguous):
        _kern.cross3(a[0].reshape(-1), a[1].reshape(-1), a[2].reshape(-1), b[0].reshape(-1), b[1].reshape(-1),
                     b[2].reshape(-1), o[0].reshape(-1), o[1].reshape(-1), o[2].reshape(-1), right_handed)
        return True
    # broadcasting or non-contiguous: the original numexpr expressions, on the already ghost-cast data
    data00, data01, data02 = a[0], a[1], a[2]
    data10, data11, data12 = b[0], b[1], b[2]
    if right_handed:
        ne.evaluate("data01*data12 - data02*data11", out=out.data[0])
        ne.evaluate("data02*data10 - data00*data12", out=out.data[1])
        ne.evaluate("data00*data11 - data01*data10", out=out.data[2])
    else:
        ne.evaluate("data02*data11 - data01*data12", out=out.data[0])
        ne.evaluate("data00*data12 - data02*data10", out=out.data[1])
        ne.evaluate("data01*data10 - data00*data11", out=out.data[2])
    return True


# ---------------------------------------------------------------------------------------------------------
# SWSH colatitude transforms

def swsh_forward(transform, gdata, cdata):
    if not ENABLED:
        return False
    from ..tools.array import apply_matrix
    m_matrices = transform._forward_SWSH_matrices
    Lmax = transform.Lmax
    for m, mg_slice, mc_slice, ell_slice in transform.m_maps:
        if abs(m) <= Lmax:
            M = m_matrices[m]
            if isinstance(M, np.ndarray):
                np.matmul(M, gdata[:, mg_slice, :, :], out=cdata[:, mc_slice, ell_slice, :])
            else:
                apply_matrix(M, gdata[:, mg_slice, :, :], axis=2, out=cdata[:, mc_slice, ell_slice, :])
    return True


def swsh_backward(transform, cdata, gdata):
    if not ENABLED:
        return False
    from ..tools.array import apply_matrix
    m_matrices = transform._backward_SWSH_matrices
    Lmax = transform.Lmax
    for m, mg_slice, mc_slice, ell_slice in transform.m_maps:
        if abs(m) > Lmax:
            gdata[:, mg_slice, :, :] = 0
        else:
            M = m_matrices[m]
            if isinstance(M, np.ndarray):
                np.matmul(M, cdata[:, mc_slice, ell_slice, :], out=gdata[:, mg_slice, :, :])
            else:
                apply_matrix(M, cdata[:, mc_slice, ell_slice, :], axis=2, out=gdata[:, mg_slice, :, :])
    return True


# ---------------------------------------------------------------------------------------------------------
# Sphere / shell / ball transforms

_ws = {}
_SPH_TYPES = ('SphereBasis', 'ShellBasis', 'BallBasis')

def _workspace(tag, shape, zero=False):
    key = (tag, tuple(shape))
    W = _ws.get(key)
    if W is None:
        from ..libraries.fftw import fftw_wrappers as fftw
        W = _ws[key] = fftw.create_array(tuple(shape), np.float64)   # aligned, zero-initialised
    elif zero:
        W.fill(0)
    return W


def _sphere(self):
    return self if type(self).__name__ == 'SphereBasis' else self.sphere_basis


def _fourier_tables(az, plan, Nc, which):
    key = ('_dfast_ftab', which, Nc, plan.N)
    tab = az.__dict__.get(key)
    if tab is None:
        Kmax, N = plan.Kmax, plan.N
        if which == 'b':
            perm = az.backward_coeff_permutation
            perm = np.arange(Nc) if perm is None else np.asarray(perm)
            src_cos = np.zeros(Kmax + 1, dtype=np.int64); src_msin = np.zeros(Kmax + 1, dtype=np.int64)
            for k in range(Kmax + 1):
                src_cos[k] = perm[2 * k]
                src_msin[k] = perm[2 * k + 1] if k else 0
            tab = (src_cos, src_msin)
        else:
            perm = az.forward_coeff_permutation
            perm = np.arange(Nc) if perm is None else np.asarray(perm)
            rescale = 1 / N
            src_k = np.zeros(Nc, dtype=np.int64); src_part = np.full(Nc, 2, dtype=np.int64); scale = np.zeros(Nc)
            for j in range(Nc):
                q = int(perm[j])
                k, part = divmod(q, 2)
                if k == 0 and part == 0:
                    src_k[j], src_part[j], scale[j] = 0, 0, rescale
                elif 1 <= k <= Kmax:
                    src_k[j], src_part[j], scale[j] = k, part, 2 * rescale
            tab = (src_k, src_part, scale)
        az.__dict__[key] = tab
    return tab


def _fourier_ok(self, field, gdata, cdata, data_axis):
    from ..core import transforms
    sph = _sphere(self)
    if sph.mmax == 0:
        return None
    az = sph.azimuth_basis
    plan = az.transform_plan(field.dist, gdata.shape[data_axis])
    if type(plan) is not transforms.FFTWRealFFT:
        return None
    if not (gdata.flags.c_contiguous and cdata.flags.c_contiguous):
        return None
    return az, plan


def _az_forward(self, field, axis, gdata, cdata):
    data_axis = len(field.tensorsig) + axis
    r = _fourier_ok(self, field, gdata, cdata, data_axis)
    if r is None:
        return False
    az, plan = r
    fplan, temp = plan._build_fftw_plan(gdata.shape, data_axis)
    Nc = cdata.shape[data_axis]
    src_k, src_part, scale = _fourier_tables(az, plan, Nc, 'f')
    fplan.forward(gdata, temp)
    npre = int(np.prod(gdata.shape[:data_axis])); npost = int(np.prod(gdata.shape[data_axis + 1:]))
    _kern.fourier_unpack(temp.reshape(-1).view(np.float64), cdata.reshape(-1), npre, temp.shape[data_axis], npost, Nc,
                         src_k, src_part, scale)
    return True


def _az_backward(self, field, axis, cdata, gdata):
    data_axis = len(field.tensorsig) + axis
    r = _fourier_ok(self, field, gdata, cdata, data_axis)
    if r is None:
        return False
    az, plan = r
    fplan, temp = plan._build_fftw_plan(gdata.shape, data_axis)
    Nc = cdata.shape[data_axis]
    if 2 * (plan.Kmax + 1) > Nc:
        return False
    src_cos, src_msin = _fourier_tables(az, plan, Nc, 'b')
    npre = int(np.prod(cdata.shape[:data_axis])); npost = int(np.prod(cdata.shape[data_axis + 1:]))
    _kern.fourier_repack(cdata.reshape(-1), temp.reshape(-1).view(np.float64), npre, Nc, npost, temp.shape[data_axis],
                         src_cos, src_msin, plan.Kmax, 1.0, 0.5)
    fplan.backward(temp, gdata)
    return True


def _uncovered_rows(sph, field, n_rows):
    """Local azimuth rows of the colatitude grid layout not written by the SWSH backward transform."""
    d = sph.__dict__.setdefault('_dfast_unc', {})
    rows = d.get((id(field.dist), n_rows))
    if rows is None:
        covered = np.zeros(n_rows, dtype=bool)
        for m, mg_slice, mc_slice, ell_slice in sph.m_maps(field.dist):
            covered[mg_slice] = True
        rows = d[(id(field.dist), n_rows)] = np.flatnonzero(~covered)
    return rows


def _colat_backward(self, field, axis, cdata, gdata):
    sph = _sphere(self)
    if sph.mmax == 0 or not gdata.flags.c_contiguous:
        return False
    rank = len(field.tensorsig)
    az_axis = rank + axis - 1
    temp = _workspace('colat_b', gdata.shape)
    unc = _uncovered_rows(sph, field, gdata.shape[az_axis])
    if unc.size:
        idx = [slice(None)] * gdata.ndim; idx[az_axis] = unc
        temp[tuple(idx)] = 0
    S = sph.spin_weights(field.tensorsig)
    for i, s in np.ndenumerate(S):
        Ntheta = gdata[i].shape[axis]
        plan = sph.transform_plan(field.dist, Ntheta, s)
        plan.backward(cdata[i], temp[i], axis)
    if gdata.shape[az_axis] % 2:
        gdata.fill(0)
    sph.backward_spin_recombination(field.tensorsig, axis, temp, gdata)
    return True


def _colat_forward(self, field, axis, gdata, cdata):
    sph = _sphere(self)
    if sph.mmax == 0 or not gdata.flags.c_contiguous:
        return False
    rank = len(field.tensorsig)
    az_axis = rank + axis - 1
    temp = _workspace('colat_f', gdata.shape, zero=bool(gdata.shape[az_axis] % 2))
    sph.forward_spin_recombination(field.tensorsig, axis, gdata, temp)
    cdata.fill(0)
    S = sph.spin_weights(field.tensorsig)
    for i, s in np.ndenumerate(S):
        Ntheta = gdata[i].shape[axis]
        plan = sph.transform_plan(field.dist, Ntheta, s)
        plan.forward(temp[i], cdata[i], axis)
    return True


def _dct_plan(gshape, axis):
    key = ('dct', tuple(gshape), axis)
    P = _ws.get(key)
    if P is None:
        from ..libraries.fftw import fftw_wrappers as fftw
        from ..core import transforms
        P = _ws[key] = fftw.DiscreteCosineTransform(np.float64, tuple(gshape), axis, flags=['FFTW_' + transforms.GET_FFTW_RIGOR().upper()])
    return P


def _radius_plan(self, field, axis, gdata):
    from ..core import transforms
    if type(self).__name__ != 'ShellBasis':
        return None
    rb = self.radial_basis
    data_axis = len(field.tensorsig) + axis
    plan = rb.transform_plan(field.dist, gdata.shape[data_axis], self.k)
    if not isinstance(plan, transforms.FFTWFastChebyshevTransform):
        return None
    if not gdata.flags.c_contiguous or data_axis != gdata.ndim - 1:
        return None
    return rb, plan


def _csr32(M):
    M = M.tocsr()
    return (np.ascontiguousarray(M.indptr, dtype=np.int32), np.ascontiguousarray(M.indices, dtype=np.int32),
            np.ascontiguousarray(M.data, dtype=np.float64))


_NOCSR = (np.zeros(1, np.int32), np.zeros(1, np.int32), np.zeros(1))


def _cheb_params(plan):
    """Parameters of a FastChebyshevTransform's resize/rescale steps for fast_kernels.cheb_backward/forward, or None
    for a direction whose method is not one of the two known variants (then the original per-component calls run)."""
    p = plan.__dict__.get('_fast_cheb')
    if p is not None:
        return p
    from ..core import transforms
    FCT = transforms.FastChebyshevTransform
    p = {}
    fb = getattr(plan.resize_rescale_backward, '__func__', None)
    if fb is FCT._resize_rescale_backward:
        p['b'] = (plan.Kmax, plan.backward_rescale_zero, plan.backward_rescale_pos, plan.KN > plan.Kmax, False, False) + _NOCSR
    elif fb is FCT._resize_rescale_backward_convert:
        K = plan.Kmax_orig
        p['b'] = (K, plan.backward_rescale_zero, plan.backward_rescale_pos, plan.KN > K, True, plan.M_orig > plan.N) + _csr32(plan.backward_conversion)
    else:
        p['b'] = None
    ff = getattr(plan.resize_rescale_forward, '__func__', None)
    if ff is FCT._resize_rescale_forward:
        p['f'] = (plan.Kmax, plan.forward_rescale_zero, plan.forward_rescale_pos, plan.KM > plan.Kmax, False) + _NOCSR
    elif ff is FCT._resize_rescale_forward_convert:
        p['f'] = (plan.Kmax, plan.forward_rescale_zero, plan.forward_rescale_pos, plan.KM > plan.Kmax, True) + _csr32(plan.forward_conversion)
    else:
        p['f'] = None
    plan.__dict__['_fast_cheb'] = p
    return p


_xtmp = [np.zeros(0)]


def _radius_forward(self, field, axis, gdata, cdata):
    r = _radius_plan(self, field, axis, gdata)
    if r is None or not cdata.flags.c_contiguous:
        return False
    rb, plan = r
    data_axis = len(field.tensorsig) + axis
    if self.k > 0:
        gdata *= rb.radial_transform_factor(field.scales[axis], data_axis, -self.k)
    rb.forward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps(field.dist))
    # one DCT over all components into a workspace; then per-component rescale/convert into cdata
    # (cdata aliases gdata, which is fully consumed by the DCT first)
    W = _workspace('rad', gdata.shape)
    g2 = gdata.reshape(-1, gdata.shape[-1]); W2 = W.reshape(g2.shape)
    _dct_plan(g2.shape, 1).forward(g2, W2)
    cp = _cheb_params(plan)['f']
    if cp is not None and hasattr(_kern, 'cheb_forward'):
        K, r0, rp, zero_bad, convert, Ap, Aj, Ax = cp
        Nc = cdata.shape[-1]
        _kern.cheb_forward(W2.reshape(-1), cdata.reshape(-1), g2.shape[0], g2.shape[1], Nc, K, r0, rp, zero_bad, convert, Nc, Ap, Aj, Ax)
        return True
    R = rb.regularity_classes(field.tensorsig)
    for i, rt in np.ndenumerate(R):
        plan.resize_rescale_forward(W[i], cdata[i], axis, plan.Kmax)
    return True


def _radius_backward(self, field, axis, cdata, gdata):
    r = _radius_plan(self, field, axis, gdata)
    if r is None or not cdata.flags.c_contiguous:
        return False
    rb, plan = r
    data_axis = len(field.tensorsig) + axis
    # per-component rescale/convert from cdata into a workspace (all reads of cdata happen before gdata is written)
    W = _workspace('rad', gdata.shape)
    cp = _cheb_params(plan)['b']
    if cp is not None and hasattr(_kern, 'cheb_backward'):
        K, r0, rp, zero_bad, convert, truncate_in, Ap, Aj, Ax = cp
        Nc, Ng = cdata.shape[-1], gdata.shape[-1]
        if _xtmp[0].size < Nc:
            _xtmp[0] = np.zeros(Nc)
        _kern.cheb_backward(cdata.reshape(-1), W.reshape(-1), cdata.size // Nc, Nc, Ng, K, r0, rp, zero_bad, convert,
                            truncate_in, Ap, Aj, Ax, _xtmp[0])
    else:
        R = rb.regularity_classes(field.tensorsig)
        for i, rt in np.ndenumerate(R):
            plan.resize_rescale_backward(cdata[i], W[i], axis, plan.Kmax)
    W2 = W.reshape(-1, W.shape[-1]); g2 = gdata.reshape(W2.shape)
    _dct_plan(W2.shape, 1).backward(W2, g2)
    rb.backward_regularity_recombination(field.tensorsig, axis, gdata, self.ell_maps(field.dist))
    if self.k > 0:
        gdata *= rb.radial_transform_factor(field.scales[axis], data_axis, self.k)
    return True


_FWD = {0: _az_forward, 1: _colat_forward, 2: _radius_forward}
_BWD = {0: _az_backward, 1: _colat_backward, 2: _radius_backward}


def md_forward(basis, field, axis, gdata, cdata):
    """Fast MultidimensionalBasis.forward_transform for sphere/shell/ball bases. Returns False if not handled."""
    if not ENABLED or field.dtype != np.float64 or gdata.dtype != np.float64 or getattr(field, 'adjoint', False):
        return False
    if type(basis).__name__ not in _SPH_TYPES:
        return False
    fn = _FWD.get(axis - field.dist.get_basis_axis(basis))
    return bool(fn is not None and fn(basis, field, axis, gdata, cdata))


def md_backward(basis, field, axis, cdata, gdata):
    """Fast MultidimensionalBasis.backward_transform for sphere/shell/ball bases. Returns False if not handled."""
    if not ENABLED or field.dtype != np.float64 or gdata.dtype != np.float64 or getattr(field, 'adjoint', False):
        return False
    if type(basis).__name__ not in _SPH_TYPES:
        return False
    fn = _BWD.get(axis - field.dist.get_basis_axis(basis))
    return bool(fn is not None and fn(basis, field, axis, cdata, gdata))


# ---------------------------------------------------------------------------------------------------------
# Regularity recombination

def _recomb_table(self, tensorsig, axis, gdata, ell_maps, transpose):
    cache = self.__dict__.setdefault('_dfast_recomb', {})
    key = (tensorsig, axis, gdata.shape, id(ell_maps), transpose)
    hit = cache.get(key)
    if hit is not None and hit[0] is ell_maps:
        return hit[1]
    rank = len(tensorsig)
    tab = None
    if axis == 2 and gdata.ndim == rank + 3:
        ncomp = int(np.prod(gdata.shape[:rank]))
        Nm, Nl, Nr = gdata.shape[rank:]
        ell_list = tuple(mp[0] for mp in ell_maps)
        Q = self.radial_recombinations(tensorsig, ell_list)
        rows, qs, qoff, qid = [], [], 0, {}
        ok = ncomp <= 64
        for ell, m_ind, ell_ind in ell_maps:
            mb, lb = _slice_bounds(m_ind, Nm), _slice_bounds(ell_ind, Nl)
            Qe = Q[ell].toarray() if sparse.issparse(Q[ell]) else np.asarray(Q[ell])
            if np.iscomplexobj(Qe):
                if np.any(Qe.imag):
                    ok = False; break
                Qe = Qe.real
            Qe = Qe.astype(np.float64)
            if mb is None or lb is None or Qe.shape != (ncomp, ncomp):
                ok = False; break
            if mb[1] <= mb[0] or lb[1] <= lb[0]:
                continue
            if ell not in qid:
                qid[ell] = qoff; qs.append(np.ascontiguousarray(Qe).reshape(-1)); qoff += ncomp * ncomp
            rows.append((mb[0], mb[1], lb[0], lb[1], qid[ell]))
        if ok:
            tab = (ncomp, Nm * Nl * Nr, np.array(rows, dtype=np.int64).reshape(-1, 5),
                   np.concatenate(qs) if qs else np.zeros(1), Nl, Nr)
    cache[key] = (ell_maps, tab)
    return tab


_rtmp = [np.zeros(0)]


def _recomb_tmp(n):
    if _rtmp[0].size < n:
        _rtmp[0] = np.zeros(n)
    return _rtmp[0]


def recombine(basis, tensorsig, axis, gdata, ell_maps, forward):
    """Fast RegularityBasis.{forward,backward}_regularity_recombination. Returns False if not handled."""
    if not ENABLED or len(tensorsig) == 0 or gdata.dtype != np.float64 or not gdata.flags.c_contiguous:
        return False
    tab = _recomb_table(basis, tensorsig, axis, gdata, ell_maps, forward)
    if tab is None:
        return False
    ncomp, csize, rows, Q, Nl, Nr = tab
    if rows.shape[0]:
        _kern.recombine_ell(gdata.reshape(-1), ncomp, csize, rows, Q, Nl, Nr, forward, _recomb_tmp(ncomp * Nl * Nr))
    return True


# ---------------------------------------------------------------------------------------------------------
# Subproblem gather / scatter

def _flat_index(view, base):
    """Flat element indices of `view` within the contiguous 1D-viewable array `base`, or None."""
    isz = base.itemsize
    off = view.__array_interface__['data'][0] - base.__array_interface__['data'][0]
    if off % isz or any(st % isz for st in view.strides):
        return None
    idx = np.full(view.shape, off // isz, dtype=np.int64)
    for ax, (n, st) in enumerate(zip(view.shape, view.strides)):
        sh = [1] * view.ndim; sh[ax] = n
        idx = idx + (np.arange(n, dtype=np.int64) * (st // isz)).reshape(sh)
    idx = idx.ravel()
    if idx.size and (idx.min() < 0 or idx.max() >= base.size):
        return None
    return idx


def _gs_table(sp, fields, which):
    cache = sp.__dict__.setdefault('_dfast_gs', {})
    key = (which, fields)
    hit = cache.get(key)
    if hit is not None:
        bufs = hit[0]
        if all(f.buffer is b for f, b in zip(fields, bufs)):
            return hit
    buffer = sp._input_buffer if which == 'in' else sp._output_buffer
    vlist = sp._input_views if which == 'in' else sp._output_views
    tab = None
    if (buffer.dtype == np.float64 and buffer.flags.c_contiguous and len(vlist) == len(fields)
            and all(f.buffer.dtype == np.float64 and f.buffer.ndim == 1 and f.buffer.flags.c_contiguous for f in fields)):
        flatbuf = buffer.reshape(-1)
        S, D, fptr, ok = [], [], [0], True
        for field, buffer_data in zip(fields, vlist):
            n = 0
            for buffer_view, field_slices in buffer_data:
                field_view = field.data[field_slices]
                if buffer_view.shape != field_view.shape:
                    ok = False; break
                d = _flat_index(buffer_view, flatbuf); s = _flat_index(field_view, field.buffer)
                if d is None or s is None:
                    ok = False; break
                S.append(s); D.append(d); n += s.size
            if not ok:
                break
            fptr.append(fptr[-1] + n)
        if ok:
            S = np.concatenate(S) if S else np.zeros(0, np.int64)
            D = np.concatenate(D) if D else np.zeros(0, np.int64)
            tab = (tuple(f.buffer for f in fields), np.array(fptr, dtype=np.int64), S, D, flatbuf)
    if tab is None:
        tab = (tuple(f.buffer for f in fields), None)
    cache[key] = tab
    return tab


def gather(sp, fields, which):
    """Copy field data into the subproblem buffer ('in' or 'out'). Returns False if not handled."""
    if not ENABLED:
        return False
    tab = _gs_table(sp, tuple(fields), which)
    if tab[1] is None:
        return False
    bufs, fptr, S, D, flatbuf = tab
    _kern.gather_idx(list(bufs), fptr, S, D, flatbuf)
    return True


def scatter(sp, fields, which):
    """Copy the subproblem buffer ('in' or 'out') back to field data. Returns False if not handled."""
    if not ENABLED:
        return False
    tab = _gs_table(sp, tuple(fields), which)
    if tab[1] is None:
        return False
    bufs, fptr, S, D, flatbuf = tab
    _kern.scatter_idx(list(bufs), fptr, S, D, flatbuf)
    return True


# ---------------------------------------------------------------------------------------------------------
# Rank order

def axis0_first_comm(comm, mesh):
    mesh = [int(m) for m in mesh if int(m) > 1]
    if len(mesh) < 2:
        return comm
    w = comm.rank
    coords = []
    for m in mesh:              # axis 0 fastest in world-rank order
        coords.append(w % m); w //= m
    key = int(np.ravel_multi_index(tuple(coords), tuple(mesh)))   # row-major cart rank of these coords
    return comm.Split(0, key)
