# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled kernels for dedalus.tools.fastpaths.

apply_ell_blocks applies, for every block of a spherical ell-operator, a small CSR radial matrix along the
last (radial) axis of the coefficient data, for all m (and ell positions) in the block. Data are flat views of
C-contiguous arrays with shape (components..., Nm, Nl, Nr); the radial axis is last.

Block table columns (int64), one row per (regindex_in, regindex_out, ell):
  0 comp_in_off   flat offset of the input component
  1 comp_out_off  flat offset of the output component
  2 m0, 3 m1      local m-axis range
  4 l0, 5 l1      local ell-axis range
  6 n0_in, 7 n0_out  radial start index in input / output
  8 kind          0: real matrix A; 1: complex matrix on (cos, -sin) pairs along m (real dtype curl)
  9 a_r           index of the real-part CSR in the matrix table
  10 a_i          index of the imaginary-part CSR (kind 1), or -1
Matrix table (int64), one row per CSR: 0 row count, 1 indptr offset, 2 indices/data offset.
"""
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t


cdef inline void _csr_apply_add(const int64_t nrow, const int64_t[::1] indptr, const int64_t[::1] indices,
                                const double[::1] data, const int64_t p0, const int64_t d0,
                                const double[::1] x, const int64_t xoff, double[::1] y, const int64_t yoff,
                                const double sign) noexcept nogil:
    cdef int64_t i, k
    cdef double acc
    for i in range(nrow):
        acc = 0.0
        for k in range(indptr[p0 + i], indptr[p0 + i + 1]):
            acc = acc + data[d0 + k] * x[xoff + indices[d0 + k]]
        y[yoff + i] += sign * acc


def apply_ell_blocks(const double[::1] x, double[::1] y,
                     const int64_t[:, ::1] blocks, const int64_t[:, ::1] mats,
                     const int64_t[::1] indptr, const int64_t[::1] indices, const double[::1] data,
                     int64_t Nl_in, int64_t Nr_in, int64_t Nl_out, int64_t Nr_out):
    """y (zeroed by caller) += blockwise radial matrices applied to x."""
    cdef int64_t b, m, l, nb = blocks.shape[0]
    cdef int64_t cin, cout, m0, m1, l0, l1, n0i, n0o, kind, ar, ai
    cdef int64_t nrow_r, p_r, d_r, nrow_i, p_i, d_i
    cdef int64_t xo, yo, xo2, yo2
    with nogil:
        for b in range(nb):
            cin = blocks[b, 0]; cout = blocks[b, 1]
            m0 = blocks[b, 2]; m1 = blocks[b, 3]; l0 = blocks[b, 4]; l1 = blocks[b, 5]
            n0i = blocks[b, 6]; n0o = blocks[b, 7]; kind = blocks[b, 8]; ar = blocks[b, 9]; ai = blocks[b, 10]
            nrow_r = 0; p_r = 0; d_r = 0
            if ar >= 0:
                nrow_r = mats[ar, 0]; p_r = mats[ar, 1]; d_r = mats[ar, 2]
            if kind == 0:
                for m in range(m0, m1):
                    for l in range(l0, l1):
                        xo = cin + (m * Nl_in + l) * Nr_in + n0i
                        yo = cout + (m * Nl_out + l) * Nr_out + n0o
                        _csr_apply_add(nrow_r, indptr, indices, data, p_r, d_r, x, xo, y, yo, 1.0)
            else:
                # pairs (cos, msin) at (m, m+1), m1 - m0 even (checked when the table is built):
                # out_cos += Ar cos - Ai msin ; out_msin += Ai cos + Ar msin
                m = m0
                while m < m1:
                    for l in range(l0, l1):
                        xo = cin + (m * Nl_in + l) * Nr_in + n0i          # cos input
                        xo2 = cin + ((m + 1) * Nl_in + l) * Nr_in + n0i   # msin input
                        yo = cout + (m * Nl_out + l) * Nr_out + n0o
                        yo2 = cout + ((m + 1) * Nl_out + l) * Nr_out + n0o
                        if ar >= 0:
                            _csr_apply_add(nrow_r, indptr, indices, data, p_r, d_r, x, xo, y, yo, 1.0)
                            _csr_apply_add(nrow_r, indptr, indices, data, p_r, d_r, x, xo2, y, yo2, 1.0)
                        if ai >= 0:
                            nrow_i = mats[ai, 0]; p_i = mats[ai, 1]; d_i = mats[ai, 2]
                            _csr_apply_add(nrow_i, indptr, indices, data, p_i, d_i, x, xo2, y, yo, -1.0)
                            _csr_apply_add(nrow_i, indptr, indices, data, p_i, d_i, x, xo, y, yo2, 1.0)
                    m = m + 2


# ---------------------------------------------------------------------------------------------------------
# Transform kernels

def recombine_ell(double[::1] data, int64_t ncomp, int64_t csize, const int64_t[:, ::1] table,
                  const double[::1] Q, int64_t Nl, int64_t Nr, bint transpose, double[::1] tmp):
    """In place: for each table row (m0, m1, l0, l1, qoff), apply the ncomp x ncomp matrix Q[qoff:] (row-major;
    its transpose if `transpose`) across the component axis of data (ncomp, Nm, Nl, Nr) for m in [m0, m1),
    l in [l0, l1), all r. For each m the (l, r) span is contiguous; tmp needs ncomp * Nl * Nr doubles."""
    cdef int64_t b, m, c, d, k, base, span, qoff, nb = table.shape[0]
    cdef double q
    cdef double* dp = &data[0]
    cdef const double* qp = &Q[0]
    cdef double* tp = &tmp[0]
    cdef double* t
    cdef double* x
    if tmp.shape[0] < ncomp * Nl * Nr:
        raise ValueError("tmp too small")
    with nogil:
        for b in range(nb):
            qoff = table[b, 4]
            span = (table[b, 3] - table[b, 2]) * Nr
            for m in range(table[b, 0], table[b, 1]):
                base = (m * Nl + table[b, 2]) * Nr
                for c in range(ncomp):
                    t = tp + c * span
                    for k in range(span):
                        t[k] = 0.0
                    for d in range(ncomp):
                        if transpose:
                            q = qp[qoff + d * ncomp + c]
                        else:
                            q = qp[qoff + c * ncomp + d]
                        if q == 0.0:
                            continue
                        x = dp + d * csize + base
                        for k in range(span):
                            t[k] = t[k] + q * x[k]
                for c in range(ncomp):
                    t = tp + c * span
                    x = dp + c * csize + base
                    for k in range(span):
                        x[k] = t[k]


def fourier_repack(const double[::1] cdata, double[::1] temp, int64_t npre, int64_t Nc, int64_t npost, int64_t Nk,
                   const int64_t[::1] src_cos, const int64_t[::1] src_msin, int64_t Kmax, double scale0, double scale):
    """temp (npre, Nk, npost) complex, as interleaved doubles: k = 0 -> cdata[src_cos[0]] * scale0 + 0j;
    1 <= k <= Kmax -> (cdata[src_cos[k]] + 1j cdata[src_msin[k]]) * scale; k > Kmax -> 0.
    cdata is (npre, Nc, npost)."""
    cdef int64_t p, k, q, ci, si, to
    with nogil:
        for p in range(npre):
            for k in range(Nk):
                to = 2 * (p * Nk + k) * npost
                if k == 0:
                    ci = (p * Nc + src_cos[0]) * npost
                    for q in range(npost):
                        temp[to + 2 * q] = cdata[ci + q] * scale0
                        temp[to + 2 * q + 1] = 0.0
                elif k <= Kmax:
                    ci = (p * Nc + src_cos[k]) * npost
                    si = (p * Nc + src_msin[k]) * npost
                    for q in range(npost):
                        temp[to + 2 * q] = cdata[ci + q] * scale
                        temp[to + 2 * q + 1] = cdata[si + q] * scale
                else:
                    for q in range(2 * npost):
                        temp[to + q] = 0.0


def fourier_unpack(const double[::1] temp, double[::1] cdata, int64_t npre, int64_t Nk, int64_t npost, int64_t Nc,
                   const int64_t[::1] src_k, const int64_t[::1] src_part, const double[::1] scale):
    """cdata (npre, Nc, npost) row j = scale[j] * (real part if src_part[j] == 0, imag part if 1) of
    temp row src_k[j] (temp is (npre, Nk, npost) complex as interleaved doubles); src_part[j] == 2 -> zero."""
    cdef int64_t p, j, q, ti, co, part
    cdef double s
    with nogil:
        for p in range(npre):
            for j in range(Nc):
                co = (p * Nc + j) * npost
                part = src_part[j]
                if part == 2:
                    for q in range(npost):
                        cdata[co + q] = 0.0
                else:
                    ti = 2 * (p * Nk + src_k[j]) * npost + part
                    s = scale[j]
                    for q in range(npost):
                        cdata[co + q] = temp[ti + 2 * q] * s


# ---------------------------------------------------------------------------------------------------------
# Subproblem gather/scatter

def gather_idx(list srcs, const int64_t[::1] fptr, const int64_t[::1] sidx, const int64_t[::1] didx, double[::1] dst):
    """dst[didx[i]] = srcs[f][sidx[i]] for i in [fptr[f], fptr[f+1])."""
    cdef Py_ssize_t f, nf = len(srcs)
    cdef int64_t i
    cdef double[::1] src
    for f in range(nf):
        src = srcs[f]
        with nogil:
            for i in range(fptr[f], fptr[f + 1]):
                dst[didx[i]] = src[sidx[i]]


def scatter_idx(list dsts, const int64_t[::1] fptr, const int64_t[::1] sidx, const int64_t[::1] didx, const double[::1] buf):
    """dsts[f][sidx[i]] = buf[didx[i]] for i in [fptr[f], fptr[f+1])."""
    cdef Py_ssize_t f, nf = len(dsts)
    cdef int64_t i
    cdef double[::1] dst
    for f in range(nf):
        dst = dsts[f]
        with nogil:
            for i in range(fptr[f], fptr[f + 1]):
                dst[sidx[i]] = buf[didx[i]]


# ---------------------------------------------------------------------------------------------------------
# Grid-space products. Same operation order as Dedalus's numexpr strings
# ("x*y - z*w" for each cross component; left-to-right sum for dot); compiled without FMA contraction.

def cross3(const double[::1] a0, const double[::1] a1, const double[::1] a2,
           const double[::1] b0, const double[::1] b1, const double[::1] b2,
           double[::1] o0, double[::1] o1, double[::1] o2, bint right_handed):
    cdef Py_ssize_t i, n = o0.shape[0]
    cdef double x0, x1, x2, y0, y1, y2
    with nogil:
        if right_handed:
            for i in range(n):
                x0 = a0[i]; x1 = a1[i]; x2 = a2[i]; y0 = b0[i]; y1 = b1[i]; y2 = b2[i]
                o0[i] = x1 * y2 - x2 * y1
                o1[i] = x2 * y0 - x0 * y2
                o2[i] = x0 * y1 - x1 * y0
        else:
            for i in range(n):
                x0 = a0[i]; x1 = a1[i]; x2 = a2[i]; y0 = b0[i]; y1 = b1[i]; y2 = b2[i]
                o0[i] = x2 * y1 - x1 * y2
                o1[i] = x0 * y2 - x2 * y0
                o2[i] = x1 * y0 - x0 * y1


def dot3(const double[::1] a0, const double[::1] a1, const double[::1] a2,
         const double[::1] b0, const double[::1] b1, const double[::1] b2, double[::1] o):
    cdef Py_ssize_t i, n = o.shape[0]
    with nogil:
        for i in range(n):
            o[i] = a0[i] * b0[i] + a1[i] * b1[i] + a2[i] * b2[i]


# ---------------------------------------------------------------------------------------------------------
# Chebyshev radial transforms (FFTW DCT with ultraspherical conversion). These reproduce, pencil by pencil and in the
# same operation order, FastChebyshevTransform's resize/rescale methods (with and without conversion) around the DCT,
# including Dedalus's solve_upper_csr and apply_csr loops, so the results are bitwise identical to the original path.

from libc.stdint cimport int32_t


def cheb_backward(const double[::1] cdata, double[::1] W, int64_t npen, int64_t Nc, int64_t Ng, int64_t K,
                  double r0, double rp, bint zero_bad, bint convert, bint truncate_in,
                  const int32_t[::1] Ap, const int32_t[::1] Aj, const double[::1] Ax, double[::1] x):
    """Coefficients (npen, Nc) -> DCT input W (npen, Ng): [truncate, upper-triangular solve,] sign of odd modes,
    rescale, zero padding. K is Kmax (no conversion) or Kmax_orig (conversion)."""
    cdef int64_t p, i, jj, ci, wo
    cdef double s
    with nogil:
        for p in range(npen):
            ci = p * Nc
            wo = p * Ng
            for i in range(Nc):
                x[i] = cdata[ci + i]
            if convert:
                if truncate_in:
                    for i in range(K + 1, Nc):
                        x[i] = 0
                for i in range(Nc - 1, -1, -1):
                    s = x[i]
                    for jj in range(Ap[i + 1] - 1, Ap[i], -1):
                        s = s - Ax[jj] * x[Aj[jj]]
                    x[i] = s / Ax[Ap[i]]
            if K > 0:
                for i in range(1, K + 1, 2):
                    x[i] = x[i] * -1
            W[wo] = x[0] * r0
            if K > 0:
                for i in range(1, K + 1):
                    W[wo + i] = x[i] * rp
                if zero_bad:
                    for i in range(K + 1, Ng):
                        W[wo + i] = 0


def cheb_forward(double[::1] W, double[::1] cdata, int64_t npen, int64_t Ng, int64_t Nc, int64_t K,
                 double r0, double rp, bint zero_bad, bint convert, int64_t n_in,
                 const int32_t[::1] Ap, const int32_t[::1] Aj, const double[::1] Ax):
    """DCT output W (npen, Ng) -> coefficients (npen, Nc): rescale, [zero bad modes,] sign of odd modes,
    [truncate to n_in and apply the CSR conversion]. W is used as scratch."""
    cdef int64_t p, i, jj, ci, wo
    cdef double s
    with nogil:
        for p in range(npen):
            ci = p * Nc
            wo = p * Ng
            if convert:
                W[wo] = W[wo] * r0
                if K > 0:
                    for i in range(1, K + 1):
                        W[wo + i] = W[wo + i] * rp
                    if zero_bad:
                        for i in range(K + 1, Ng):
                            W[wo + i] = 0
                    for i in range(1, K + 1, 2):
                        W[wo + i] = W[wo + i] * -1
                for i in range(Nc):
                    s = 0.0
                    for jj in range(Ap[i], Ap[i + 1]):
                        s = s + Ax[jj] * W[wo + Aj[jj]]
                    cdata[ci + i] = s
            else:
                cdata[ci] = W[wo] * r0
                if K > 0:
                    for i in range(1, K + 1):
                        cdata[ci + i] = W[wo + i] * rp
                    if zero_bad:
                        for i in range(K + 1, Nc):
                            cdata[ci + i] = 0
                    for i in range(1, K + 1, 2):
                        cdata[ci + i] = cdata[ci + i] * -1
