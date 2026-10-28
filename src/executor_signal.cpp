#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>

#ifdef USE_NVIDIA
#include <cufft.h>
#endif

// CONVOLVE / FLIP (opcodes 150-151), BATCHED_MATMUL (170). Each result chare
// gathers one box per source view, then computes locally (direct sum, overlap-save
// FFT, batched GEMM; cuFFT/cuBLAS on NVIDIA). The result has any decomposition.

#ifdef USE_KOKKOS
#define CT_SIG_FN KOKKOS_INLINE_FUNCTION
#else
#define CT_SIG_FN inline
#endif

namespace signal_detail {

enum ConvMethod : int { CONV_DIRECT = 0, CONV_FFT = 1 };

/// A strided view of a source array, in view-index coordinates.
template <int N>
struct SrcView {
    int name = -1;
    ArrayDecomp<N> dec;
    std::array<int, N> vstart, vstep, vsize; // vstart in decomp-global coords
};

template <int N>
struct SigGeom {
    Opcode op = Opcode::NOOP;
    int nin = 0;
    SrcView<N> in[2]; // CONVOLVE: a, v; FLIP: x
    ArrayDecomp<N> out;
    int method = CONV_DIRECT;
    std::array<int, N> shift{}; // CONVOLVE: out[j] = sum_k v[k] a[j + shift - k]
    int flip_mask = 0;
    int dims[2][3] = {}; // BATCHED_MATMUL: dims of a (batch, m, k) and b (batch, n, k)
};

ASTNode* signal_root(DAGNode* node) {
    for (ASTNode* r : node->ast->roots)
        if (is_signal_op(static_cast<Opcode>(r->opcode)))
            return r;
    return nullptr;
}

template <int N>
bool make_view(ArrayDAGGroup* g, ASTNode* root, int i, SrcView<N>& v) {
    v.name = root->operands[i]->result_name;
    auto it = g->array_meta.find(v.name);
    if (it == g->array_meta.end() || it->second.ndims != N)
        return false;
    v.dec = it->second.template decomp<N>();
    Region* r = root->get_operand_region(i);
    for (int d = 0; d < N; ++d) {
        if (r == nullptr || r->is_global) {
            v.vstart[d] = v.dec.offset[d];
            v.vstep[d] = 1;
            v.vsize[d] = v.dec.global_shape[d];
        } else {
            auto* ar = static_cast<ArrayRegion<N>*>(r);
            v.vstart[d] = ar->start[d] + v.dec.offset[d];
            v.vstep[d] = ar->step[d];
            v.vsize[d] = ar->size(d);
        }
    }
    return true;
}

/// CONVOLVE operands: a, v, method, shift[N], shape[N]; FLIP operands: x, mask.
template <int N>
bool make_geom(ArrayDAGGroup* g, ASTNode* root, SigGeom<N>& s) {
    s.op = static_cast<Opcode>(root->opcode);
    s.nin = s.op == Opcode::FLIP ? 1 : 2;
    for (int i = 0; i < s.nin; ++i)
        if (!make_view<N>(g, root, i, s.in[i]))
            return false;
    auto out_it = g->array_meta.find(root->result_name);
    if (out_it == g->array_meta.end())
        return false;
    s.out = out_it->second.template decomp<N>();
    if (s.op == Opcode::CONVOLVE) {
        s.method = (int)root->operands[2]->scalar;
        for (int d = 0; d < N; ++d)
            s.shift[d] = (int)root->operands[3 + d]->scalar;
    } else if (s.op == Opcode::BATCHED_MATMUL) {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                s.dims[i][j] = (int)root->operands[2 + 3 * i + j]->scalar;
    } else {
        s.flip_mask = (int)root->operands[1]->scalar;
    }
    return true;
}

/// View-index range [lo, hi) along axis d held by source chare c.
template <int N>
bool src_block_1d(const SrcView<N>& v, int d, int c, int& lo, int& hi) {
    return v.dec.view_block(d, c, v.vstart[d], v.vstep[d], v.vsize[d], lo, hi);
}

template <int N>
void chare_span(const ArrayDecomp<N>& dec, int d, int& first, int& last) {
    first = dec.offset[d] / dec.tile;
    last = (dec.offset[d] + dec.global_shape[d] - 1) / dec.tile;
}

/// Result index range [lo, hi) (array coords) of result chare c along d.
template <int N>
void out_range_1d(const ArrayDecomp<N>& dec, int d, int c, int& lo, int& hi) {
    lo = std::max(c * dec.tile, dec.offset[d]) - dec.offset[d];
    hi = std::min((c + 1) * dec.tile, dec.offset[d] + dec.global_shape[d]) - dec.offset[d];
}

/// Box of source view i (view-index coords, may exceed it) that the result
/// range [olo, ohi) along d reads.
template <int N>
void need_1d(const SigGeom<N>& s, int i, int d, int olo, int ohi, int& lo, int& hi) {
    if (s.op == Opcode::CONVOLVE) {
        int f = s.in[1].vsize[d];
        if (i == 0) {
            lo = olo + s.shift[d] - (f - 1);
            hi = ohi + s.shift[d];
        } else {
            lo = 0;
            hi = f;
        }
    } else if ((s.flip_mask >> d) & 1) {
        lo = s.in[0].vsize[d] - ohi;
        hi = s.in[0].vsize[d] - olo;
    } else {
        lo = olo;
        hi = ohi;
    }
}

/// CONVOLVE: f(T{}) for the compute dtypes (the frontend casts other dtypes).
template <typename F>
void with_conv_dtype(DType dt, F&& f) {
    switch (dt) {
    case DType::FLOAT32: f(float{}); break;
    case DType::FLOAT64: f(double{}); break;
    case DType::INT32: f(int32_t{}); break;
    case DType::INT64: f(int64_t{}); break;
    default: CkAbort("CONVOLVE: unsupported dtype %d", (int)dt);
    }
}

/// FLIP moves elements only: f(T{}) with a type of dt's element size.
template <typename F>
void with_elem_type(DType dt, F&& f) {
    switch (dtype_size(dt)) {
    case 1: f(uint8_t{}); break;
    case 2: f(uint16_t{}); break;
    case 4: f(uint32_t{}); break;
    case 8: f(uint64_t{}); break;
    default: f(cn_c128{}); break;
    }
}

/// with_conv_dtype or with_elem_type, by op; the BLAS types for BATCHED_MATMUL.
template <typename F>
void with_sig_type(Opcode op, DType dt, F&& f) {
    if (op == Opcode::FLIP)
        with_elem_type(dt, f);
    else if (op == Opcode::BATCHED_MATMUL) {
        if (!cn_dispatch_blas(dt, f))
            CkAbort("batched matmul: unsupported dtype %d", (int)dt);
    } else
        with_conv_dtype(dt, f);
}

// ---- device/host buffers and kernels ----

/// Strided box copy: dst[dbase + sum i_d*ds_d] = src[sbase + sum i_d*ss_d] over
/// the box ext (row-major order); strides may be negative.
template <int N, typename T>
void copy_box(PartitionImpl<N>* partition, T* dst, int64_t dbase, const int64_t* dstride,
              const T* src, int64_t sbase, const int64_t* sstride, const int64_t* extent) {
    int64_t ext[N], ds[N], ss[N];
    int64_t total = 1;
    for (int d = 0; d < N; ++d) {
        ext[d] = extent[d];
        ds[d] = dstride[d];
        ss[d] = sstride[d];
        total *= ext[d];
    }
    if (total <= 0)
        return;
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, total), KOKKOS_LAMBDA(int64_t k) {
            int64_t rem = k, di = dbase, si = sbase;
            for (int d = N - 1; d >= 0; --d) {
                int64_t c = rem % ext[d];
                rem /= ext[d];
                di += c * ds[d];
                si += c * ss[d];
            }
            dst[di] = src[si];
        });
#else
    (void)partition;
    for (int64_t k = 0; k < total; ++k) {
        int64_t rem = k, di = dbase, si = sbase;
        for (int d = N - 1; d >= 0; --d) {
            int64_t c = rem % ext[d];
            rem /= ext[d];
            di += c * ds[d];
            si += c * ss[d];
        }
        dst[di] = src[si];
    }
#endif
}

template <int N, typename T>
void fill_zero(PartitionImpl<N>* partition, T* p, int64_t n) {
#ifdef USE_KOKKOS
    Kokkos::parallel_for(CT_COMPUTE_POLICY(partition, n), KOKKOS_LAMBDA(int64_t k) { p[k] = T(0); });
#else
    (void)partition;
    std::fill(p, p + n, T(0));
#endif
}

template <int N>
void row_major_strides(const int64_t* ext, int64_t* stride) {
    stride[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d)
        stride[d] = stride[d + 1] * ext[d + 1];
}

/// Direct convolution out[j] = sum_k v[k] * w[j + f - 1 - k] (w: extent len + f - 1),
/// with N-D shapes padded to 3-D (leading extents 1).
template <typename T>
struct DirectConv {
    T* out;
    const T* w;
    const T* v;
    int64_t len[3], f[3], ws[3];

    CT_SIG_FN void operator()(int64_t idx) const {
        int64_t j2 = idx % len[2], j1 = (idx / len[2]) % len[1], j0 = idx / (len[2] * len[1]);
        int64_t base = (j0 + f[0] - 1) * ws[0] + (j1 + f[1] - 1) * ws[1] + (j2 + f[2] - 1);
        T acc = T(0);
        for (int64_t k0 = 0; k0 < f[0]; ++k0)
            for (int64_t k1 = 0; k1 < f[1]; ++k1) {
                const T* vk = v + (k0 * f[1] + k1) * f[2];
                const T* wk = w + base - k0 * ws[0] - k1 * ws[1];
                for (int64_t k2 = 0; k2 < f[2]; ++k2)
                    acc += vk[k2] * wk[-k2];
            }
        out[idx] = acc;
    }
};

template <int N, typename T>
void direct_conv(PartitionImpl<N>* partition, T* out, const T* w, const T* v, const int64_t* len,
                 const int64_t* f) {
    DirectConv<T> k{out, w, v, {1, 1, 1}, {1, 1, 1}, {0, 0, 1}};
    for (int d = 0; d < N; ++d) {
        k.len[3 - N + d] = len[d];
        k.f[3 - N + d] = f[d];
    }
    k.ws[1] = k.len[2] + k.f[2] - 1;
    k.ws[0] = k.ws[1] * (k.len[1] + k.f[1] - 1);
    int64_t total = k.len[0] * k.len[1] * k.len[2];
#ifdef USE_KOKKOS
    Kokkos::parallel_for(CT_COMPUTE_POLICY(partition, total), k);
#else
    // Row by row: the output row and the input rows it reads stay in cache.
    (void)partition;
    (void)total;
    int64_t n2 = k.len[2], f2 = k.f[2];
    for (int64_t j0 = 0; j0 < k.len[0]; ++j0)
        for (int64_t j1 = 0; j1 < k.len[1]; ++j1) {
            T* orow = out + (j0 * k.len[1] + j1) * n2;
            std::fill(orow, orow + n2, T(0));
            for (int64_t k0 = 0; k0 < k.f[0]; ++k0)
                for (int64_t k1 = 0; k1 < k.f[1]; ++k1) {
                    const T* wrow = w + (j0 + k.f[0] - 1 - k0) * k.ws[0] +
                                    (j1 + k.f[1] - 1 - k1) * k.ws[1];
                    const T* vk = v + (k0 * k.f[1] + k1) * f2;
                    for (int64_t k2 = 0; k2 < f2; ++k2) {
                        T c = vk[k2];
                        const T* wr = wrow + (f2 - 1 - k2);
                        for (int64_t j2 = 0; j2 < n2; ++j2)
                            orow[j2] += c * wr[j2];
                    }
                }
        }
#endif
}

/// Smallest n' >= n whose prime factors are all in {2, 3, 5, 7}.
int64_t fft_size(int64_t n) {
    for (int64_t m = std::max<int64_t>(n, 1);; ++m) {
        int64_t r = m;
        for (int64_t p : {2, 3, 5, 7})
            while (r % p == 0)
                r /= p;
        if (r == 1)
            return m;
    }
}

#ifndef USE_NVIDIA
// ---- host FFT (mixed radix 2/3/5/7, double precision) ----

using cplx = std::complex<double>;

/// Out-of-place recursive Cooley-Tukey: y = DFT_n(x[0], x[xs], ...).
/// tw[j * tws] = exp(-2 pi i j / n).
void fft_rec(const cplx* x, int64_t xs, cplx* y, int64_t n, const int* fac, const cplx* tw,
             int64_t tws) {
    if (n == 1) {
        y[0] = x[0];
        return;
    }
    int p = fac[0];
    int64_t m = n / p;
    for (int r = 0; r < p; ++r)
        fft_rec(x + r * xs, xs * p, y + r * m, m, fac + 1, tw, tws * p);
    cplx t[7];
    for (int64_t k = 0; k < m; ++k) {
        for (int r = 0; r < p; ++r)
            t[r] = y[r * m + k] * tw[((r * k) % n) * tws];
        for (int q = 0; q < p; ++q) {
            cplx acc = t[0];
            for (int r = 1; r < p; ++r)
                acc += t[r] * tw[((int64_t(r) * q % p) * m) * tws];
            y[k + q * m] = acc;
        }
    }
}

/// In-place forward N-D FFT of a row-major array with extents L.
template <int N>
void fftn(std::vector<cplx>& a, const int64_t* L) {
    int64_t stride[N];
    row_major_strides<N>(L, stride);
    int64_t total = static_cast<int64_t>(a.size());
    for (int d = 0; d < N; ++d) {
        int64_t n = L[d], s = stride[d];
        if (n == 1)
            continue;
        std::vector<int> fac;
        int64_t r = n;
        for (int p : {2, 3, 5, 7})
            while (r % p == 0) {
                fac.push_back(p);
                r /= p;
            }
        std::vector<cplx> tw(n), in(n), out(n);
        for (int64_t j = 0; j < n; ++j)
            tw[j] = std::polar(1.0, -2.0 * M_PI * double(j) / double(n));
        for (int64_t blk = 0; blk < total; blk += n * s)
            for (int64_t i0 = 0; i0 < s; ++i0) {
                cplx* line = a.data() + blk + i0;
                for (int64_t i = 0; i < n; ++i)
                    in[i] = line[i * s];
                fft_rec(in.data(), 1, out.data(), n, fac.data(), tw.data(), 1);
                for (int64_t i = 0; i < n; ++i)
                    line[i * s] = out[i];
            }
    }
}

/// FFT convolution on the host: out = (w conv v)[f - 1 + j], j < len. Both real
/// inputs share one complex transform (w + i v).
template <int N, typename T>
void fft_conv(T* out, const T* w, const T* v, const int64_t* len, const int64_t* f) {
    int64_t L[N], Ls[N], wext[N];
    int64_t total = 1;
    for (int d = 0; d < N; ++d) {
        wext[d] = len[d] + f[d] - 1;
        L[d] = fft_size(wext[d]);
        total *= L[d];
    }
    row_major_strides<N>(L, Ls);
    auto offset = [&](const int64_t* ext, int64_t k, int64_t shift) {
        int64_t li = 0;
        for (int d = N - 1; d >= 0; --d) {
            li += (k % ext[d] + shift * (f[d] - 1)) * Ls[d];
            k /= ext[d];
        }
        return li;
    };
    std::vector<cplx> Z(total), P(total);
    int64_t nw = 1, nv = 1, no = 1;
    for (int d = 0; d < N; ++d) {
        nw *= wext[d];
        nv *= f[d];
        no *= len[d];
    }
    for (int64_t k = 0; k < nw; ++k)
        Z[offset(wext, k, 0)].real(static_cast<double>(w[k]));
    for (int64_t k = 0; k < nv; ++k)
        Z[offset(f, k, 0)].imag(static_cast<double>(v[k]));
    fftn<N>(Z, L);
    // W(k) = (Z(k) + conj Z(-k)) / 2, V(k) = (Z(k) - conj Z(-k)) / 2i; P = conj(W V) / total
    // (conjugated so the inverse is a forward transform).
    for (int64_t k = 0; k < total; ++k) {
        int64_t rem = k, kn = 0;
        for (int d = N - 1; d >= 0; --d) {
            int64_t c = rem % L[d];
            rem /= L[d];
            kn += ((L[d] - c) % L[d]) * Ls[d];
        }
        cplx zc = std::conj(Z[kn]);
        cplx W = 0.5 * (Z[k] + zc), V = cplx(0.0, -0.5) * (Z[k] - zc);
        P[k] = std::conj(W * V) / double(total);
    }
    fftn<N>(P, L);
    for (int64_t k = 0; k < no; ++k)
        out[k] = static_cast<T>(P[offset(len, k, 1)].real());
}
#endif

#ifdef USE_NVIDIA
// ---- cuFFT convolution (real-to-complex, on the chare's compute stream) ----

#define CT_CUFFT_CHECK(x)                                                                          \
    do {                                                                                           \
        cufftResult r_ = (x);                                                                      \
        if (r_ != CUFFT_SUCCESS)                                                                   \
            CkAbort("cuFFT error %d at %s:%d", (int)r_, __FILE__, __LINE__);                       \
    } while (0)

template <typename T>
struct CufftTypes;
template <>
struct CufftTypes<float> {
    using C = cufftComplex;
    static constexpr cufftType r2c = CUFFT_R2C, c2r = CUFFT_C2R;
    static cufftResult fwd(cufftHandle p, float* i, C* o) { return cufftExecR2C(p, i, o); }
    static cufftResult inv(cufftHandle p, C* i, float* o) { return cufftExecC2R(p, i, o); }
};
template <>
struct CufftTypes<double> {
    using C = cufftDoubleComplex;
    static constexpr cufftType r2c = CUFFT_D2Z, c2r = CUFFT_Z2D;
    static cufftResult fwd(cufftHandle p, double* i, C* o) { return cufftExecD2Z(p, i, o); }
    static cufftResult inv(cufftHandle p, C* i, double* o) { return cufftExecZ2D(p, i, o); }
};

/// a[k] *= b[k] * scale (complex).
template <int N, typename T, typename C>
void spectrum_mul(PartitionImpl<N>* partition, C* a, const C* b, int64_t n, T scale) {
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, n), KOKKOS_LAMBDA(int64_t k) {
            T re = a[k].x * b[k].x - a[k].y * b[k].y;
            T im = a[k].x * b[k].y + a[k].y * b[k].x;
            a[k].x = re * scale;
            a[k].y = im * scale;
        });
}

/// out = (w conv v)[f - 1 + j], j < len; w, v are device buffers.
template <int N, typename T>
void fft_conv_gpu(PartitionImpl<N>* partition, T* out, const T* w, const T* v,
                  const int64_t* len, const int64_t* f) {
    using CT = CufftTypes<T>;
    using C = typename CT::C;
    int64_t L[N], Ls[N], wext[N], ws[N], fs[N];
    int64_t total = 1;
    for (int d = 0; d < N; ++d) {
        wext[d] = len[d] + f[d] - 1;
        L[d] = fft_size(wext[d]);
        total *= L[d];
    }
    row_major_strides<N>(L, Ls);
    row_major_strides<N>(wext, ws);
    row_major_strides<N>(f, fs);
    int64_t ctotal = total / L[N - 1] * (L[N - 1] / 2 + 1);

    T* wp = reinterpret_cast<T*>(alloc_buffer(total * sizeof(T)));
    T* vp = reinterpret_cast<T*>(alloc_buffer(total * sizeof(T)));
    C* wf = reinterpret_cast<C*>(alloc_buffer(ctotal * sizeof(C)));
    C* vf = reinterpret_cast<C*>(alloc_buffer(ctotal * sizeof(C)));
    fill_zero<N, T>(partition, wp, total);
    fill_zero<N, T>(partition, vp, total);
    copy_box<N, T>(partition, wp, 0, Ls, w, 0, ws, wext);
    copy_box<N, T>(partition, vp, 0, Ls, v, 0, fs, f);

    int dims[N];
    for (int d = 0; d < N; ++d)
        dims[d] = static_cast<int>(L[d]);
    cufftHandle fwd, inv;
    CT_CUFFT_CHECK(cufftPlanMany(&fwd, N, dims, nullptr, 1, 0, nullptr, 1, 0, CT::r2c, 1));
    CT_CUFFT_CHECK(cufftPlanMany(&inv, N, dims, nullptr, 1, 0, nullptr, 1, 0, CT::c2r, 1));
    CT_CUFFT_CHECK(cufftSetStream(fwd, partition->compute_stream_raw));
    CT_CUFFT_CHECK(cufftSetStream(inv, partition->compute_stream_raw));
    CT_CUFFT_CHECK(CT::fwd(fwd, wp, wf));
    CT_CUFFT_CHECK(CT::fwd(fwd, vp, vf));
    spectrum_mul<N, T, C>(partition, wf, vf, ctotal, T(1) / T(total));
    CT_CUFFT_CHECK(CT::inv(inv, wf, wp));

    int64_t base = 0;
    for (int d = 0; d < N; ++d)
        base += (f[d] - 1) * Ls[d];
    int64_t os[N];
    row_major_strides<N>(len, os);
    copy_box<N, T>(partition, out, 0, os, wp, base, Ls, len);
    compute_fence(partition);
    cufftDestroy(fwd);
    cufftDestroy(inv);
    free_buffer(reinterpret_cast<char*>(wp));
    free_buffer(reinterpret_cast<char*>(vp));
    free_buffer(reinterpret_cast<char*>(wf));
    free_buffer(reinterpret_cast<char*>(vf));
}
#endif

// ---- send side ----

/// Send each result chare the part of source view i this chare holds.
template <int N, typename T>
void send_input(PartitionImpl<N>* partition, int node_id, const SigGeom<N>& s, int i) {
    const SrcView<N>& v = s.in[i];
    auto it = partition->arrays.find(v.name);
    if (it == partition->arrays.end() || it->second->local_size() == 0)
        return;
    CTArrayBase<N>* src = it->second;
    auto me = partition->nd_index();
    std::array<int, N> blo, bhi;
    for (int d = 0; d < N; ++d)
        if (!src_block_1d<N>(v, d, me[d], blo[d], bhi[d]))
            return;

    // Per axis: (result chare, lo, hi) for every result chare reading [blo, bhi).
    std::array<std::vector<std::array<int, 3>>, N> targets;
    for (int d = 0; d < N; ++d) {
        int first, last;
        chare_span<N>(s.out, d, first, last);
        for (int c = first; c <= last; ++c) {
            int olo, ohi, nlo, nhi;
            out_range_1d<N>(s.out, d, c, olo, ohi);
            need_1d<N>(s, i, d, olo, ohi, nlo, nhi);
            int lo = std::max(nlo, blo[d]), hi = std::min(nhi, bhi[d]);
            if (lo < hi)
                targets[d].push_back({c, lo, hi});
        }
        if (targets[d].empty())
            return;
    }

    int64_t tsize[N], lstride[N], sstride[N];
    for (int d = 0; d < N; ++d)
        tsize[d] = src->region.size(d);
    row_major_strides<N>(tsize, lstride);
    for (int d = 0; d < N; ++d)
        sstride[d] = (int64_t)v.vstep[d] * lstride[d];
    const T* src_ptr = static_cast<const T*>(src->device_data_ptr());

    std::array<size_t, N> t{};
    while (true) {
        std::array<int, N> ci;
        ArrayRegion<N> box;
        int64_t ext[N], pstride[N], base = 0, count = 1;
        for (int d = 0; d < N; ++d) {
            const auto& e = targets[d][t[d]];
            ci[d] = e[0];
            box.start[d] = e[1];
            box.stop[d] = e[2];
            box.step[d] = 1;
            ext[d] = e[2] - e[1];
            count *= ext[d];
            int cstart = v.dec.chare_start_global(d, me[d]);
            base += (int64_t)(v.vstart[d] + e[1] * v.vstep[d] - cstart) * lstride[d];
        }
        row_major_strides<N>(ext, pstride);
        int64_t bytes = count * (int64_t)sizeof(T);
        char* buf = alloc_buffer(bytes);
        copy_box<N, T>(partition, reinterpret_cast<T*>(buf), 0, pstride, src_ptr, base, sstride,
                       ext);
        compute_fence(partition);
        ChareIndex<N> target;
        int region_data[N * 3];
        for (int d = 0; d < N; ++d) {
            target.idx[d] = ci[d];
            region_data[d * 3 + 0] = box.start[d];
            region_data[d * 3 + 1] = box.stop[d];
            region_data[d * 3 + 2] = 1;
        }
#ifndef NDEBUG
        partition->comm_bytes_sent += bytes;
#endif
#ifdef USE_KOKKOS
        device_pack_send<N, N>(partition, partition->thisProxy, target, node_id, i, v.name,
                               region_data, bytes, buf);
#else
        proxy_at<N>(partition->thisProxy, target)
            .receive_data(node_id, i, v.name, N, region_data, bytes, buf);
        delete[] buf;
#endif

        int d = N - 1;
        for (; d >= 0; --d) {
            if (++t[d] < targets[d].size())
                break;
            t[d] = 0;
        }
        if (d < 0)
            break;
    }
}

/// Messages a result chare with range [olo, ohi) receives for source view i.
template <int N>
int expected_msgs(const SigGeom<N>& s, int i, const std::array<int, N>& olo,
                  const std::array<int, N>& ohi) {
    const SrcView<N>& v = s.in[i];
    int total = 1;
    for (int d = 0; d < N; ++d) {
        int nlo, nhi, first, last, count = 0;
        need_1d<N>(s, i, d, olo[d], ohi[d], nlo, nhi);
        chare_span<N>(v.dec, d, first, last);
        for (int c = first; c <= last; ++c) {
            int lo, hi;
            if (src_block_1d<N>(v, d, c, lo, hi) && lo < nhi && hi > nlo)
                ++count;
        }
        total *= count;
    }
    return total;
}

// ---- receive side ----

/// Dense box [lo, lo + ext) of source view i, zero outside the view.
template <int N, typename T>
T* assemble(PartitionImpl<N>* partition, PendingComm<N>& comm, int i, const int64_t* lo,
            const int64_t* ext, bool zero = true) {
    int64_t n = 1, stride[N];
    for (int d = 0; d < N; ++d)
        n *= ext[d];
    row_major_strides<N>(ext, stride);
    T* box = reinterpret_cast<T*>(alloc_buffer(n * (int64_t)sizeof(T)));
    if (zero)
        fill_zero<N, T>(partition, box, n);
    auto rb_it = comm.remote_buffers.find(i);
    if (rb_it == comm.remote_buffers.end())
        return box;
    for (auto& rb : rb_it->second) {
        int64_t bext[N], bstride[N], base = 0;
        for (int d = 0; d < N; ++d) {
            bext[d] = rb.region.stop[d] - rb.region.start[d];
            base += (rb.region.start[d] - lo[d]) * stride[d];
        }
        row_major_strides<N>(bext, bstride);
        copy_box<N, T>(partition, box, base, stride, reinterpret_cast<const T*>(rb.data), 0,
                       bstride, bext);
    }
    return box;
}

template <int N, typename T>
void finish_flip(PartitionImpl<N>* partition, const SigGeom<N>& s, PendingComm<N>& comm, T* out,
                 const std::array<int, N>& olo, const std::array<int, N>& ohi) {
    int64_t len[N], lo[N], stride[N], sstride[N], base = 0;
    for (int d = 0; d < N; ++d) {
        len[d] = ohi[d] - olo[d];
        int nlo, nhi;
        need_1d<N>(s, 0, d, olo[d], ohi[d], nlo, nhi);
        lo[d] = nlo;
    }
    T* box = assemble<N, T>(partition, comm, 0, lo, len);
    row_major_strides<N>(len, stride);
    for (int d = 0; d < N; ++d) {
        bool f = (s.flip_mask >> d) & 1;
        sstride[d] = f ? -stride[d] : stride[d];
        if (f)
            base += (len[d] - 1) * stride[d];
    }
    copy_box<N, T>(partition, out, 0, stride, box, base, sstride, len);
    compute_fence(partition);
    free_buffer(reinterpret_cast<char*>(box));
}

template <int N, typename T>
void finish_conv(PartitionImpl<N>* partition, const SigGeom<N>& s, PendingComm<N>& comm, T* out,
                 const std::array<int, N>& olo, const std::array<int, N>& ohi) {
    int64_t len[N], wlo[N], wext[N], f[N], zero[N] = {};
    for (int d = 0; d < N; ++d) {
        len[d] = ohi[d] - olo[d];
        f[d] = s.in[1].vsize[d];
        wlo[d] = olo[d] + s.shift[d] - (f[d] - 1);
        wext[d] = len[d] + f[d] - 1;
    }
    T* w = assemble<N, T>(partition, comm, 0, wlo, wext);
    T* v = assemble<N, T>(partition, comm, 1, zero, f);
    if (s.method == CONV_FFT && std::is_floating_point<T>::value) {
#ifdef USE_NVIDIA
        if constexpr (std::is_floating_point<T>::value)
            fft_conv_gpu<N, T>(partition, out, w, v, len, f);
#elif defined(USE_KOKKOS)
        // Host FFT on a staged copy.
        int64_t nw = 1, nv = 1, no = 1;
        for (int d = 0; d < N; ++d) {
            nw *= wext[d];
            nv *= f[d];
            no *= len[d];
        }
        compute_fence(partition);
        std::vector<T> hw(nw), hv(nv), ho(no);
        using U = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
        Kokkos::deep_copy(Kokkos::View<T*, HostSpace, U>(hw.data(), nw),
                          Kokkos::View<T*, DeviceSpace, U>(w, nw));
        Kokkos::deep_copy(Kokkos::View<T*, HostSpace, U>(hv.data(), nv),
                          Kokkos::View<T*, DeviceSpace, U>(v, nv));
        fft_conv<N, T>(ho.data(), hw.data(), hv.data(), len, f);
        Kokkos::deep_copy(Kokkos::View<T*, DeviceSpace, U>(out, no),
                          Kokkos::View<T*, HostSpace, U>(ho.data(), no));
#else
        fft_conv<N, T>(out, w, v, len, f);
#endif
    } else {
        direct_conv<N, T>(partition, out, w, v, len, f);
    }
    compute_fence(partition);
    free_buffer(reinterpret_cast<char*>(w));
    free_buffer(reinterpret_cast<char*>(v));
}


// ---- BATCHED_MATMUL ----

/// View-index range [lo, hi) along dim d of operand i that result range [olo, ohi) reads.
template <int N>
void bmm_need(const SigGeom<N>& s, int i, int d, const std::array<int, N>& olo,
              const std::array<int, N>& ohi, int& lo, int& hi) {
    const int* dm = s.dims[i];
    int e = d == dm[0] ? 0 : d == dm[1] ? 1 + i : -1;
    if (e < 0) {
        lo = 0;
        hi = s.in[i].vsize[d];
    } else {
        lo = olo[e];
        hi = ohi[e];
    }
}

/// True if this chare holds all of operand i its own result tile reads, with unit
/// last-dim step: the GEMM then reads it in place and nothing is sent.
template <int N>
bool bmm_local(const SigGeom<N>& s, int i, const std::array<int, N>& me,
               const std::array<int, N>& olo, const std::array<int, N>& ohi) {
    const SrcView<N>& v = s.in[i];
    if (v.vstep[N - 1] != 1)
        return false;
    for (int d = 0; d < N; ++d) {
        int lo, hi, nlo, nhi;
        if (olo[d] >= ohi[d] || !src_block_1d<N>(v, d, me[d], lo, hi))
            return false;
        bmm_need<N>(s, i, d, olo, ohi, nlo, nhi);
        if (lo > nlo || hi < nhi)
            return false;
    }
    return true;
}

/// Send each result chare the part of operand i this chare holds.
template <int N, typename T>
void send_bmm_input(PartitionImpl<N>* partition, int node_id, const SigGeom<N>& s, int i) {
    const SrcView<N>& v = s.in[i];
    auto it = partition->arrays.find(v.name);
    if (it == partition->arrays.end() || it->second->local_size() == 0)
        return;
    CTArrayBase<N>* src = it->second;
    auto me = partition->nd_index();
    std::array<int, N> blo, bhi;
    for (int d = 0; d < N; ++d)
        if (!src_block_1d<N>(v, d, me[d], blo[d], bhi[d]))
            return;
    // Per result dim e: (result chare, lo, hi) on the operand dim read along e, or the
    // whole block when no operand dim maps to e.
    const int* dm = s.dims[i];
    int src_of[3] = {dm[0], i == 0 ? dm[1] : -1, i == 0 ? -1 : dm[1]};
    std::array<std::vector<std::array<int, 3>>, N> targets;
    for (int e = 0; e < N; ++e) {
        int first, last;
        chare_span<N>(s.out, e, first, last);
        int sd = src_of[e];
        for (int c = first; c <= last; ++c) {
            int olo, ohi;
            out_range_1d<N>(s.out, e, c, olo, ohi);
            if (olo >= ohi)
                continue;
            if (sd < 0) {
                targets[e].push_back({c, 0, 0});
                continue;
            }
            int lo = std::max(olo, blo[sd]), hi = std::min(ohi, bhi[sd]);
            if (lo < hi)
                targets[e].push_back({c, lo, hi});
        }
        if (targets[e].empty())
            return;
    }

    int64_t tsize[N], lstride[N], sstride[N];
    for (int d = 0; d < N; ++d)
        tsize[d] = src->region.size(d);
    row_major_strides<N>(tsize, lstride);
    for (int d = 0; d < N; ++d)
        sstride[d] = (int64_t)v.vstep[d] * lstride[d];
    const T* src_ptr = static_cast<const T*>(src->device_data_ptr());

    std::array<size_t, N> t{};
    while (true) {
        std::array<int, N> ci, olo, ohi;
        int lo[N], hi[N];
        for (int d = 0; d < N; ++d) {
            lo[d] = blo[d];
            hi[d] = bhi[d];
        }
        for (int e = 0; e < N; ++e) {
            const auto& te = targets[e][t[e]];
            ci[e] = te[0];
            out_range_1d<N>(s.out, e, ci[e], olo[e], ohi[e]);
            if (src_of[e] >= 0) {
                lo[src_of[e]] = te[1];
                hi[src_of[e]] = te[2];
            }
        }
        if (ci != me || !bmm_local<N>(s, i, me, olo, ohi)) {
            int64_t ext[N], pstride[N], base = 0, count = 1;
            ChareIndex<N> target;
            int region_data[N * 3];
            for (int d = 0; d < N; ++d) {
                ext[d] = hi[d] - lo[d];
                count *= ext[d];
                int cstart = v.dec.chare_start_global(d, me[d]);
                base += (int64_t)(v.vstart[d] + lo[d] * v.vstep[d] - cstart) * lstride[d];
                target.idx[d] = ci[d];
                region_data[d * 3 + 0] = lo[d];
                region_data[d * 3 + 1] = hi[d];
                region_data[d * 3 + 2] = 1;
            }
            row_major_strides<N>(ext, pstride);
            int64_t bytes = count * (int64_t)sizeof(T);
            char* buf = alloc_buffer(bytes);
            copy_box<N, T>(partition, reinterpret_cast<T*>(buf), 0, pstride, src_ptr, base,
                           sstride, ext);
            compute_fence(partition);
#ifndef NDEBUG
            partition->comm_bytes_sent += bytes;
#endif
#ifdef USE_KOKKOS
            device_pack_send<N, N>(partition, partition->thisProxy, target, node_id, i, v.name,
                                   region_data, bytes, buf);
#else
            proxy_at<N>(partition->thisProxy, target)
                .receive_data(node_id, i, v.name, N, region_data, bytes, buf);
            delete[] buf;
#endif
        }
        int e = N - 1;
        for (; e >= 0; --e) {
            if (++t[e] < targets[e].size())
                break;
            t[e] = 0;
        }
        if (e < 0)
            break;
    }
}

/// Messages result chare `me` (range [olo, ohi)) receives for operand i.
template <int N>
int bmm_expected_msgs(const SigGeom<N>& s, int i, const std::array<int, N>& me,
                      const std::array<int, N>& olo, const std::array<int, N>& ohi) {
    if (bmm_local<N>(s, i, me, olo, ohi))
        return 0;
    const SrcView<N>& v = s.in[i];
    int total = 1;
    for (int d = 0; d < N; ++d) {
        int nlo, nhi, first, last, count = 0;
        bmm_need<N>(s, i, d, olo, ohi, nlo, nhi);
        chare_span<N>(v.dec, d, first, last);
        for (int c = first; c <= last; ++c) {
            int lo, hi;
            if (src_block_1d<N>(v, d, c, lo, hi) && lo < nhi && hi > nlo)
                ++count;
        }
        total *= count;
    }
    return total;
}

/// Naive batched product, one thread per result element (types cuBLAS lacks).
template <int N, typename T>
void bmm_naive(PartitionImpl<N>* partition, int64_t bs, int64_t m, int64_t n, int64_t k,
               const T* a, const int64_t* as, const T* b, const int64_t* bst, T* c) {
    int64_t a0 = as[0], a1 = as[1], a2 = as[2], b0 = bst[0], b1 = bst[1], b2 = bst[2];
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, bs * m * n), KOKKOS_LAMBDA(int64_t idx) {
            int64_t j = idx % n, r = (idx / n) % m, q = idx / (m * n);
            const T* ap = a + q * a0 + r * a1;
            const T* bp = b + q * b0 + j * b2;
            T acc = T(0);
            for (int64_t t = 0; t < k; ++t)
                acc += ap[t * a2] * bp[t * b1];
            c[idx] = acc;
        });
#else
    (void)partition;
    for (int64_t idx = 0; idx < bs * m * n; ++idx) {
        int64_t j = idx % n, r = (idx / n) % m, q = idx / (m * n);
        const T* ap = a + q * a0 + r * a1;
        const T* bp = b + q * b0 + j * b2;
        T acc = T(0);
        for (int64_t t = 0; t < k; ++t)
            acc += ap[t * a2] * bp[t * b1];
        c[idx] = acc;
    }
#endif
}

#ifdef USE_NVIDIA
template <typename T>
struct CublasBatched;
#define CT_CUBLAS_BATCHED(T, CT, X)                                                                \
    template <>                                                                                    \
    struct CublasBatched<T> {                                                                      \
        static cublasStatus_t run(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,    \
                                  int m, int n, int k, const T* a, int lda, long long sa,          \
                                  const T* b, int ldb, long long sb, T* c, int ldc, long long sc,  \
                                  int count) {                                                     \
            const CT one = CT{1}, zero = CT{0};                                                    \
            return cublas##X##gemmStridedBatched(                                                  \
                h, ta, tb, m, n, k, &one, reinterpret_cast<const CT*>(a), lda, sa,                 \
                reinterpret_cast<const CT*>(b), ldb, sb, &zero, reinterpret_cast<CT*>(c), ldc, sc, \
                count);                                                                            \
        }                                                                                          \
    };
CT_CUBLAS_BATCHED(float, float, S)
CT_CUBLAS_BATCHED(double, double, D)
CT_CUBLAS_BATCHED(cn_c64, cuComplex, C)
CT_CUBLAS_BATCHED(cn_c128, cuDoubleComplex, Z)
#undef CT_CUBLAS_BATCHED

/// cuBLAS operation and leading dim of a (rows x cols) operand with element strides
/// (rs, cs) seen column-major; false unless one stride is 1 and the other valid.
inline bool cublas_layout(int64_t rows, int64_t cols, int64_t rs, int64_t cs,
                          cublasOperation_t& op, int64_t& ld) {
    if (cs == 1 && rs >= std::max<int64_t>(1, cols)) {
        op = CUBLAS_OP_N; // row-major (rows x cols) == column-major (cols x rows)
        ld = rs;
        return true;
    }
    if (rs == 1 && cs >= std::max<int64_t>(1, rows)) {
        op = CUBLAS_OP_T;
        ld = cs;
        return true;
    }
    return false;
}
#endif

/// c[q] = a[q] * b[q] for q < bs: a[q] is m x k with element strides (as[1], as[2]),
/// b[q] k x n with (bst[1], bst[2]), batch strides as[0], bst[0]; c dense (bs, m, n).
template <int N, typename T>
void batched_gemm(PartitionImpl<N>* partition, int64_t bs, int64_t m, int64_t n, int64_t k,
                  const T* a, const int64_t* as, const T* b, const int64_t* bst, T* c) {
    if (bs <= 0 || m <= 0 || n <= 0)
        return;
#if defined(USE_NVIDIA)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, cn_c64> || std::is_same_v<T, cn_c128>) {
        // column-major view of the row-major product: C^T = B^T * A^T
        cublasOperation_t oa, ob;
        int64_t lda, ldb;
        if (k > 0 && cublas_layout(m, k, as[1], as[2], oa, lda) &&
            cublas_layout(k, n, bst[1], bst[2], ob, ldb)) {
            // inputs may come from work on the default instance's stream
            Kokkos::DefaultExecutionSpace().fence("ct_bmm inputs");
            cublasHandle_t h = ct_cublas();
            cublasSetStream(h, partition->compute_stream_raw);
            cublasStatus_t st = CublasBatched<T>::run(h, ob, oa, (int)n, (int)m, (int)k, b,
                                                      (int)ldb, bst[0], a, (int)lda, as[0], c,
                                                      (int)n, m * n, (int)bs);
            if (st != CUBLAS_STATUS_SUCCESS)
                CkAbort("cuBLAS batched gemm failed (%d)", (int)st);
            return;
        }
    }
    bmm_naive<N, T>(partition, bs, m, n, k, a, as, b, bst, c);
#elif defined(USE_KOKKOS)
    bmm_naive<N, T>(partition, bs, m, n, k, a, as, b, bst, c);
#else
    (void)partition;
    using RM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using CM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using OS = Eigen::OuterStride<>;
    using GS = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    // f(map) with the cheapest Eigen map of a (rows x cols) operand
    auto with_map = [](const T* p, int64_t rows, int64_t cols, int64_t rs, int64_t cs,
                       auto&& f) {
        if (cs == 1 && rs >= cols)
            f(Eigen::Map<const RM, 0, OS>(p, rows, cols, OS(rs)));
        else if (rs == 1 && cs >= rows)
            f(Eigen::Map<const CM, 0, OS>(p, rows, cols, OS(cs)));
        else
            f(Eigen::Map<const RM, 0, GS>(p, rows, cols, GS(rs, cs)));
    };
    for (int64_t q = 0; q < bs; ++q) {
        Eigen::Map<RM> C(c + q * m * n, m, n);
        with_map(a + q * as[0], m, k, as[1], as[2], [&](const auto& A) {
            with_map(b + q * bst[0], k, n, bst[1], bst[2],
                     [&](const auto& B) { C.noalias() = A * B; });
        });
    }
#endif
}

template <int N, typename T>
void finish_bmm(PartitionImpl<N>* partition, const SigGeom<N>& s, PendingComm<N>& comm, T* out,
                const std::array<int, N>& olo, const std::array<int, N>& ohi) {
    auto me = partition->nd_index();
    const T* ptr[2];
    int64_t str[2][3];
    T* staged[2] = {nullptr, nullptr};
    int64_t ksize = 1;
    for (int i = 0; i < 2; ++i) {
        const SrcView<N>& v = s.in[i];
        int64_t lo[N], ext[N], stride[N];
        for (int d = 0; d < N; ++d) {
            int nlo, nhi;
            bmm_need<N>(s, i, d, olo, ohi, nlo, nhi);
            lo[d] = nlo;
            ext[d] = nhi - nlo;
        }
        if (bmm_local<N>(s, i, me, olo, ohi)) {
            CTArrayBase<N>* src = partition->arrays.at(v.name);
            int64_t tsize[N], lstride[N], base = 0;
            for (int d = 0; d < N; ++d)
                tsize[d] = src->region.size(d);
            row_major_strides<N>(tsize, lstride);
            for (int d = 0; d < N; ++d) {
                int cstart = v.dec.chare_start_global(d, me[d]);
                base += (int64_t)(v.vstart[d] + lo[d] * v.vstep[d] - cstart) * lstride[d];
                stride[d] = (int64_t)v.vstep[d] * lstride[d];
            }
            ptr[i] = static_cast<const T*>(src->device_data_ptr()) + base;
        } else {
            staged[i] = assemble<N, T>(partition, comm, i, lo, ext, false);
            row_major_strides<N>(ext, stride);
            ptr[i] = staged[i];
        }
        const int* dm = s.dims[i];
        // a: (batch, m, k); b: (batch, k, n)
        str[i][0] = stride[dm[0]];
        str[i][1] = stride[i == 0 ? dm[1] : dm[2]];
        str[i][2] = stride[i == 0 ? dm[2] : dm[1]];
        ksize = ext[dm[2]];
    }
    batched_gemm<N, T>(partition, ohi[0] - olo[0], ohi[1] - olo[1], ohi[2] - olo[2], ksize,
                       ptr[0], str[0], ptr[1], str[1], out);
    if (staged[0] || staged[1]) {
        compute_fence(partition);
        for (T* p : staged)
            if (p)
                free_buffer(reinterpret_cast<char*>(p));
    }
}

} // namespace signal_detail

using namespace signal_detail;

template <int N>
void ArrayDAGExecutorND<N>::execute_signal_node(DAGNode* node) {
    auto* g = static_cast<ArrayDAGGroup*>(group);
    ASTNode* root = signal_root(node);
    SigGeom<N> s;
    if (root == nullptr || root->ndims != N || !make_geom<N>(g, root, s)) {
        node_finished(node->id);
        return;
    }
    DType dt = root->dtype;
    for (int i = 0; i < s.nin; ++i) {
        auto it = partition->arrays.find(s.in[i].name);
        if (it != partition->arrays.end() && it->second->dtype != dt)
            CkAbort("CONVOLVE/FLIP: operand dtype %d != result dtype %d",
                    (int)it->second->dtype, (int)dt);
    }
    for (int i = 0; i < s.nin; ++i)
        with_sig_type(s.op, dt, [&](auto tag) {
            using T = decltype(tag);
            if constexpr (N == 3) {
                if (s.op == Opcode::BATCHED_MATMUL) {
                    send_bmm_input<N, T>(partition, node->id, s, i);
                    return;
                }
            }
            send_input<N, T>(partition, node->id, s, i);
        });

    // Receive side: chares holding part of the result.
    auto me = partition->nd_index();
    std::array<int, N> olo, ohi, zeros{}, sizes{}, ones{};
    for (int d = 0; d < N; ++d) {
        int first, last;
        chare_span<N>(s.out, d, first, last);
        out_range_1d<N>(s.out, d, me[d], olo[d], ohi[d]);
        if (me[d] < first || me[d] > last || ohi[d] <= olo[d]) {
            node_finished(node->id);
            return;
        }
        sizes[d] = ohi[d] - olo[d];
        ones[d] = 1;
    }
    partition->ensure_array(ArrayRegion<N>(zeros, sizes, ones), s.out.global_shape,
                            root->result_name, dt, s.out);
    int expected = 0;
    for (int i = 0; i < s.nin; ++i)
        expected += s.op == Opcode::BATCHED_MATMUL ? bmm_expected_msgs<N>(s, i, me, olo, ohi)
                                                   : expected_msgs<N>(s, i, olo, ohi);

    DBG_PRINT("[PE %d] execute_signal_node<%d>: node=%d op=%d result=%d expected=%d\n",
              CkMyPe(), N, node->id, root->opcode, root->result_name, expected);

    auto pre_it = pending.find(node->id);
    int pre_arrived = 0;
    std::unordered_map<int, std::vector<RemoteBuffer<N>>> pre_buffers;
    if (pre_it != pending.end()) {
        pre_arrived = -(pre_it->second.expected_msgs);
        pre_buffers = std::move(pre_it->second.remote_buffers);
    }
    int remaining = expected - pre_arrived;
    pending[node->id] = {node, remaining, {}, std::move(pre_buffers)};
    if (remaining <= 0)
        on_comm_done(node->id);
}

template <int N>
void ArrayDAGExecutorND<N>::finish_signal_node(PendingComm<N>& comm) {
    auto* g = static_cast<ArrayDAGGroup*>(group);
    ASTNode* root = signal_root(comm.node);
    SigGeom<N> s;
    make_geom<N>(g, root, s);
    auto me = partition->nd_index();
    std::array<int, N> olo, ohi;
    for (int d = 0; d < N; ++d)
        out_range_1d<N>(s.out, d, me[d], olo[d], ohi[d]);
    CTArrayBase<N>* out = partition->arrays.at(root->result_name);
    if (s.op == Opcode::FLIP)
        with_elem_type(out->dtype, [&](auto tag) {
            using T = decltype(tag);
            finish_flip<N, T>(partition, s, comm, static_cast<T*>(out->device_data_ptr()), olo,
                              ohi);
        });
    else if (s.op == Opcode::BATCHED_MATMUL)
        with_sig_type(s.op, out->dtype, [&](auto tag) {
            using T = decltype(tag);
            if constexpr (N == 3)
                finish_bmm<N, T>(partition, s, comm, static_cast<T*>(out->device_data_ptr()),
                                 olo, ohi);
        });
    else
        with_conv_dtype(out->dtype, [&](auto tag) {
            using T = decltype(tag);
            finish_conv<N, T>(partition, s, comm, static_cast<T*>(out->device_data_ptr()), olo,
                              ohi);
        });
}

template void ArrayDAGExecutorND<1>::execute_signal_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_signal_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_signal_node(DAGNode*);
template void ArrayDAGExecutorND<1>::finish_signal_node(PendingComm<1>&);
template void ArrayDAGExecutorND<2>::finish_signal_node(PendingComm<2>&);
template void ArrayDAGExecutorND<3>::finish_signal_node(PendingComm<3>&);
