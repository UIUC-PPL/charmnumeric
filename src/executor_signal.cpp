#include "backend_internal.hpp"
#include "dispatch.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <complex>
#include <deque>
#include <map>
#include <tuple>
#include <unordered_map>
#include <type_traits>
#include <vector>

#ifdef USE_NVIDIA
#include <cufft.h>
#endif

// CONVOLVE / FLIP (opcodes 150-151), BATCHED_MATMUL (170). Each result chare
// gathers one box per source view (CONVOLVE's filter: once per PE), then computes
// locally (direct sum, overlap-save FFT, batched GEMM; cuFFT/cuBLAS on NVIDIA). The
// result has any decomposition.

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

#ifdef USE_NVIDIA
/// copy_box as one cudaMemcpy/2D/3D on the compute stream: unit last-dim strides and
/// positive pitches, at most two outer dims of extent > 1; false if it does not apply.
template <int N, typename T>
bool copy_box_memcpy(PartitionImpl<N>* partition, T* dst, const int64_t* ds, const T* src,
                     const int64_t* ss, const int64_t* ext) {
    if (ds[N - 1] != 1 || ss[N - 1] != 1)
        return false;
    int64_t e[2], dp[2], sp[2];
    int k = 0;
    for (int d = 0; d < N - 1; ++d) {
        if (ext[d] == 1)
            continue;
        if (k == 2 || ds[d] < ext[N - 1] || ss[d] < ext[N - 1])
            return false;
        e[k] = ext[d];
        dp[k] = ds[d];
        sp[k] = ss[d];
        ++k;
    }
    const size_t sz = sizeof(T), w = size_t(ext[N - 1]) * sz;
    cudaStream_t st = partition->compute_stream_raw;
    cudaError_t err;
    if (k == 0) {
        err = cudaMemcpyAsync(dst, src, w, cudaMemcpyDeviceToDevice, st);
    } else if (k == 1) {
        err = cudaMemcpy2DAsync(dst, dp[0] * sz, src, sp[0] * sz, w, e[0],
                                cudaMemcpyDeviceToDevice, st);
    } else {
        // outer stride a whole number of rows, the rows of one slice in its pitch
        if (dp[0] % dp[1] || sp[0] % sp[1] || dp[0] / dp[1] < e[1] || sp[0] / sp[1] < e[1])
            return false;
        cudaMemcpy3DParms p = {};
        p.srcPtr = make_cudaPitchedPtr(const_cast<T*>(src), sp[1] * sz, w, sp[0] / sp[1]);
        p.dstPtr = make_cudaPitchedPtr(dst, dp[1] * sz, w, dp[0] / dp[1]);
        p.extent = make_cudaExtent(w, e[1], e[0]);
        p.kind = cudaMemcpyDeviceToDevice;
        err = cudaMemcpy3DAsync(&p, st);
    }
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}
#endif

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
#ifdef USE_NVIDIA
    if (copy_box_memcpy<N, T>(partition, dst + dbase, ds, src + sbase, ss, ext))
        return;
#endif
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
    int64_t len = ext[N - 1], dl = ds[N - 1], sl = ss[N - 1];
    for (int64_t r = 0; r < total / len; ++r) {
        int64_t rem = r, di = dbase, si = sbase;
        for (int d = N - 2; d >= 0; --d) {
            int64_t c = rem % ext[d];
            rem /= ext[d];
            di += c * ds[d];
            si += c * ss[d];
        }
        T* dp = dst + di;
        const T* sp = src + si;
        if (dl == 1 && sl == 1)
            std::copy(sp, sp + len, dp);
        else
            for (int64_t j = 0; j < len; ++j)
                dp[j * dl] = sp[j * sl];
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
        comm_after_compute(partition);
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
        ct_send_data(proxy_at<N>(partition->thisProxy, target), node_id, i, v.name, N,
                     region_data, bytes, buf);
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
    free_buffer(partition, reinterpret_cast<char*>(box));
}

// ---- BATCHED_MATMUL ----
// Result chare R (range [olo, ohi): batch, m, n) computes C[t] = sum over panels P of
// A[t, m, P] * B[t, n, P]^T. R's batch range and the contraction range are cut at every
// chare boundary of either operand, the batch segments further into chunks Q so that a
// (Q, P) block stays under bmm_msg_bytes(). A block is one piece per source chare along
// its rows dim: sent to R, or read in place when R holds it. R runs the GEMMs of a chunk
// in panel order as soon as their blocks are complete, while the rest still arrives.

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

/// Message and block size bound: 64 MB (an IPC slab block, a pooled buffer), or
/// CNUM_BMM_MSG_KB (testing).
inline int64_t bmm_msg_bytes() {
    static const int64_t b = [] {
        const char* e = getenv("CNUM_BMM_MSG_KB");
        int64_t kb = e ? atoll(e) : 0;
        return kb > 0 ? kb << 10 : int64_t(64) << 20;
    }();
    return b;
}

/// Packed outgoing pieces are sent once they reach this many bytes.
constexpr int64_t kBmmFlushBytes = int64_t(256) << 20;

/// Source chare of view v that holds view index x along dim d, or -1.
template <int N>
int src_chare_at(const SrcView<N>& v, int d, int x) {
    int first, last;
    chare_span<N>(v.dec, d, first, last);
    for (int c = first; c <= last; ++c) {
        int lo, hi;
        if (src_block_1d<N>(v, d, c, lo, hi) && lo <= x && x < hi)
            return c;
    }
    return -1;
}

/// Batch chunk bounds (qcut, result coords) and contraction panel bounds (pcut) of a
/// result range; rows[i]: the view range of operand i's rows dim.
struct BmmPlan {
    std::vector<int> qcut, pcut;
    int rows[2][2];
    int nq() const { return (int)qcut.size() - 1; }
    int np() const { return (int)pcut.size() - 1; }
};

template <int N>
BmmPlan bmm_plan(const SigGeom<N>& s, const std::array<int, N>& olo,
                 const std::array<int, N>& ohi, int esize) {
    BmmPlan pl;
    const int K = s.in[0].vsize[s.dims[0][2]];
    std::vector<int> bc = {olo[0], ohi[0]};
    pl.pcut = {0, K};
    for (int i = 0; i < 2; ++i) {
        const SrcView<N>& v = s.in[i];
        auto cuts = [&](int d, int lo, int hi, std::vector<int>& out) {
            int first, last;
            chare_span<N>(v.dec, d, first, last);
            for (int c = first; c <= last; ++c) {
                int a, b;
                if (!src_block_1d<N>(v, d, c, a, b))
                    continue;
                for (int x : {a, b})
                    if (x > lo && x < hi)
                        out.push_back(x);
            }
        };
        cuts(s.dims[i][0], olo[0], ohi[0], bc);
        cuts(s.dims[i][2], 0, K, pl.pcut);
        pl.rows[i][0] = olo[1 + i];
        pl.rows[i][1] = ohi[1 + i];
    }
    for (auto* c : {&bc, &pl.pcut}) {
        std::sort(c->begin(), c->end());
        c->erase(std::unique(c->begin(), c->end()), c->end());
    }
    const int64_t cap = bmm_msg_bytes(), rows = std::max(ohi[1] - olo[1], ohi[2] - olo[2]);
    // panels longer than one batch slice of `cap` allows are cut further
    const int64_t kmax = std::max<int64_t>(1, cap / std::max<int64_t>(1, rows * esize));
    std::vector<int> pc = {pl.pcut[0]};
    int64_t maxp = 1;
    for (size_t j = 0; j + 1 < pl.pcut.size(); ++j) {
        int64_t len = pl.pcut[j + 1] - pl.pcut[j], np = (len + kmax - 1) / kmax;
        for (int64_t t = 1; t <= np; ++t)
            pc.push_back(pl.pcut[j] + (int)(len * t / np));
        maxp = std::max(maxp, (len + np - 1) / np);
    }
    pl.pcut = pc;
    const int64_t per = std::max<int64_t>(1, cap / (rows * maxp * esize)); // batches per chunk
    pl.qcut = {bc[0]};
    for (size_t j = 0; j + 1 < bc.size(); ++j) {
        int64_t len = bc[j + 1] - bc[j], nq = (len + per - 1) / per;
        for (int64_t q = 1; q <= nq; ++q)
            pl.qcut.push_back(bc[j] + (int)(len * q / nq));
    }
    return pl;
}

/// One source chare's part of a block: chare index and view-index box.
template <int N>
struct BmmPiece {
    std::array<int, N> src;
    std::array<int, N> lo, hi;
};

/// Pieces of block (q, p) of operand i, one per source chare along its rows dim.
template <int N>
void bmm_pieces(const SigGeom<N>& s, const BmmPlan& pl, int i, int q, int p,
                std::vector<BmmPiece<N>>& out) {
    out.clear();
    const SrcView<N>& v = s.in[i];
    const int db = s.dims[i][0], dr = s.dims[i][1], dk = s.dims[i][2];
    BmmPiece<N> pc;
    pc.lo[db] = pl.qcut[q];
    pc.hi[db] = pl.qcut[q + 1];
    pc.lo[dk] = pl.pcut[p];
    pc.hi[dk] = pl.pcut[p + 1];
    pc.src[db] = src_chare_at<N>(v, db, pc.lo[db]);
    pc.src[dk] = src_chare_at<N>(v, dk, pc.lo[dk]);
    const int rlo = pl.rows[i][0], rhi = pl.rows[i][1];
    int first, last;
    chare_span<N>(v.dec, dr, first, last);
    for (int c = first; c <= last; ++c) {
        int a, b;
        if (!src_block_1d<N>(v, dr, c, a, b) || b <= rlo || a >= rhi)
            continue;
        pc.src[dr] = c;
        pc.lo[dr] = std::max(a, rlo);
        pc.hi[dr] = std::min(b, rhi);
        out.push_back(pc);
    }
}

/// Element offset and strides (per view dim) of view index lo in this chare's tile of v.
template <int N>
int64_t bmm_local_offset(PartitionImpl<N>* partition, const SrcView<N>& v,
                         const std::array<int, N>& lo, int64_t* stride) {
    CTArrayBase<N>* src = partition->arrays.at(v.name);
    auto me = partition->nd_index();
    int64_t tsize[N], lstride[N], base = 0;
    for (int d = 0; d < N; ++d)
        tsize[d] = src->region.size(d);
    row_major_strides<N>(tsize, lstride);
    for (int d = 0; d < N; ++d) {
        int cstart = v.dec.chare_start_global(d, me[d]);
        base += (int64_t)(v.vstart[d] + lo[d] * v.vstep[d] - cstart) * lstride[d];
        stride[d] = (int64_t)v.vstep[d] * lstride[d];
    }
    return base;
}

/// Send the pieces of operand i this chare holds to the other result chares reading them.
template <int N, typename T>
void send_bmm_input(PartitionImpl<N>* partition, int node_id, const SigGeom<N>& s, int i) {
    const SrcView<N>& v = s.in[i];
    auto it = partition->arrays.find(v.name);
    if (it == partition->arrays.end() || it->second->local_size() == 0)
        return;
    auto me = partition->nd_index();
    std::array<int, N> blo, bhi;
    for (int d = 0; d < N; ++d)
        if (!src_block_1d<N>(v, d, me[d], blo[d], bhi[d]))
            return;
    const int db = s.dims[i][0], dr = s.dims[i][1], dk = s.dims[i][2];
    // result chares whose batch and (operand i) rows ranges meet this block
    std::array<std::vector<int>, N> rc;
    for (int e = 0; e < N; ++e) {
        int first, last;
        chare_span<N>(s.out, e, first, last);
        for (int c = first; c <= last; ++c) {
            int olo, ohi;
            out_range_1d<N>(s.out, e, c, olo, ohi);
            int d = e == 0 ? db : e == 1 + i ? dr : -1;
            if (olo < ohi && (d < 0 || (ohi > blo[d] && olo < bhi[d])))
                rc[e].push_back(c);
        }
        if (rc[e].empty())
            return;
    }
    int64_t sstride[N];
    const T* src_ptr = static_cast<const T*>(it->second->device_data_ptr());

    struct Msg {
        ChareIndex<N> target;
        int region_data[N * 3];
        int64_t bytes;
        char* buf;
    };
    std::vector<Msg> msgs;
    int64_t packed = 0;
    // send what is packed so far (bounds the packed but unsent bytes)
    auto flush = [&] {
        if (msgs.empty())
            return;
        compute_fence(partition);
        for (Msg& m : msgs) {
#ifndef NDEBUG
            partition->comm_bytes_sent += m.bytes;
#endif
#ifdef USE_KOKKOS
            device_pack_send<N, N>(partition, partition->thisProxy, m.target, node_id, i, v.name,
                                   m.region_data, m.bytes, m.buf);
#else
            ct_send_data(proxy_at<N>(partition->thisProxy, m.target), node_id, i, v.name, N,
                         m.region_data, m.bytes, m.buf);
            free_buffer(partition, m.buf);
#endif
        }
        msgs.clear();
        packed = 0;
    };
    std::vector<BmmPiece<N>> pcs;
    std::array<size_t, N> t{};
    while (true) {
        std::array<int, N> ci, olo, ohi;
        for (int e = 0; e < N; ++e) {
            ci[e] = rc[e][t[e]];
            out_range_1d<N>(s.out, e, ci[e], olo[e], ohi[e]);
        }
        if (ci != me) {
            BmmPlan pl = bmm_plan<N>(s, olo, ohi, (int)sizeof(T));
            for (int q = 0; q < pl.nq(); ++q) {
                if (pl.qcut[q] < blo[db] || pl.qcut[q] >= bhi[db])
                    continue;
                for (int p = 0; p < pl.np(); ++p) {
                    if (pl.pcut[p] < blo[dk] || pl.pcut[p] >= bhi[dk])
                        continue;
                    bmm_pieces<N>(s, pl, i, q, p, pcs);
                    for (const BmmPiece<N>& pc : pcs) {
                        if (pc.src != me)
                            continue;
                        Msg m;
                        int64_t ext[N], pstride[N], n = 1;
                        for (int d = 0; d < N; ++d) {
                            ext[d] = pc.hi[d] - pc.lo[d];
                            n *= ext[d];
                            m.target.idx[d] = ci[d];
                            m.region_data[d * 3 + 0] = pc.lo[d];
                            m.region_data[d * 3 + 1] = pc.hi[d];
                            m.region_data[d * 3 + 2] = 1;
                        }
                        row_major_strides<N>(ext, pstride);
                        int64_t base = bmm_local_offset<N>(partition, v, pc.lo, sstride);
                        m.bytes = n * (int64_t)sizeof(T);
#ifdef USE_KOKKOS
                        m.buf = static_cast<char*>(comm_alloc(m.bytes));
#else
                        m.buf = alloc_buffer(m.bytes);
#endif
                        copy_box<N, T>(partition, reinterpret_cast<T*>(m.buf), 0, pstride,
                                       src_ptr, base, sstride, ext);
                        msgs.push_back(m);
                        packed += m.bytes;
                        if (packed >= kBmmFlushBytes)
                            flush();
                    }
                }
            }
        }
        int e = N - 1;
        for (; e >= 0; --e) {
            if (++t[e] < rc[e].size())
                break;
            t[e] = 0;
        }
        if (e < 0)
            break;
    }
    flush();
}

/// Accumulator of the naive product: unsigned for ints, so overflow wraps (as in NumPy).
template <typename T, bool = std::is_integral_v<T>>
struct BmmAcc {
    using type = T;
};
template <typename T>
struct BmmAcc<T, true> {
    using type = std::make_unsigned_t<T>;
};

/// Naive batched product, one thread per result element (types cuBLAS lacks); `acc`
/// adds to c.
template <int N, typename T>
void bmm_naive(PartitionImpl<N>* partition, int64_t bs, int64_t m, int64_t n, int64_t k,
               const T* a, const int64_t* as, const T* b, const int64_t* bst, T* c, bool acc) {
    using U = typename BmmAcc<T>::type;
    int64_t a0 = as[0], a1 = as[1], a2 = as[2], b0 = bst[0], b1 = bst[1], b2 = bst[2];
#ifdef USE_KOKKOS
    Kokkos::parallel_for(
        CT_COMPUTE_POLICY(partition, bs * m * n), KOKKOS_LAMBDA(int64_t idx) {
            int64_t j = idx % n, r = (idx / n) % m, q = idx / (m * n);
            const T* ap = a + q * a0 + r * a1;
            const T* bp = b + q * b0 + j * b2;
            U sum = acc ? U(c[idx]) : U(0);
            for (int64_t t = 0; t < k; ++t)
                sum += U(ap[t * a2]) * U(bp[t * b1]);
            c[idx] = T(sum);
        });
#else
    (void)partition;
    for (int64_t idx = 0; idx < bs * m * n; ++idx) {
        int64_t j = idx % n, r = (idx / n) % m, q = idx / (m * n);
        const T* ap = a + q * a0 + r * a1;
        const T* bp = b + q * b0 + j * b2;
        U sum = acc ? U(c[idx]) : U(0);
        for (int64_t t = 0; t < k; ++t)
            sum += U(ap[t * a2]) * U(bp[t * b1]);
        c[idx] = T(sum);
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
                                  int count, bool acc) {                                           \
            const CT one = CT{1}, beta = acc ? CT{1} : CT{0};                                      \
            return cublas##X##gemmStridedBatched(                                                  \
                h, ta, tb, m, n, k, &one, reinterpret_cast<const CT*>(a), lda, sa,                 \
                reinterpret_cast<const CT*>(b), ldb, sb, &beta, reinterpret_cast<CT*>(c), ldc, sc, \
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
    if (std::max(rs, cs) > INT_MAX)
        return false;
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

#if !defined(USE_KOKKOS)
template <typename T>
void bmm_eigen(int64_t bs, int64_t m, int64_t n, int64_t k, const T* a, const int64_t* as,
               const T* b, const int64_t* bst, T* c, bool acc) {
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
            with_map(b + q * bst[0], k, n, bst[1], bst[2], [&](const auto& B) {
                if (acc)
                    C.noalias() += A * B;
                else
                    C.noalias() = A * B;
            });
        });
    }
}
#endif

/// c[q] (+)= a[q] * b[q] for q < bs: a[q] is m x k with element strides (as[1], as[2]),
/// b[q] k x n with (bst[1], bst[2]), batch strides as[0], bst[0]; c dense (bs, m, n); `acc`
/// adds to c.
template <int N, typename T>
void batched_gemm(PartitionImpl<N>* partition, int64_t bs, int64_t m, int64_t n, int64_t k,
                  const T* a, const int64_t* as, const T* b, const int64_t* bst, T* c,
                  bool acc) {
    if (bs <= 0 || m <= 0 || n <= 0)
        return;
#if defined(USE_NVIDIA)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, cn_c64> || std::is_same_v<T, cn_c128>) {
        // column-major view of the row-major product: C^T = B^T * A^T
        cublasOperation_t oa, ob;
        int64_t lda, ldb;
        bool fits = std::max({bs, m, n, k}) <= INT_MAX;
        if (fits && k > 0 && cublas_layout(m, k, as[1], as[2], oa, lda) &&
            cublas_layout(k, n, bst[1], bst[2], ob, ldb)) {
            cublasHandle_t h = ct_cublas();
            cublasSetStream(h, partition->compute_stream_raw);
            cublasStatus_t st = CublasBatched<T>::run(h, ob, oa, (int)n, (int)m, (int)k, b,
                                                      (int)ldb, bst[0], a, (int)lda, as[0], c,
                                                      (int)n, m * n, (int)bs, acc);
            if (st != CUBLAS_STATUS_SUCCESS)
                CkAbort("cuBLAS batched gemm failed (%d)", (int)st);
            return;
        }
        static std::atomic<bool> warned{false};
        if (k > 0 && !warned.exchange(true))
            CkPrintf("[PE %d] batched matmul: layout or size not supported by cuBLAS, using the "
                     "naive kernel\n", CkMyPe());
    }
    bmm_naive<N, T>(partition, bs, m, n, k, a, as, b, bst, c, acc);
#elif defined(USE_KOKKOS)
    bmm_naive<N, T>(partition, bs, m, n, k, a, as, b, bst, c, acc);
#else
    (void)partition;
    // signed integers wrap: multiply as unsigned (signed overflow is undefined behaviour)
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        using U = std::make_unsigned_t<T>;
        bmm_eigen<U>(bs, m, n, k, reinterpret_cast<const U*>(a), as,
                     reinterpret_cast<const U*>(b), bst, reinterpret_cast<U*>(c), acc);
    } else {
        bmm_eigen<T>(bs, m, n, k, a, as, b, bst, c, acc);
    }
#endif
}

/// Receive-side state of a BATCHED_MATMUL node on a result chare.
struct BmmState {
    SigGeom<3> geom;
    BmmPlan plan;
    std::array<int, 3> olo, ohi;
    std::vector<int> need[2], got[2]; // per block q * np + p: pieces still to arrive / arrived
    std::vector<int> next;            // per chunk: next panel to multiply
    bool eager = true;                // multiply as blocks complete (false: output aliases an input)
};

inline std::map<std::pair<const void*, int>, BmmState>& bmm_states() {
    thread_local std::map<std::pair<const void*, int>, BmmState> states;
    return states;
}

constexpr int kBmmScratch = 2; // remote_buffers index of scratch blocks (freed with the node)

/// Pointer and (batch, rows, k) strides of block (q, p) of operand i: a received piece or the
/// local tile in place when one piece covers it, else the pieces copied into a scratch box.
template <int N, typename T>
const T* bmm_block(PartitionImpl<N>* partition, const SigGeom<N>& s, const BmmState& st,
                   PendingComm<N>& comm, int i, int q, int p, int64_t* str, bool copy_local) {
    std::vector<BmmPiece<N>> pcs;
    bmm_pieces<N>(s, st.plan, i, q, p, pcs);
    auto me = partition->nd_index();
    const SrcView<N>& v = s.in[i];
    const int* dm = s.dims[i];
    auto received = [&](const BmmPiece<N>& pc) -> const T* {
        for (auto& rb : comm.remote_buffers[i])
            if (rb.region.start == pc.lo && rb.region.stop == pc.hi)
                return reinterpret_cast<const T*>(rb.data);
        CkAbort("batched matmul: missing operand piece");
        return nullptr;
    };
    int64_t stride[N];
    const T* ptr;
    // in place only with a unit last-dim step (a GEMM operand layout)
    if (pcs.size() == 1 && !((copy_local || v.vstep[N - 1] != 1) && pcs[0].src == me)) {
        const BmmPiece<N>& pc = pcs[0];
        if (pc.src == me) {
            int64_t base = bmm_local_offset<N>(partition, v, pc.lo, stride);
            ptr = static_cast<const T*>(partition->arrays.at(v.name)->device_data_ptr()) + base;
        } else {
            int64_t ext[N];
            for (int d = 0; d < N; ++d)
                ext[d] = pc.hi[d] - pc.lo[d];
            row_major_strides<N>(ext, stride);
            ptr = received(pc);
        }
    } else {
        std::array<int, N> lo = pcs[0].lo, hi = pcs[0].hi;
        lo[dm[1]] = st.plan.rows[i][0];
        hi[dm[1]] = st.plan.rows[i][1];
        int64_t ext[N], n = 1;
        for (int d = 0; d < N; ++d) {
            ext[d] = hi[d] - lo[d];
            n *= ext[d];
        }
        row_major_strides<N>(ext, stride);
#ifdef USE_KOKKOS
        T* box = static_cast<T*>(comm_alloc(n * (int64_t)sizeof(T)));
#else
        T* box = reinterpret_cast<T*>(alloc_buffer(n * (int64_t)sizeof(T)));
#endif
        comm.remote_buffers[kBmmScratch].push_back(
            {reinterpret_cast<char*>(box), ArrayRegion<N>(), n * (int64_t)sizeof(T)});
        for (const BmmPiece<N>& pc : pcs) {
            int64_t pext[N], pstr[N], sstr[N], dbase = 0, sbase;
            for (int d = 0; d < N; ++d) {
                pext[d] = pc.hi[d] - pc.lo[d];
                dbase += (pc.lo[d] - lo[d]) * stride[d];
            }
            const T* src;
            if (pc.src == me) {
                sbase = bmm_local_offset<N>(partition, v, pc.lo, sstr);
                src = static_cast<const T*>(partition->arrays.at(v.name)->device_data_ptr());
            } else {
                row_major_strides<N>(pext, pstr);
                std::copy(pstr, pstr + N, sstr);
                sbase = 0;
                src = received(pc);
            }
            copy_box<N, T>(partition, box, dbase, stride, src, sbase, sstr, pext);
        }
        ptr = box;
    }
    // a: (batch, m, k); b: (batch, k, n)
    str[0] = stride[dm[0]];
    str[1] = stride[i == 0 ? dm[1] : dm[2]];
    str[2] = stride[i == 0 ? dm[2] : dm[1]];
    return ptr;
}

/// Move the received pieces of block (q, p) and the scratch boxes, all read by GEMMs already
/// queued, to `done`.
template <int N>
void bmm_release(PartitionImpl<N>* partition, const SigGeom<N>& s, const BmmState& st,
                 PendingComm<N>& comm, int q, int p,
                 std::unordered_map<int, std::vector<RemoteBuffer<N>>>& done) {
    auto me = partition->nd_index();
    std::vector<BmmPiece<N>> pcs;
    for (int i = 0; i < 2; ++i) {
        bmm_pieces<N>(s, st.plan, i, q, p, pcs);
        auto& list = comm.remote_buffers[i];
        for (const BmmPiece<N>& pc : pcs) {
            if (pc.src == me)
                continue;
            for (size_t j = 0; j < list.size(); ++j)
                if (list[j].region.start == pc.lo && list[j].region.stop == pc.hi) {
                    done[i].push_back(list[j]);
                    list.erase(list.begin() + j);
                    break;
                }
        }
    }
    auto sc = comm.remote_buffers.find(kBmmScratch);
    if (sc != comm.remote_buffers.end()) {
        for (auto& rb : sc->second)
            done[kBmmScratch].push_back(rb);
        comm.remote_buffers.erase(sc);
    }
}

/// Run the GEMMs whose blocks are complete, in panel order per chunk; all of them if `all`.
template <int N, typename T>
void bmm_progress(PartitionImpl<N>* partition, const SigGeom<N>& s, BmmState& st,
                  PendingComm<N>& comm, T* out, bool all) {
    const int np = st.plan.np();
    const int64_t m = st.ohi[1] - st.olo[1], n = st.ohi[2] - st.olo[2];
    std::vector<std::array<const T*, 2>> ptrs;
    std::vector<std::array<int64_t, 6>> strs;
    std::unordered_map<int, std::vector<RemoteBuffer<N>>> done; // freed once the GEMMs ran
    if (all && !st.eager) {
        // the output aliases an input: stage every block before the first GEMM writes
        for (int q = 0; q < st.plan.nq(); ++q)
            for (int p = 0; p < np; ++p) {
                std::array<int64_t, 6> sv;
                ptrs.push_back({bmm_block<N, T>(partition, s, st, comm, 0, q, p, &sv[0], true),
                                bmm_block<N, T>(partition, s, st, comm, 1, q, p, &sv[3], true)});
                strs.push_back(sv);
            }
    }
    if (!all && !st.eager)
        return;
    for (int q = 0; q < st.plan.nq(); ++q) {
        const int64_t bs = st.plan.qcut[q + 1] - st.plan.qcut[q];
        T* c = out + (int64_t)(st.plan.qcut[q] - st.olo[0]) * m * n;
        for (int& p = st.next[q]; p < np; ++p) {
            int b = q * np + p;
            if (st.got[0][b] < st.need[0][b] || st.got[1][b] < st.need[1][b])
                break;
            int64_t as[3], bst[3];
            const T *a, *bp;
            if (!ptrs.empty()) {
                a = ptrs[b][0];
                bp = ptrs[b][1];
                std::copy(&strs[b][0], &strs[b][3], as);
                std::copy(&strs[b][3], &strs[b][6], bst);
            } else {
                a = bmm_block<N, T>(partition, s, st, comm, 0, q, p, as, false);
                bp = bmm_block<N, T>(partition, s, st, comm, 1, q, p, bst, false);
            }
            batched_gemm<N, T>(partition, bs, m, n, st.plan.pcut[p + 1] - st.plan.pcut[p], a,
                               as, bp, bst, c, p > 0);
            if (st.eager)
                bmm_release<N>(partition, s, st, comm, q, p, done);
        }
    }
    if (done.empty())
        return;
#ifdef USE_NVIDIA
    auto* bufs = new RemoteBuffers<N>(std::move(done));
    CkCallback cb(free_remote_buffers_cb<N>, bufs);
    hapiAddCallback(partition->compute_stream_raw, &cb);
#elif defined(USE_KOKKOS)
    for (auto& [i, list] : done)
        for (auto& rb : list)
            comm.remote_buffers[kBmmScratch].push_back(rb);
#else
    for (auto& [i, list] : done)
        for (auto& rb : list)
            delete[] rb.data;
#endif
}

/// Count a received piece of operand i against its block.
inline void bmm_count(BmmState& st, const SigGeom<3>& s, int i, const ArrayRegion<3>& r) {
    const auto& pl = st.plan;
    int q = int(std::upper_bound(pl.qcut.begin(), pl.qcut.end(), r.start[s.dims[i][0]]) -
                pl.qcut.begin()) - 1;
    int p = int(std::upper_bound(pl.pcut.begin(), pl.pcut.end(), r.start[s.dims[i][2]]) -
                pl.pcut.begin()) - 1;
    ++st.got[i][q * pl.np() + p];
}

/// Set up the receive state of result chare `me` (range [olo, ohi)); returns the number of
/// messages it receives.
template <int N>
int bmm_start(PartitionImpl<N>* partition, int node_id, const SigGeom<N>& s, int result_name,
              const std::array<int, N>& olo, const std::array<int, N>& ohi, int esize) {
    BmmState& st = bmm_states()[{partition, node_id}];
    st = BmmState{};
    st.geom = s;
    st.plan = bmm_plan<N>(s, olo, ohi, esize);
    for (int d = 0; d < 3; ++d) {
        st.olo[d] = olo[d];
        st.ohi[d] = ohi[d];
    }
    st.eager = result_name != s.in[0].name && result_name != s.in[1].name;
    const int nb = st.plan.nq() * st.plan.np();
    auto me = partition->nd_index();
    int total = 0;
    std::vector<BmmPiece<N>> pcs;
    for (int i = 0; i < 2; ++i) {
        st.need[i].assign(nb, 0);
        st.got[i].assign(nb, 0);
        for (int q = 0; q < st.plan.nq(); ++q)
            for (int p = 0; p < st.plan.np(); ++p) {
                bmm_pieces<N>(s, st.plan, i, q, p, pcs);
                for (const auto& pc : pcs)
                    st.need[i][q * st.plan.np() + p] += pc.src != me;
            }
        for (int x : st.need[i])
            total += x;
    }
    st.next.assign(st.plan.nq(), 0);
    return total;
}

// ---- CONVOLVE: one filter copy per PE, overlap-save FFT blocks ----
//
// The filter goes only to the first result chare of each PE (row-major order). That chare
// prepares what every result chare of its PE reads: the dense filter (direct method), or
// its spectrum at the node's FFT block size, as soon as the filter is in, and then tells
// the others.
// FFTs run over blocks of at most kConvBlockMax[N] per dim (overlap-save), batched, with
// cuFFT plans cached per shape.

/// Result chares of a CONVOLVE node by PE: the leader (first) of each and this PE's count.
template <int N>
struct ConvPeers {
    std::vector<std::array<int, N>> leader; // by PE; leader[pe][0] < 0: none
    int local = 0;                          // result chares on this PE
};

/// Depends only on the result decomposition and the PE count, which make the key: node ids
/// restart when the frontend resets.
template <int N>
const ConvPeers<N>& conv_peers(const SigGeom<N>& s) {
    using Key = std::tuple<std::array<int, N>, std::array<int, N>, int, int>;
    thread_local std::deque<std::pair<Key, ConvPeers<N>>> cache;
    const Key key{s.out.offset, s.out.global_shape, s.out.tile, CkNumPes()};
    for (auto& [k, p] : cache)
        if (k == key)
            return p;
    if (cache.size() >= 8)
        cache.pop_front();
    cache.emplace_back(key, ConvPeers<N>{});
    ConvPeers<N>& p = cache.back().second;
    std::array<int, N> none;
    none.fill(-1);
    p.leader.assign(CkNumPes(), none);
    std::array<int, N> first, last, ci;
    for (int d = 0; d < N; ++d)
        chare_span<N>(s.out, d, first[d], last[d]);
    ci = first;
    while (true) {
        int pe = chare_home_pe(N, ci.data(), CkNumPes());
        if (p.leader[pe][0] < 0)
            p.leader[pe] = ci;
        p.local += pe == CkMyPe();
        int d = N - 1;
        for (; d >= 0; --d) {
            if (++ci[d] <= last[d])
                break;
            ci[d] = first[d];
        }
        if (d < 0)
            break;
    }
    return p;
}

template <int N>
bool conv_is_leader(const ConvPeers<N>& p, const std::array<int, N>& me) {
    return p.leader[CkMyPe()] == me;
}

/// FFT blocks of a CONVOLVE node, the same on every chare: L (FFT extents) and B (outputs
/// per block, L >= B + f - 1).
template <int N>
struct ConvBlocks {
    int64_t f[N], L[N], B[N];
    int64_t lvol = 1, lc = 1; // real and half-spectrum (R2C) volumes of a block
};

constexpr int64_t kConvBlockMax[3] = {int64_t(1) << 22, 2048, 256};
constexpr int64_t kConvChunkBytes = int64_t(256) << 20;

/// Largest FFT block extent per dim: kConvBlockMax, or CNUM_CONV_BLOCK (testing).
inline int64_t conv_block_max(int ndims) {
    static const int64_t over = [] {
        const char* e = getenv("CNUM_CONV_BLOCK");
        return e ? atoll(e) : 0;
    }();
    return over > 0 ? over : kConvBlockMax[ndims - 1];
}

/// The node's FFT blocks, the same on every chare. Blocks hold outputs of at least f per dim
/// (or about f / 4 if a block would pass kConvChunkBytes); false if even those do, and the
/// node then runs the direct method.
template <int N>
bool conv_blocks(const SigGeom<N>& s, int64_t esz, ConvBlocks<N>& cb) {
    for (int tier = 0; tier < 2; ++tier) {
        cb = ConvBlocks<N>{};
        for (int d = 0; d < N; ++d) {
            int64_t f = s.in[1].vsize[d];
            int64_t t = std::min<int64_t>(s.out.tile, s.out.global_shape[d]); // longest tile
            int64_t lmax = std::max(conv_block_max(N), tier == 0 ? 2 * f - 1 : f + (f + 3) / 4);
            if (t + f - 1 <= lmax) {
                cb.B[d] = t;
            } else {
                int64_t nb = (t + (lmax - f + 1) - 1) / (lmax - f + 1);
                cb.B[d] = (t + nb - 1) / nb;
            }
            cb.f[d] = f;
            cb.L[d] = fft_size(cb.B[d] + f - 1);
            cb.lvol *= cb.L[d];
            cb.lc *= d == N - 1 ? cb.L[d] / 2 + 1 : cb.L[d];
        }
        if (cb.lvol * esz <= kConvChunkBytes)
            return true;
    }
    return false;
}

/// FFT method for this node (else direct), and its blocks.
template <int N, typename T>
bool conv_fft(const SigGeom<N>& s, ConvBlocks<N>& cb) {
    return s.method == CONV_FFT && std::is_floating_point<T>::value &&
           conv_blocks<N>(s, sizeof(T), cb);
}

/// N-D extents padded to 3-D (leading 1s).
template <int N>
std::array<int64_t, 3> pad3(const int64_t* a) {
    std::array<int64_t, 3> r = {1, 1, 1};
    for (int d = 0; d < N; ++d)
        r[3 - N + d] = a[d];
    return r;
}

/// Block g of the grid nb (row-major), its block coordinates.
CT_SIG_FN void block_coords(int64_t g, const int64_t* nb, int64_t* b) {
    b[2] = g % nb[2];
    b[1] = (g / nb[2]) % nb[1];
    b[0] = g / (nb[2] * nb[1]);
}

/// xb[q] = the window w (extents ext) at block first + q, zero past its end.
template <typename T>
struct ConvGather {
    T* xb;
    const T* w;
    int64_t L[3], B[3], ext[3], nb[3], first;

    CT_SIG_FN void operator()(int64_t idx) const {
        const int64_t lvol = L[0] * L[1] * L[2], r = idx % lvol;
        int64_t b[3];
        block_coords(first + idx / lvol, nb, b);
        const int64_t t0 = b[0] * B[0] + r / (L[1] * L[2]), t1 = b[1] * B[1] + (r / L[2]) % L[1],
                      t2 = b[2] * B[2] + r % L[2];
        xb[idx] = (t0 < ext[0] && t1 < ext[1] && t2 < ext[2])
                      ? w[(t0 * ext[1] + t1) * ext[2] + t2]
                      : T(0);
    }
};

/// out at block first + q = yb[q] from offset f - 1, clipped to len.
template <typename T>
struct ConvScatter {
    T* out;
    const T* yb;
    int64_t L[3], B[3], f[3], len[3], nb[3], first;

    CT_SIG_FN void operator()(int64_t idx) const {
        const int64_t bvol = B[0] * B[1] * B[2], r = idx % bvol, q = idx / bvol;
        int64_t b[3];
        block_coords(first + q, nb, b);
        const int64_t j0 = r / (B[1] * B[2]), j1 = (r / B[2]) % B[1], j2 = r % B[2];
        const int64_t o0 = b[0] * B[0] + j0, o1 = b[1] * B[1] + j1, o2 = b[2] * B[2] + j2;
        if (o0 >= len[0] || o1 >= len[1] || o2 >= len[2])
            return;
        out[(o0 * len[1] + o1) * len[2] + o2] =
            yb[q * L[0] * L[1] * L[2] + ((j0 + f[0] - 1) * L[1] + j1 + f[1] - 1) * L[2] + j2 +
               f[2] - 1];
    }
};

#ifdef USE_NVIDIA
/// x[k] *= v[k % lc] * scale (complex), for a batch of half spectra.
template <typename T, typename C>
struct ConvSpecMul {
    C* x;
    const C* v;
    int64_t lc;
    T scale;

    CT_SIG_FN void operator()(int64_t k) const {
        const C b = v[k % lc];
        const C a = x[k];
        x[k].x = (a.x * b.x - a.y * b.y) * scale;
        x[k].y = (a.x * b.y + a.y * b.x) * scale;
    }
};

struct FftPlan {
    cufftHandle h;
    size_t work;
    uint64_t used;
};

/// A cuFFT plan for `batch` dense row-major transforms of extents n, made once per stream
/// and shape (at most kMaxPlans per PE): cuFFT is thread-safe only across plans, so chares
/// never share one between streams. It has no work area: fft_run passes one per call.
template <int N>
FftPlan& fft_plan(cudaStream_t stream, const int64_t* n, int64_t batch, cufftType type) {
    constexpr size_t kMaxPlans = 64;
    using Key = std::tuple<uintptr_t, int, std::array<int64_t, 3>, int64_t, int>;
    thread_local std::map<Key, FftPlan> plans;
    thread_local uint64_t clock = 0;
    Key key{reinterpret_cast<uintptr_t>(stream), N, pad3<N>(n), batch, (int)type};
    auto it = plans.find(key);
    if (it != plans.end()) {
        it->second.used = ++clock;
        return it->second;
    }
    if (plans.size() >= kMaxPlans) {
        auto old = plans.begin();
        for (auto j = plans.begin(); j != plans.end(); ++j)
            if (j->second.used < old->second.used)
                old = j;
        // FFTs queued with the plan may still be running on its stream
        cufftHandle h = old->second.h;
        ct_after_stream(reinterpret_cast<cudaStream_t>(std::get<0>(old->first)),
                        [h] { cufftDestroy(h); });
        plans.erase(old);
    }
    FftPlan p{0, 0, ++clock};
    long long dims[N];
    for (int d = 0; d < N; ++d)
        dims[d] = n[d];
    CT_CUFFT_CHECK(cufftCreate(&p.h));
    CT_CUFFT_CHECK(cufftSetAutoAllocation(p.h, 0));
    CT_CUFFT_CHECK(cufftMakePlanMany64(p.h, N, dims, nullptr, 1, 0, nullptr, 1, 0, type, batch,
                                       &p.work));
    return plans.emplace(key, p).first->second;
}

/// Runs plan p on the chare's compute stream with a pooled work area.
template <int N, typename F>
void fft_run(PartitionImpl<N>* partition, FftPlan& p, F&& exec) {
    cudaStream_t st = partition->compute_stream_raw;
    void* wa = p.work ? comm_alloc((int64_t)p.work) : nullptr;
    if (wa)
        CT_CUFFT_CHECK(cufftSetWorkArea(p.h, wa));
    CT_CUFFT_CHECK(cufftSetStream(p.h, st));
    CT_CUFFT_CHECK(exec(p.h));
    comm_free_after(st, wa);
}

/// Half spectrum of the filter v (dense, extents f) zero-padded to one block.
template <int N, typename T>
char* filter_spectrum(PartitionImpl<N>* partition, const T* v, const ConvBlocks<N>& cb) {
    using CT = CufftTypes<T>;
    using C = typename CT::C;
    int64_t Ls[N], fs[N];
    row_major_strides<N>(cb.L, Ls);
    row_major_strides<N>(cb.f, fs);
    T* vp = static_cast<T*>(comm_alloc(cb.lvol * (int64_t)sizeof(T)));
    C* spec = static_cast<C*>(comm_alloc(cb.lc * (int64_t)sizeof(C)));
    fill_zero<N, T>(partition, vp, cb.lvol);
    copy_box<N, T>(partition, vp, 0, Ls, v, 0, fs, cb.f);
    fft_run(partition, fft_plan<N>(partition->compute_stream_raw, cb.L, 1, CT::r2c),
            [&](cufftHandle h) { return CT::fwd(h, vp, spec); });
    comm_free_after(partition->compute_stream_raw, vp);
    return reinterpret_cast<char*>(spec);
}

/// out (extents len) = valid part of w (extents len + f - 1) convolved with the filter whose
/// half spectrum is spec, block by block in batches.
template <int N, typename T>
void conv_fft_blocks(PartitionImpl<N>* partition, T* out, const T* w, const int64_t* len,
                     const ConvBlocks<N>& cb, const char* spec, int local) {
    using CT = CufftTypes<T>;
    using C = typename CT::C;
    int64_t ext[N], nb[N], nbt = 1;
    for (int d = 0; d < N; ++d) {
        ext[d] = len[d] + cb.f[d] - 1;
        nb[d] = (len[d] + cb.B[d] - 1) / cb.B[d];
        nbt *= nb[d];
    }
    // a PE's chares queue their batches without a host wait: they share the budget
    static const int64_t budget = device_budget("CNUM_CONV_FFT_MB", 8);
    const int64_t chunk = std::min(kConvChunkBytes, budget / 2 / std::max(1, local));
    const int64_t q =
        std::max<int64_t>(1, std::min<int64_t>(nbt, chunk / (cb.lvol * (int64_t)sizeof(T))));
    T* xb = static_cast<T*>(comm_alloc(q * cb.lvol * (int64_t)sizeof(T)));
    C* xf = static_cast<C*>(comm_alloc(q * cb.lc * (int64_t)sizeof(C)));
    auto L3 = pad3<N>(cb.L), B3 = pad3<N>(cb.B), f3 = pad3<N>(cb.f), e3 = pad3<N>(ext),
         n3 = pad3<N>(nb), len3 = pad3<N>(len);
    int64_t bvol = B3[0] * B3[1] * B3[2];
    for (int64_t first = 0; first < nbt; first += q) {
        const int64_t k = std::min(q, nbt - first);
        ConvGather<T> g{xb, w, {L3[0], L3[1], L3[2]}, {B3[0], B3[1], B3[2]},
                        {e3[0], e3[1], e3[2]}, {n3[0], n3[1], n3[2]}, first};
        Kokkos::parallel_for("ct_conv_gather", CT_COMPUTE_POLICY(partition, k * cb.lvol), g);
        fft_run(partition, fft_plan<N>(partition->compute_stream_raw, cb.L, k, CT::r2c),
                [&](cufftHandle h) { return CT::fwd(h, xb, xf); });
        Kokkos::parallel_for("ct_conv_mul", CT_COMPUTE_POLICY(partition, k * cb.lc),
                             ConvSpecMul<T, C>{xf, reinterpret_cast<const C*>(spec), cb.lc,
                                               T(1) / T(cb.lvol)});
        fft_run(partition, fft_plan<N>(partition->compute_stream_raw, cb.L, k, CT::c2r),
                [&](cufftHandle h) { return CT::inv(h, xf, xb); });
        ConvScatter<T> sc{out, xb, {L3[0], L3[1], L3[2]}, {B3[0], B3[1], B3[2]},
                          {f3[0], f3[1], f3[2]}, {len3[0], len3[1], len3[2]},
                          {n3[0], n3[1], n3[2]}, first};
        Kokkos::parallel_for("ct_conv_scatter", CT_COMPUTE_POLICY(partition, k * bvol), sc);
    }
    comm_free_after(partition->compute_stream_raw, xb);
    comm_free_after(partition->compute_stream_raw, xf);
}
#else
/// Host spectrum of the filter v (dense, extents f) zero-padded to one block.
template <int N, typename T>
void filter_spectrum_host(const T* v, const ConvBlocks<N>& cb, std::vector<cplx>& spec) {
    int64_t Ls[N], fs[N], n = 1;
    row_major_strides<N>(cb.L, Ls);
    row_major_strides<N>(cb.f, fs);
    for (int d = 0; d < N; ++d)
        n *= cb.f[d];
    spec.assign(cb.lvol, cplx(0.0));
    for (int64_t k = 0; k < n; ++k) {
        int64_t li = 0;
        for (int d = 0; d < N; ++d)
            li += (k / fs[d] % cb.f[d]) * Ls[d];
        spec[li] = cplx(static_cast<double>(v[k]), 0.0);
    }
    fftn<N>(spec, cb.L);
}

/// Host overlap-save: like conv_fft_blocks, one block at a time.
template <int N, typename T>
void conv_fft_blocks_host(T* out, const T* w, const int64_t* len, const ConvBlocks<N>& cb,
                          const std::vector<cplx>& spec) {
    int64_t ext[N], nb[N], nbt = 1;
    for (int d = 0; d < N; ++d) {
        ext[d] = len[d] + cb.f[d] - 1;
        nb[d] = (len[d] + cb.B[d] - 1) / cb.B[d];
        nbt *= nb[d];
    }
    auto L3 = pad3<N>(cb.L), B3 = pad3<N>(cb.B), f3 = pad3<N>(cb.f), e3 = pad3<N>(ext),
         n3 = pad3<N>(nb), len3 = pad3<N>(len);
    std::vector<cplx> z(cb.lvol);
    std::vector<T> xb(cb.lvol);
    for (int64_t g = 0; g < nbt; ++g) {
        ConvGather<T> ga{xb.data(), w, {L3[0], L3[1], L3[2]}, {B3[0], B3[1], B3[2]},
                         {e3[0], e3[1], e3[2]}, {n3[0], n3[1], n3[2]}, g};
        for (int64_t i = 0; i < cb.lvol; ++i) {
            ga(i);
            z[i] = cplx(static_cast<double>(xb[i]), 0.0);
        }
        fftn<N>(z, cb.L);
        // inverse as a forward transform of the conjugate
        for (int64_t i = 0; i < cb.lvol; ++i)
            z[i] = std::conj(z[i] * spec[i]) / double(cb.lvol);
        fftn<N>(z, cb.L);
        for (int64_t i = 0; i < cb.lvol; ++i)
            xb[i] = static_cast<T>(z[i].real());
        ConvScatter<T> sc{out, xb.data(), {L3[0], L3[1], L3[2]}, {B3[0], B3[1], B3[2]},
                          {f3[0], f3[1], f3[2]}, {len3[0], len3[1], len3[2]},
                          {n3[0], n3[1], n3[2]}, g};
        for (int64_t i = 0; i < B3[0] * B3[1] * B3[2]; ++i)
            sc(i);
    }
}
#endif

/// What the result chares of one PE share: the dense filter (direct) or its spectrum (FFT).
struct ConvShared {
    char* data = nullptr; // device memory under USE_KOKKOS, else host
#ifndef USE_NVIDIA
    std::vector<cplx> spec; // host FFT
#endif
    int refs = 0;
#ifdef USE_NVIDIA
    cudaEvent_t ready = nullptr;    // recorded on the leader's compute stream
    std::vector<cudaEvent_t> done;  // recorded on the compute streams of released users
#endif
};

inline std::unordered_map<int, ConvShared>& conv_shared() {
    thread_local std::unordered_map<int, ConvShared> m;
    return m;
}

/// Frees a buffer once the work queued so far on the chare's compute stream has run.
template <int N>
void sig_free_after_compute(PartitionImpl<N>* partition, char* p) {
#if defined(USE_NVIDIA)
    comm_free_after(partition->compute_stream_raw, p);
#elif defined(USE_KOKKOS)
    (void)partition;
    Kokkos::fence("ct_conv_free");
    comm_free(p);
#else
    (void)partition;
    delete[] p;
#endif
}

/// A result chare is done reading the shared entry of node_id.
template <int N>
void conv_release(PartitionImpl<N>* partition, int node_id) {
    auto it = conv_shared().find(node_id);
    ConvShared& sh = it->second;
#ifdef USE_NVIDIA
    cudaStream_t st = partition->compute_stream_raw;
    if (--sh.refs > 0) {
        cudaEvent_t ev;
        hapiCheck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        hapiCheck(cudaEventRecord(ev, st));
        sh.done.push_back(ev);
        return;
    }
    for (cudaEvent_t ev : sh.done) {
        hapiCheck(cudaStreamWaitEvent(st, ev, 0));
        hapiCheck(cudaEventDestroy(ev));
    }
    if (sh.ready)
        hapiCheck(cudaEventDestroy(sh.ready));
#else
    if (--sh.refs > 0)
        return;
#endif
    if (sh.data)
        sig_free_after_compute<N>(partition, sh.data);
    conv_shared().erase(it);
}

/// Tells result chare ci (on this PE) that this PE's shared entry for node_id is ready. A
/// message, so Charm++ holds it until a chare inserted this epoch exists.
template <int N>
void conv_notify(PartitionImpl<N>* partition, int node_id, const std::array<int, N>& ci) {
    ChareIndex<N> target;
    for (int d = 0; d < N; ++d)
        target.idx[d] = ci[d];
    proxy_at<N>(partition->thisProxy, target).conv_ready(node_id);
}

/// Send this chare's block of the filter to the leader of every PE that holds result chares.
template <int N, typename T>
void send_filter(PartitionImpl<N>* partition, int node_id, const SigGeom<N>& s) {
    const SrcView<N>& v = s.in[1];
    auto it = partition->arrays.find(v.name);
    if (it == partition->arrays.end() || it->second->local_size() == 0)
        return;
    CTArrayBase<N>* src = it->second;
    auto me = partition->nd_index();
    int64_t ext[N], tsize[N], lstride[N], sstride[N], pstride[N], base = 0, count = 1;
    int region_data[N * 3];
    for (int d = 0; d < N; ++d) {
        int lo, hi;
        if (!src_block_1d<N>(v, d, me[d], lo, hi))
            return;
        ext[d] = hi - lo;
        count *= ext[d];
        tsize[d] = src->region.size(d);
        region_data[3 * d] = lo;
        region_data[3 * d + 1] = hi;
        region_data[3 * d + 2] = 1;
    }
    row_major_strides<N>(tsize, lstride);
    row_major_strides<N>(ext, pstride);
    for (int d = 0; d < N; ++d) {
        sstride[d] = (int64_t)v.vstep[d] * lstride[d];
        base += (int64_t)(v.vstart[d] + region_data[3 * d] * v.vstep[d] -
                          v.dec.chare_start_global(d, me[d])) *
                lstride[d];
    }
    const T* src_ptr = static_cast<const T*>(src->device_data_ptr());
    const int64_t bytes = count * (int64_t)sizeof(T);
    const ConvPeers<N>& peers = conv_peers<N>(s);
    std::vector<ChareIndex<N>> targets;
    for (const auto& l : peers.leader) {
        if (l[0] < 0)
            continue;
        ChareIndex<N> target;
        for (int d = 0; d < N; ++d)
            target.idx[d] = l[d];
        targets.push_back(target);
    }
    if (targets.empty())
        return;
    // one packed copy for every leader: it is read-only until the last send frees it
    char* buf = alloc_buffer(bytes);
    copy_box<N, T>(partition, reinterpret_cast<T*>(buf), 0, pstride, src_ptr, base, sstride, ext);
#ifdef USE_NVIDIA
    partition->join_streams(); // the sends start on the comm stream
#elif defined(USE_KOKKOS)
    Kokkos::fence("ct_conv_filter");
#endif
#ifdef USE_KOKKOS
    comm_share(buf, (int)targets.size());
#endif
    for (auto& target : targets) {
#ifndef NDEBUG
        partition->comm_bytes_sent += bytes;
#endif
#ifdef USE_KOKKOS
        device_pack_send<N, N>(partition, partition->thisProxy, target, node_id, 1, v.name,
                               region_data, bytes, buf);
#else
        ct_send_data(proxy_at<N>(partition->thisProxy, target), node_id, 1, v.name, N,
                     region_data, bytes, buf);
#endif
    }
#ifndef USE_KOKKOS
    delete[] buf;
#endif
}

/// The PE's leader, once its filter pieces are in: builds the shared filter or spectrum
/// and tells the PE's other result chares.
template <int N, typename T>
void conv_build_shared(PartitionImpl<N>* partition, int node_id, const SigGeom<N>& s,
                       PendingComm<N>& comm) {
    int64_t f[N], zero[N] = {};
    for (int d = 0; d < N; ++d)
        f[d] = s.in[1].vsize[d];
    const ConvPeers<N>& peers = conv_peers<N>(s);
    T* v = assemble<N, T>(partition, comm, 1, zero, f);
    auto [sit, fresh] = conv_shared().try_emplace(node_id);
    if (!fresh)
        CkAbort("CONVOLVE: PE %d already has a shared filter for node %d", CkMyPe(), node_id);
    ConvShared& sh = sit->second;
    sh.refs = peers.local;
    ConvBlocks<N> cb;
    if (!conv_fft<N, T>(s, cb)) {
        sh.data = reinterpret_cast<char*>(v);
    } else {
        if constexpr (std::is_floating_point<T>::value) {
#if defined(USE_NVIDIA)
            sh.data = filter_spectrum<N, T>(partition, v, cb);
#elif defined(USE_KOKKOS)
            int64_t nv = 1;
            for (int d = 0; d < N; ++d)
                nv *= f[d];
            std::vector<T> hv(nv);
            using U = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
            Kokkos::fence("ct_conv_filter");
            Kokkos::deep_copy(Kokkos::View<T*, HostSpace, U>(hv.data(), nv),
                              Kokkos::View<T*, DeviceSpace, U>(v, nv));
            filter_spectrum_host<N, T>(hv.data(), cb, sh.spec);
#else
            filter_spectrum_host<N, T>(v, cb, sh.spec);
#endif
        }
        sig_free_after_compute<N>(partition, reinterpret_cast<char*>(v));
    }
#ifdef USE_NVIDIA
    hapiCheck(cudaEventCreateWithFlags(&sh.ready, cudaEventDisableTiming));
    hapiCheck(cudaEventRecord(sh.ready, partition->compute_stream_raw));
#endif
    const auto me = partition->nd_index();
    std::array<int, N> first, last, ci;
    for (int d = 0; d < N; ++d)
        chare_span<N>(s.out, d, first[d], last[d]);
    ci = first;
    while (true) {
        if (ci != me && chare_home_pe(N, ci.data(), CkNumPes()) == CkMyPe())
            conv_notify<N>(partition, node_id, ci);
        int d = N - 1;
        for (; d >= 0; --d) {
            if (++ci[d] <= last[d])
                break;
            ci[d] = first[d];
        }
        if (d < 0)
            break;
    }
}

/// CONVOLVE on a result chare (range [olo, ohi)), once the PE's shared entry is built.
template <int N, typename T>
void finish_conv_shared(PartitionImpl<N>* partition, int node_id, const SigGeom<N>& s,
                        PendingComm<N>& comm, T* out, const std::array<int, N>& olo,
                        const std::array<int, N>& ohi) {
    int64_t len[N], wlo[N], wext[N], f[N];
    for (int d = 0; d < N; ++d) {
        len[d] = ohi[d] - olo[d];
        f[d] = s.in[1].vsize[d];
        wlo[d] = olo[d] + s.shift[d] - (f[d] - 1);
        wext[d] = len[d] + f[d] - 1;
    }
    ConvBlocks<N> cb;
    const bool fft = conv_fft<N, T>(s, cb);
    auto it = conv_shared().find(node_id);
    if (it == conv_shared().end())
        CkAbort("CONVOLVE: no shared filter for node %d on PE %d", node_id, CkMyPe());
    ConvShared& sh = it->second;
#ifdef USE_NVIDIA
    hapiCheck(cudaStreamWaitEvent(partition->compute_stream_raw, sh.ready, 0));
#endif
    T* w = assemble<N, T>(partition, comm, 0, wlo, wext);
    if (fft) {
        if constexpr (std::is_floating_point<T>::value) {
#if defined(USE_NVIDIA)
            conv_fft_blocks<N, T>(partition, out, w, len, cb, sh.data, conv_peers<N>(s).local);
#elif defined(USE_KOKKOS)
            int64_t nw = 1, no = 1;
            for (int d = 0; d < N; ++d) {
                nw *= wext[d];
                no *= len[d];
            }
            std::vector<T> hw(nw), ho(no);
            using U = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
            Kokkos::fence("ct_conv_window");
            Kokkos::deep_copy(Kokkos::View<T*, HostSpace, U>(hw.data(), nw),
                              Kokkos::View<T*, DeviceSpace, U>(w, nw));
            conv_fft_blocks_host<N, T>(ho.data(), hw.data(), len, cb, sh.spec);
            Kokkos::deep_copy(Kokkos::View<T*, DeviceSpace, U>(out, no),
                              Kokkos::View<T*, HostSpace, U>(ho.data(), no));
#else
            conv_fft_blocks_host<N, T>(out, w, len, cb, sh.spec);
#endif
        }
    } else {
        direct_conv<N, T>(partition, out, w, reinterpret_cast<const T*>(sh.data), len, f);
    }
    sig_free_after_compute<N>(partition, reinterpret_cast<char*>(w));
    conv_release<N>(partition, node_id);
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
#ifdef USE_NVIDIA
    // inputs may come from work on other streams; GEMMs then run without host fences
    if (s.op == Opcode::BATCHED_MATMUL)
        partition->compute_after_others();
#endif
    for (int i = 0; i < s.nin; ++i)
        with_sig_type(s.op, dt, [&](auto tag) {
            using T = decltype(tag);
            if constexpr (N == 3) {
                if (s.op == Opcode::BATCHED_MATMUL) {
                    send_bmm_input<N, T>(partition, node->id, s, i);
                    return;
                }
            }
            if (s.op == Opcode::CONVOLVE && i == 1)
                send_filter<N, T>(partition, node->id, s);
            else
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
    // finish_conv / finish_flip write the whole tile
    CTArrayBase<N>* res = partition->ensure_array(ArrayRegion<N>(zeros, sizes, ones),
                                                  s.out.global_shape, root->result_name, dt,
                                                  s.out, false);
    for (int d = 0; d < N; ++d)
        if (res->region.size(d) != sizes[d] || res->dtype != dt)
            CkAbort("CONVOLVE/FLIP/BATCHED_MATMUL: result %d exists with another tile or dtype",
                    root->result_name);
    const bool bmm = s.op == Opcode::BATCHED_MATMUL;
    int expected = 0;
    if constexpr (N == 3) {
        if (bmm)
            expected = bmm_start<N>(partition, node->id, s, root->result_name, olo, ohi,
                                    dtype_size(dt));
    }
    const bool conv = s.op == Opcode::CONVOLVE;
    const bool leader = conv && conv_is_leader<N>(conv_peers<N>(s), me);
    for (int i = 0; i < s.nin && !bmm; ++i)
        if (!conv || i == 0 || leader)
            expected += expected_msgs<N>(s, i, olo, ohi);
    if (conv && !leader)
        ++expected; // the PE's leader: shared entry ready

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
    pending[node->id] = {node, remaining, {}, std::move(pre_buffers), bmm};
    if constexpr (N == 3) {
        if (bmm) {
            BmmState& st = bmm_states().at({partition, node->id});
            PendingComm<N>& comm = pending[node->id];
            for (int i = 0; i < 2; ++i)
                for (auto& rb : comm.remote_buffers[i])
                    bmm_count(st, s, i, rb.region);
            CTArrayBase<N>* out = partition->arrays.at(root->result_name);
            with_sig_type(s.op, dt, [&](auto tag) {
                using T = decltype(tag);
                T* o = static_cast<T*>(out->device_data_ptr());
                if (st.plan.np() == 0)
                    fill_zero<N, T>(partition, o, out->local_size());
                else
                    bmm_progress<N, T>(partition, s, st, comm, o, false);
            });
        }
    }
    if (conv && leader) {
        PendingComm<N>& comm = pending[node->id];
        auto rb = comm.remote_buffers.find(1);
        const int have = rb == comm.remote_buffers.end() ? 0 : (int)rb->second.size();
        comm.conv_filter_left = std::max(0, expected_msgs<N>(s, 1, olo, ohi) - have);
        if (comm.conv_filter_left == 0)
            on_conv_filter(node->id);
    }
    if (remaining <= 0)
        on_comm_done(node->id);
}

template <int N>
void ArrayDAGExecutorND<N>::on_conv_filter(int node_id) {
    PendingComm<N>& comm = pending.at(node_id);
    ASTNode* root = signal_root(comm.node);
    SigGeom<N> s;
    make_geom<N>(static_cast<ArrayDAGGroup*>(group), root, s);
    with_conv_dtype(root->dtype, [&](auto tag) {
        conv_build_shared<N, decltype(tag)>(partition, node_id, s, comm);
    });
}

template <int N>
void ArrayDAGExecutorND<N>::on_signal_receive(int node_id, int input_index) {
    if constexpr (N == 3) {
        auto sit = bmm_states().find({partition, node_id});
        auto pit = pending.find(node_id);
        if (sit == bmm_states().end() || pit == pending.end() || input_index > 1)
            return;
        BmmState& st = sit->second;
        PendingComm<N>& comm = pit->second;
        bmm_count(st, st.geom, input_index, comm.remote_buffers[input_index].back().region);
        CTArrayBase<N>* out = partition->arrays.at(signal_root(comm.node)->result_name);
        with_sig_type(st.geom.op, out->dtype, [&](auto tag) {
            using T = decltype(tag);
            bmm_progress<N, T>(partition, st.geom, st, comm,
                               static_cast<T*>(out->device_data_ptr()), false);
        });
    }
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
    else if (s.op == Opcode::BATCHED_MATMUL) {
        if constexpr (N == 3) {
            auto sit = bmm_states().find({partition, comm.node->id});
            if (sit == bmm_states().end())
                CkAbort("batched matmul: no receive state");
            BmmState& st = sit->second;
            with_sig_type(s.op, out->dtype, [&](auto tag) {
                using T = decltype(tag);
                bmm_progress<N, T>(partition, s, st, comm,
                                   static_cast<T*>(out->device_data_ptr()), true);
            });
            for (int q = 0; q < st.plan.nq(); ++q)
                if (st.next[q] < st.plan.np())
                    CkAbort("batched matmul: operand blocks missing at completion");
            bmm_states().erase(sit);
        }
    }
    else
        with_conv_dtype(out->dtype, [&](auto tag) {
            using T = decltype(tag);
            finish_conv_shared<N, T>(partition, comm.node->id, s, comm,
                                     static_cast<T*>(out->device_data_ptr()), olo, ohi);
        });
}

template <int N>
void PartitionImpl<N>::conv_ready(int node_id) {
    PendingComm<N>& comm = executor->pending[node_id];
    if (comm.expected_msgs > 0) {
        if (--comm.expected_msgs == 0)
            executor->on_comm_done(node_id);
    } else {
        --comm.expected_msgs;
    }
}

template void PartitionImpl<1>::conv_ready(int);
template void PartitionImpl<2>::conv_ready(int);
template void PartitionImpl<3>::conv_ready(int);
template void ArrayDAGExecutorND<1>::execute_signal_node(DAGNode*);
template void ArrayDAGExecutorND<2>::execute_signal_node(DAGNode*);
template void ArrayDAGExecutorND<3>::execute_signal_node(DAGNode*);
template void ArrayDAGExecutorND<1>::on_signal_receive(int, int);
template void ArrayDAGExecutorND<2>::on_signal_receive(int, int);
template void ArrayDAGExecutorND<3>::on_signal_receive(int, int);
template void ArrayDAGExecutorND<1>::on_conv_filter(int);
template void ArrayDAGExecutorND<2>::on_conv_filter(int);
template void ArrayDAGExecutorND<3>::on_conv_filter(int);
template void ArrayDAGExecutorND<1>::finish_signal_node(PendingComm<1>&);
template void ArrayDAGExecutorND<2>::finish_signal_node(PendingComm<2>&);
template void ArrayDAGExecutorND<3>::finish_signal_node(PendingComm<3>&);
