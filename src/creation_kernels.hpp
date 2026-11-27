#pragma once

// Per-element value functions for array-creation ops (RANDOM, IOTA), shared by
// CPU loops and Kokkos kernels. The d* / f* helpers use _rn intrinsics on device
// so no FMA contraction occurs. charmnumeric/_rng_reference.py mirrors this file.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "dtypes.hpp"

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#define CT_HD KOKKOS_INLINE_FUNCTION
#else
#define CT_HD inline
#endif

namespace ct_create {

// ---- Exactly rounded arithmetic (no FMA contraction on the device) ----

CT_HD double dadd(double a, double b) {
#ifdef __CUDA_ARCH__
    return __dadd_rn(a, b);
#else
    return a + b;
#endif
}
CT_HD double dsub(double a, double b) {
#ifdef __CUDA_ARCH__
    return __dsub_rn(a, b);
#else
    return a - b;
#endif
}
CT_HD double dmul(double a, double b) {
#ifdef __CUDA_ARCH__
    return __dmul_rn(a, b);
#else
    return a * b;
#endif
}
CT_HD double ddiv(double a, double b) {
#ifdef __CUDA_ARCH__
    return __ddiv_rn(a, b);
#else
    return a / b;
#endif
}
CT_HD float fadd(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
CT_HD float fmul(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
CT_HD float fdiv(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

CT_HD double bits_to_double(uint64_t u) {
    double d;
    memcpy(&d, &u, sizeof(d));
    return d;
}
CT_HD uint64_t double_to_bits(double d) {
    uint64_t u;
    memcpy(&u, &d, sizeof(u));
    return u;
}

// ---- Philox4x32-10 ----

struct Philox4 {
    uint32_t v[4];
};

CT_HD Philox4 philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0,
                            uint32_t k1) {
    for (int r = 0; r < 10; ++r) {
        if (r > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        uint64_t p0 = (uint64_t)0xD2511F53u * c0;
        uint64_t p1 = (uint64_t)0xCD9E8D57u * c2;
        uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
        uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
        uint32_t n0 = hi1 ^ c1 ^ k0;
        uint32_t n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
    }
    return Philox4{{c0, c1, c2, c3}};
}

CT_HD Philox4 rng_block(uint64_t g, uint64_t seed, uint64_t stream, uint32_t attempt) {
    return philox4x32_10((uint32_t)g, (uint32_t)(g >> 32), (uint32_t)stream, attempt,
                         (uint32_t)seed, (uint32_t)(seed >> 32));
}

CT_HD uint64_t word_a(const Philox4& x) {
    return ((uint64_t)x.v[1] << 32) | x.v[0];
}
CT_HD uint64_t word_b(const Philox4& x) {
    return ((uint64_t)x.v[3] << 32) | x.v[2];
}

/// Uniform double in [0, 1) with 53 random bits.
CT_HD double u01(uint64_t w) {
    return (double)(w >> 11) * (1.0 / 9007199254740992.0);
}

CT_HD uint64_t mulhi64(uint64_t a, uint64_t b) {
#ifdef __CUDA_ARCH__
    return __umul64hi(a, b);
#else
    return (uint64_t)(((unsigned __int128)a * b) >> 64);
#endif
}

// ---- Software log and cos(2*pi*u) (fixed operation sequence) ----

/// Natural log for normal positive x: x = m * 2^e with m in [sqrt(1/2), sqrt(2)),
/// log(m) = 2 atanh(s), s = (m-1)/(m+1), as a degree-23 odd series in s.
CT_HD double soft_log(double x) {
    uint64_t bits = double_to_bits(x);
    int e = (int)((bits >> 52) & 0x7ff) - 1023;
    double m = bits_to_double((bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL);
    if (m > 1.4142135623730951) {
        m = dmul(m, 0.5);
        e += 1;
    }
    double f = dsub(m, 1.0);
    double s = ddiv(f, dadd(2.0, f));
    double s2 = dmul(s, s);
    double p = ddiv(1.0, 23.0);
    for (int k = 10; k >= 0; --k)
        p = dadd(dmul(p, s2), ddiv(1.0, (double)(2 * k + 1)));
    double ed = (double)e;
    return dadd(dmul(ed, 6.93147180369123816490e-01),
                dadd(dmul(ed, 1.90821492927058770002e-10), dmul(dmul(2.0, s), p)));
}

/// cos(2*pi*u) for u in [0, 1): exact quadrant reduction, then Taylor series
/// for sin/cos on [0, pi/4].
CT_HD double soft_cos2pi(double u) {
    double t = dmul(u, 4.0);
    int q = (int)t;
    double f = dsub(t, (double)q);
    bool flip = f > 0.5;
    double g = flip ? dsub(1.0, f) : f;
    double r = dmul(g, 1.5707963267948966);
    double r2 = dmul(r, r);
    const double cc[10] = {1.0,
                           -1.0 / 2.0,
                           1.0 / 24.0,
                           -1.0 / 720.0,
                           1.0 / 40320.0,
                           -1.0 / 3628800.0,
                           1.0 / 479001600.0,
                           -1.0 / 87178291200.0,
                           1.0 / 20922789888000.0,
                           -1.0 / 6402373705728000.0};
    const double sc[10] = {1.0,
                           -1.0 / 6.0,
                           1.0 / 120.0,
                           -1.0 / 5040.0,
                           1.0 / 362880.0,
                           -1.0 / 39916800.0,
                           1.0 / 6227020800.0,
                           -1.0 / 1307674368000.0,
                           1.0 / 355687428096000.0,
                           -1.0 / 121645100408832000.0};
    double c = cc[9];
    double s = sc[9];
    for (int k = 8; k >= 0; --k) {
        c = dadd(dmul(c, r2), cc[k]);
        s = dadd(dmul(s, r2), sc[k]);
    }
    s = dmul(s, r);
    double cos_t = flip ? s : c;
    double sin_t = flip ? c : s;
    switch (q) {
    case 0:
        return cos_t;
    case 1:
        return -sin_t;
    case 2:
        return -cos_t;
    default:
        return sin_t;
    }
}

// ---- RANDOM ----

enum RandomKind : int { RANDOM_U01 = 0, RANDOM_UNIFORM = 1, RANDOM_NORMAL = 2, RANDOM_INT = 3 };

struct RandomParams {
    int kind;
    uint64_t seed;
    uint64_t stream;
    uint64_t a; // bit pattern: double (low / loc) or int64 (randint low)
    uint64_t b; // bit pattern: double (high-low / scale) or int64 (randint high)
    int64_t index_strides[3];
};

template <typename T>
CT_HD T random_value(const RandomParams& p, uint64_t g) {
    Philox4 x = rng_block(g, p.seed, p.stream, 0);
    if (p.kind == RANDOM_INT) {
        uint64_t low = p.a;
        uint64_t range = p.b - p.a;
        uint64_t w = word_a(x);
        if (range == 0) // the full 2^64 range: every word is a valid draw
            return cn_cast<T>((int64_t)(w + low));
        uint64_t thresh = (0 - range) % range;
        uint64_t hi = mulhi64(w, range);
        uint64_t lo = w * range;
        uint32_t draw = 1;
        while (lo < thresh) {
            if (draw % 2 == 0)
                x = rng_block(g, p.seed, p.stream, draw / 2);
            w = (draw % 2 == 0) ? word_a(x) : word_b(x);
            hi = mulhi64(w, range);
            lo = w * range;
            ++draw;
        }
        return cn_cast<T>((int64_t)(hi + low));
    }
    if constexpr (std::is_floating_point_v<T>) {
        if (p.kind == RANDOM_U01) {
            if constexpr (std::is_same_v<T, float>)
                return fmul((float)(x.v[0] >> 8), 1.0f / 16777216.0f);
            else
                return u01(word_a(x));
        }
        double a = bits_to_double(p.a);
        double b = bits_to_double(p.b);
        if (p.kind == RANDOM_UNIFORM)
            return (T)dadd(a, dmul(b, u01(word_a(x))));
        // RANDOM_NORMAL: Box-Muller with u1 in (0, 1]
        double u1 = dsub(1.0, u01(word_a(x)));
        double u2 = u01(word_b(x));
        double z = dmul(sqrt(dmul(-2.0, soft_log(u1))), soft_cos2pi(u2));
        return (T)dadd(a, dmul(b, z));
    } else {
        return T(0); // float kinds on integer dtypes are rejected by the frontend
    }
}

// ---- IOTA (arange / linspace / indices / eye) ----

enum IotaMode : int { IOTA_ARANGE = 0, IOTA_LINSPACE = 1, IOTA_INDICES = 2, IOTA_EYE = 3 };

struct IotaParams {
    int mode;
    int axis;
    int flags;   // LINSPACE: bit0 endpoint, bit1 step==0, bit2 compute in float32
    int64_t num; // LINSPACE: number of samples
    int64_t div; // LINSPACE: num-1 (endpoint) or num
    uint64_t v0; // ARANGE: first value; LINSPACE: start; EYE: k (int64)
    uint64_t v1; // ARANGE: second value; LINSPACE: stop
    uint64_t v2; // LINSPACE: step
    uint64_t v3; // LINSPACE: delta = stop - start
};

/// Value of type T from a 64-bit parameter word (double bits or int64).
template <typename T>
CT_HD T param_value(uint64_t bits) {
    if constexpr (cn_is_real_float_v<T> || cn_is_complex_v<T>)
        return cn_cast<T>(bits_to_double(bits));
    else
        return cn_cast<T>((int64_t)bits);
}

template <typename C>
CT_HD C c_add(C a, C b) {
    if constexpr (std::is_same_v<C, float>)
        return fadd(a, b);
    else
        return dadd(a, b);
}
template <typename C>
CT_HD C c_mul(C a, C b) {
    if constexpr (std::is_same_v<C, float>)
        return fmul(a, b);
    else
        return dmul(a, b);
}
template <typename C>
CT_HD C c_div(C a, C b) {
    if constexpr (std::is_same_v<C, float>)
        return fdiv(a, b);
    else
        return ddiv(a, b);
}

/// numpy.linspace element i, evaluated in compute type C (float or double).
template <typename C>
CT_HD double linspace_value(const IotaParams& p, int64_t i) {
    C start = (C)bits_to_double(p.v0);
    C stop = (C)bits_to_double(p.v1);
    C step = (C)bits_to_double(p.v2);
    C delta = (C)bits_to_double(p.v3);
    C y = (C)i;
    if (p.div > 0) {
        if (p.flags & 2)
            y = c_mul<C>(c_div<C>(y, (C)p.div), delta);
        else
            y = c_mul<C>(y, step);
    } else {
        y = c_mul<C>(y, delta);
    }
    y = c_add<C>(y, start);
    if ((p.flags & 1) && p.num > 1 && i == p.num - 1)
        y = stop;
    return (double)y;
}

/// Value at array index idx (idx[d] for d < N; unused entries are 0).
template <typename T>
CT_HD T iota_value(const IotaParams& p, const int64_t* idx) {
    switch (p.mode) {
    case IOTA_ARANGE: {
        // numpy's fill: out[0] = v0, out[1] = v1, out[i] = v0 + i * (v1 - v0) in T.
        int64_t i = idx[p.axis];
        T v0 = param_value<T>(p.v0);
        if (i == 0)
            return v0;
        T v1 = param_value<T>(p.v1);
        if (i == 1)
            return v1;
        if constexpr (std::is_same_v<T, float>) {
            return fadd(v0, fmul((float)i, fadd(v1, -v0)));
        } else if constexpr (std::is_same_v<T, double>) {
            return dadd(v0, dmul((double)i, dsub(v1, v0)));
        } else if constexpr (cn_is_half_v<T> || cn_is_complex_v<T>) {
            double d0 = bits_to_double(p.v0), d1 = bits_to_double(p.v1);
            return cn_cast<T>(dadd(d0, dmul((double)i, dsub(d1, d0))));
        } else {
            uint64_t delta = (uint64_t)p.v1 - (uint64_t)p.v0;
            return cn_cast<T>((int64_t)((uint64_t)p.v0 + (uint64_t)i * delta));
        }
    }
    case IOTA_LINSPACE: {
        int64_t i = idx[p.axis];
        double y = (p.flags & 4) ? linspace_value<float>(p, i) : linspace_value<double>(p, i);
        if constexpr (cn_is_real_float_v<T> || cn_is_complex_v<T>)
            return cn_cast<T>(y);
        else
            return cn_cast<T>(floor(y));
    }
    case IOTA_INDICES:
        // numpy.indices: out[k, i0, i1, ...] = i_k
        return cn_cast<T>(idx[1 + idx[0]]);
    case IOTA_EYE:
        return (idx[1] - idx[0] == (int64_t)p.v0) ? T(1) : T(0);
    default:
        return T(0);
    }
}

} // namespace ct_create
