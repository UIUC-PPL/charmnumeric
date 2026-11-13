#pragma once

// Compile-time DType dispatch and host/device element helpers (NumPy semantics).
// Dtype switches should go through cn_dispatch*.

#include "dtype_info.hpp"

#include <cmath>
#include <cstdint>
#include <type_traits>

#ifdef USE_KOKKOS
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#define CN_HD KOKKOS_INLINE_FUNCTION
using cn_half = Kokkos::Experimental::half_t;
template <typename F>
using cn_complex = Kokkos::complex<F>;
#else
#include <complex>
#define CN_HD inline
using cn_half = _Float16;
template <typename F>
using cn_complex = std::complex<F>;
#endif
using cn_c64 = cn_complex<float>;
using cn_c128 = cn_complex<double>;

template <typename T>
constexpr DType dtype_of();
#define CN_DTYPE_OF(T, D)                                                                          \
    template <>                                                                                    \
    constexpr DType dtype_of<T>() {                                                                \
        return DType::D;                                                                           \
    }
CN_DTYPE_OF(float, FLOAT32)
CN_DTYPE_OF(double, FLOAT64)
CN_DTYPE_OF(int32_t, INT32)
CN_DTYPE_OF(int64_t, INT64)
CN_DTYPE_OF(bool, BOOL)
CN_DTYPE_OF(int8_t, INT8)
CN_DTYPE_OF(int16_t, INT16)
CN_DTYPE_OF(uint8_t, UINT8)
CN_DTYPE_OF(uint16_t, UINT16)
CN_DTYPE_OF(uint32_t, UINT32)
CN_DTYPE_OF(uint64_t, UINT64)
CN_DTYPE_OF(cn_half, FLOAT16)
CN_DTYPE_OF(cn_c64, COMPLEX64)
CN_DTYPE_OF(cn_c128, COMPLEX128)
#undef CN_DTYPE_OF

template <typename T>
struct cn_is_complex : std::false_type {};
template <typename F>
struct cn_is_complex<cn_complex<F>> : std::true_type {};
template <typename T>
inline constexpr bool cn_is_complex_v = cn_is_complex<T>::value;

template <typename T>
inline constexpr bool cn_is_half_v = std::is_same_v<T, cn_half>;

/// float, double or half.
template <typename T>
inline constexpr bool cn_is_real_float_v = std::is_floating_point_v<T> || cn_is_half_v<T>;

/// Types the BLAS-backed executors (matvec, matmul, dot, linalg) support.
template <typename T>
inline constexpr bool cn_is_blas_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int32_t> ||
    std::is_same_v<T, int64_t> || cn_is_complex_v<T>;

/// X-macro over every element type (explicit instantiations).
#define CN_FOR_EACH_TYPE(X)                                                                        \
    X(float) X(double) X(int32_t) X(int64_t) X(bool) X(int8_t) X(int16_t) X(uint8_t) X(uint16_t)  \
        X(uint32_t) X(uint64_t) X(cn_half) X(cn_c64) X(cn_c128)

/// Calls f(T{}) for dt's element type. f must not contain device lambdas (nvcc).
template <typename F>
void cn_dispatch(DType dt, F&& f) {
    switch (dt) {
    case DType::FLOAT32: f(float{}); break;
    case DType::FLOAT64: f(double{}); break;
    case DType::INT32: f(int32_t{}); break;
    case DType::INT64: f(int64_t{}); break;
    case DType::BOOL: f(bool{}); break;
    case DType::INT8: f(int8_t{}); break;
    case DType::INT16: f(int16_t{}); break;
    case DType::UINT8: f(uint8_t{}); break;
    case DType::UINT16: f(uint16_t{}); break;
    case DType::UINT32: f(uint32_t{}); break;
    case DType::UINT64: f(uint64_t{}); break;
    case DType::FLOAT16: f(cn_half{}); break;
    case DType::COMPLEX64: f(cn_c64{}); break;
    case DType::COMPLEX128: f(cn_c128{}); break;
    }
}

/// cn_dispatch restricted to the BLAS types (cn_is_blas_v); false for other dtypes.
template <typename F>
bool cn_dispatch_blas(DType dt, F&& f) {
    switch (dt) {
    case DType::FLOAT32: f(float{}); return true;
    case DType::FLOAT64: f(double{}); return true;
    case DType::INT32: f(int32_t{}); return true;
    case DType::INT64: f(int64_t{}); return true;
    case DType::COMPLEX64: f(cn_c64{}); return true;
    case DType::COMPLEX128: f(cn_c128{}); return true;
    default: return false;
    }
}

// ---- element helpers (host and device) ----

template <typename T>
CN_HD T cn_real_part(const T& x) {
    return x;
}
template <typename F>
CN_HD F cn_real_part(const cn_complex<F>& x) {
    return x.real();
}

template <typename T>
CN_HD bool cn_isnan(const T& x) {
    if constexpr (cn_is_complex_v<T>)
        return x.real() != x.real() || x.imag() != x.imag();
    else if constexpr (cn_is_half_v<T>)
        return static_cast<float>(x) != static_cast<float>(x);
    else
        return x != x;
}

/// x != 0 (NaN is true, like bool(nan)).
template <typename T>
CN_HD bool cn_nonzero(const T& x) {
    if constexpr (cn_is_complex_v<T>)
        return x.real() != 0 || x.imag() != 0;
    else if constexpr (cn_is_half_v<T>)
        return static_cast<float>(x) != 0.0f;
    else
        return x != T(0);
}

/// Real value as double (complex: real part).
template <typename T>
CN_HD double cn_to_double(const T& x) {
    if constexpr (cn_is_complex_v<T>)
        return static_cast<double>(x.real());
    else if constexpr (cn_is_half_v<T>)
        return static_cast<double>(static_cast<float>(x));
    else
        return static_cast<double>(x);
}

/// NumPy astype: complex -> real drops the imaginary part, x -> bool is x != 0,
/// half goes through float.
template <typename To, typename From>
CN_HD To cn_cast(const From& x) {
    if constexpr (std::is_same_v<To, From>) {
        return x;
    } else if constexpr (std::is_same_v<To, bool>) {
        return cn_nonzero(x);
    } else if constexpr (cn_is_complex_v<To>) {
        using F = typename To::value_type;
        if constexpr (cn_is_complex_v<From>)
            return To(static_cast<F>(x.real()), static_cast<F>(x.imag()));
        else
            return To(cn_cast<F>(x), F(0));
    } else if constexpr (cn_is_complex_v<From>) {
        return cn_cast<To>(x.real());
    } else if constexpr (cn_is_half_v<To>) {
#ifdef USE_KOKKOS
        if constexpr (std::is_same_v<From, bool>)
            return To(x ? 1.0f : 0.0f);
        else
            return Kokkos::Experimental::cast_to_half(x);
#else
        return static_cast<To>(x);
#endif
    } else if constexpr (cn_is_half_v<From>) {
        return static_cast<To>(static_cast<float>(x));
    } else {
        return static_cast<To>(x);
    }
}

/// NumPy's sort order (npysort *_LT): NaNs last; complex is lexicographic, with
/// NaN parts ordered as R+Rj < R+nanj < nan+Rj < nan+nanj.
template <typename T>
CN_HD bool cn_sort_lt(const T& a, const T& b) {
    if constexpr (cn_is_complex_v<T>) {
        auto ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
        if (ar < br)
            return ai == ai || bi != bi;
        if (ar > br)
            return bi != bi && ai == ai;
        if (ar == br || (ar != ar && br != br))
            return ai < bi || (bi != bi && ai == ai);
        return br != br;
    } else if constexpr (cn_is_half_v<T>) {
        float x = static_cast<float>(a), y = static_cast<float>(b);
        return x < y || (y != y && x == x);
    } else {
        return a < b || (b != b && a == a);
    }
}
