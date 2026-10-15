#pragma once

// DType predicates and NumPy type promotion (no Kokkos/MLIR dependencies).

#include <charmtyles/core/ast.hpp>

inline bool dtype_is_complex(DType dt) {
    return dt == DType::COMPLEX64 || dt == DType::COMPLEX128;
}
inline bool dtype_is_float(DType dt) {
    return dt == DType::FLOAT16 || dt == DType::FLOAT32 || dt == DType::FLOAT64;
}
inline bool dtype_is_unsigned(DType dt) {
    return dt == DType::UINT8 || dt == DType::UINT16 || dt == DType::UINT32 ||
           dt == DType::UINT64;
}
inline bool dtype_is_int(DType dt) { // signed or unsigned integer, not bool
    return dt == DType::INT8 || dt == DType::INT16 || dt == DType::INT32 || dt == DType::INT64 ||
           dtype_is_unsigned(dt);
}

/// numpy.promote_types for all pairs (row/column order = DType codes).
inline DType dtype_promote(DType a, DType b) {
    static const char* table[NUM_DTYPES] = {
        // f4 f8 i4 i8 b i1 i2 u1 u2 u4 u8 f2 c8 c16
        "\x00\x01\x01\x01\x00\x00\x00\x00\x00\x01\x01\x00\x0c\x0d",
        "\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x0d\x0d",
        "\x01\x01\x02\x03\x02\x02\x02\x02\x02\x03\x01\x01\x0d\x0d",
        "\x01\x01\x03\x03\x03\x03\x03\x03\x03\x03\x01\x01\x0d\x0d",
        "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d",
        "\x00\x01\x02\x03\x05\x05\x06\x06\x02\x03\x01\x0b\x0c\x0d",
        "\x00\x01\x02\x03\x06\x06\x06\x06\x02\x03\x01\x00\x0c\x0d",
        "\x00\x01\x02\x03\x07\x06\x06\x07\x08\x09\x0a\x0b\x0c\x0d",
        "\x00\x01\x02\x03\x08\x02\x02\x08\x08\x09\x0a\x00\x0c\x0d",
        "\x01\x01\x03\x03\x09\x03\x03\x09\x09\x09\x0a\x01\x0d\x0d",
        "\x01\x01\x01\x01\x0a\x01\x01\x0a\x0a\x0a\x0a\x01\x0d\x0d",
        "\x00\x01\x01\x01\x0b\x0b\x00\x0b\x00\x01\x01\x0b\x0c\x0d",
        "\x0c\x0d\x0d\x0d\x0c\x0c\x0c\x0c\x0c\x0d\x0d\x0c\x0c\x0d",
        "\x0d\x0d\x0d\x0d\x0d\x0d\x0d\x0d\x0d\x0d\x0d\x0d\x0d\x0d",
    };
    return static_cast<DType>(table[static_cast<int>(a)][static_cast<int>(b)]);
}
