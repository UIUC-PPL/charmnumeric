#pragma once

enum class Opcode {
    COPY = -5,
    SET_REGION = -4,
    GET_REGION = -3,
    CREATE = -2,
    NOOP = -1,
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    MATMUL = 4,
    TANH = 5,
    EXP = 6,
    TILE = 7,
    REDUCE = 8,
    MATMATMUL = 9,
    DIAG = 10,

    // Elementwise math (20-49). Comparisons produce bool; greater/greater_equal
    // are less/less_equal with swapped operands.
    SQRT = 20,
    LOG = 21,
    LOG2 = 22,
    LOG10 = 23,
    LOG1P = 24,
    EXP2 = 25,
    EXPM1 = 26,
    ABS = 27,
    FLOOR = 28,
    CEIL = 29,
    RINT = 30, // round half to even
    ISFINITE = 31,
    SIN = 32,
    COS = 33,
    SIGN = 34,
    ISNAN = 35,
    ISINF = 36,
    POW = 37,
    MAXIMUM = 38, // NaN-propagating
    MINIMUM = 39, // NaN-propagating
    FMAX = 40,    // NaN-ignoring
    FMIN = 41,    // NaN-ignoring
    MOD = 42,     // Python/NumPy remainder: sign of the divisor
    FLOOR_DIV = 43,
    LESS = 44,
    LESS_EQUAL = 45,
    EQUAL = 46,
    NOT_EQUAL = 47,
    WHERE = 48, // where(cond, x, y): cond != 0 selects x
    CLIP = 49,  // clip(x, lo, hi) = minimum(maximum(x, lo), hi)
    // Array creation (50-59), filled chare-locally.
    UPLOAD = 50, // host data carried in the DAG (ASTNode::payload)
    RANDOM = 51, // counter-based RNG
    IOTA = 52,   // arange / linspace / indices / eye
    // Reductions (executor_reduction.cpp). Opcode range 60-69.
    REDUCTION = 60, // full or axis reduction of one operand (sum, max, argmin, ...)
    BINCOUNT = 61,  // histogram of a non-negative int vector (optionally weighted)
    // Transpose (70): materialized axis permutation (executor_transpose.cpp).
    TRANSPOSE = 70,
    // 80-89: executor_castscansort.cpp (80 is unused)
    SCAN = 81,    // operands (x, axis, kind); kind: 0 cumsum, 1 cumprod, 2 nancumsum, 3 nancumprod
    SORT = 82,    // operands (x, axis); sorted copy along axis (NaNs last)
    ARGSORT = 83, // operands (x, axis); stable int64 indices along axis
    // Dense linear algebra (100-119, executor_linalg.cpp).
    LINALG = 100, // factorization/solve on one chare; kind is a scalar operand
    // Data-dependent indexing (executor_indexing.cpp). Opcode range 120-139.
    GATHER = 120,  // out[e] = src.flat[L[b] + sum(slice coords * strides)]
    SCATTER = 121, // dst.flat[L[b] + ...] = v[e] (in place; result name unused)
    EXPAND = 122,  // flat positions i of a view, each repeated count(x[i]) times
    // 140-149: executor_shape.cpp. Operands (x, target shape...).
    RESHAPE = 140,   // C-order reshape into a new array
    BROADCAST = 141, // numpy.broadcast_to, materialized
    // Signal processing (150-159, executor_signal.cpp).
    CONVOLVE = 150, // operands (a, v, method, shift[nd], shape[nd]); method: 0 direct, 1 fft
    FLIP = 151,     // operands (x, axis_mask)
    // Elementwise dtype ops (160-169): fused astype, complex parts, bitwise.
    CONVERT = 160, // astype: result dtype is the node dtype
    REAL = 161,
    IMAG = 162,
    CONJ = 163,
    BITWISE_AND = 164, // integers and bool
    BITWISE_OR = 165,
    BITWISE_XOR = 166,
    INVERT = 167, // ~x; logical not on bool
    LEFT_SHIFT = 168,
    RIGHT_SHIFT = 169 // arithmetic for signed ints
};

/// Returns true for the ops handled by execute_signal_node.
inline bool is_signal_op(Opcode op) {
    return op == Opcode::CONVOLVE || op == Opcode::FLIP;
}

/// Returns true for ops handled by execute_index_node.
inline bool is_index_op(Opcode op) {
    return op == Opcode::GATHER || op == Opcode::SCATTER || op == Opcode::EXPAND;
}

/// Returns true for ops handled by execute_reduction_node.
inline bool is_reduction_op(Opcode op) {
    return op == Opcode::REDUCTION || op == Opcode::BINCOUNT;
}

/// Returns true for ops handled by execute_shape_node.
inline bool is_shape_op(Opcode op) {
    return op == Opcode::RESHAPE || op == Opcode::BROADCAST;
}

/// Returns true for the ops that create and fill a new array (not CREATE).
inline bool is_fill_creation(Opcode op) {
    return op == Opcode::UPLOAD || op == Opcode::RANDOM || op == Opcode::IOTA;
}

/// Returns true for every op that defines a fresh array from its region
/// operand (CREATE and the fill-creation ops).
inline bool is_creation(Opcode op) {
    return op == Opcode::CREATE || is_fill_creation(op);
}

/// Returns true for binary elementwise ops.
inline bool is_binary_elementwise(Opcode op) {
    switch (op) {
    case Opcode::ADD:
    case Opcode::SUB:
    case Opcode::MUL:
    case Opcode::DIV:
    case Opcode::POW:
    case Opcode::MAXIMUM:
    case Opcode::MINIMUM:
    case Opcode::FMAX:
    case Opcode::FMIN:
    case Opcode::MOD:
    case Opcode::FLOOR_DIV:
    case Opcode::LESS:
    case Opcode::LESS_EQUAL:
    case Opcode::EQUAL:
    case Opcode::NOT_EQUAL:
    case Opcode::BITWISE_AND:
    case Opcode::BITWISE_OR:
    case Opcode::BITWISE_XOR:
    case Opcode::LEFT_SHIFT:
    case Opcode::RIGHT_SHIFT:
        return true;
    default:
        return false;
    }
}

/// Returns true for unary elementwise ops.
inline bool is_unary_elementwise(Opcode op) {
    switch (op) {
    case Opcode::TANH:
    case Opcode::EXP:
    case Opcode::SQRT:
    case Opcode::LOG:
    case Opcode::LOG2:
    case Opcode::LOG10:
    case Opcode::LOG1P:
    case Opcode::EXP2:
    case Opcode::EXPM1:
    case Opcode::ABS:
    case Opcode::FLOOR:
    case Opcode::CEIL:
    case Opcode::RINT:
    case Opcode::ISFINITE:
    case Opcode::SIN:
    case Opcode::COS:
    case Opcode::SIGN:
    case Opcode::ISNAN:
    case Opcode::ISINF:
    case Opcode::CONVERT:
    case Opcode::REAL:
    case Opcode::IMAG:
    case Opcode::CONJ:
    case Opcode::INVERT:
        return true;
    default:
        return false;
    }
}

inline bool is_ternary_elementwise(Opcode op) {
    switch (op) {
    case Opcode::WHERE:
    case Opcode::CLIP:
        return true;
    default:
        return false;
    }
}

/// Result has the input's shape.
inline bool is_castscansort(Opcode op) {
    switch (op) {
    case Opcode::SCAN:
    case Opcode::SORT:
    case Opcode::ARGSORT:
        return true;
    default:
        return false;
    }
}

inline bool is_elementwise(Opcode op) {
    return is_binary_elementwise(op) || is_unary_elementwise(op) || is_ternary_elementwise(op);
}
