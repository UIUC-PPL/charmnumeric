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

    // Elementwise math (opcode range 20-49). Comparisons and predicates
    // produce 0/1 masks in the kernel's element type (there is no bool dtype).
    // greater/greater_equal are less/less_equal with swapped operands.
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
    // Array creation (50-59): create a new array and fill it chare-locally.
    UPLOAD = 50, // host data carried in the DAG (ASTNode::payload)
    RANDOM = 51, // counter-based RNG
    IOTA = 52,   // arange / linspace / indices / eye
    // Reductions (executor_reduction.cpp). Opcode range 60-69.
    REDUCTION = 60, // full or axis reduction of one operand (sum, max, argmin, ...)
    BINCOUNT = 61   // histogram of a non-negative int vector (optionally weighted)
};

/// Returns true for ops handled by execute_reduction_node.
inline bool is_reduction_op(Opcode op) {
    return op == Opcode::REDUCTION || op == Opcode::BINCOUNT;
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
        return true;
    default:
        return false;
    }
}

/// Returns true for ternary elementwise ops.
inline bool is_ternary_elementwise(Opcode op) {
    switch (op) {
    case Opcode::WHERE:
    case Opcode::CLIP:
        return true;
    default:
        return false;
    }
}

/// Returns true for any elementwise op (unary, binary or ternary).
inline bool is_elementwise(Opcode op) {
    return is_binary_elementwise(op) || is_unary_elementwise(op) || is_ternary_elementwise(op);
}
