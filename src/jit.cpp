#include "jit.hpp"
#include "dtype_info.hpp"
#include "jit_api.hpp"

#include "mlir/IR/Verifier.h"

#include <chrono>
#include <limits>

#include "llvm/Support/FormatVariadic.h"

#ifdef JIT_ENABLE_GPU_BACKEND
#include "llvm/IR/LegacyPassManager.h"
#endif
#ifdef USE_NVIDIA
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#endif

namespace {
void declareExternalCallees(mlir::ModuleOp module);
mlir::Type storageType(mlir::MLIRContext* ctx, DType dt);
mlir::Value convertValue(mlir::OpBuilder& b, mlir::Location loc, mlir::Value v, DType from,
                         DType to);
} // namespace

namespace {
// CNUM_JIT_TIME=1 prints each kernel compile's phase times (ms) to stderr.
struct JitTimer {
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    std::string line;
    void mark(const char* what) {
        auto now = std::chrono::steady_clock::now();
        line += llvm::formatv(" {0}={1:F1}", what,
                              std::chrono::duration<double, std::milli>(now - t).count());
        t = now;
    }
};
thread_local JitTimer* jitTimer = nullptr;
void jitMark(const char* what) {
    if (jitTimer)
        jitTimer->mark(what);
}
} // namespace

namespace {
// One context per process: per-kernel creation dominated compile time.
mlir::MLIRContext& sharedContext() {
    static mlir::MLIRContext* ctx = [] {
        mlir::DialectRegistry registry;
        mlir::registerAllToLLVMIRTranslations(registry);
#ifdef USE_NVIDIA
        mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
        mlir::registerAllExtensions(registry);
        mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
#endif
        auto* c = new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
        c->getOrLoadDialect<mlir::linalg::LinalgDialect>();
        c->getOrLoadDialect<mlir::affine::AffineDialect>();
        c->getOrLoadDialect<mlir::gpu::GPUDialect>();
        c->getOrLoadDialect<mlir::arith::ArithDialect>();
        c->getOrLoadDialect<mlir::memref::MemRefDialect>();
        c->getOrLoadDialect<mlir::func::FuncDialect>();
        c->getOrLoadDialect<mlir::scf::SCFDialect>();
        c->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        c->getOrLoadDialect<mlir::NVVM::NVVMDialect>();
        c->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
        c->getOrLoadDialect<mlir::spirv::SPIRVDialect>();
        c->getOrLoadDialect<mlir::math::MathDialect>();
        c->getOrLoadDialect<mlir::complex::ComplexDialect>();
        return c;
    }();
    return *ctx;
}
} // namespace

MLIRJitCompiler::MLIRJitCompiler() : context(sharedContext()) {
    module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    builder = std::make_unique<mlir::OpBuilder>(&context);
}

void MLIRJitCompiler::buildFromAST(void* astPtr) {
    using namespace mlir;
    AST* ast = static_cast<AST*>(astPtr);
    auto loc = builder->getUnknownLoc();

    // --- 1. Collect leaf nodes ---
    std::vector<LeafUse> memrefLeaves;     // non-scalar leaves → memref func args
    std::vector<ASTNode*> scalarLeaves;    // scalar leaves → scalar func args
    std::vector<ASTNode*> broadcastLeaves; // size-1 arrays → scalar func args
    collectLeaves(ast, memrefLeaves, scalarLeaves, broadcastLeaves);

    // Scalar values are runtime arguments, so kernels differing only in constants are shared.
    {
        std::unordered_set<int> seen;
        std::vector<ASTNode*> unique;
        for (ASTNode* sn : scalarLeaves)
            if (seen.insert(sn->result_name).second)
                unique.push_back(sn);
        scalarLeaves.swap(unique);
    }

    // Collect non-temp root operations as output slots.
    // Each gets a caller-provided output memref appended after the input args.
    std::vector<ASTNode*> outputRoots;
    for (ASTNode* op : ast->roots) {
        auto opc = static_cast<Opcode>(op->opcode);
        if (opc != Opcode::NOOP && opc != Opcode::CREATE && !op->is_temp)
            outputRoots.push_back(op);
    }

    // Each memref, broadcast scalar and output has its own storageType; scalar leaves are f64.
    auto dynamicStridedMemRefType = [&](int64_t rank, DType dt) {
        SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
        SmallVector<int64_t> dynStrides(rank, ShapedType::kDynamic);
        auto layout = StridedLayoutAttr::get(&context, ShapedType::kDynamic, dynStrides);
        return MemRefType::get(dynShape, storageType(&context, dt), layout);
    };

    // --- 2. Create a func.func: memref inputs, broadcast scalars, then output memrefs ---
    SmallVector<Type> argTypes;
    for (const LeafUse& leaf : memrefLeaves)
        argTypes.push_back(dynamicStridedMemRefType(leaf.operand->ndims, leaf.operand->dtype));
    // Broadcast leaves are scalar function arguments (runtime values, not constants)
    for (ASTNode* leaf : broadcastLeaves)
        argTypes.push_back(storageType(&context, leaf->dtype));
    for (size_t i = 0; i < scalarLeaves.size(); ++i)
        argTypes.push_back(builder->getF64Type());
    for (ASTNode* root : outputRoots)
        argTypes.push_back(dynamicStridedMemRefType(root->ndims, root->dtype));
    auto funcType = FunctionType::get(&context, argTypes, /*results=*/{});
    auto funcOp = func::FuncOp::create(loc, "fused_kernel", funcType);
    module->push_back(funcOp);

    Block* entry = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entry);

    // Seed memref leaves with block arguments
    nodeValues.clear();
    leafValues.clear();
    tempValues.clear();
    for (size_t i = 0; i < memrefLeaves.size(); ++i)
        leafValues[leafUseKey(memrefLeaves[i].operand, memrefLeaves[i].region)] =
            entry->getArgument(i);

    // Seed broadcast leaves with their scalar block arguments
    for (size_t i = 0; i < broadcastLeaves.size(); ++i)
        nodeValues[broadcastLeaves[i]->result_name] =
            entry->getArgument(memrefLeaves.size() + i);

    // Seed scalar leaves with their scalar block arguments
    for (size_t i = 0; i < scalarLeaves.size(); ++i)
        nodeValues[scalarLeaves[i]->result_name] =
            entry->getArgument(memrefLeaves.size() + broadcastLeaves.size() + i);

    // --- 3. Process each operation sequentially (flat list) ---
    for (ASTNode* op : ast->roots)
        emitOp(op, loc);

    // --- 3b. Copy each non-temp root result into its caller-provided output arg ---
    //
    // Use linalg.generic instead of memref.copy: the default-layout memref
    // type (memref<?x?xf32>) causes memref.copy to lower to a flat memcpy,
    // which is incorrect when source and destination have different strides
    // (e.g., a 14×14 contiguous temp copied into a 14×14 view of a 16×16
    // array with stride [16,1]).  A linalg.generic copy lowers to affine
    // loops that respect each operand's stride descriptor and participates
    // in affine loop fusion, eliminating the intermediate allocation.
    size_t outArgIdx = memrefLeaves.size() + broadcastLeaves.size() + scalarLeaves.size();
    for (ASTNode* root : outputRoots) {
        Value result = nodeValues[root->result_name];
        Value outArg = entry->getArgument(outArgIdx++);
        if (mlir::isa<MemRefType>(result.getType())) {
            auto outType = mlir::cast<MemRefType>(outArg.getType());
            int64_t rank = outType.getRank();
            auto identityMap = builder->getMultiDimIdentityMap(rank);
            SmallVector<AffineMap, 2> copyMaps = {identityMap, identityMap};
            SmallVector<utils::IteratorType> copyIters(rank, utils::IteratorType::parallel);
            linalg::GenericOp::create(*builder, loc, TypeRange{}, ValueRange{result},
                                               ValueRange{outArg}, copyMaps, copyIters,
                                               [](OpBuilder& nb, Location nl, ValueRange args) {
                                                   linalg::YieldOp::create(nb, nl, args[0]);
                                               });
        } else {
            // Scalar result (e.g., SET_REGION with scalar RHS) — fill output
            linalg::FillOp::create(*builder, loc, result, outArg);
        }
    }

    // --- 3c. Declare external math callees (fmod) used by op bodies ---
    declareExternalCallees(*module);

    // --- 4. Deallocate temporaries (after all consumers have read) ---
    for (mlir::Value tmp : tempValues)
        memref::DeallocOp::create(*builder, loc, tmp);

    // --- 5. Terminate the function body ---
    func::ReturnOp::create(*builder, loc);
}

bool MLIRJitCompiler::optimizeAndFuse() {
    mlir::PassManager pm(&context);
    pm.enableVerifier(false); // the module is verified once, after building
    // Complex arithmetic becomes arith/math on the parts (the GPU pipeline
    // only lowers complex.create/re/im).
    pm.addPass(mlir::createConvertComplexToStandardPass());
    auto& funcPM = pm.nest<mlir::func::FuncOp>();

#if defined(USE_NVIDIA) || defined(USE_AMD) || defined(USE_INTEL)
    // GPU path: fuse at Linalg level, keeping linalg.generic ops intact
    // for later conversion to scf.parallel → gpu.launch.
    funcPM.addPass(mlir::createLinalgElementwiseOpFusionPass());
#else
    // CPU path: convert to Affine and use the powerful Affine Fusion pass.
    // A. Convert Linalg to Affine loops
    funcPM.addPass(mlir::createConvertLinalgToAffineLoopsPass());

    // B. Maximal Loop Fusion: removes intermediate array results
    // by merging producer loops into consumer loops.
    funcPM.addPass(mlir::affine::createLoopFusionPass(0,          /* fastMemorySpace */
                                                      UINT64_MAX, /* localBufSizeThreshold */
                                                      true,       /* maximalFusion */
                                                      mlir::affine::FusionMode::ProducerConsumer));

    // C. Scalar Replacement: Promotes intermediate MemRefs to registers/SSA
    funcPM.addPass(mlir::affine::createAffineScalarReplacementPass());
#endif

    return mlir::succeeded(pm.run(*module));
}

#if !defined(USE_AMD) && !defined(USE_INTEL)
namespace {
// Half-precision conversions LLVM emits as libgcc calls on x86 (no F16C); the
// static libgcc is not visible to the JIT.
extern "C" _Float16 cn_truncdfhf2(double x) { return (_Float16)x; }
extern "C" _Float16 cn_truncsfhf2(float x) { return (_Float16)x; }
extern "C" float cn_extendhfsf2(_Float16 x) { return (float)x; }
extern "C" double cn_extendhfdf2(_Float16 x) { return (double)x; }

llvm::orc::SymbolMap halfRuntimeSymbols(llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap map;
    auto add = [&](const char* name, void* fn) {
        map[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(fn),
                               llvm::JITSymbolFlags::Exported};
    };
    add("__truncdfhf2", (void*)&cn_truncdfhf2);
    add("__truncsfhf2", (void*)&cn_truncsfhf2);
    add("__extendhfsf2", (void*)&cn_extendhfsf2);
    add("__extendhfdf2", (void*)&cn_extendhfdf2);
#ifdef USE_NVIDIA
    // libdevice names for host-side scalar ops
    add("__nv_fmodf", (void*)static_cast<float (*)(float, float)>(&fmodf));
    add("__nv_fmod", (void*)static_cast<double (*)(double, double)>(&fmod));
#endif
    return map;
}
} // namespace
#endif

std::unique_ptr<mlir::ExecutionEngine> MLIRJitCompiler::generateCPU() {
    if (!lowerToCPU())
        return nullptr;

    // Initialize native target (required for JIT on the host CPU)
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::ExecutionEngineOptions engineOptions;
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) {
        llvm::errs() << "Failed to create ExecutionEngine\n";
        return nullptr;
    }
#if !defined(USE_AMD) && !defined(USE_INTEL)
    (*maybeEngine)->registerSymbols(halfRuntimeSymbols);
#endif

    return std::move(*maybeEngine);
}

bool MLIRJitCompiler::loadCPU(std::unique_ptr<mlir::ExecutionEngine>& engine,
                              const char* kernelName, void** moduleOut, void** functionOut) {
    if (!engine)
        return false;

    auto funcPtrOrErr = engine->lookupPacked(kernelName);
    if (!funcPtrOrErr)
        return false;

    *moduleOut = (void*)engine.release();
    *functionOut = (void*)*funcPtrOrErr;
    return true;
}

bool MLIRJitCompiler::verify() { return mlir::succeeded(mlir::verify(*module)); }

void MLIRJitCompiler::dumpMLIR() { module->dump(); }

void MLIRJitCompiler::printModule(llvm::raw_ostream& os) { module->print(os); }

// ---- Private helpers ----

void MLIRJitCompiler::collectLeaves(AST* ast, std::vector<LeafUse>& memrefLeaves,
                                    std::vector<ASTNode*>& scalarLeaves,
                                    std::vector<ASTNode*>& broadcastLeaves) {
    // Build set of result_names produced internally by roots in this fused kernel.
    // Operands referencing these names are internal intermediates, not external inputs.
    std::unordered_set<int> internalResults;
    for (ASTNode* root : ast->roots)
        internalResults.insert(root->result_name);

    std::unordered_set<int> seenBroadcasts;

    for (ASTNode* root : ast->roots) {
        auto root_opc = static_cast<Opcode>(root->opcode);

        for (int op_idx = 0; op_idx < (int)root->operands.size(); ++op_idx) {
            ASTNode* operand = root->operands[op_idx];
            if (!operand)
                continue;

            if (internalResults.count(operand->result_name))
                continue;

            // Skip SET_REGION's target operand (operands[0]) — it's a write
            // destination handled by the copy-back, not a JIT input.
            if (root_opc == Opcode::SET_REGION && op_idx == 0)
                continue;

            auto op = static_cast<Opcode>(operand->opcode);
            if (op == Opcode::NOOP || op == Opcode::CREATE) {
                if (operand->is_scalar) {
                    scalarLeaves.push_back(operand);
                } else if (operand->is_broadcast) {
                    if (seenBroadcasts.insert(operand->result_name).second)
                        broadcastLeaves.push_back(operand);
                } else {
                    memrefLeaves.push_back({operand, root->get_operand_region(op_idx)});
                }
            }
        }
    }
}

std::string MLIRJitCompiler::leafUseKey(ASTNode* operand, Region* region) const {
    std::string key = std::to_string(operand->result_name) + "|" + std::to_string(operand->ndims);
    if (region == nullptr)
        return key + "|null";
    if (region->is_global)
        return key + "|global";

    switch (operand->ndims) {
    case 1:
        return key + "|" + fmt_region(*static_cast<ArrayRegion<1>*>(region));
    case 2:
        return key + "|" + fmt_region(*static_cast<ArrayRegion<2>*>(region));
    case 3:
        return key + "|" + fmt_region(*static_cast<ArrayRegion<3>*>(region));
    default:
        return key + "|nd=" + std::to_string(operand->ndims);
    }
}

mlir::Value MLIRJitCompiler::lookupOperand(ASTNode* operand, Region* region) {
    // Broadcast uses are scalar block arguments seeded into nodeValues.
    if (region != nullptr && !operand->is_broadcast) {
        auto leafIt = leafValues.find(leafUseKey(operand, region));
        if (leafIt != leafValues.end())
            return leafIt->second;
    }
    auto it = nodeValues.find(operand->result_name);
    if (it != nodeValues.end())
        return it->second;
    return {}; // should not happen if AST is well-formed
}

namespace {
/// i1 inside op bodies (comparison and predicate results).
constexpr DType DT_PRED = static_cast<DType>(100);

enum class BodyKind { Float, SInt, UInt, Bool, Complex };

BodyKind bodyKind(DType dt) {
    if (dtype_is_float(dt))
        return BodyKind::Float;
    if (dtype_is_complex(dt))
        return BodyKind::Complex;
    if (dt == DType::BOOL)
        return BodyKind::Bool;
    return dtype_is_unsigned(dt) ? BodyKind::UInt : BodyKind::SInt;
}

bool isComparison(Opcode op) {
    return op == Opcode::LESS || op == Opcode::LESS_EQUAL || op == Opcode::EQUAL ||
           op == Opcode::NOT_EQUAL;
}

bool isPredicate(Opcode op) {
    return isComparison(op) || op == Opcode::ISNAN || op == Opcode::ISINF ||
           op == Opcode::ISFINITE;
}

/// Unary ops that compute in their operand's dtype; the result dtype may differ.
bool computesInOperandType(Opcode op) {
    return op == Opcode::ISNAN || op == Opcode::ISINF || op == Opcode::ISFINITE ||
           op == Opcode::ABS || op == Opcode::REAL || op == Opcode::IMAG || op == Opcode::CONVERT;
}

/// The registry body for a compute kind, or nullptr if the op has none.
template <typename Entry>
const typename Entry::Fn* pickBody(const Entry& e, BodyKind k) {
    const typename Entry::Fn* f = nullptr;
    switch (k) {
    case BodyKind::Float: f = &e.float_body; break;
    case BodyKind::SInt: f = &e.int_body; break;
    case BodyKind::UInt: f = e.uint_body ? &e.uint_body : &e.int_body; break;
    case BodyKind::Bool: f = e.bool_body ? &e.bool_body : &e.int_body; break;
    case BodyKind::Complex: f = &e.complex_body; break;
    }
    return (f && *f) ? f : nullptr;
}

DType realOf(DType dt) {
    return dt == DType::COMPLEX64 ? DType::FLOAT32 : DType::FLOAT64;
}
} // namespace

mlir::Value MLIRJitCompiler::allocLike(mlir::Value arrayOperand, DType dt) {
    using namespace mlir;
    auto loc = builder->getUnknownLoc();
    int64_t rank = mlir::cast<MemRefType>(arrayOperand.getType()).getRank();
    SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
    SmallVector<Value> dimSizes;
    for (int64_t d = 0; d < rank; ++d)
        dimSizes.push_back(memref::DimOp::create(*builder, loc, arrayOperand, d).getResult());
    auto type = MemRefType::get(dynShape, storageType(&context, dt));
    return memref::AllocOp::create(*builder, loc, type, ValueRange(dimSizes));
}

mlir::Value MLIRJitCompiler::convertArray(mlir::Value v, DType from, DType to, bool is_temp) {
    if (from == to)
        return v;
    mlir::Value out = allocLike(v, to);
    emitGenericUnaryOp(v, out, [&](mlir::OpBuilder& b, mlir::Location l, mlir::Value x) {
        return convertValue(b, l, x, from, to);
    });
    if (is_temp)
        tempValues.push_back(out);
    return out;
}

void MLIRJitCompiler::emitOp(ASTNode* node, mlir::Location loc) {
    using namespace mlir;

    auto op = static_cast<Opcode>(node->opcode);
    Value result;
    // Scalar leaves are weak f64 values: they take no part in type promotion.
    auto operandDType = [](ASTNode* operand) {
        return operand->is_scalar ? DType::FLOAT64 : operand->dtype;
    };

    switch (op) {
    case Opcode::NOOP:
    case Opcode::CREATE:
        return;

    case Opcode::SET_REGION: {
        ASTNode* rhs = node->operands[1];
        Value v = lookupOperand(rhs, node->get_operand_region(1));
        if (mlir::isa<MemRefType>(v.getType()))
            result = convertArray(v, operandDType(rhs), node->dtype, true);
        else
            result = convertValue(*builder, loc, v, operandDType(rhs), node->dtype);
        break;
    }

    default: {
        const BinaryOpEntry* be = nullptr;
        const UnaryOpEntry* ue = nullptr;
        const TernaryOpEntry* te = nullptr;
        int arity = 0;
        if (auto it = binaryOpRegistry().find(node->opcode); it != binaryOpRegistry().end()) {
            be = &it->second;
            arity = 2;
        } else if (auto it2 = unaryOpRegistry().find(node->opcode); it2 != unaryOpRegistry().end()) {
            ue = &it2->second;
            arity = 1;
        } else if (auto it3 = ternaryOpRegistry().find(node->opcode);
                   it3 != ternaryOpRegistry().end()) {
            te = &it3->second;
            arity = 3;
        } else {
            llvm::errs() << "Unknown opcode in JIT emitOp: " << node->opcode << "\n";
            return;
        }

        SmallVector<Value, 3> ins;
        SmallVector<DType, 3> dts;
        for (int i = 0; i < arity; ++i) {
            ins.push_back(lookupOperand(node->operands[i], node->get_operand_region(i)));
            dts.push_back(operandDType(node->operands[i]));
        }

        // Compute dtype C: comparisons promote non-scalar operands; some unary ops keep input dtype.
        DType R = node->dtype;
        DType C = R;
        if (isComparison(op)) {
            bool any = false;
            for (int i = 0; i < arity; ++i)
                if (!node->operands[i]->is_scalar) {
                    C = any ? dtype_promote(C, dts[i]) : dts[i];
                    any = true;
                }
            if (!any)
                C = DType::FLOAT64;
        } else if (computesInOperandType(op)) {
            C = dts[0];
        }
        DType K = C == DType::FLOAT16 ? DType::FLOAT32 : C; // half math runs in float
        BodyKind kind = bodyKind(K);

        std::function<Value(OpBuilder&, Location, ValueRange)> body;
        if (be) {
            if (auto* f = pickBody(*be, kind))
                body = [f](OpBuilder& b, Location l, ValueRange a) { return (*f)(b, l, a[0], a[1]); };
        } else if (ue) {
            if (auto* f = pickBody(*ue, kind))
                body = [f](OpBuilder& b, Location l, ValueRange a) { return (*f)(b, l, a[0]); };
        } else if (auto* f = pickBody(*te, kind)) {
            body = [f](OpBuilder& b, Location l, ValueRange a) {
                return (*f)(b, l, a[0], a[1], a[2]);
            };
        }
        if (!body)
            llvm::report_fatal_error(llvm::Twine("JIT: opcode ") + llvm::Twine(node->opcode) +
                                     " is not defined for dtype " + llvm::Twine((int)C));

        // Operands convert to K (where's condition to i1); the body result
        // (i1 for predicates, real for complex abs/real/imag, 0/1 i8 for
        // bool math) converts to R.
        SmallVector<DType, 3> tgt(arity, K);
        if (op == Opcode::WHERE)
            tgt[0] = DT_PRED;
        DType RB = K;
        if (isPredicate(op))
            RB = DT_PRED;
        else if ((op == Opcode::ABS || op == Opcode::REAL || op == Opcode::IMAG) &&
                 dtype_is_complex(K))
            RB = realOf(K);
        else if (K == DType::BOOL)
            RB = DType::INT8;
        auto typed = [&](OpBuilder& b, Location l, ValueRange args) -> Value {
            SmallVector<Value, 3> conv;
            for (int i = 0; i < arity; ++i)
                conv.push_back(convertValue(b, l, args[i], dts[i], tgt[i]));
            return convertValue(b, l, body(b, l, conv), RB, R);
        };

        Value arrayOperand;
        for (Value v : ins)
            if (!arrayOperand && mlir::isa<MemRefType>(v.getType()))
                arrayOperand = v;
        if (!arrayOperand) {
            // Scalar and broadcast operands only: emit a plain scalar op
            result = typed(*builder, loc, ins);
            break;
        }

        result = allocLike(arrayOperand, R);
        if (arity == 2)
            emitGenericArrayOp(ins[0], ins[1], result, [&](OpBuilder& b, Location l, Value x, Value y) {
                return typed(b, l, ValueRange{x, y});
            });
        else if (arity == 1)
            emitGenericUnaryOp(ins[0], result, [&](OpBuilder& b, Location l, Value x) {
                return typed(b, l, ValueRange{x});
            });
        else
            emitGenericNaryOp(ins, result, typed);

        // freed at the end; root results are copied to their outputs first
        tempValues.push_back(result);
        break;
    }
    }

    nodeValues[node->result_name] = result;
}

bool MLIRJitCompiler::lowerToCPU() {
    mlir::PassManager pm(&context);
    pm.enableVerifier(false); // the module is verified once, after building
    // Lower any remaining Linalg ops to Affine loops (no-op if optimizeAndFuse() ran first)
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    // Lower Affine to SCF (structured control flow)
    pm.addPass(mlir::createLowerAffinePass());
    // Lower SCF to CF (unstructured control flow)
    pm.addPass(mlir::createSCFToControlFlowPass());
    // Decompose memref.subview and similar ops into lower-level memref ops
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    // Lower CF, Arith, MemRef, Func to LLVM dialect
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertComplexToStandardPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertComplexToLLVMPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    // Reconcile unrealized conversion casts left by partial lowerings
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    return mlir::succeeded(pm.run(*module));
}

#ifdef USE_NVIDIA

#ifndef MLIR_CUDA_RUNTIME_LIB
#error "MLIR_CUDA_RUNTIME_LIB must point to libmlir_cuda_runtime.so"
#endif

namespace {
using mlir::Operation;

/// Memref -> function argument it provably shares a shape with (identity-map linalg ops).
llvm::DenseMap<mlir::Value, mlir::Value> sameShapeArgs(mlir::func::FuncOp f) {
    using namespace mlir;
    llvm::DenseMap<Value, Value> parent;
    std::function<Value(Value)> find = [&](Value v) -> Value {
        auto it = parent.find(v);
        if (it == parent.end() || it->second == v)
            return v;
        Value r = find(it->second);
        parent[v] = r;
        return r;
    };
    f.walk([&](linalg::LinalgOp op) {
        if (!llvm::all_of(op.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }))
            return;
        Value root;
        for (Value v : op->getOperands()) {
            if (!isa<MemRefType>(v.getType()))
                continue;
            Value r = find(v);
            if (!root) {
                root = r;
                continue;
            }
            if (r == root)
                continue;
            if (isa<BlockArgument>(r))
                std::swap(r, root);
            parent[r] = root;
        }
    });
    llvm::DenseMap<Value, Value> out;
    for (auto& [v, unused] : parent) {
        Value r = find(v);
        if (r != v && isa<BlockArgument>(r))
            out[v] = r;
    }
    return out;
}

bool sameIvs(mlir::ValueRange idx, mlir::scf::ParallelOp loop) {
    auto ivs = loop.getInductionVars();
    if (idx.size() != ivs.size())
        return false;
    for (size_t i = 0; i < ivs.size(); ++i)
        if (idx[i] != ivs[i])
            return false;
    return true;
}

mlir::Value memrefRoot(mlir::Value v) {
    while (auto view = v.getDefiningOp<mlir::ViewLikeOpInterface>())
        v = view.getViewSource();
    return v;
}

struct MemAccess {
    mlir::Value mem;
    bool write;
    bool identity;
};

/// Loads/stores of a loop body; false if it has other side effects.
bool collectAccesses(mlir::scf::ParallelOp loop, llvm::SmallVectorImpl<MemAccess>& out) {
    using namespace mlir;
    bool ok = true;
    loop.getBody()->walk([&](Operation* op) {
        if (auto ld = dyn_cast<memref::LoadOp>(op))
            out.push_back({ld.getMemRef(), false, sameIvs(ld.getIndices(), loop)});
        else if (auto st = dyn_cast<memref::StoreOp>(op))
            out.push_back({st.getMemRef(), true, sameIvs(st.getIndices(), loop)});
        else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() && !isa<func::CallOp>(op) &&
                 !isMemoryEffectFree(op))
            ok = false;
    });
    return ok;
}

/// Adjacent loops can share one body if every buffer written by either one is
/// accessed by both only at the iteration's own index. Distinct kernel
/// arguments never alias: outputs are fresh arrays.
bool canFuse(mlir::scf::ParallelOp a, mlir::scf::ParallelOp b) {
    if (a.getNumResults() || b.getNumResults() || a.getLowerBound() != b.getLowerBound() ||
        a.getUpperBound() != b.getUpperBound() || a.getStep() != b.getStep())
        return false;
    llvm::SmallVector<MemAccess> accA, accB;
    if (!collectAccesses(a, accA) || !collectAccesses(b, accB))
        return false;
    auto conflicts = [](llvm::ArrayRef<MemAccess> x, llvm::ArrayRef<MemAccess> y) {
        for (const MemAccess& p : x)
            for (const MemAccess& q : y)
                if ((p.write || q.write) && memrefRoot(p.mem) == memrefRoot(q.mem) &&
                    !(p.identity && q.identity && p.mem == q.mem))
                    return true;
        return false;
    };
    return !conflicts(accA, accB);
}

/// Fuse the elementwise loops of the kernel function into as few loops as
/// possible and drop the temporaries that fusion makes thread-local.
void fuseElementwiseLoops(mlir::func::FuncOp f) {
    using namespace mlir;
    Block& body = f.getBody().front();

    // Hoist allocations and scalar code above the first loop so loops become adjacent.
    scf::ParallelOp first;
    for (Operation& op : body)
        if ((first = dyn_cast<scf::ParallelOp>(op)))
            break;
    if (!first)
        return;
    auto definedAbove = [&](Value v) {
        Operation* d = v.getDefiningOp();
        return !d || d->getBlock() != &body || d->isBeforeInBlock(first);
    };
    for (Operation& op : llvm::make_early_inc_range(
             llvm::make_range(std::next(first->getIterator()), body.end()))) {
        if (op.getNumRegions() || op.hasTrait<OpTrait::IsTerminator>() ||
            !(isMemoryEffectFree(&op) || isa<memref::AllocOp, memref::AllocaOp, func::CallOp>(op)))
            continue;
        if (llvm::all_of(op.getOperands(), definedAbove))
            op.moveBefore(first);
    }

    for (Operation& op : body) {
        auto a = dyn_cast<scf::ParallelOp>(op);
        if (!a)
            continue;
        while (auto b = dyn_cast_or_null<scf::ParallelOp>(a->getNextNode())) {
            if (!canFuse(a, b))
                break;
            for (auto [ivB, ivA] : llvm::zip(b.getInductionVars(), a.getInductionVars()))
                ivB.replaceAllUsesWith(ivA);
            Block* src = b.getBody();
            a.getBody()->getOperations().splice(a.getBody()->getTerminator()->getIterator(),
                                                src->getOperations(), src->begin(),
                                                src->getTerminator()->getIterator());
            b.erase();
        }
    }

    // Forward stores to temporaries into later loads of the same element.
    f.walk([&](scf::ParallelOp loop) {
        llvm::DenseMap<Value, memref::StoreOp> last;
        for (Operation& op : llvm::make_early_inc_range(*loop.getBody())) {
            if (auto st = dyn_cast<memref::StoreOp>(op)) {
                if (st.getMemRef().getDefiningOp<memref::AllocOp>())
                    last[st.getMemRef()] = st;
            } else if (auto ld = dyn_cast<memref::LoadOp>(op)) {
                auto it = last.find(ld.getMemRef());
                if (it != last.end() && llvm::equal(it->second.getIndices(), ld.getIndices())) {
                    ld.getResult().replaceAllUsesWith(it->second.getValueToStore());
                    ld.erase();
                }
            } else if (op.getNumRegions()) {
                op.walk([&](memref::StoreOp st) { last.erase(st.getMemRef()); });
            }
        }
    });

    // Temporaries that are only written are dead.
    for (Operation& op : llvm::make_early_inc_range(body)) {
        auto alloc = dyn_cast<memref::AllocOp>(op);
        if (!alloc || !llvm::all_of(alloc->getUsers(), [](Operation* u) {
                return isa<memref::StoreOp, memref::DeallocOp>(u);
            }))
            continue;
        for (Operation* u : llvm::make_early_inc_range(alloc->getUsers()))
            u->erase();
        alloc.erase();
    }
}
} // namespace

bool MLIRJitCompiler::lowerToNVIDIAHost() {
    using namespace mlir;

    // 1. Every linalg op becomes one multi-dim scf.parallel; adjacent loops over
    //    the same index space are fused. The rest is tiled so the innermost
    //    tile maps to threads and the rest to blocks. Tile sizes keep blockDim
    //    within limits (x,y <= 1024, z <= 64).
    int64_t rank = 1;
    module->walk([&](linalg::LinalgOp op) {
        rank = std::max<int64_t>(rank, op.getNumLoops());
    });
    SmallVector<int64_t> tiles(rank, 1);
    tiles.back() = (rank >= 3) ? 64 : 256;

    auto kernelFn = module->lookupSymbol<func::FuncOp>("fused_kernel");
    auto shapeArg = sameShapeArgs(kernelFn);
    PassManager pm0(&context);
    pm0.addPass(createConvertLinalgToParallelLoopsPass());
    if (failed(pm0.run(*module)))
        return false;
    // Loop bounds of same-shaped arrays become the same SSA values.
    kernelFn.walk([&](memref::DimOp d) {
        auto it = shapeArg.find(d.getSource());
        if (it != shapeArg.end())
            d.getSourceMutable().assign(it->second);
    });
    PassManager pm1(&context);
    pm1.addPass(createCanonicalizerPass());
    pm1.addPass(createCSEPass());
    if (failed(pm1.run(*module)))
        return false;
    fuseElementwiseLoops(kernelFn);

    PassManager pm(&context);
    pm.nest<func::FuncOp>().addPass(createParallelLoopTilingPass(tiles, /*noMinMaxBounds=*/true));
    pm.nest<func::FuncOp>().addPass(createGpuMapParallelLoopsPass());
    pm.addPass(createConvertParallelLoopToGpuPass());
    if (failed(pm.run(*module)))
        return false;

    // 2. Scalar code (e.g. ops on broadcast or scalar leaves) runs in the
    //    kernels, so the host only launches.
    auto sinkable = [](Operation* op) {
        return op->getNumRegions() == 0 && !op->hasTrait<OpTrait::IsTerminator>() &&
               !isa<memref::AllocOp, memref::AllocaOp>(op) &&
               (isMemoryEffectFree(op) || isa<func::CallOp>(op));
    };
    module->walk([&](gpu::LaunchOp launch) { (void)sinkOperationsIntoLaunchOp(launch, sinkable); });
    for (Operation& op : llvm::make_early_inc_range(llvm::reverse(kernelFn.getBody().front())))
        if (op.use_empty() && sinkable(&op))
            op.erase();

    // Launches are asynchronous: the host must not read what kernels write.
    bool hostLoad = false;
    kernelFn.walk([&](memref::LoadOp ld) {
        hostLoad |= !ld->getParentOfType<gpu::LaunchOp>();
    });
    if (hostLoad) {
        llvm::errs() << "lowerToNVIDIAHost: host-side load left after kernel outlining\n";
        return false;
    }

    // 3. Remaining temporaries are device memory, unless the host touches them
    //    (then managed). gpu.alloc/dealloc lower only in async form (or sync
    //    host_shared alloc), so wrap them in gpu.wait.
    SmallVector<Operation*> allocs;
    SmallVector<memref::DeallocOp> deallocs;
    module->walk([&](Operation* op) {
        if (op->getParentOfType<gpu::LaunchOp>())
            return;
        if (isa<memref::AllocOp, memref::AllocaOp>(op))
            allocs.push_back(op);
        else if (auto d = dyn_cast<memref::DeallocOp>(op))
            deallocs.push_back(d);
    });
    OpBuilder b(&context);
    Type tokTy = gpu::AsyncTokenType::get(&context);
    auto emitDealloc = [&](Location loc, Value mem) {
        Value t0 = gpu::WaitOp::create(b, loc, tokTy, ValueRange{}).getAsyncToken();
        Value t1 = gpu::DeallocOp::create(b, loc, tokTy, ValueRange{t0}, mem).getAsyncToken();
        gpu::WaitOp::create(b, loc, Type(), ValueRange{t1});
    };
    for (Operation* op : allocs) {
        b.setInsertionPoint(op);
        auto type = cast<MemRefType>(op->getResult(0).getType());
        ValueRange dynSizes, symOps;
        if (auto a = dyn_cast<memref::AllocOp>(op)) {
            dynSizes = a.getDynamicSizes();
            symOps = a.getSymbolOperands();
        } else {
            auto aa = cast<memref::AllocaOp>(op);
            dynSizes = aa.getDynamicSizes();
            symOps = aa.getSymbolOperands();
        }
        bool hostUse = llvm::any_of(op->getUsers(), [](Operation* u) {
            return !isa<memref::DeallocOp>(u) && !u->getParentOfType<gpu::LaunchOp>();
        });
        Value mem;
        if (hostUse) {
            mem = gpu::AllocOp::create(b, op->getLoc(), type, /*asyncToken=*/Type(), ValueRange{},
                                       dynSizes, symOps, /*hostShared=*/true)
                      .getMemref();
        } else {
            Value t0 = gpu::WaitOp::create(b, op->getLoc(), tokTy, ValueRange{}).getAsyncToken();
            auto ga = gpu::AllocOp::create(b, op->getLoc(), type, tokTy, ValueRange{t0}, dynSizes,
                                           symOps, /*hostShared=*/false);
            gpu::WaitOp::create(b, op->getLoc(), Type(), ValueRange{ga.getAsyncToken()});
            mem = ga.getMemref();
        }
        op->getResult(0).replaceAllUsesWith(mem);
        // allocas have no dealloc: free them before the function returns
        if (isa<memref::AllocaOp>(op)) {
            auto func = mem.getParentRegion()->getParentOfType<func::FuncOp>();
            func.walk([&](func::ReturnOp ret) {
                b.setInsertionPoint(ret);
                emitDealloc(ret.getLoc(), mem);
            });
        }
        op->erase();
    }
    for (memref::DeallocOp op : deallocs) {
        b.setInsertionPoint(op);
        emitDealloc(op.getLoc(), op.getMemref());
        op.erase();
    }

    // 4. NVVM pipeline: kernels embed PTX (driver JITs it); host uses mgpu* calls.
    jitMark("gpu-map");
    PassManager pm2(&context);
    gpu::GPUToNVVMPipelineOptions opts;
    opts.cubinChip = "sm_80";
    opts.cubinFeatures = "+ptx80";
    opts.cubinFormat = "isa";
    gpu::buildLowerToNVVMPassPipeline(pm2, opts);
    if (failed(pm2.run(*module))) {
        if (getenv("CNUM_JIT_DUMP"))
            module->dump();
        return false;
    }
    jitMark("nvvm");
    return true;
}

namespace {
// Redirects the launcher's mgpu* stream/alloc calls to the calling chare's
// compute stream, without host syncs.
thread_local CUstream jitStream = nullptr;

extern "C" CUstream cn_mgpuStreamCreate() { return jitStream; }
extern "C" void cn_mgpuStreamSynchronize(CUstream s) {
    if (s != jitStream)
        cuStreamSynchronize(s);
}
extern "C" void cn_mgpuStreamDestroy(CUstream) {}
extern "C" void* cn_mgpuMemAlloc(uint64_t bytes, CUstream s, bool hostShared) {
    CUdeviceptr p = 0;
    if (bytes == 0)
        return nullptr;
    CUresult r = hostShared ? cuMemAllocManaged(&p, bytes, CU_MEM_ATTACH_GLOBAL)
                            : cuMemAllocAsync(&p, bytes, s);
    if (r != CUDA_SUCCESS)
        llvm::report_fatal_error(llvm::Twine("JIT temporary allocation failed: ") + llvm::Twine(r));
    return reinterpret_cast<void*>(p);
}
extern "C" void cn_mgpuMemFree(void* ptr, CUstream s) {
    if (!ptr)
        return;
    unsigned managed = 0;
    auto p = reinterpret_cast<CUdeviceptr>(ptr);
    cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, p);
    if (managed) {
        cuStreamSynchronize(s);
        cuMemFree(p);
    } else {
        cuMemFreeAsync(p, s);
    }
}

llvm::orc::SymbolMap streamRuntimeSymbols(llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap map = halfRuntimeSymbols(interner);
    auto add = [&](const char* name, void* fn) {
        map[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(fn),
                               llvm::JITSymbolFlags::Exported};
    };
    add("mgpuStreamCreate", (void*)&cn_mgpuStreamCreate);
    add("mgpuStreamSynchronize", (void*)&cn_mgpuStreamSynchronize);
    add("mgpuStreamDestroy", (void*)&cn_mgpuStreamDestroy);
    add("mgpuMemAlloc", (void*)&cn_mgpuMemAlloc);
    add("mgpuMemFree", (void*)&cn_mgpuMemFree);
    return map;
}
} // namespace

void jit_set_stream(void* stream) { jitStream = static_cast<CUstream>(stream); }

std::unique_ptr<mlir::ExecutionEngine> MLIRJitCompiler::generateNVIDIAHost() {
    if (!lowerToNVIDIAHost()) {
        llvm::errs() << "generateNVIDIAHost: GPU lowering failed\n";
        return nullptr;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::ExecutionEngineOptions engineOptions;
    static const llvm::StringRef runtimeLibs[] = {MLIR_CUDA_RUNTIME_LIB};
    engineOptions.sharedLibPaths = runtimeLibs;
    // Host only launches: O0. jitCodeGenOptLevel alone is ignored; the target machine counts.
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
    std::unique_ptr<llvm::TargetMachine> tm;
    if (auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost()) {
        jtmb->setCodeGenOptLevel(llvm::CodeGenOptLevel::None);
        if (auto t = jtmb->createTargetMachine())
            tm = std::move(*t);
        else
            llvm::consumeError(t.takeError());
    } else {
        llvm::consumeError(jtmb.takeError());
    }
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions, std::move(tm));
    if (!maybeEngine) {
        llvm::errs() << "generateNVIDIAHost: failed to create ExecutionEngine: "
                     << llvm::toString(maybeEngine.takeError()) << "\n";
        return nullptr;
    }
    (*maybeEngine)->registerSymbols(streamRuntimeSymbols);
    // Runs ctors that load the embedded PTX; we call the packed pointer, not invokePacked().
    jitMark("engine");
    (*maybeEngine)->initialize();
    jitMark("init");
    return std::move(*maybeEngine);
}

#endif // USE_NVIDIA

// ---- Elementwise op registries ----

namespace {
using mlir::Location;
using mlir::OpBuilder;
using mlir::Value;
namespace arith = mlir::arith;
namespace math = mlir::math;

/// Constant of the same (scalar) type as `like`.
Value constLike(OpBuilder& b, Location loc, Value like, double v) {
    mlir::Type t = like.getType();
    mlir::TypedAttr attr;
    if (mlir::isa<mlir::FloatType>(t))
        attr = b.getFloatAttr(t, v);
    else
        attr = b.getIntegerAttr(t, static_cast<int64_t>(v));
    return arith::ConstantOp::create(b, loc, attr).getResult();
}

Value fcmp(OpBuilder& b, Location loc, arith::CmpFPredicate p, Value x, Value y) {
    return arith::CmpFOp::create(b, loc, p, x, y).getResult();
}

Value icmp(OpBuilder& b, Location loc, arith::CmpIPredicate p, Value x, Value y) {
    return arith::CmpIOp::create(b, loc, p, x, y).getResult();
}

Value sel(OpBuilder& b, Location loc, Value c, Value t, Value f) {
    return arith::SelectOp::create(b, loc, c, t, f).getResult();
}

/// i1: v != 0 (NaN counts as true, like NumPy's bool(nan)).
Value isNonzero(OpBuilder& b, Location loc, Value v) {
    if (v.getType().isInteger(1))
        return v;
    Value zero = constLike(b, loc, v, 0.0);
    if (mlir::isa<mlir::FloatType>(v.getType()))
        return fcmp(b, loc, arith::CmpFPredicate::UNE, v, zero);
    return icmp(b, loc, arith::CmpIPredicate::ne, v, zero);
}

/// Float compare op (i1 result).
BinaryOpEntry::Fn fcmpPred(arith::CmpFPredicate p) {
    return [p](OpBuilder& b, Location loc, Value x, Value y) { return fcmp(b, loc, p, x, y); };
}

/// Integer compare op (i1 result).
BinaryOpEntry::Fn icmpPred(arith::CmpIPredicate p) {
    return [p](OpBuilder& b, Location loc, Value x, Value y) { return icmp(b, loc, p, x, y); };
}

/// Integer x**e by square-and-multiply over every exponent bit (branch-free,
/// so it lowers the same on CPU and GPU). Wraps on overflow like NumPy. For
/// signed e < 0 the result is 1 for x == 1, +-1 for x == -1 and 0 otherwise
/// (NumPy raises instead).
Value intPow(OpBuilder& b, Location loc, Value x, Value e, bool isUnsigned = false) {
    unsigned bits = mlir::cast<mlir::IntegerType>(x.getType()).getWidth();
    Value one = constLike(b, loc, x, 1.0);
    Value zero = constLike(b, loc, x, 0.0);
    Value r = one, base = x;
    unsigned nbits = isUnsigned ? bits : bits - 1;
    for (unsigned i = 0; i < nbits; ++i) {
        Value sh = arith::ShRUIOp::create(b, loc, e, constLike(b, loc, e, i)).getResult();
        Value bit = arith::TruncIOp::create(b, loc, b.getI1Type(), sh).getResult();
        r = sel(b, loc, bit, arith::MulIOp::create(b, loc, r, base).getResult(), r);
        if (i + 1 < nbits)
            base = arith::MulIOp::create(b, loc, base, base).getResult();
    }
    if (isUnsigned)
        return r;
    Value eOdd = arith::TruncIOp::create(b, loc, b.getI1Type(), e).getResult();
    Value minusOne = constLike(b, loc, x, -1.0);
    Value negRes = sel(b, loc, icmp(b, loc, arith::CmpIPredicate::eq, x, one), one,
                       sel(b, loc, icmp(b, loc, arith::CmpIPredicate::eq, x, minusOne),
                           sel(b, loc, eOdd, minusOne, one), zero));
    return sel(b, loc, icmp(b, loc, arith::CmpIPredicate::slt, e, zero), negRes, r);
}

/// Integer floor division / remainder with NumPy semantics (floor rounding,
/// remainder has the sign of the divisor). Division by zero gives 0 (NumPy
/// warns and returns 0); b == -1 is special-cased so INT_MIN // -1 wraps
/// instead of trapping.
Value intDivMod(OpBuilder& b, Location loc, Value x, Value y, bool wantMod) {
    Value zero = constLike(b, loc, x, 0.0);
    Value one = constLike(b, loc, x, 1.0);
    Value minusOne = constLike(b, loc, x, -1.0);
    Value yZero = icmp(b, loc, arith::CmpIPredicate::eq, y, zero);
    Value yNegOne = icmp(b, loc, arith::CmpIPredicate::eq, y, minusOne);
    Value bad = arith::OrIOp::create(b, loc, yZero, yNegOne).getResult();
    Value ySafe = sel(b, loc, bad, one, y);
    Value q = arith::DivSIOp::create(b, loc, x, ySafe).getResult();
    Value r = arith::RemSIOp::create(b, loc, x, ySafe).getResult();
    Value rNonzero = icmp(b, loc, arith::CmpIPredicate::ne, r, zero);
    Value signDiffers =
        arith::XOrIOp::create(b, loc, icmp(b, loc, arith::CmpIPredicate::slt, r, zero),
                              icmp(b, loc, arith::CmpIPredicate::slt, y, zero))
            .getResult();
    Value fix = arith::AndIOp::create(b, loc, rNonzero, signDiffers).getResult();
    if (wantMod) {
        Value rFixed = sel(b, loc, fix, arith::AddIOp::create(b, loc, r, y).getResult(), r);
        return sel(b, loc, bad, zero, rFixed);
    }
    Value qFixed = sel(b, loc, fix, arith::SubIOp::create(b, loc, q, one).getResult(), q);
    Value negX = arith::SubIOp::create(b, loc, zero, x).getResult();
    return sel(b, loc, yZero, zero, sel(b, loc, yNegOne, negX, qFixed));
}

/// C fmod. arith.remf is unusable (llvm.frem is illegal in the NVVM pipeline), so
/// this calls libm fmod on host and libdevice __nv_fmod on NVIDIA.
Value fmodCall(OpBuilder& b, Location loc, Value x, Value y) {
    mlir::Type t = x.getType();
    bool isF32 = t.isF32();
#ifdef USE_NVIDIA
    llvm::StringRef name = isF32 ? "__nv_fmodf" : "__nv_fmod";
#else
    llvm::StringRef name = isF32 ? "fmodf" : "fmod";
#endif
    return mlir::func::CallOp::create(b, loc, name, mlir::TypeRange{t}, mlir::ValueRange{x, y})
        .getResult(0);
}

/// Declares private callees (e.g. fmod) that are called but not defined.
void declareExternalCallees(mlir::ModuleOp module) {
    OpBuilder b(module.getContext());
    b.setInsertionPointToStart(module.getBody());
    module.walk([&](mlir::func::CallOp call) {
        if (module.lookupSymbol(call.getCalleeAttr()))
            return;
        auto fn = mlir::func::FuncOp::create(b, call.getLoc(), call.getCallee(),
                                             call.getCalleeType());
        fn.setPrivate();
    });
}

/// Float remainder with NumPy semantics (npy_remainder): fmod, then shift
/// into the divisor's sign; a zero result takes the divisor's sign.
Value floatMod(OpBuilder& b, Location loc, Value x, Value y) {
    Value zero = constLike(b, loc, x, 0.0);
    Value m = fmodCall(b, loc, x, y);
    Value mNonzero = fcmp(b, loc, arith::CmpFPredicate::UNE, m, zero);
    Value signDiffers =
        arith::XOrIOp::create(b, loc, fcmp(b, loc, arith::CmpFPredicate::OLT, y, zero),
                              fcmp(b, loc, arith::CmpFPredicate::OLT, m, zero))
            .getResult();
    Value fix = arith::AndIOp::create(b, loc, mNonzero, signDiffers).getResult();
    Value mFixed = sel(b, loc, fix, arith::AddFOp::create(b, loc, m, y).getResult(), m);
    Value signedZero = math::CopySignOp::create(b, loc, zero, y).getResult();
    return sel(b, loc, mNonzero, mFixed, signedZero);
}

/// Float floor division with NumPy semantics (npy_floor_divide / npy_divmod).
Value floatFloorDiv(OpBuilder& b, Location loc, Value x, Value y) {
    Value zero = constLike(b, loc, x, 0.0);
    Value one = constLike(b, loc, x, 1.0);
    Value half = constLike(b, loc, x, 0.5);
    Value quot = arith::DivFOp::create(b, loc, x, y).getResult();
    Value m = fmodCall(b, loc, x, y);
    Value div = arith::DivFOp::create(b, loc, arith::SubFOp::create(b, loc, x, m), y).getResult();
    Value mNonzero = fcmp(b, loc, arith::CmpFPredicate::UNE, m, zero);
    Value signDiffers =
        arith::XOrIOp::create(b, loc, fcmp(b, loc, arith::CmpFPredicate::OLT, y, zero),
                              fcmp(b, loc, arith::CmpFPredicate::OLT, m, zero))
            .getResult();
    Value fix = arith::AndIOp::create(b, loc, mNonzero, signDiffers).getResult();
    div = sel(b, loc, fix, arith::SubFOp::create(b, loc, div, one).getResult(), div);
    Value fl = math::FloorOp::create(b, loc, div).getResult();
    Value roundUp =
        fcmp(b, loc, arith::CmpFPredicate::OGT, arith::SubFOp::create(b, loc, div, fl), half);
    fl = sel(b, loc, roundUp, arith::AddFOp::create(b, loc, fl, one).getResult(), fl);
    Value signedZero = math::CopySignOp::create(b, loc, zero, quot).getResult();
    Value res = sel(b, loc, fcmp(b, loc, arith::CmpFPredicate::UNE, div, zero), fl, signedZero);
    return sel(b, loc, fcmp(b, loc, arith::CmpFPredicate::OEQ, y, zero), quot, res);
}

/// log1p(x) = log(u) * x / (u - 1) with u = 1 + x (Goldberg); exact for tiny x.
/// The CPU math-to-llvm lowering of math.log1p is the naive log(1 + x).
Value log1pAccurate(OpBuilder& b, Location loc, Value x) {
    Value one = constLike(b, loc, x, 1.0);
    Value inf = constLike(b, loc, x, std::numeric_limits<double>::infinity());
    Value u = arith::AddFOp::create(b, loc, one, x).getResult();
    Value logU = math::LogOp::create(b, loc, u).getResult();
    Value ratio =
        arith::DivFOp::create(b, loc, x, arith::SubFOp::create(b, loc, u, one)).getResult();
    Value general = arith::MulFOp::create(b, loc, logU, ratio).getResult();
    Value r = sel(b, loc, fcmp(b, loc, arith::CmpFPredicate::OEQ, u, inf), u, general);
    return sel(b, loc, fcmp(b, loc, arith::CmpFPredicate::OEQ, u, one), x, r);
}

/// expm1(x) = (u - 1) * x / log(u) with u = exp(x) (Kahan); exact for tiny x.
/// The CPU math-to-llvm lowering of math.expm1 is the naive exp(x) - 1.
Value expm1Accurate(OpBuilder& b, Location loc, Value x) {
    Value one = constLike(b, loc, x, 1.0);
    Value minusOne = constLike(b, loc, x, -1.0);
    Value inf = constLike(b, loc, x, std::numeric_limits<double>::infinity());
    Value u = math::ExpOp::create(b, loc, x).getResult();
    Value um1 = arith::SubFOp::create(b, loc, u, one).getResult();
    Value ratio =
        arith::DivFOp::create(b, loc, x, math::LogOp::create(b, loc, u)).getResult();
    Value r = arith::MulFOp::create(b, loc, um1, ratio).getResult();
    r = sel(b, loc, fcmp(b, loc, arith::CmpFPredicate::OEQ, u, inf), u, r);
    r = sel(b, loc, fcmp(b, loc, arith::CmpFPredicate::OEQ, um1, minusOne), minusOne, r);
    return sel(b, loc, fcmp(b, loc, arith::CmpFPredicate::OEQ, u, one), x, r);
}

template <typename OpT>
UnaryOpEntry::Fn mathUnary() {
    return [](OpBuilder& b, Location loc, Value v) { return OpT::create(b, loc, v).getResult(); };
}

template <typename OpT>
BinaryOpEntry::Fn arithBinary() {
    return [](OpBuilder& b, Location loc, Value x, Value y) {
        return OpT::create(b, loc, x, y).getResult();
    };
}

Value identityUnary(OpBuilder&, Location, Value v) { return v; }

/// Integer shift with NumPy semantics: a count outside [0, width) gives 0
/// (sign fill for signed >>) instead of an undefined shift.
Value intShift(OpBuilder& b, Location loc, Value x, Value y, bool left, bool isSigned) {
    unsigned bits = mlir::cast<mlir::IntegerType>(x.getType()).getWidth();
    Value inRange = icmp(b, loc, arith::CmpIPredicate::ult, y, constLike(b, loc, y, bits));
    Value cnt = sel(b, loc, inRange, y, constLike(b, loc, y, 0.0));
    if (left)
        return sel(b, loc, inRange, arith::ShLIOp::create(b, loc, x, cnt).getResult(),
                   constLike(b, loc, x, 0.0));
    if (!isSigned)
        return sel(b, loc, inRange, arith::ShRUIOp::create(b, loc, x, cnt).getResult(),
                   constLike(b, loc, x, 0.0));
    Value sign = arith::ShRSIOp::create(b, loc, x, constLike(b, loc, x, bits - 1)).getResult();
    return sel(b, loc, inRange, arith::ShRSIOp::create(b, loc, x, cnt).getResult(), sign);
}

/// Unsigned floor division / remainder; division by zero gives 0 (as NumPy).
Value uintDivMod(OpBuilder& b, Location loc, Value x, Value y, bool wantMod) {
    Value zero = constLike(b, loc, x, 0.0);
    Value yZero = icmp(b, loc, arith::CmpIPredicate::eq, y, zero);
    Value ySafe = sel(b, loc, yZero, constLike(b, loc, x, 1.0), y);
    Value r = wantMod ? arith::RemUIOp::create(b, loc, x, ySafe).getResult()
                      : arith::DivUIOp::create(b, loc, x, ySafe).getResult();
    return sel(b, loc, yZero, zero, r);
}

// ---- complex helpers (complex<f32|f64> values) ----

Value re(OpBuilder& b, Location loc, Value z) {
    return mlir::complex::ReOp::create(b, loc, z).getResult();
}
Value im(OpBuilder& b, Location loc, Value z) {
    return mlir::complex::ImOp::create(b, loc, z).getResult();
}
Value mkComplex(OpBuilder& b, Location loc, mlir::Type t, Value r, Value i) {
    return mlir::complex::CreateOp::create(b, loc, t, r, i).getResult();
}
Value fisnan(OpBuilder& b, Location loc, Value x) {
    return fcmp(b, loc, arith::CmpFPredicate::UNO, x, x);
}
Value andI(OpBuilder& b, Location loc, Value x, Value y) {
    return arith::AndIOp::create(b, loc, x, y).getResult();
}
Value orI(OpBuilder& b, Location loc, Value x, Value y) {
    return arith::OrIOp::create(b, loc, x, y).getResult();
}
Value notI(OpBuilder& b, Location loc, Value x) {
    return arith::XOrIOp::create(b, loc, x, constLike(b, loc, x, 1.0)).getResult();
}
/// select on complex values, part by part.
Value selC(OpBuilder& b, Location loc, Value c, Value x, Value y) {
    return mkComplex(b, loc, x.getType(), sel(b, loc, c, re(b, loc, x), re(b, loc, y)),
                     sel(b, loc, c, im(b, loc, x), im(b, loc, y)));
}
/// z with both parts scaled by s.
Value scaleC(OpBuilder& b, Location loc, Value z, double s) {
    Value r = re(b, loc, z), i = im(b, loc, z);
    Value k = constLike(b, loc, r, s);
    return mkComplex(b, loc, z.getType(), arith::MulFOp::create(b, loc, r, k).getResult(),
                     arith::MulFOp::create(b, loc, i, k).getResult());
}
/// NumPy's complex orderings: CLT/CLE are lexicographic and false on NaN;
/// CGE/CLE with NaN guards select for maximum/minimum.
Value clt(OpBuilder& b, Location loc, Value x, Value y, bool orEqual) {
    Value xr = re(b, loc, x), xi = im(b, loc, x), yr = re(b, loc, y), yi = im(b, loc, y);
    return orI(b, loc, fcmp(b, loc, arith::CmpFPredicate::OLT, xr, yr),
               andI(b, loc, fcmp(b, loc, arith::CmpFPredicate::OEQ, xr, yr),
                    fcmp(b, loc, orEqual ? arith::CmpFPredicate::OLE : arith::CmpFPredicate::OLT,
                         xi, yi)));
}
Value cAnyNan(OpBuilder& b, Location loc, Value z) {
    return orI(b, loc, fisnan(b, loc, re(b, loc, z)), fisnan(b, loc, im(b, loc, z)));
}
/// x >= y (ge) or x <= y (!ge) as in NumPy's CGE/CLE.
Value cge(OpBuilder& b, Location loc, Value x, Value y, bool ge) {
    Value xr = re(b, loc, x), xi = im(b, loc, x), yr = re(b, loc, y), yi = im(b, loc, y);
    Value strict = fcmp(b, loc, ge ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OLT, xr, yr);
    Value noNan = notI(b, loc, orI(b, loc, fisnan(b, loc, xi), fisnan(b, loc, yi)));
    Value eq = andI(b, loc, fcmp(b, loc, arith::CmpFPredicate::OEQ, xr, yr),
                    fcmp(b, loc, ge ? arith::CmpFPredicate::OGE : arith::CmpFPredicate::OLE, xi, yi));
    return orI(b, loc, andI(b, loc, strict, noNan), eq);
}
/// maximum/minimum (propagate NaN of x first) and fmax/fmin (ignore NaN of y).
BinaryOpEntry::Fn cMinMax(bool max, bool fm) {
    return [max, fm](OpBuilder& b, Location loc, Value x, Value y) {
        Value nan = cAnyNan(b, loc, fm ? y : x);
        return selC(b, loc, orI(b, loc, nan, cge(b, loc, x, y, max)), x, y);
    };
}
/// Predicate on both parts of z combined with and/or.
UnaryOpEntry::Fn cParts(UnaryOpEntry::Fn realPred, bool both) {
    return [realPred, both](OpBuilder& b, Location loc, Value z) {
        Value pr = realPred(b, loc, re(b, loc, z)), pi = realPred(b, loc, im(b, loc, z));
        return both ? andI(b, loc, pr, pi) : orI(b, loc, pr, pi);
    };
}
Value fIsInf(OpBuilder& b, Location loc, Value v) {
    Value a = math::AbsFOp::create(b, loc, v).getResult();
    Value inf = constLike(b, loc, v, std::numeric_limits<double>::infinity());
    return fcmp(b, loc, arith::CmpFPredicate::OEQ, a, inf);
}
Value fIsFinite(OpBuilder& b, Location loc, Value v) {
    Value a = math::AbsFOp::create(b, loc, v).getResult();
    Value inf = constLike(b, loc, v, std::numeric_limits<double>::infinity());
    return fcmp(b, loc, arith::CmpFPredicate::OLT, a, inf);
}
Value constI1(OpBuilder& b, Location loc, bool v) {
    return arith::ConstantOp::create(b, loc, b.getBoolAttr(v)).getResult();
}

template <typename OpT>
UnaryOpEntry::Fn complexUnary() {
    return [](OpBuilder& b, Location loc, Value z) { return OpT::create(b, loc, z).getResult(); };
}
mlir::Type storageType(mlir::MLIRContext* ctx, DType dt) {
    switch (dt) {
    case DType::FLOAT32: return mlir::Float32Type::get(ctx);
    case DType::FLOAT64: return mlir::Float64Type::get(ctx);
    case DType::FLOAT16: return mlir::Float16Type::get(ctx);
    case DType::BOOL:
    case DType::INT8:
    case DType::UINT8: return mlir::IntegerType::get(ctx, 8);
    case DType::INT16:
    case DType::UINT16: return mlir::IntegerType::get(ctx, 16);
    case DType::INT32:
    case DType::UINT32: return mlir::IntegerType::get(ctx, 32);
    case DType::INT64:
    case DType::UINT64: return mlir::IntegerType::get(ctx, 64);
    case DType::COMPLEX64: return mlir::ComplexType::get(mlir::Float32Type::get(ctx));
    case DType::COMPLEX128: return mlir::ComplexType::get(mlir::Float64Type::get(ctx));
    default: return mlir::IntegerType::get(ctx, 1); // DT_PRED
    }
}

/// Convert an element from dtype `from` to `to` with NumPy astype semantics
/// (x -> bool is x != 0, complex -> real takes the real part, float -> int
/// truncates). DT_PRED (i1) converts like bool.
Value convertValue(OpBuilder& b, Location loc, Value v, DType from, DType to) {
    if (from == to)
        return v;
    mlir::MLIRContext* ctx = b.getContext();
    mlir::Type tt = storageType(ctx, to);
    bool fromC = dtype_is_complex(from), toC = dtype_is_complex(to);
    if (to == DType::BOOL || to == DT_PRED) {
        Value p = fromC ? orI(b, loc, isNonzero(b, loc, re(b, loc, v)), isNonzero(b, loc, im(b, loc, v)))
                        : isNonzero(b, loc, v);
        return to == DT_PRED ? p : arith::ExtUIOp::create(b, loc, tt, p).getResult();
    }
    if (fromC && toC) {
        DType fr = realOf(from), tr = realOf(to);
        return mkComplex(b, loc, tt, convertValue(b, loc, re(b, loc, v), fr, tr),
                         convertValue(b, loc, im(b, loc, v), fr, tr));
    }
    if (fromC)
        return convertValue(b, loc, re(b, loc, v), realOf(from), to);
    if (toC) {
        DType tr = realOf(to);
        Value r = convertValue(b, loc, v, from, tr);
        return mkComplex(b, loc, tt, r, constLike(b, loc, r, 0.0));
    }
    bool fromF = dtype_is_float(from), toF = dtype_is_float(to);
    bool fromU = dtype_is_unsigned(from) || from == DType::BOOL || from == DT_PRED;
    unsigned fw = v.getType().getIntOrFloatBitWidth(), tw = tt.getIntOrFloatBitWidth();
    if (fromF && toF) {
        if (fw < tw)
            return arith::ExtFOp::create(b, loc, tt, v).getResult();
        if (fw > tw)
            return arith::TruncFOp::create(b, loc, tt, v).getResult();
        return v;
    }
    if (!fromF && !toF) {
        if (fw < tw)
            return fromU ? arith::ExtUIOp::create(b, loc, tt, v).getResult()
                         : arith::ExtSIOp::create(b, loc, tt, v).getResult();
        if (fw > tw)
            return arith::TruncIOp::create(b, loc, tt, v).getResult();
        return v;
    }
    if (toF)
        return fromU ? arith::UIToFPOp::create(b, loc, tt, v).getResult()
                     : arith::SIToFPOp::create(b, loc, tt, v).getResult();
    // float -> int: through int64 (wraps like a C cast on x86); uint64 values
    // >= 2^63 need fptoui
    Value i64 = arith::FPToSIOp::create(b, loc, b.getI64Type(), v).getResult();
    if (to == DType::UINT64) {
        Value big = fcmp(b, loc, arith::CmpFPredicate::OGE, v, constLike(b, loc, v, 9223372036854775808.0));
        i64 = sel(b, loc, big, arith::FPToUIOp::create(b, loc, b.getI64Type(), v).getResult(), i64);
    }
    return tw < 64 ? arith::TruncIOp::create(b, loc, tt, i64).getResult() : i64;
}
} // namespace

const std::unordered_map<int, BinaryOpEntry>& MLIRJitCompiler::binaryOpRegistry() {
    namespace cx = mlir::complex;
    using P = arith::CmpIPredicate;
    static const std::unordered_map<int, BinaryOpEntry> registry = {
        // {float, int, uint, complex, bool}; missing uint/bool bodies use int
        {(int)Opcode::ADD,
         {arithBinary<arith::AddFOp>(), arithBinary<arith::AddIOp>(), nullptr,
          arithBinary<cx::AddOp>()}},
        {(int)Opcode::SUB,
         {arithBinary<arith::SubFOp>(), arithBinary<arith::SubIOp>(), nullptr,
          arithBinary<cx::SubOp>()}},
        {(int)Opcode::MUL,
         {arithBinary<arith::MulFOp>(), arithBinary<arith::MulIOp>(), nullptr,
          arithBinary<cx::MulOp>()}},
        {(int)Opcode::DIV,
         {arithBinary<arith::DivFOp>(), arithBinary<arith::DivSIOp>(),
          arithBinary<arith::DivUIOp>(), arithBinary<cx::DivOp>()}},
        {(int)Opcode::POW,
         {arithBinary<math::PowFOp>(),
          [](OpBuilder& b, Location loc, Value x, Value e) { return intPow(b, loc, x, e); },
          [](OpBuilder& b, Location loc, Value x, Value e) { return intPow(b, loc, x, e, true); },
          arithBinary<cx::PowOp>()}},
        {(int)Opcode::MAXIMUM,
         {arithBinary<arith::MaximumFOp>(), arithBinary<arith::MaxSIOp>(),
          arithBinary<arith::MaxUIOp>(), cMinMax(true, false)}},
        {(int)Opcode::MINIMUM,
         {arithBinary<arith::MinimumFOp>(), arithBinary<arith::MinSIOp>(),
          arithBinary<arith::MinUIOp>(), cMinMax(false, false)}},
        {(int)Opcode::FMAX,
         {arithBinary<arith::MaxNumFOp>(), arithBinary<arith::MaxSIOp>(),
          arithBinary<arith::MaxUIOp>(), cMinMax(true, true)}},
        {(int)Opcode::FMIN,
         {arithBinary<arith::MinNumFOp>(), arithBinary<arith::MinSIOp>(),
          arithBinary<arith::MinUIOp>(), cMinMax(false, true)}},
        {(int)Opcode::MOD,
         {floatMod,
          [](OpBuilder& b, Location loc, Value x, Value y) { return intDivMod(b, loc, x, y, true); },
          [](OpBuilder& b, Location loc, Value x, Value y) { return uintDivMod(b, loc, x, y, true); }}},
        {(int)Opcode::FLOOR_DIV,
         {floatFloorDiv,
          [](OpBuilder& b, Location loc, Value x, Value y) { return intDivMod(b, loc, x, y, false); },
          [](OpBuilder& b, Location loc, Value x, Value y) {
              return uintDivMod(b, loc, x, y, false);
          }}},
        // Comparisons (i1): ordered for <, <=, == (NaN -> false), unordered for !=;
        // complex < and <= are lexicographic
        {(int)Opcode::LESS,
         {fcmpPred(arith::CmpFPredicate::OLT), icmpPred(P::slt), icmpPred(P::ult),
          [](OpBuilder& b, Location loc, Value x, Value y) { return clt(b, loc, x, y, false); }}},
        {(int)Opcode::LESS_EQUAL,
         {fcmpPred(arith::CmpFPredicate::OLE), icmpPred(P::sle), icmpPred(P::ule),
          [](OpBuilder& b, Location loc, Value x, Value y) { return clt(b, loc, x, y, true); }}},
        {(int)Opcode::EQUAL,
         {fcmpPred(arith::CmpFPredicate::OEQ), icmpPred(P::eq), nullptr,
          arithBinary<cx::EqualOp>()}},
        {(int)Opcode::NOT_EQUAL,
         {fcmpPred(arith::CmpFPredicate::UNE), icmpPred(P::ne), nullptr,
          arithBinary<cx::NotEqualOp>()}},
        // Bitwise ops: integers and bool only
        {(int)Opcode::BITWISE_AND, {nullptr, arithBinary<arith::AndIOp>()}},
        {(int)Opcode::BITWISE_OR, {nullptr, arithBinary<arith::OrIOp>()}},
        {(int)Opcode::BITWISE_XOR, {nullptr, arithBinary<arith::XOrIOp>()}},
        {(int)Opcode::LEFT_SHIFT,
         {nullptr,
          [](OpBuilder& b, Location loc, Value x, Value y) { return intShift(b, loc, x, y, true, true); },
          [](OpBuilder& b, Location loc, Value x, Value y) { return intShift(b, loc, x, y, true, false); }}},
        {(int)Opcode::RIGHT_SHIFT,
         {nullptr,
          [](OpBuilder& b, Location loc, Value x, Value y) { return intShift(b, loc, x, y, false, true); },
          [](OpBuilder& b, Location loc, Value x, Value y) { return intShift(b, loc, x, y, false, false); }}},
    };
    return registry;
}

const std::unordered_map<int, UnaryOpEntry>& MLIRJitCompiler::unaryOpRegistry() {
    namespace cx = mlir::complex;
    static const std::unordered_map<int, UnaryOpEntry> registry = {
        // {float, int, uint, complex, bool}. Float-only math has no int body:
        // the frontend gives integer inputs a float result dtype (as NumPy).
        {(int)Opcode::TANH, {mathUnary<math::TanhOp>(), nullptr, nullptr, complexUnary<cx::TanhOp>()}},
        {(int)Opcode::EXP, {mathUnary<math::ExpOp>(), nullptr, nullptr, complexUnary<cx::ExpOp>()}},
        {(int)Opcode::SQRT, {mathUnary<math::SqrtOp>(), nullptr, nullptr, complexUnary<cx::SqrtOp>()}},
        {(int)Opcode::LOG, {mathUnary<math::LogOp>(), nullptr, nullptr, complexUnary<cx::LogOp>()}},
        {(int)Opcode::LOG2,
         {mathUnary<math::Log2Op>(), nullptr, nullptr,
          [](OpBuilder& b, Location loc, Value z) {
              return scaleC(b, loc, cx::LogOp::create(b, loc, z).getResult(), 1.0 / M_LN2);
          }}},
        {(int)Opcode::LOG10,
         {mathUnary<math::Log10Op>(), nullptr, nullptr,
          [](OpBuilder& b, Location loc, Value z) {
              return scaleC(b, loc, cx::LogOp::create(b, loc, z).getResult(), 1.0 / M_LN10);
          }}},
        {(int)Opcode::LOG1P, {log1pAccurate, nullptr, nullptr, complexUnary<cx::Log1pOp>()}},
        {(int)Opcode::EXP2,
         {mathUnary<math::Exp2Op>(), nullptr, nullptr,
          [](OpBuilder& b, Location loc, Value z) {
              return cx::ExpOp::create(b, loc, scaleC(b, loc, z, M_LN2)).getResult();
          }}},
        {(int)Opcode::EXPM1, {expm1Accurate, nullptr, nullptr, complexUnary<cx::Expm1Op>()}},
        {(int)Opcode::SIN, {mathUnary<math::SinOp>(), nullptr, nullptr, complexUnary<cx::SinOp>()}},
        {(int)Opcode::COS, {mathUnary<math::CosOp>(), nullptr, nullptr, complexUnary<cx::CosOp>()}},
        // Integer inputs keep their dtype (rounding is the identity on ints);
        // complex abs is real (the result converts to the node dtype)
        {(int)Opcode::ABS,
         {mathUnary<math::AbsFOp>(), mathUnary<math::AbsIOp>(), identityUnary,
          [](OpBuilder& b, Location loc, Value z) {
              // hypot semantics: an infinite part gives inf even if the other is NaN
              Value a = cx::AbsOp::create(b, loc, z).getResult();
              Value inf = orI(b, loc, fIsInf(b, loc, re(b, loc, z)), fIsInf(b, loc, im(b, loc, z)));
              return sel(b, loc, inf, constLike(b, loc, a, std::numeric_limits<double>::infinity()),
                         a);
          }}},
        {(int)Opcode::FLOOR, {mathUnary<math::FloorOp>(), identityUnary}},
        {(int)Opcode::CEIL, {mathUnary<math::CeilOp>(), identityUnary}},
        {(int)Opcode::RINT,
         {mathUnary<math::RoundEvenOp>(), identityUnary, nullptr,
          [](OpBuilder& b, Location loc, Value z) {
              return mkComplex(b, loc, z.getType(),
                               math::RoundEvenOp::create(b, loc, re(b, loc, z)).getResult(),
                               math::RoundEvenOp::create(b, loc, im(b, loc, z)).getResult());
          }}},
        {(int)Opcode::SIGN,
         {[](OpBuilder& b, Location loc, Value v) {
              // sign(+-0) = +-0 and sign(NaN) = NaN, as in NumPy
              Value zero = constLike(b, loc, v, 0.0);
              Value pos = fcmp(b, loc, arith::CmpFPredicate::OGT, v, zero);
              Value neg = fcmp(b, loc, arith::CmpFPredicate::OLT, v, zero);
              return sel(b, loc, pos, constLike(b, loc, v, 1.0),
                         sel(b, loc, neg, constLike(b, loc, v, -1.0), v));
          },
          [](OpBuilder& b, Location loc, Value v) {
              Value zero = constLike(b, loc, v, 0.0);
              Value pos = icmp(b, loc, arith::CmpIPredicate::sgt, v, zero);
              Value neg = icmp(b, loc, arith::CmpIPredicate::slt, v, zero);
              return sel(b, loc, pos, constLike(b, loc, v, 1.0),
                         sel(b, loc, neg, constLike(b, loc, v, -1.0), zero));
          },
          [](OpBuilder& b, Location loc, Value v) {
              Value zero = constLike(b, loc, v, 0.0);
              return sel(b, loc, icmp(b, loc, arith::CmpIPredicate::ne, v, zero),
                         constLike(b, loc, v, 1.0), zero);
          },
          complexUnary<cx::SignOp>()}},
        // Predicates (i1)
        {(int)Opcode::ISNAN,
         {fisnan, [](OpBuilder& b, Location loc, Value) { return constI1(b, loc, false); },
          nullptr, cParts(fisnan, false)}},
        {(int)Opcode::ISINF,
         {fIsInf, [](OpBuilder& b, Location loc, Value) { return constI1(b, loc, false); },
          nullptr, cParts(fIsInf, false)}},
        {(int)Opcode::ISFINITE,
         {fIsFinite, [](OpBuilder& b, Location loc, Value) { return constI1(b, loc, true); },
          nullptr, cParts(fIsFinite, true)}},
        // astype (the conversion happens around the body), real/imag/conj
        {(int)Opcode::CONVERT, {identityUnary, identityUnary, nullptr, identityUnary}},
        {(int)Opcode::REAL, {identityUnary, identityUnary, nullptr, complexUnary<cx::ReOp>()}},
        {(int)Opcode::IMAG,
         {[](OpBuilder& b, Location loc, Value v) { return constLike(b, loc, v, 0.0); },
          [](OpBuilder& b, Location loc, Value v) { return constLike(b, loc, v, 0.0); }, nullptr,
          complexUnary<cx::ImOp>()}},
        {(int)Opcode::CONJ, {identityUnary, identityUnary, nullptr, complexUnary<cx::ConjOp>()}},
        // ~x: bitwise not; logical not on bool
        {(int)Opcode::INVERT,
         {nullptr,
          [](OpBuilder& b, Location loc, Value v) {
              return arith::XOrIOp::create(b, loc, v, constLike(b, loc, v, -1.0)).getResult();
          },
          nullptr, nullptr,
          [](OpBuilder& b, Location loc, Value v) {
              return arith::XOrIOp::create(b, loc, v, constLike(b, loc, v, 1.0)).getResult();
          }}},
    };
    return registry;
}

const std::unordered_map<int, TernaryOpEntry>& MLIRJitCompiler::ternaryOpRegistry() {
    static const std::unordered_map<int, TernaryOpEntry> registry = {
        // where(cond, x, y): cond arrives as i1 (nonzero, NaN included, is true)
        {(int)Opcode::WHERE,
         {[](OpBuilder& b, Location loc, Value c, Value x, Value y) { return sel(b, loc, c, x, y); },
          [](OpBuilder& b, Location loc, Value c, Value x, Value y) { return sel(b, loc, c, x, y); },
          nullptr,
          [](OpBuilder& b, Location loc, Value c, Value x, Value y) {
              return selC(b, loc, c, x, y);
          }}},
        // clip(x, lo, hi) = minimum(maximum(x, lo), hi), NaN-propagating like NumPy
        {(int)Opcode::CLIP,
         {[](OpBuilder& b, Location loc, Value x, Value lo, Value hi) {
              Value m = arith::MaximumFOp::create(b, loc, x, lo).getResult();
              return arith::MinimumFOp::create(b, loc, m, hi).getResult();
          },
          [](OpBuilder& b, Location loc, Value x, Value lo, Value hi) {
              Value m = arith::MaxSIOp::create(b, loc, x, lo).getResult();
              return arith::MinSIOp::create(b, loc, m, hi).getResult();
          },
          [](OpBuilder& b, Location loc, Value x, Value lo, Value hi) {
              Value m = arith::MaxUIOp::create(b, loc, x, lo).getResult();
              return arith::MinUIOp::create(b, loc, m, hi).getResult();
          }}},
    };
    return registry;
}

#ifdef JIT_ENABLE_GPU_BACKEND

std::string MLIRJitCompiler::generateAMD() {
    if (!lowerToGPU())
        return "";

    // Lower ops inside gpu.module to LLVM + ROCDL dialects
    mlir::PassManager pm(&context);
    auto& gpuPM = pm.nest<mlir::gpu::GPUModuleOp>();
    gpuPM.addPass(mlir::createConvertMathToLLVMPass());
    gpuPM.addPass(mlir::createArithToLLVMConversionPass());
    gpuPM.addPass(mlir::createConvertIndexToLLVMPass());
    gpuPM.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    gpuPM.addPass(mlir::createConvertFuncToLLVMPass());
    gpuPM.addPass(mlir::createConvertGpuOpsToROCDLOps());
    gpuPM.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(pm.run(*module)))
        return "";

    return serializeToGCN();
}

std::string MLIRJitCompiler::generateIntel() {
    if (!lowerToGPU())
        return "";

    // Convert GPU module contents to SPIR-V
    mlir::PassManager pm(&context);
    pm.addPass(mlir::createConvertGPUToSPIRVPass());
    auto& spvPM = pm.nest<mlir::spirv::ModuleOp>();
    spvPM.addPass(mlir::spirv::createSPIRVLowerABIAttributesPass());
    spvPM.addPass(mlir::spirv::createSPIRVUpdateVCEPass());
    if (mlir::failed(pm.run(*module)))
        return "";

    return serializeToSPIRV();
}

bool MLIRJitCompiler::loadAMD(const std::string& gcn, const char* kernelName, void** moduleOut,
                              void** functionOut) {
#if defined(USE_AMD)
    hipModule_t hipModule;
    hipFunction_t hipFunction;
    hipError_t res;

    res = hipModuleLoadData(&hipModule, gcn.data());
    if (res != hipSuccess)
        return false;

    res = hipModuleGetFunction(&hipFunction, hipModule, kernelName);
    if (res != hipSuccess)
        return false;

    *moduleOut = (void*)hipModule;
    *functionOut = (void*)hipFunction;
    return true;
#else
    return false;
#endif
}

bool MLIRJitCompiler::loadIntel(const std::string& spirv, const char* kernelName, void** moduleOut,
                                void** functionOut) {
#if defined(USE_INTEL)
    // Initialize Level Zero
    ze_result_t res = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (res != ZE_RESULT_SUCCESS)
        return false;

    // Get driver
    uint32_t driverCount = 1;
    ze_driver_handle_t driver;
    res = zeDriverGet(&driverCount, &driver);
    if (res != ZE_RESULT_SUCCESS)
        return false;

    // Get device
    uint32_t deviceCount = 1;
    ze_device_handle_t device;
    res = zeDeviceGet(driver, &deviceCount, &device);
    if (res != ZE_RESULT_SUCCESS)
        return false;

    // Create context
    ze_context_desc_t ctxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t zeContext;
    res = zeContextCreate(driver, &ctxDesc, &zeContext);
    if (res != ZE_RESULT_SUCCESS)
        return false;

    // Create module from SPIR-V binary
    ze_module_desc_t moduleDesc = {};
    moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.inputSize = spirv.size();
    moduleDesc.pInputModule = reinterpret_cast<const uint8_t*>(spirv.data());

    ze_module_handle_t zeModule;
    ze_module_build_log_handle_t buildLog;
    res = zeModuleCreate(zeContext, device, &moduleDesc, &zeModule, &buildLog);
    if (res != ZE_RESULT_SUCCESS)
        return false;

    // Get kernel
    ze_kernel_desc_t kernelDesc = {};
    kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernelDesc.pKernelName = kernelName;
    ze_kernel_handle_t zeKernel;
    res = zeKernelCreate(zeModule, &kernelDesc, &zeKernel);
    if (res != ZE_RESULT_SUCCESS)
        return false;

    *moduleOut = reinterpret_cast<void*>(zeModule);
    *functionOut = reinterpret_cast<void*>(zeKernel);
    return true;
#else
    return false;
#endif
}

bool MLIRJitCompiler::lowerToGPU() {
    mlir::PassManager pm(&context);
    // Convert linalg.generic to scf.parallel (NOT affine.for)
    pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
    // Annotate scf.parallel loops with GPU mapping attributes (block/thread dims)
    pm.nest<mlir::func::FuncOp>().addPass(mlir::createGpuMapParallelLoopsPass());
    // Convert annotated scf.parallel to gpu.launch
    pm.addPass(mlir::createConvertParallelLoopToGpuPass());
    // Outline gpu.launch bodies into gpu.func inside gpu.module
    pm.addPass(mlir::createGpuKernelOutliningPass());
    return mlir::succeeded(pm.run(*module));
}

std::string MLIRJitCompiler::translateToTarget(llvm::StringRef triple, llvm::StringRef cpu,
                                               llvm::StringRef features,
                                               llvm::CodeGenFileType fileType) {
    // 1. Translate MLIR module to LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    if (!llvmModule)
        return "";

    llvm::Triple targetTriple(triple);
    llvmModule->setTargetTriple(targetTriple);

    // 2. Look up the LLVM target
    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target)
        return "";

    // 3. Create a TargetMachine
    llvm::TargetOptions opts;
    std::unique_ptr<llvm::TargetMachine> tm(
        target->createTargetMachine(targetTriple, cpu, features, opts, llvm::Reloc::Model::PIC_));
    if (!tm)
        return "";

    llvmModule->setDataLayout(tm->createDataLayout());

    // 4. Emit to a string (assembly or object depending on fileType)
    llvm::SmallString<0> outBuf;
    llvm::raw_svector_ostream os(outBuf);
    llvm::legacy::PassManager pass;
    if (tm->addPassesToEmitFile(pass, os, nullptr, fileType))
        return "";

    pass.run(*llvmModule);
    return std::string(outBuf.str());
}

std::string MLIRJitCompiler::serializeToGCN() {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    mlir::registerROCDLDialectTranslation(context);

    // AMD HSA requires an ELF code object, not assembly text
    return translateToTarget(
        /*triple=*/"amdgcn-amd-amdhsa",
        /*cpu=*/"gfx908",
        /*features=*/"", llvm::CodeGenFileType::ObjectFile);
}

std::string MLIRJitCompiler::serializeToSPIRV() {
    std::string result;

    // Walk the module looking for spirv.module ops and serialize each one
    module->walk([&](mlir::spirv::ModuleOp spvModule) {
        llvm::SmallVector<uint32_t, 0> binary;
        if (mlir::succeeded(mlir::spirv::serialize(spvModule, binary))) {
            // Append raw SPIR-V words to the output
            result.append(reinterpret_cast<const char*>(binary.data()),
                          binary.size() * sizeof(uint32_t));
        }
    });

    return result;
}

#endif // JIT_ENABLE_GPU_BACKEND


namespace {
std::unordered_map<std::string, std::pair<void*, void*>>& kernelCache() {
    static std::unordered_map<std::string, std::pair<void*, void*>> cache;
    return cache;
}
} // namespace

static void* jit_compile_ast_impl(void* ast, void** moduleHandleOut);

void* jit_compile_ast(void* ast, void** moduleHandleOut) {
    static const bool timeIt = getenv("CNUM_JIT_TIME") != nullptr;
    JitTimer timer;
    jitTimer = timeIt ? &timer : nullptr;
    void* fn = jit_compile_ast_impl(ast, moduleHandleOut);
    if (timeIt && !timer.line.empty())
        llvm::errs() << "jit:" << timer.line << "\n";
    jitTimer = nullptr;
    return fn;
}

static void* jit_compile_ast_impl(void* ast, void** moduleHandleOut) {
    MLIRJitCompiler node_jit;
    node_jit.buildFromAST(ast);
    // Kernels take every array as a dynamic memref, so equal IR means an equal kernel.
    std::string key;
    {
        llvm::raw_string_ostream os(key);
        node_jit.printModule(os);
    }
    auto hit = kernelCache().find(key);
    if (hit != kernelCache().end()) {
        *moduleHandleOut = hit->second.first;
        return hit->second.second;
    }
    jitMark("build");
    if (!node_jit.verify())
        return nullptr;
    node_jit.optimizeAndFuse();
    jitMark("opt");
    void* moduleHandle = nullptr;
    void* funcPtr = nullptr;

#if defined(USE_NVIDIA)
    auto engine = node_jit.generateNVIDIAHost();
    if (!engine)
        return nullptr;
    if (!node_jit.loadCPU(engine, "fused_kernel", &moduleHandle, &funcPtr))
        return nullptr;
#elif defined(USE_AMD)
    std::string gcn = node_jit.generateAMD();
    if (gcn.empty())
        return nullptr;
    if (!node_jit.loadAMD(gcn, "fused_kernel", &moduleHandle, &funcPtr))
        return nullptr;
#elif defined(USE_INTEL)
    std::string spirv = node_jit.generateIntel();
    if (spirv.empty())
        return nullptr;
    if (!node_jit.loadIntel(spirv, "fused_kernel", &moduleHandle, &funcPtr))
        return nullptr;
#else
    auto engine = node_jit.generateCPU();
    if (!engine)
        return nullptr;
    if (!node_jit.loadCPU(engine, "fused_kernel", &moduleHandle, &funcPtr))
        return nullptr;
#endif

    jitMark("load");
    kernelCache().emplace(std::move(key), std::make_pair(moduleHandle, funcPtr));
    *moduleHandleOut = moduleHandle;
    return funcPtr;
}

void jit_free_module(void* moduleHandle) {
#if !defined(USE_NVIDIA) && !defined(USE_AMD) && !defined(USE_INTEL)
    if (!moduleHandle)
        return;
    for (auto it = kernelCache().begin(); it != kernelCache().end();) {
        if (it->second.first == moduleHandle)
            it = kernelCache().erase(it);
        else
            ++it;
    }
    delete static_cast<mlir::ExecutionEngine*>(moduleHandle);
#endif
}
