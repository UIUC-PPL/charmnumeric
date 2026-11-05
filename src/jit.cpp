#include <mutex>
#include "jit.hpp"
#include "dtype_info.hpp"
#include "jit_api.hpp"

#include "mlir/IR/Verifier.h"

#include <chrono>
#include <limits>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <sched.h>
#include <set>
#include <cerrno>
#include <climits>
#include <csignal>
#include <ctime>
#include <fcntl.h>
#include <link.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/TargetParser/Host.h"

#ifdef JIT_ENABLE_GPU_BACKEND
#include "llvm/IR/LegacyPassManager.h"
#endif
#ifdef USE_NVIDIA
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include <numeric>
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
// Set while jit_prewarm runs a cached kernel's constructors: the first CUDA failure there
// is recorded here instead of aborting (-1: not a CUDA error), with the modules loaded so far.
struct SoftLoad {
    int error = 0;
    std::vector<void*> modules;
};
thread_local SoftLoad* jitSoftLoad = nullptr;
void jitMark(const char* what) {
    if (jitTimer)
        jitTimer->mark(what);
}
} // namespace

namespace {
/// What a kernel's module handle points to: the engine that compiled it, or
/// the JIT that loaded it from the kernel cache.
struct JitModule {
    std::unique_ptr<mlir::ExecutionEngine> engine;
    std::unique_ptr<llvm::orc::LLJIT> jit;
};

bool jitCacheEnabled();
const std::string& cacheDir();
const std::string& cubinDir();

/// Unique temporary name next to a cache entry (servers on several nodes share the cache).
std::string cacheTmpName(const std::string& path) {
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    static std::atomic<unsigned> n{0}; // compile threads of a process
    return llvm::formatv("{0}.{1}.{2}.{3}.tmp", path, host, getpid(), n++).str();
}

/// Publishes tmp as path unless path exists: an entry is never replaced, since
/// other servers may be reading it.
void publishCacheFile(const std::string& tmp, const std::string& path) {
    if (link(tmp.c_str(), path.c_str()) != 0 && errno != EEXIST &&
        !llvm::sys::fs::exists(path) && rename(tmp.c_str(), path.c_str()) == 0)
        return; // no hard links on this file system
    unlink(tmp.c_str());
}

/// Reads a cache entry into memory (no mmap: the file system is shared).
llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> readCacheFile(const std::string& path) {
    return llvm::MemoryBuffer::getFile(path, /*IsText=*/false, /*RequiresNullTerminator=*/false,
                                       /*IsVolatile=*/true);
}

/// Refreshes an entry's mtime (recency for jit_prewarm and eviction) at most hourly, since
/// the file system is shared.
void touchCacheEntry(const std::string& path, std::time_t mtime = 0) {
    struct stat st;
    if (!mtime && stat(path.c_str(), &st) == 0)
        mtime = st.st_mtime;
    if (mtime && std::time(nullptr) - mtime > 3600)
        utimes(path.c_str(), nullptr);
}
} // namespace

namespace {
// One context per thread (compiles run on CompilePool threads), kept: per-kernel creation
// dominated compile time.
mlir::MLIRContext& sharedContext() {
    thread_local mlir::MLIRContext* ctx = [] {
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
    engineOptions.enableObjectDump = jitCacheEnabled();
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

    *moduleOut = new JitModule{std::move(engine), nullptr};
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

/// Kernel parameter bytes of loops fused into one: each array it captures once host scalar
/// code is sunk is a (3 + 2 * rank) * 8 byte descriptor, each scalar at most 8 bytes.
/// Temporaries used only by these loops become thread-local and cost nothing.
int64_t paramBytes(llvm::ArrayRef<mlir::scf::ParallelOp> loops) {
    using namespace mlir;
    auto inside = [&](Operation* op) {
        return llvm::any_of(loops, [&](scf::ParallelOp l) { return l->isAncestor(op); });
    };
    llvm::SetVector<Value> vals;
    for (scf::ParallelOp l : loops)
        l->walk([&](Operation* op) {
            for (Value v : op->getOperands())
                if (!inside(v.getDefiningOp() ? v.getDefiningOp()
                                              : v.getParentBlock()->getParentOp()))
                    vals.insert(v);
        });
    int64_t bytes = 0;
    for (size_t i = 0; i < vals.size(); ++i) {
        Value v = vals[i];
        Operation* d = v.getDefiningOp();
        if (auto t = dyn_cast<MemRefType>(v.getType())) {
            bool local = d && isa<memref::AllocOp, memref::AllocaOp>(d) &&
                         llvm::all_of(v.getUsers(), [&](Operation* u) {
                             return inside(u) || isa<memref::DeallocOp>(u);
                         });
            if (!local)
                bytes += (3 + 2 * t.getRank()) * 8;
        } else if (d && d->getNumRegions() == 0 &&
                   (isMemoryEffectFree(d) || isa<func::CallOp>(d))) {
            vals.insert(d->operand_begin(), d->operand_end());
        } else {
            bytes += 8;
        }
    }
    return bytes;
}

/// Moves each loop that copies a temporary into an output right after the loop that last
/// writes the temporary, so a kernel split between them needs no extra array.
void hoistOutputCopies(mlir::Block& body) {
    using namespace mlir;
    SmallVector<scf::ParallelOp> loops;
    for (Operation& op : body)
        if (auto l = dyn_cast<scf::ParallelOp>(op))
            loops.push_back(l);
    for (scf::ParallelOp c : loops) {
        Block* b = c.getBody();
        if (b->getOperations().size() != 3)
            continue;
        auto ld = dyn_cast<memref::LoadOp>(b->front());
        auto st = dyn_cast<memref::StoreOp>(*std::next(b->begin()));
        if (!ld || !st || st.getValueToStore() != ld.getResult() ||
            !isa<BlockArgument>(st.getMemRef()) || !ld.getMemRef().getDefiningOp<memref::AllocOp>())
            continue;
        Value src = ld.getMemRef(), dst = st.getMemRef();
        for (Operation* p = c->getPrevNode(); p; p = p->getPrevNode()) {
            auto l = dyn_cast<scf::ParallelOp>(p);
            SmallVector<MemAccess> acc;
            if (!l || !collectAccesses(l, acc) ||
                llvm::any_of(acc, [&](const MemAccess& m) { return memrefRoot(m.mem) == dst; }))
                break;
            if (llvm::any_of(acc, [&](const MemAccess& m) {
                    return m.write && memrefRoot(m.mem) == src;
                })) {
                if (p != c->getPrevNode() && llvm::equal(l.getLowerBound(), c.getLowerBound()) &&
                    llvm::equal(l.getUpperBound(), c.getUpperBound()) &&
                    llvm::equal(l.getStep(), c.getStep()))
                    c->moveAfter(l);
                break;
            }
        }
    }
}

/// Fuse the elementwise loops of the kernel function into as few loops as
/// possible, each passing at most maxParamBytes of kernel parameters, and drop
/// the temporaries that fusion makes thread-local.
void fuseElementwiseLoops(mlir::func::FuncOp f, int64_t maxParamBytes) {
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
    hoistOutputCopies(body);

    for (Operation& op : body) {
        auto a = dyn_cast<scf::ParallelOp>(op);
        if (!a)
            continue;
        while (auto b = dyn_cast_or_null<scf::ParallelOp>(a->getNextNode())) {
            if (!canFuse(a, b) || paramBytes({a, b}) > maxParamBytes)
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

bool cnEnsureContext();

/// The current device's sm with PTX ISA max(8.0, the sm's minimum), if both
/// LLVM and the driver support them; else sm_80 PTX, which the driver JITs forward.
/// Kernels embed a cubin for the device: sm and driver.
struct NvTarget {
    std::string chip, features, device;
    int driver;
};
NvTarget nvTarget() {
    thread_local CUdevice cachedDev = -1;
    thread_local NvTarget t;
    CUdevice dev = 0;
    cnEnsureContext();
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS)
        dev = 0;
    if (dev == cachedDev)
        return t;
    int major = 8, minor = 0, driver = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDriverGetVersion(&driver);
    int sm = major * 10 + minor;
    static const std::pair<int, int> smPtx[] = {{86, 71},  {87, 74},  {88, 90},  {89, 78},
                                                {90, 78},  {100, 86}, {101, 86}, {103, 88},
                                                {110, 90}, {120, 87}, {121, 88}};
    static const std::pair<int, int> driverPtx[] = {{12000, 80}, {12010, 81}, {12020, 82},
                                                    {12030, 83}, {12040, 84}, {12050, 85},
                                                    {12080, 87}, {12090, 88}, {13000, 90}};
    int need = sm > 121 ? 1000 : 70, have = 0;
    for (auto [s, v] : smPtx)
        if (s == sm)
            need = v;
    for (auto [d, v] : driverPtx)
        if (driver >= d)
            have = v;
    int ptx = std::max(need, std::min(80, have));
    std::string chip = "sm_" + std::to_string(sm), feat = "ptx" + std::to_string(ptx);

    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    std::string err;
    llvm::Triple triple("nvptx64-nvidia-cuda");
    bool ok = ptx <= have;
    if (ok) {
        const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, err);
        std::unique_ptr<llvm::MCSubtargetInfo> sti(
            target ? target->createMCSubtargetInfo(triple, chip, "") : nullptr);
        ok = sti && sti->isCPUStringValid(chip) &&
             llvm::any_of(sti->getAllProcessorFeatures(),
                          [&](const llvm::SubtargetFeatureKV& f) { return feat == f.Key; });
    }
    std::string device = llvm::formatv("{0} driver {1}", chip, driver).str();
    t = ok ? NvTarget{chip, "+" + feat, device, driver}
           : NvTarget{"sm_80", "+ptx80", device, driver};
    cachedDev = dev;
    return t;
}

/// Kernel parameter bytes a fused loop may use: ptxas allows 4352 below PTX ISA 8.1 or
/// sm_70 and 32764 otherwise, less a margin for launch bounds. CNUM_JIT_MAX_PARAM_BYTES
/// overrides it.
int64_t paramBudget() {
    static const int64_t env = [] {
        const char* e = getenv("CNUM_JIT_MAX_PARAM_BYTES");
        return e && *e ? std::max(0LL, atoll(e)) : -1LL;
    }();
    if (env >= 0)
        return env;
    NvTarget t = nvTarget();
    int ptx = atoi(t.features.c_str() + 4), sm = atoi(t.chip.c_str() + 3);
    return (ptx >= 81 && sm >= 70 ? 32764 : 4352) - 128;
}
} // namespace

bool MLIRJitCompiler::lowerToNVIDIAHost() {
    using namespace mlir;

    // 1. Every linalg op becomes one multi-dim scf.parallel; adjacent loops over
    //    the same index space are fused. Each loop is then linearized and tiled
    //    by 256: blocks on grid x (up to 2^31-1), threads on block x.
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
    fuseElementwiseLoops(kernelFn, fuseLoops ? paramBudget() : 0);
    IRRewriter rewriter(&context);
    for (Operation& op : llvm::make_early_inc_range(kernelFn.getBody().front())) {
        auto loop = dyn_cast<scf::ParallelOp>(op);
        if (!loop || loop.getNumLoops() < 2 || loop.getNumReductions())
            continue;
        std::vector<unsigned> dims(loop.getNumLoops());
        std::iota(dims.begin(), dims.end(), 0u);
        collapseParallelLoops(rewriter, loop, {dims});
    }

    PassManager pm(&context);
    pm.nest<func::FuncOp>().addPass(createParallelLoopTilingPass({256}, /*noMinMaxBounds=*/true));
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

    // 4. NVVM pipeline: kernels embed PTX (generateNVIDIAHost assembles it); the host
    //    uses mgpu* calls.
    jitMark("gpu-map");
    PassManager pm2(&context);
    gpu::GPUToNVVMPipelineOptions opts;
    NvTarget target = nvTarget();
    opts.cubinChip = target.chip;
    opts.cubinFeatures = target.features;
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

// Modules load and kernels launch in the calling thread's current context (the
// chare's device). The MLIR wrappers would use device 0's primary context, which
// differs once Kokkos picks a device per local rank (e.g. SLURM_LOCALID under srun).
void cnCheck(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS)
        llvm::report_fatal_error(llvm::Twine(what) + " failed: " + llvm::Twine(r));
}
/// cnCheck, but under jit_prewarm a failure is recorded and returns false.
bool cnOk(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS)
        return true;
    if (jitSoftLoad) {
        if (!jitSoftLoad->error)
            jitSoftLoad->error = r;
        return false;
    }
    cnCheck(r, what);
    return false;
}
/// The entry itself is bad (not e.g. out of memory): remove it.
bool cnBadImage(int r) {
    return r == CUDA_ERROR_INVALID_IMAGE || r == CUDA_ERROR_NO_BINARY_FOR_GPU ||
           r == CUDA_ERROR_INVALID_PTX || r == CUDA_ERROR_INVALID_SOURCE ||
           r == CUDA_ERROR_UNSUPPORTED_PTX_VERSION;
}
bool cnEnsureContext() {
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx)
        return true;
    if (jitSoftLoad) { // jit_prewarm only loads into an existing context
        cnOk(CUDA_ERROR_INVALID_CONTEXT, "");
        return false;
    }
    CUdevice dev;
    cnCheck(cuInit(0), "cuInit");
    cnCheck(cuDeviceGet(&dev, 0), "cuDeviceGet");
    cnCheck(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    cnCheck(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
    return true;
}
CUmodule cnLoaded(CUmodule m) {
    if (jitSoftLoad)
        jitSoftLoad->modules.push_back(m);
    return m;
}
extern "C" CUmodule cn_mgpuModuleLoad(void* data, size_t) {
    CUmodule m = nullptr;
    if (!cnEnsureContext() || !cnOk(cuModuleLoadData(&m, data), "cuModuleLoadData"))
        return nullptr;
    return cnLoaded(m);
}
/// Assembles PTX for the current context's device with the driver's JIT; log gets its errors.
CUresult cnPtxToCubin(llvm::StringRef ptx, int optLevel, std::string& cubin, std::string& log) {
    char buf[4096] = {0};
    CUjit_option opts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                           CU_JIT_OPTIMIZATION_LEVEL};
    void* vals[] = {buf, reinterpret_cast<void*>(sizeof(buf)),
                    reinterpret_cast<void*>(static_cast<intptr_t>(optLevel))};
    CUlinkState link;
    CUresult r = cuLinkCreate(3, opts, vals, &link);
    if (r != CUDA_SUCCESS)
        return r;
    std::string text = ptx.str(); // NUL-terminated
    void* out = nullptr;
    size_t size = 0;
    r = cuLinkAddData(link, CU_JIT_INPUT_PTX, text.data(), text.size() + 1, "fused_kernel", 0,
                      nullptr, nullptr);
    if (r == CUDA_SUCCESS)
        r = cuLinkComplete(link, &out, &size);
    if (r == CUDA_SUCCESS)
        cubin.assign(static_cast<const char*>(out), size);
    log = buf;
    cuLinkDestroy(link);
    return r;
}
/// Loads PTX (kernels whose cubin could not be embedded). It is assembled once per device
/// architecture and driver; the cubin goes to the kernel cache (shared by all server builds).
extern "C" CUmodule cn_mgpuModuleLoadJIT(void* data, int optLevel, size_t) {
    CUdevice dev;
    int cc[2] = {0, 0};
    if (!cnEnsureContext() || !cnOk(cuCtxGetDevice(&dev), "cuCtxGetDevice"))
        return nullptr;
    cuDeviceGetAttribute(&cc[0], CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&cc[1], CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    std::string path;
    if (!cubinDir().empty()) {
        int driver = 0;
        cuDriverGetVersion(&driver);
        llvm::SHA256 h;
        h.update(llvm::StringRef(static_cast<const char*>(data)));
        h.update(llvm::formatv("sm_{0}{1} O{2} driver {3}", cc[0], cc[1], optLevel, driver).str());
        path = cubinDir() + "/" + llvm::toHex(h.final(), true) + ".cubin";
        if (auto buf = readCacheFile(path)) {
            CUmodule m = nullptr;
            CUresult r = cuModuleLoadData(&m, (*buf)->getBufferStart());
            if (r == CUDA_SUCCESS) {
                touchCacheEntry(path);
                return cnLoaded(m);
            }
            if (cnBadImage(r))
                unlink(path.c_str()); // corrupt entry: republished below
            else if (!cnOk(r, "cuModuleLoadData"))
                return nullptr;
        }
    }
    std::string cubin, log;
    CUresult r =
        cnPtxToCubin(llvm::StringRef(static_cast<const char*>(data)), optLevel, cubin, log);
    if (r != CUDA_SUCCESS) {
        if (jitSoftLoad) {
            cnOk(r, "");
            return nullptr;
        }
        llvm::report_fatal_error(llvm::Twine("PTX JIT failed: ") + log);
    }
    CUmodule m = nullptr;
    if (!cnOk(cuModuleLoadData(&m, cubin.data()), "cuModuleLoadData"))
        return nullptr;
    if (!path.empty()) {
        std::string tmp = cacheTmpName(path);
        std::error_code ec;
        llvm::raw_fd_ostream os(tmp, ec);
        if (!ec) {
            os << cubin;
            os.close();
            if (os.has_error()) {
                os.clear_error();
                unlink(tmp.c_str());
            } else {
                publishCacheFile(tmp, path);
            }
        }
    }
    return cnLoaded(m);
}
extern "C" void cn_mgpuModuleUnload(CUmodule m) {
    if (m)
        cuModuleUnload(m);
}
extern "C" CUfunction cn_mgpuModuleGetFunction(CUmodule m, const char* name) {
    CUfunction f = nullptr;
    if (!m && jitSoftLoad)
        return nullptr;
    cnCheck(cuModuleGetFunction(&f, m, name), "cuModuleGetFunction");
    return f;
}
extern "C" void cn_mgpuLaunchKernel(CUfunction f, intptr_t gx, intptr_t gy, intptr_t gz,
                                    intptr_t bx, intptr_t by, intptr_t bz, int32_t smem,
                                    CUstream s, void** params, void** extra, size_t) {
    if (gx <= 0 || gy <= 0 || gz <= 0)
        return; // empty iteration space
    cnCheck(cuLaunchKernel(f, gx, gy, gz, bx, by, bz, smem, s, params, extra), "cuLaunchKernel");
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
    add("mgpuModuleLoad", (void*)&cn_mgpuModuleLoad);
    add("mgpuModuleLoadJIT", (void*)&cn_mgpuModuleLoadJIT);
    add("mgpuModuleUnload", (void*)&cn_mgpuModuleUnload);
    add("mgpuModuleGetFunction", (void*)&cn_mgpuModuleGetFunction);
    add("mgpuLaunchKernel", (void*)&cn_mgpuLaunchKernel);
    return map;
}
} // namespace

void jit_set_stream(void* stream) { jitStream = static_cast<CUstream>(stream); }

std::unique_ptr<mlir::ExecutionEngine> MLIRJitCompiler::generateNVIDIAHost() {
    if (!lowerToNVIDIAHost()) {
        llvm::errs() << "generateNVIDIAHost: GPU lowering failed\n";
        return nullptr;
    }
    // Embed cubins instead of PTX, so loading does not run the driver's JIT. A kernel
    // ptxas rejects fails the compile instead of aborting at load.
    bool assembled = true;
    module->walk([&](mlir::gpu::BinaryOp bin) {
        llvm::SmallVector<mlir::Attribute> objects;
        for (mlir::Attribute a : bin.getObjectsAttr()) {
            auto obj = mlir::dyn_cast<mlir::gpu::ObjectAttr>(a);
            std::string cubin, log;
            int opt = 2; // as mgpuModuleLoadJIT gets it
            if (auto props = obj ? obj.getProperties() : nullptr)
                if (auto o = mlir::dyn_cast_or_null<mlir::IntegerAttr>(props.get("O")))
                    opt = o.getInt();
            if (obj && obj.getFormat() == mlir::gpu::CompilationTarget::Assembly) {
                if (cnPtxToCubin(obj.getObject().getValue(), opt, cubin, log) != CUDA_SUCCESS) {
                    llvm::errs() << "generateNVIDIAHost: ptxas failed: " << log << "\n";
                    assembled = false;
                    continue;
                }
                a = mlir::gpu::ObjectAttr::get(&context, obj.getTarget(),
                                               mlir::gpu::CompilationTarget::Binary,
                                               mlir::StringAttr::get(&context, cubin),
                                               obj.getProperties(), obj.getKernels());
            }
            objects.push_back(a);
        }
        bin.setObjectsAttr(mlir::ArrayAttr::get(&context, objects));
    });
    if (!assembled)
        return nullptr;
    jitMark("cubin");

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::ExecutionEngineOptions engineOptions;
    static const llvm::StringRef runtimeLibs[] = {MLIR_CUDA_RUNTIME_LIB};
    engineOptions.sharedLibPaths = runtimeLibs;
    // Host only launches: O0. jitCodeGenOptLevel alone is ignored; the target machine counts.
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
    engineOptions.enableObjectDump = jitCacheEnabled();
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
/// Float maximum/minimum (NaN of x first) and fmax/fmin (ignore NaN of y). Ties
/// give y, so signed zeros match NumPy.
BinaryOpEntry::Fn fMinMax(bool max, bool fm) {
    return [max, fm](OpBuilder& b, Location loc, Value x, Value y) {
        Value gt = fcmp(b, loc, max ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OLT, x, y);
        return sel(b, loc, orI(b, loc, gt, fisnan(b, loc, fm ? y : x)), x, y);
    };
}
Value fIsInf(OpBuilder& b, Location loc, Value v);
/// NumPy's sign(z): z/|z|, +0 at 0, a unit along a single infinite part.
Value cSign(OpBuilder& b, Location loc, Value z) {
    Value r = re(b, loc, z), i = im(b, loc, z);
    Value zero = constLike(b, loc, r, 0.0), one = constLike(b, loc, r, 1.0);
    Value nan = constLike(b, loc, r, std::numeric_limits<double>::quiet_NaN());
    Value a = mlir::complex::AbsOp::create(b, loc, z).getResult();
    Value isZero = fcmp(b, loc, arith::CmpFPredicate::OEQ, a, zero);
    Value nr = sel(b, loc, isZero, zero, arith::DivFOp::create(b, loc, r, a).getResult());
    Value ni = sel(b, loc, isZero, zero, arith::DivFOp::create(b, loc, i, a).getResult());
    Value rInf = fIsInf(b, loc, r), iInf = fIsInf(b, loc, i);
    Value rUnit = math::CopySignOp::create(b, loc, one, r).getResult();
    Value iUnit = math::CopySignOp::create(b, loc, one, i).getResult();
    nr = sel(b, loc, rInf, sel(b, loc, iInf, nan, rUnit), sel(b, loc, iInf, zero, nr));
    ni = sel(b, loc, iInf, sel(b, loc, rInf, nan, iUnit), sel(b, loc, rInf, zero, ni));
    return mkComplex(b, loc, z.getType(), nr, ni);
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
         {fMinMax(true, false), arithBinary<arith::MaxSIOp>(),
          arithBinary<arith::MaxUIOp>(), cMinMax(true, false)}},
        {(int)Opcode::MINIMUM,
         {fMinMax(false, false), arithBinary<arith::MinSIOp>(),
          arithBinary<arith::MinUIOp>(), cMinMax(false, false)}},
        {(int)Opcode::FMAX,
         {fMinMax(true, true), arithBinary<arith::MaxSIOp>(),
          arithBinary<arith::MaxUIOp>(), cMinMax(true, true)}},
        {(int)Opcode::FMIN,
         {fMinMax(false, true), arithBinary<arith::MinSIOp>(),
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
              // sign(+-0) = +0 and sign(NaN) = NaN, as in NumPy
              Value zero = constLike(b, loc, v, 0.0);
              Value pos = fcmp(b, loc, arith::CmpFPredicate::OGT, v, zero);
              Value neg = fcmp(b, loc, arith::CmpFPredicate::OLT, v, zero);
              Value isZero = fcmp(b, loc, arith::CmpFPredicate::OEQ, v, zero);
              return sel(b, loc, pos, constLike(b, loc, v, 1.0),
                         sel(b, loc, neg, constLike(b, loc, v, -1.0),
                             sel(b, loc, isZero, zero, v)));
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
          cSign}},
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
        // clip(x, lo, hi) = minimum(hi, maximum(lo, x)), NaN-propagating like
        // NumPy; ties keep x (signed zeros)
        {(int)Opcode::CLIP,
         {[](OpBuilder& b, Location loc, Value x, Value lo, Value hi) {
              Value m = fMinMax(true, false)(b, loc, lo, x);
              return fMinMax(false, false)(b, loc, hi, m);
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


#if !defined(USE_AMD) && !defined(USE_INTEL)
// ---- Persistent kernel cache ----
// Compiled kernels are object files under $CNUM_JIT_CACHE_DIR (default
// ${XDG_CACHE_HOME:-$HOME/.cache}/charmnumeric/jit, so it outlives node-local
// /tmp), one directory per server build (GNU build id) and target, named by a
// hash of the kernel IR and the target. CNUM_JIT_CACHE=0 turns it off.
namespace {

int buildIdNote(struct dl_phdr_info* info, size_t, void* out) {
    // the first entry is the executable
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const ElfW(Phdr)& ph = info->dlpi_phdr[i];
        if (ph.p_type != PT_NOTE)
            continue;
        const char* p = reinterpret_cast<const char*>(info->dlpi_addr + ph.p_vaddr);
        const char* end = p + ph.p_memsz;
        while (p + sizeof(ElfW(Nhdr)) <= end) {
            const auto* nh = reinterpret_cast<const ElfW(Nhdr)*>(p);
            const char* name = p + sizeof(ElfW(Nhdr));
            const char* desc = name + ((nh->n_namesz + 3) & ~3u);
            if (nh->n_type == NT_GNU_BUILD_ID && nh->n_namesz == 4 && memcmp(name, "GNU", 4) == 0) {
                *static_cast<std::string*>(out) = llvm::toHex(
                    llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t*>(desc), nh->n_descsz),
                    true);
                return 1;
            }
            p = desc + ((nh->n_descsz + 3) & ~3u);
        }
    }
    return 1;
}

std::string serverBuildId() {
    std::string id;
    dl_iterate_phdr(buildIdNote, &id);
    struct stat st;
    if (id.empty() && stat("/proc/self/exe", &st) == 0)
        id = llvm::formatv("{0}-{1}", st.st_size, st.st_mtime).str();
    return id;
}

/// The cache root; "" when the cache is off.
std::string cacheBase() {
    const char* on = getenv("CNUM_JIT_CACHE");
    if (on && std::string(on) == "0")
        return {};
    std::string d;
    const char* dir = getenv("CNUM_JIT_CACHE_DIR");
    const char* xdg = getenv("XDG_CACHE_HOME");
    const char* home = getenv("HOME");
    if (dir && *dir)
        d = dir;
    else if (xdg && *xdg)
        d = std::string(xdg) + "/charmnumeric/jit";
    else if (home && *home)
        d = std::string(home) + "/.cache/charmnumeric/jit";
    else {
        d = llvm::formatv("/tmp/cnum-jit-{0}", getuid()).str();
        struct stat st;
        if ((mkdir(d.c_str(), 0700) != 0 && errno != EEXIST) || lstat(d.c_str(), &st) != 0 ||
            !S_ISDIR(st.st_mode) || st.st_uid != getuid() ||
            ((st.st_mode & 077) && chmod(d.c_str(), 0700) != 0))
            return {};
    }
    // one spelling (absolute, no dots or trailing separators), so paths built on it compare
    llvm::SmallString<256> p(d);
    if (llvm::sys::fs::make_absolute(p))
        return {};
    llvm::sys::path::remove_dots(p, /*remove_dot_dot=*/true);
    while (p.size() > 1 && llvm::sys::path::is_separator(p.back()))
        p.pop_back();
    return std::string(p);
}

// Marks a dedicated cache root; evictCache only runs on marked roots.
constexpr const char* kCacheMarker = ".cnum-jit-cache";

bool isLowerHex(llvm::StringRef n) {
    return llvm::all_of(n, [](char c) { return llvm::isDigit(c) || (c >= 'a' && c <= 'f'); });
}

/// A serverBuildId(): GNU build id, or "<size>-<mtime>".
bool isBuildIdName(llvm::StringRef n) {
    if ((n.size() == 16 || n.size() == 32 || n.size() == 40 || n.size() == 64) && isLowerHex(n))
        return true;
    auto [a, b] = n.split('-');
    return !a.empty() && !b.empty() && llvm::all_of(a, llvm::isDigit) &&
           llvm::all_of(b, llvm::isDigit);
}

/// A targetCacheDir() tag: [sm_<n>+ptx<n>-[sm_<n>-][d<driver>-]]<host CPU>.
bool isTargetName(llvm::StringRef n) {
    auto sm = [](llvm::StringRef& s) {
        if (!s.consume_front("sm_") || s.empty() || !llvm::isDigit(s.front()))
            return false;
        s = s.drop_while(llvm::isDigit);
        if (!s.empty() && s.front() >= 'a' && s.front() <= 'z')
            s = s.drop_front();
        return true;
    };
    auto number = [](llvm::StringRef& s) {
        size_t k = s.find_if_not(llvm::isDigit);
        bool ok = k > 0 && k != llvm::StringRef::npos && s[k] == '-';
        s = ok ? s.drop_front(k + 1) : s;
        return ok;
    };
    if (n.starts_with("sm_")) {
        if (!sm(n) || !n.consume_front("+ptx") || !number(n))
            return false;
        llvm::StringRef t = n;
        if (sm(t) && t.consume_front("-"))
            n = t;
        t = n;
        if (t.consume_front("d") && number(t))
            n = t;
    }
    return !n.empty() && n.size() <= 64 && llvm::isAlnum(n.front()) &&
           llvm::all_of(n, [](char c) {
               return llvm::isDigit(c) || (c >= 'a' && c <= 'z') || c == '-' || c == '_' ||
                      c == '.';
           });
}

/// What a file of a cache directory is; eviction deletes nothing else.
enum class CacheFile { Other, Entry, Temp };
/// Entries are <sha256>.o and <sha256>.cubin; temporaries are their cacheTmpName()s and
/// compile claims (<sha256>.o.lock).
CacheFile cacheFileKind(llvm::StringRef n) {
    if (n.size() < 66 || !isLowerHex(n.take_front(64)))
        return CacheFile::Other;
    llvm::StringRef rest = n.drop_front(64);
    if (rest == ".o" || rest == ".cubin")
        return CacheFile::Entry;
    if (rest == ".o.lock")
        return CacheFile::Temp;
    if (!(rest.consume_front(".o.") || rest.consume_front(".cubin.")) ||
        !rest.consume_back(".tmp"))
        return CacheFile::Other;
    // <host>.<pid>[.<n>]
    auto digits = [](llvm::StringRef x) { return !x.empty() && llvm::all_of(x, llvm::isDigit); };
    auto [h1, n1] = rest.rsplit('.');
    if (!digits(n1))
        return CacheFile::Other;
    auto [h2, n2] = h1.rsplit('.');
    llvm::StringRef host = digits(n2) && !h2.empty() ? h2 : h1;
    bool ok = !host.empty() && llvm::all_of(host, [](char c) {
        return llvm::isAlnum(c) || c == '-' || c == '_' || c == '.';
    });
    return ok ? CacheFile::Temp : CacheFile::Other;
}

/// True if dir holds only cache files (fileOnly) or, recursively, the cache layout: cubin/,
/// build directories with target directories (and entries of the layout before them).
bool looksLikeCache(const std::string& dir, int level) {
    namespace fs = llvm::sys::fs;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec, false), end; it != end && !ec; it.increment(ec)) {
        fs::file_status st;
        if (fs::status(it->path(), st, /*Follow=*/false))
            return false;
        llvm::StringRef n = llvm::sys::path::filename(it->path());
        bool isDir = st.type() == fs::file_type::directory_file;
        bool ok;
        if (level == 0) // the root
            ok = isDir ? (n == "cubin" ? looksLikeCache(it->path(), 2)
                                       : isBuildIdName(n) && looksLikeCache(it->path(), 1))
                       : st.type() == fs::file_type::regular_file &&
                             (n == ".last-evict" || n == ".evict-lock");
        else if (isDir) // a build directory's target directory
            ok = level == 1 && isTargetName(n) && looksLikeCache(it->path(), 2);
        else
            ok = st.type() == fs::file_type::regular_file &&
                 cacheFileKind(n) != CacheFile::Other;
        if (!ok)
            return false;
    }
    return !ec;
}

/// Marks base as a dedicated cache when it is new or empty, when it has exactly the cache
/// layout (a cache from before the marker), or with CNUM_JIT_CACHE_ADOPT=1. Any other
/// directory is used but never evicted.
void markCacheBase(const std::string& base) {
    static std::once_flag once;
    std::call_once(once, [&] {
        namespace fs = llvm::sys::fs;
        std::string marker = base + "/" + kCacheMarker;
        if (fs::exists(marker))
            return;
        const char* adopt = getenv("CNUM_JIT_CACHE_ADOPT");
        if (fs::exists(base) && !(adopt && std::string(adopt) == "1") &&
            !looksLikeCache(base, 0))
            return;
        if (fs::create_directories(base))
            return;
        int fd = open(marker.c_str(), O_CREAT | O_WRONLY, 0644);
        if (fd >= 0)
            close(fd);
    });
}

/// base/sub, created; "" when the cache is off or the directory cannot be made.
std::string cacheSubdir(const std::string& sub) {
    std::string d = cacheBase();
    if (d.empty())
        return {};
    markCacheBase(d);
    d += "/" + sub;
    if (llvm::sys::fs::create_directories(d))
        return {};
    return d;
}

const std::string& cacheDir() {
    static const std::string dir = cacheSubdir(serverBuildId());
    return dir;
}

const std::string& cubinDir() {
    static const std::string dir = cacheSubdir("cubin");
    return dir;
}

/// cacheDir()/<GPU target>-<host CPU>, created; "" if that fails.
std::string targetCacheDir() {
    std::string tag = llvm::sys::getHostCPUName().str();
#ifdef USE_NVIDIA
    NvTarget target = nvTarget();
    llvm::StringRef sm = llvm::StringRef(target.device).split(' ').first;
    tag = target.chip + target.features + (sm == target.chip ? "" : "-" + sm.str()) + "-d" +
          std::to_string(target.driver) + "-" + tag;
#endif
    std::string dir = cacheDir() + "/" + tag;
    thread_local std::string made;
    if (dir != made) {
        if (llvm::sys::fs::create_directories(dir))
            return {};
        made = dir;
    }
    return dir;
}

std::string cachePath(const std::string& ir) {
    if (cacheDir().empty())
        return {};
    std::string dir = targetCacheDir();
    if (dir.empty())
        return {};
    llvm::SHA256 h;
    h.update(ir);
#ifdef USE_NVIDIA
    NvTarget target = nvTarget();
    h.update("nvidia " + target.chip + " " + target.features + " cubin " + target.device +
             " params " + std::to_string(paramBudget()));
#endif
    h.update(llvm::sys::getHostCPUName());
    return dir + "/" + llvm::toHex(h.final(), true) + ".o";
}

llvm::orc::SymbolMap runtimeSymbols(llvm::orc::MangleAndInterner interner) {
#ifdef USE_NVIDIA
    return streamRuntimeSymbols(interner);
#else
    return halfRuntimeSymbols(interner);
#endif
}

/// Loads a cached kernel object (running its constructors, which load the GPU
/// binary); nullptr if that fails. With soft, CUDA failures in the constructors fail the
/// load instead of aborting. Entries that fail to load are removed.
void* loadCachedKernel(const std::string& path, void** moduleOut, bool soft = false,
                       bool* found = nullptr) {
    auto buf = readCacheFile(path);
    if (!buf)
        return nullptr;
    if (found)
        *found = true;
    // ctors were turned into __orc_init_func.* functions when the object was built
    std::vector<std::string> inits;
    {
        auto obj = llvm::object::ObjectFile::createObjectFile((*buf)->getMemBufferRef());
        if (!obj) {
            llvm::consumeError(obj.takeError());
            unlink(path.c_str()); // corrupt entry: let the next compile republish it
            return nullptr;
        }
        for (const auto& sym : (*obj)->symbols()) {
            auto name = sym.getName();
            if (name && name->starts_with("__orc_init_func."))
                inits.push_back(name->str());
            else if (!name)
                llvm::consumeError(name.takeError());
        }
    }
    jitMark("c-obj");
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
#ifdef USE_NVIDIA
    static const bool cudaRuntime =
        !llvm::sys::DynamicLibrary::LoadLibraryPermanently(MLIR_CUDA_RUNTIME_LIB);
    if (!cudaRuntime)
        return nullptr;
#endif
    auto tmb = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmb) {
        llvm::consumeError(tmb.takeError());
        return nullptr;
    }
    auto jit =
        llvm::orc::LLJITBuilder()
            .setJITTargetMachineBuilder(std::move(*tmb))
            .setObjectLinkingLayerCreator(
                [](llvm::orc::ExecutionSession& es)
                    -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
                    return std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
                        es, [](const llvm::MemoryBuffer&) {
                            return std::make_unique<llvm::SectionMemoryManager>();
                        });
                })
            .create();
    if (!jit) {
        llvm::consumeError(jit.takeError());
        return nullptr;
    }
    jitMark("c-jit");
    auto& jd = (*jit)->getMainJITDylib();
    llvm::orc::MangleAndInterner mangle((*jit)->getExecutionSession(), (*jit)->getDataLayout());
    llvm::Error err = jd.define(llvm::orc::absoluteSymbols(runtimeSymbols(mangle)));
    // libc/libm and the runtime library, as ExecutionEngine resolves them
    if (!err) {
        auto gen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            (*jit)->getDataLayout().getGlobalPrefix());
        if (gen)
            jd.addGenerator(std::move(*gen));
        else
            err = gen.takeError();
    }
    if (err) {
        llvm::consumeError(std::move(err));
        return nullptr;
    }
    if (llvm::Error add = (*jit)->addObjectFile(std::move(*buf))) {
        llvm::consumeError(std::move(add));
        unlink(path.c_str());
        return nullptr;
    }
    auto fn = (*jit)->lookup("_mlir_fused_kernel");
    if (!fn) {
        llvm::consumeError(fn.takeError());
        unlink(path.c_str());
        return nullptr;
    }
    jitMark("c-link");
    SoftLoad sl;
    jitSoftLoad = soft ? &sl : nullptr;
    for (const std::string& name : inits) {
        auto init = (*jit)->lookup(name);
        if (!init) {
            llvm::consumeError(init.takeError());
            sl.error = -1;
            break;
        }
        init->toPtr<void (*)()>()();
        if (sl.error)
            break;
    }
    jitSoftLoad = nullptr;
    if (sl.error) {
        bool bad = sl.error < 0;
#ifdef USE_NVIDIA
        bad = bad || cnBadImage(sl.error);
        for (void* m : sl.modules)
            cuModuleUnload(static_cast<CUmodule>(m));
#endif
        if (bad) // else transient (e.g. out of memory): keep the entry
            unlink(path.c_str());
        return nullptr;
    }
    *moduleOut = new JitModule{nullptr, std::move(*jit)};
    return fn->toPtr<void*>();
}

/// Writes a compiled kernel's object into the cache.
void storeCachedKernel(const std::string& path, void* moduleHandle) {
    auto* jm = static_cast<JitModule*>(moduleHandle);
    if (path.empty() || !jm->engine)
        return;
    std::string tmp = cacheTmpName(path);
    jm->engine->dumpToObjectFile(tmp);
    uint64_t size = 0;
    if (llvm::sys::fs::file_size(tmp, size) || size == 0) {
        unlink(tmp.c_str()); // failed or partial dump, or eviction removed the directory
        if (llvm::sys::fs::create_directories(llvm::sys::path::parent_path(path)))
            return;
        jm->engine->dumpToObjectFile(tmp);
        if (llvm::sys::fs::file_size(tmp, size) || size == 0) {
            unlink(tmp.c_str());
            return;
        }
    }
    publishCacheFile(tmp, path);
}

bool jitCacheEnabled() { return !cacheDir().empty(); }

} // namespace
#else
namespace {
bool jitCacheEnabled() { return false; }
} // namespace
#endif

namespace {
std::unordered_map<std::string, std::pair<void*, void*>>& kernelCache() {
    static std::unordered_map<std::string, std::pair<void*, void*>> cache;
    return cache;
}
struct PreloadedKernel {
    void* module;
    void* fn;
    std::time_t mtime;
};
/// Kernels loaded by jit_prewarm, by cache path, until their first compile.
std::unordered_map<std::string, PreloadedKernel>& preloadedKernels() {
    static std::unordered_map<std::string, PreloadedKernel> cache;
    return cache;
}
// The context and caches are per process; SMP PEs (threads) must not use them concurrently.
std::mutex& jitMutex() {
    static std::mutex m;
    return m;
}
} // namespace

namespace {
const bool jitTimeIt = getenv("CNUM_JIT_TIME") != nullptr;

/// What a waiter saw of another server's claim.
struct LockWatch {
    struct timespec mtime = {0, 0};
    std::chrono::steady_clock::time_point since;
    bool seen = false;
};

/// Compiles ast; the kernel goes to the disk cache at path unless that is "".
void* compileKernel(void* ast, const std::string& path, void** moduleHandleOut) {
    MLIRJitCompiler node_jit;
    node_jit.buildFromAST(ast);
    if (!node_jit.verify())
        return nullptr;
    node_jit.optimizeAndFuse();
    jitMark("opt");
    void* moduleHandle = nullptr;
    void* funcPtr = nullptr;

#if defined(USE_NVIDIA)
    auto engine = node_jit.generateNVIDIAHost();
    if (!engine) {
        MLIRJitCompiler unfused;
        unfused.fuseLoops = false;
        unfused.buildFromAST(ast);
        unfused.optimizeAndFuse();
        engine = unfused.generateNVIDIAHost();
        if (!engine)
            return nullptr;
        llvm::errs() << "JIT: fused kernel failed to compile; running it as one kernel per op\n";
    }
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
#if !defined(USE_AMD) && !defined(USE_INTEL)
    storeCachedKernel(path, moduleHandle);
    jitMark("store");
#endif
    *moduleHandleOut = moduleHandle;
    return funcPtr;
}

#if !defined(USE_AMD) && !defined(USE_INTEL)
/// A pre-warmed or disk-cached kernel; nullptr if there is none or it does not load. CUDA
/// failures do not abort: invalid entries are removed, and *kept is set when an entry
/// stays that could not be loaded (e.g. out of memory).
void* loadCached(const std::string& path, void** moduleHandleOut, bool* kept = nullptr) {
    if (path.empty())
        return nullptr;
    void* fn = nullptr;
    std::time_t mtime = 0;
    auto pre = preloadedKernels().find(path);
    if (pre != preloadedKernels().end()) {
        *moduleHandleOut = pre->second.module;
        fn = pre->second.fn;
        mtime = pre->second.mtime;
        preloadedKernels().erase(pre);
    } else {
        bool found = false;
        fn = loadCachedKernel(path, moduleHandleOut, /*soft=*/true, &found);
        if (!fn) {
            if (kept && found)
                *kept = llvm::sys::fs::exists(path);
            return nullptr;
        }
    }
    touchCacheEntry(path, mtime);
    jitMark("cache");
    return fn;
}

/// The process's CPUs at startup, before Charm++ may pin the PE thread (+setcpuaffinity):
/// compile threads run on all of them.
struct StartupCpus {
    cpu_set_t set;
    bool ok;
    StartupCpus() { ok = sched_getaffinity(0, sizeof(set), &set) == 0; }
} startupCpus;

/// Threads that compile kernels, made on demand and kept (each has its own MLIR context);
/// big stacks, since LLVM codegen recursion grows with loop count. CNUM_JIT_THREADS sets
/// their number (default: this process's CPUs, at most 8).
class CompilePool {
  public:
    static CompilePool& get() {
        static CompilePool* pool = new CompilePool;
        return *pool;
    }
    int size() const { return max_; }
    /// Runs the jobs, on the calling thread if no thread can be made.
    void run(std::vector<std::function<void()>>& jobs) {
        std::unique_lock<std::mutex> lk(m_);
        while (threads_ < max_ && threads_ < static_cast<int>(jobs.size()) && spawn())
            ++threads_;
        if (threads_ == 0) {
            lk.unlock();
            for (auto& job : jobs)
                job();
            return;
        }
        for (auto& job : jobs)
            queue_.push_back(&job);
        pending_ += jobs.size();
        work_.notify_all();
        done_.wait(lk, [&] { return pending_ == 0; });
    }

  private:
    CompilePool() {
        int n = 8;
        if (startupCpus.ok)
            n = std::min(n, CPU_COUNT(&startupCpus.set));
        if (const char* e = getenv("CNUM_JIT_THREADS"))
            if (atoi(e) > 0)
                n = atoi(e);
        max_ = std::max(n, 1);
    }
    bool spawn() {
        pthread_attr_t attr;
        pthread_t t;
        if (pthread_attr_init(&attr) != 0)
            return false;
        // the threads take no signals (they belong to the Charm++ scheduler thread)
        sigset_t all, old;
        sigfillset(&all);
        pthread_sigmask(SIG_BLOCK, &all, &old);
        auto body = [](void* pool) -> void* {
            static_cast<CompilePool*>(pool)->work();
            return nullptr;
        };
        bool ok = pthread_attr_setstacksize(&attr, size_t(1) << 30) == 0 &&
                  (!startupCpus.ok || pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t),
                                                                  &startupCpus.set) == 0) &&
                  pthread_create(&t, &attr, body, this) == 0;
        pthread_sigmask(SIG_SETMASK, &old, nullptr);
        pthread_attr_destroy(&attr);
        if (ok)
            pthread_detach(t);
        return ok;
    }
    void work() {
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            work_.wait(lk, [&] { return !queue_.empty(); });
            std::function<void()>* job = queue_.front();
            queue_.pop_front();
            lk.unlock();
            try {
                (*job)();
            } catch (...) { // the job's kernel stays missing
            }
            lk.lock();
            if (--pending_ == 0)
                done_.notify_all();
        }
    }
    std::mutex m_;
    std::condition_variable work_, done_;
    std::deque<std::function<void()>*> queue_;
    size_t pending_ = 0;
    int threads_ = 0, max_ = 1;
};

const std::string& hostName() {
    static const std::string name = [] {
        char host[256] = {0};
        gethostname(host, sizeof(host) - 1);
        return std::string(host);
    }();
    return name;
}

/// Compile claims: <entry>.lock, created exclusively and held while a server compiles the
/// entry, so the servers sharing a cache compile each kernel once. Holders refresh the
/// lock's mtime every 2 s.
class Claims {
  public:
    static Claims& get() {
        static Claims* claims = new Claims;
        return *claims;
    }
    /// The claim's fd; -1 if another server holds it, -2 if it cannot be made here.
    int take(const std::string& lock) {
        int fd = open(lock.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
        if (fd < 0)
            return errno == EEXIST ? -1 : -2;
        std::string id = hostName() + " " + std::to_string(getpid()) + "\n";
        ssize_t written = write(fd, id.data(), id.size());
        (void)written;
        std::lock_guard<std::mutex> g(m_);
        held_.insert(lock);
        if (!beating_)
            beating_ = startHeartbeat();
        return fd;
    }
    /// Removes the lock unless it was broken and taken by someone else meanwhile.
    void release(int fd, const std::string& lock) {
        if (fd < 0)
            return;
        struct stat a, b;
        if (fstat(fd, &a) == 0 && stat(lock.c_str(), &b) == 0 && a.st_ino == b.st_ino &&
            a.st_dev == b.st_dev)
            unlink(lock.c_str());
        close(fd);
        std::lock_guard<std::mutex> g(m_);
        held_.erase(lock);
    }

  private:
    bool startHeartbeat() {
        sigset_t all, old;
        sigfillset(&all);
        pthread_sigmask(SIG_BLOCK, &all, &old);
        bool ok = true;
        try {
            std::thread([this] {
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    std::set<std::string> held;
                    {
                        std::lock_guard<std::mutex> g(m_);
                        held = held_;
                    }
                    for (const std::string& lock : held)
                        utimes(lock.c_str(), nullptr);
                }
            }).detach();
        } catch (...) {
            ok = false;
        }
        pthread_sigmask(SIG_SETMASK, &old, nullptr);
        return ok;
    }
    std::mutex m_;
    std::set<std::string> held_;
    bool beating_ = false;
};

/// False once the process is gone, exiting (a killed GPU process can take many seconds to
/// tear down) or a zombie (not reaped yet, e.g. under a launcher).
bool processAlive(int pid) {
    if (kill(pid, 0) != 0)
        return errno != ESRCH;
    auto stat = llvm::MemoryBuffer::getFileAsStream("/proc/" + std::to_string(pid) + "/stat");
    if (!stat)
        return true;
    // after "(comm)": state ppid pgrp session tty_nr tpgid flags
    llvm::SmallVector<llvm::StringRef, 8> f;
    (*stat)->getBuffer().rsplit(')').second.ltrim().split(f, ' ', 7);
    unsigned long flags = 0;
    constexpr unsigned long PF_EXITING = 0x4;
    if (f.size() > 6 && !f[6].getAsInteger(10, flags) && (flags & PF_EXITING))
        return false;
    return f.empty() || (!f[0].starts_with("Z") && !f[0].starts_with("X"));
}

/// True if the claim lock is gone, or stale and then removed: its holder died on this host,
/// or its mtime stood still for 30 s (no heartbeat; 6 s for a holder on this host or a lock
/// without its holder, which died between creating and writing it), or for 5 s and it is
/// 2 min old (by the local clock, so only with an observed standstill).
bool lockFree(const std::string& lock, LockWatch& w) {
    int fd = open(lock.c_str(), O_RDONLY); // not stat: opening revalidates NFS attributes
    if (fd < 0)
        return errno == ENOENT;
    struct stat st;
    char buf[300] = {0};
    bool ok = fstat(fd, &st) == 0;
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (!ok)
        return false;
    auto now = std::chrono::steady_clock::now();
    bool stale = false;
    llvm::StringRef id(buf, n > 0 ? n : 0);
    auto [host, pid] = id.trim().split(' ');
    int p = 0;
    bool here = host == hostName();
    if (here && !pid.getAsInteger(10, p) && p > 0 && !processAlive(p))
        stale = true;
    if (!w.seen || st.st_mtim.tv_sec != w.mtime.tv_sec || st.st_mtim.tv_nsec != w.mtime.tv_nsec) {
        w = {st.st_mtim, now, true};
    } else if (now - w.since > std::chrono::seconds(here || id.trim().empty() ? 6 : 30) ||
               (now - w.since > std::chrono::seconds(5) &&
                std::time(nullptr) - st.st_mtime > 120)) {
        stale = true;
    }
    if (stale)
        unlink(lock.c_str());
    return stale;
}
#endif
} // namespace

void* jit_compile_ast(void* ast, void** moduleHandleOut) {
    void* fn = nullptr;
    jit_compile_batch(&ast, 1, &fn, moduleHandleOut);
    return fn;
}

void jit_compile_batch(void* const* asts, int n, void** fns, void** moduleHandles) {
    std::lock_guard<std::mutex> lock(jitMutex());
    struct Item {
        std::string key, path;
        int same = -1; // index of the item with the same kernel
        bool local = false; // compile without a claim
        int fd = -1;
        LockWatch watch;
        std::chrono::steady_clock::time_point waitStart;
    };
    std::vector<Item> items(n);
    std::unordered_map<std::string, int> first;
    std::vector<int> todo;
    for (int i = 0; i < n; ++i) {
        Item& it = items[i];
        fns[i] = moduleHandles[i] = nullptr;
        JitTimer timer;
        jitTimer = jitTimeIt ? &timer : nullptr;
        {
            MLIRJitCompiler node_jit;
            node_jit.buildFromAST(asts[i]);
            // Kernels take every array as a dynamic memref, so equal IR means an equal kernel.
            llvm::raw_string_ostream os(it.key);
            node_jit.printModule(os);
        }
        auto hit = kernelCache().find(it.key);
        auto dup = first.find(it.key);
        if (hit != kernelCache().end()) {
            moduleHandles[i] = hit->second.first;
            fns[i] = hit->second.second;
        } else if (dup != first.end()) {
            it.same = dup->second;
        } else {
            first.emplace(it.key, i);
            jitMark("build");
#if !defined(USE_AMD) && !defined(USE_INTEL)
            it.path = cachePath(it.key);
            fns[i] = loadCached(it.path, &moduleHandles[i], &it.local);
#endif
            if (fns[i])
                kernelCache().emplace(it.key, std::make_pair(moduleHandles[i], fns[i]));
            else
                todo.push_back(i);
        }
        jitTimer = nullptr;
        if (jitTimeIt && !timer.line.empty() && !std::count(todo.begin(), todo.end(), i))
            llvm::errs() << "jit:" + timer.line + "\n";
    }

    if (!todo.empty()) {
        auto t0 = std::chrono::steady_clock::now();
        int compiled = 0, rounds = 0, width = 1;
#if !defined(USE_AMD) && !defined(USE_INTEL)
        // Servers sharing the cache split the kernels: each claims up to a pool's worth,
        // compiles them in parallel and loads the others' once published. A waiter breaks
        // stale claims (lockFree), and compiles without a claim after CNUM_JIT_WAIT_S (600) s
        // or when a published entry does not load here.
        width = CompilePool::get().size();
        static const double maxWait = [] {
            const char* e = getenv("CNUM_JIT_WAIT_S");
            return e && atof(e) > 0 ? atof(e) : 600.0;
        }();
        // target registration is not thread-safe (a race can corrupt the registry's list)
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
#ifdef USE_NVIDIA
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();
        cnEnsureContext();
        CUcontext ctx = nullptr;
        cuCtxGetCurrent(&ctx);
#endif
        std::rotate(todo.begin(), todo.begin() + getpid() % todo.size(), todo.end());
        for (int i : todo)
            items[i].waitStart = t0;
        auto pause = std::chrono::milliseconds(10); // between polls; doubles up to 200 ms
        while (!todo.empty()) {
            auto now = std::chrono::steady_clock::now();
            std::vector<int> mine, waiting;
            for (int i : todo) {
                Item& it = items[i];
                std::string lockPath = it.path + ".lock";
                // opened, not stat'ed: no stale negative NFS lookups
                if (rounds > 0 && !it.local &&
                    (fns[i] = loadCached(it.path, &moduleHandles[i], &it.local)))
                    continue;
                if (static_cast<int>(mine.size()) >= width) {
                    waiting.push_back(i);
                    continue;
                }
                int fd = -2;
                if (!it.path.empty() && !it.local) {
                    fd = rounds == 0 || lockFree(lockPath, it.watch)
                             ? Claims::get().take(lockPath)
                             : -1;
                    if (fd == -1 &&
                        std::chrono::duration<double>(now - it.waitStart).count() > maxWait)
                        fd = -2;
                }
                if (fd == -1) {
                    waiting.push_back(i);
                    continue;
                }
                // claimed: the entry may have been published since it was looked up
                if (fd >= 0 && (fns[i] = loadCached(it.path, &moduleHandles[i], &it.local))) {
                    Claims::get().release(fd, lockPath);
                    continue;
                }
                it.fd = fd;
                mine.push_back(i);
            }
            if (!mine.empty()) {
                std::vector<std::function<void()>> jobs;
                for (int i : mine)
                    jobs.push_back([&, i] {
                        struct Release { // the claim, also if the compile throws
                            int fd;
                            std::string lock;
                            ~Release() { Claims::get().release(fd, lock); }
                        } release{items[i].fd, items[i].path + ".lock"};
#ifdef USE_NVIDIA
                        cuCtxSetCurrent(ctx);
#endif
                        JitTimer timer;
                        jitTimer = jitTimeIt ? &timer : nullptr;
                        fns[i] = compileKernel(asts[i], items[i].path, &moduleHandles[i]);
                        jitTimer = nullptr;
                        if (jitTimeIt)
                            llvm::errs() << "jit:" + timer.line + "\n";
                    });
                CompilePool::get().run(jobs);
                compiled += mine.size();
            }
            if (!waiting.empty() && mine.empty() && waiting.size() == todo.size()) {
                std::this_thread::sleep_for(pause);
                pause = std::min(pause * 2, std::chrono::milliseconds(200));
            } else {
                pause = std::chrono::milliseconds(10);
            }
            todo.swap(waiting);
            ++rounds;
        }
#else
        for (int i : todo)
            fns[i] = compileKernel(asts[i], "", &moduleHandles[i]);
        compiled = todo.size();
#endif
        for (int i = 0; i < n; ++i)
            if (items[i].same < 0 && fns[i])
                kernelCache().emplace(items[i].key, std::make_pair(moduleHandles[i], fns[i]));
        if (jitTimeIt)
            llvm::errs() << llvm::formatv("jit batch: {0} kernels, {1} compiled here, {2} rounds, "
                                          "{3:F1} ms, {4} threads\n",
                                          n, compiled, rounds,
                                          std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now() - t0)
                                              .count(),
                                          width);
    }
    for (int i = 0; i < n; ++i)
        if (items[i].same >= 0) {
            fns[i] = fns[items[i].same];
            moduleHandles[i] = moduleHandles[items[i].same];
        }
}

int jit_prewarm(double budget_ms) {
#if !defined(USE_AMD) && !defined(USE_INTEL)
    static std::once_flag once; // SMP: the PEs of a process share the kernel cache
    std::call_once(once, [budget_ms] {
        if (budget_ms <= 0 || cacheDir().empty())
            return;
        auto t0 = std::chrono::steady_clock::now();
        auto spent = [&] {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();
        };
        std::string dir = targetCacheDir();
        if (dir.empty())
            return;
        // examine at most kScan entries, within the budget (stat per entry)
        constexpr int kScan = 1024;
        std::vector<std::pair<llvm::sys::TimePoint<>, std::string>> files;
        std::error_code ec;
        int seen = 0;
        for (llvm::sys::fs::directory_iterator it(dir, ec), end;
             it != end && !ec && ++seen <= kScan && spent() < budget_ms; it.increment(ec)) {
            llvm::sys::fs::file_status st;
            llvm::StringRef name = llvm::sys::path::filename(it->path());
            if (cacheFileKind(name) == CacheFile::Entry && name.ends_with(".o") &&
                !llvm::sys::fs::status(it->path(), st))
                files.emplace_back(st.getLastModificationTime(), it->path());
        }
        std::sort(files.rbegin(), files.rend());
        std::lock_guard<std::mutex> lock(jitMutex());
        for (const auto& [mtime, path] : files) {
            if (spent() >= budget_ms)
                break;
            void* module = nullptr;
            if (void* fn = loadCachedKernel(path, &module, /*soft=*/true))
                preloadedKernels().emplace(
                    path, PreloadedKernel{module, fn, llvm::sys::toTimeT(mtime)});
        }
    });
    std::lock_guard<std::mutex> lock(jitMutex());
    return static_cast<int>(preloadedKernels().size());
#else
    return 0;
#endif
}

#if !defined(USE_AMD) && !defined(USE_INTEL)
namespace {
/// See jit_cache_maintain (jit_api.hpp).
void evictCache(std::string base, std::string current) {
    namespace fs = llvm::sys::fs;
    using TimePoint = llvm::sys::TimePoint<>;
    auto now = std::chrono::system_clock::now();
    auto age = [&](TimePoint t) { return std::chrono::duration<double>(now - t).count(); };
    // .last-evict: the last completed pass. .evict-lock: a pass is running (or its server
    // exited first; then it is taken over after 10 minutes and the next pass resumes).
    std::string stamp = base + "/.last-evict", lock = base + "/.evict-lock";
    fs::file_status st;
    char real[PATH_MAX], home[PATH_MAX];
    const char* h = getenv("HOME");
    if (!llvm::sys::path::is_absolute(base) || !realpath(base.c_str(), real) ||
        std::string(real) == "/" || (h && realpath(h, home) && std::string(real) == home))
        return;
    if (!fs::exists(base + "/" + kCacheMarker)) {
        fprintf(stderr,
                "jit cache: %s is not marked %s (it held other files when first used); "
                "it is not evicted\n",
                base.c_str(), kCacheMarker);
        return;
    }
    if (!fs::status(stamp, st) && age(st.getLastModificationTime()) < 3600)
        return;
    int fd = open(lock.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0 && !fs::status(lock, st) && age(st.getLastModificationTime()) > 600) {
        unlink(lock.c_str());
        fd = open(lock.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    }
    if (fd < 0)
        return;
    close(fd);

    auto env = [](const char* name, double dflt) {
        const char* v = getenv(name);
        char* end = nullptr;
        double x = v ? strtod(v, &end) : 0;
        return v && end != v && *end == '\0' && x > 0 ? x : dflt;
    };
    double maxAge = env("CNUM_JIT_CACHE_DAYS", 7) * 86400;
    size_t maxFiles = static_cast<size_t>(env("CNUM_JIT_CACHE_MAX_FILES", 5000));
    size_t removed = 0;
    auto remove = [&](const std::string& p) { removed += !fs::remove(p); };

    // Entries of dir go to out and temporaries an hour old are removed; other files are
    // left alone. Returns the newest mtime.
    std::vector<std::pair<TimePoint, std::string>> entries;
    auto scan = [&](const std::string& dir, decltype(entries)& out,
                    std::vector<std::string>* subdirs) {
        TimePoint newest{};
        if (!fs::status(dir, st))
            newest = st.getLastModificationTime();
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec, false), end; it != end && !ec; it.increment(ec)) {
            const std::string& p = it->path();
            if (fs::status(p, st, false))
                continue;
            if (st.type() == fs::file_type::directory_file) {
                if (subdirs)
                    subdirs->push_back(p);
                continue;
            }
            TimePoint mt = st.getLastModificationTime();
            CacheFile kind = st.type() == fs::file_type::regular_file
                                 ? cacheFileKind(llvm::sys::path::filename(p))
                                 : CacheFile::Other;
            if (kind == CacheFile::Temp && age(mt) > 3600) {
                remove(p);
            } else if (kind == CacheFile::Entry) {
                newest = std::max(newest, mt);
                out.emplace_back(mt, p);
            }
        }
        return newest;
    };

    // Entries unused for maxAge go, the rest count towards maxFiles.
    auto keepRecent = [&](const decltype(entries)& files) {
        for (const auto& f : files) {
            if (age(f.first) > maxAge)
                remove(f.second);
            else
                entries.push_back(f);
        }
    };
    std::vector<std::string> dirs;
    decltype(entries) none;
    scan(base, none, &dirs);
    // Only build, target and cubin/ directories are entered, only cache files are deleted,
    // and directories go only once empty.
    for (const std::string& d : dirs) {
        llvm::StringRef name = llvm::sys::path::filename(d);
        decltype(entries) files;
        if (name == "cubin") {
            if (age(scan(d, files, nullptr)) > maxAge) { // unused (e.g. by newer servers)
                for (const auto& f : files)
                    remove(f.second);
                rmdir(d.c_str());
            } else {
                keepRecent(files);
            }
            continue;
        }
        if (!isBuildIdName(name))
            continue;
        std::vector<std::string> targets;
        scan(d, files, &targets); // entries here predate the target directories
        keepRecent(files);
        for (const std::string& t : targets) {
            if (!isTargetName(llvm::sys::path::filename(t)))
                continue;
            files.clear();
            if (age(scan(t, files, nullptr)) > maxAge) {
                for (const auto& f : files)
                    remove(f.second);
                rmdir(t.c_str());
                continue;
            }
            entries.insert(entries.end(), files.begin(), files.end());
        }
        if (name != llvm::sys::path::filename(current))
            rmdir(d.c_str());
    }
    if (entries.size() > maxFiles) {
        std::sort(entries.begin(), entries.end());
        for (size_t i = 0; i < entries.size() - maxFiles; ++i)
            remove(entries[i].second);
    }
    fd = open(stamp.c_str(), O_CREAT | O_WRONLY, 0644);
    if (fd >= 0) {
        close(fd);
        utimes(stamp.c_str(), nullptr);
    }
    unlink(lock.c_str());
    if (getenv("CNUM_JIT_TIME"))
        fprintf(stderr, "jit cache: %zu entries, %zu removed (max %zu files, %.0f days)\n",
                std::min(entries.size(), maxFiles), removed, maxFiles, maxAge / 86400);
}
} // namespace
#endif

void jit_cache_maintain() {
#if !defined(USE_AMD) && !defined(USE_INTEL)
    const char* on = getenv("CNUM_JIT_CACHE_EVICT");
    std::string base = cacheBase();
    if ((on && std::string(on) == "0") || base.empty() || cacheDir().empty())
        return;
    // the thread takes no signals (they belong to the Charm++ scheduler thread)
    sigset_t all, old;
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, &old);
    try {
        std::thread([base, current = cacheDir()] {
            try {
                evictCache(base, current);
            } catch (...) {
            }
        }).detach();
    } catch (...) {
    }
    pthread_sigmask(SIG_SETMASK, &old, nullptr);
#endif
}

void jit_free_module(void* moduleHandle) {
#if !defined(USE_NVIDIA) && !defined(USE_AMD) && !defined(USE_INTEL)
    if (!moduleHandle)
        return;
    std::lock_guard<std::mutex> lock(jitMutex());
    bool cached = false; // modules are shared by the PEs of a process: free once
    for (auto it = kernelCache().begin(); it != kernelCache().end();) {
        if (it->second.first == moduleHandle) {
            it = kernelCache().erase(it);
            cached = true;
        } else
            ++it;
    }
    if (cached)
        delete static_cast<JitModule*>(moduleHandle);
#endif
}
