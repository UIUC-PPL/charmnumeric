#include "jit.hpp"
#include "jit_api.hpp"

#include "mlir/IR/Verifier.h"

#include <limits>

#ifdef JIT_ENABLE_GPU_BACKEND
#include "llvm/IR/LegacyPassManager.h"
#endif
#ifdef USE_NVIDIA
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#endif

namespace {
void declareExternalCallees(mlir::ModuleOp module);
}

namespace {
// One context per process: creating one (dialect loading, thread pool) per kernel dominated compile time.
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

    // Determine element type from the AST's dtype
    mlir::Type elemType;
    DType astDtype = DType::FLOAT32;
    for (ASTNode* op : ast->roots)
        if (op->dtype != DType::FLOAT32) {
            astDtype = op->dtype;
            break;
        }
    switch (astDtype) {
    case DType::FLOAT64:
        elemType = Float64Type::get(&context);
        break;
    case DType::INT32:
        elemType = IntegerType::get(&context, 32);
        break;
    case DType::INT64:
        elemType = IntegerType::get(&context, 64);
        break;
    default:
        elemType = Float32Type::get(&context);
        break;
    }

    auto dynamicStridedMemRefType = [&](int64_t rank) {
        SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
        SmallVector<int64_t> dynStrides(rank, ShapedType::kDynamic);
        auto layout = StridedLayoutAttr::get(&context, ShapedType::kDynamic, dynStrides);
        return MemRefType::get(dynShape, elemType, layout);
    };

    // --- 2. Create a func.func: memref inputs, broadcast scalars, then output memrefs ---
    SmallVector<Type> argTypes;
    for (const LeafUse& leaf : memrefLeaves)
        argTypes.push_back(dynamicStridedMemRefType(leaf.operand->ndims));
    // Broadcast leaves are scalar function arguments (runtime values, not constants)
    for (size_t i = 0; i < broadcastLeaves.size(); ++i)
        argTypes.push_back(elemType);
    for (size_t i = 0; i < scalarLeaves.size(); ++i)
        argTypes.push_back(elemType);
    for (ASTNode* root : outputRoots)
        argTypes.push_back(dynamicStridedMemRefType(root->ndims));
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

void MLIRJitCompiler::emitOp(ASTNode* node, mlir::Location loc) {
    using namespace mlir;

    auto op = static_cast<Opcode>(node->opcode);
    Value result;

    switch (op) {
    case Opcode::NOOP:
    case Opcode::CREATE:
        return;

    case Opcode::SET_REGION: {
        result = lookupOperand(node->operands[1], node->get_operand_region(1));
        break;
    }

    default: {
        // Binary elementwise ops
        auto& binReg = binaryOpRegistry();
        auto binIt = binReg.find(node->opcode);
        if (binIt != binReg.end()) {
            Value lhs = lookupOperand(node->operands[0], node->get_operand_region(0));
            Value rhs = lookupOperand(node->operands[1], node->get_operand_region(1));

            bool lhsIsScalar = !mlir::isa<MemRefType>(lhs.getType());
            bool rhsIsScalar = !mlir::isa<MemRefType>(rhs.getType());

            auto elemType = lhsIsScalar ? lhs.getType() :
                mlir::cast<MemRefType>(lhs.getType()).getElementType();
            bool isInt = mlir::isa<IntegerType>(elemType);
            const auto& entry = binIt->second;
            const auto& bodyFn = (isInt && entry.int_body) ? entry.int_body : entry.float_body;

            if (lhsIsScalar && rhsIsScalar) {
                // Both operands are scalars — emit a plain scalar op
                result = bodyFn(*builder, loc, lhs, rhs);
                break;
            }

            Value arrayOperand = lhsIsScalar ? rhs : lhs;
            auto srcType = mlir::cast<MemRefType>(arrayOperand.getType());
            int64_t rank = srcType.getRank();
            SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
            auto outType = MemRefType::get(dynShape, elemType);

            SmallVector<Value> dimSizes;
            for (int64_t d = 0; d < rank; ++d)
                dimSizes.push_back(memref::DimOp::create(*builder, loc, arrayOperand, d).getResult());
            result = memref::AllocOp::create(*builder, loc, outType, ValueRange(dimSizes));

            emitGenericArrayOp(lhs, rhs, result, bodyFn);

            if (node->is_temp)
                tempValues.push_back(result);
            break;
        }

        // Unary elementwise ops
        auto& unReg = unaryOpRegistry();
        auto unIt = unReg.find(node->opcode);
        if (unIt != unReg.end()) {
            Value input = lookupOperand(node->operands[0], node->get_operand_region(0));
            bool inputIsScalar = !mlir::isa<MemRefType>(input.getType());
            auto elemType = inputIsScalar ? input.getType()
                                          : mlir::cast<MemRefType>(input.getType()).getElementType();
            bool isInt = mlir::isa<IntegerType>(elemType);
            const auto& entry = unIt->second;
            const auto& bodyFn = (isInt && entry.int_body) ? entry.int_body : entry.float_body;

            if (inputIsScalar) {
                // Broadcast (size-1) or scalar input — emit a plain scalar op
                result = bodyFn(*builder, loc, input);
                break;
            }

            auto srcType = mlir::cast<MemRefType>(input.getType());
            int64_t rank = srcType.getRank();
            SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
            auto outType = MemRefType::get(dynShape, elemType);

            SmallVector<Value> dimSizes;
            for (int64_t d = 0; d < rank; ++d)
                dimSizes.push_back(memref::DimOp::create(*builder, loc, input, d).getResult());
            result = memref::AllocOp::create(*builder, loc, outType, ValueRange(dimSizes));

            emitGenericUnaryOp(input, result, bodyFn);

            if (node->is_temp)
                tempValues.push_back(result);
            break;
        }

        // Ternary elementwise ops (where, clip); any operand may be a scalar
        auto& terReg = ternaryOpRegistry();
        auto terIt = terReg.find(node->opcode);
        if (terIt != terReg.end()) {
            SmallVector<Value, 3> ins;
            Value arrayOperand;
            for (int i = 0; i < 3; ++i) {
                ins.push_back(lookupOperand(node->operands[i], node->get_operand_region(i)));
                if (!arrayOperand && mlir::isa<MemRefType>(ins.back().getType()))
                    arrayOperand = ins.back();
            }
            // Single element type per kernel: take it from the value operand
            Value typeSrc = ins[1];
            auto elemType = mlir::isa<MemRefType>(typeSrc.getType())
                                ? mlir::cast<MemRefType>(typeSrc.getType()).getElementType()
                                : typeSrc.getType();
            bool isInt = mlir::isa<IntegerType>(elemType);
            const auto& entry = terIt->second;
            const auto& bodyFn = (isInt && entry.int_body) ? entry.int_body : entry.float_body;
            auto naryBody = [&](OpBuilder& b, Location l, ValueRange args) {
                return bodyFn(b, l, args[0], args[1], args[2]);
            };

            if (!arrayOperand) {
                // All operands are scalars — emit a plain scalar op
                result = naryBody(*builder, loc, ins);
                break;
            }

            int64_t rank = mlir::cast<MemRefType>(arrayOperand.getType()).getRank();
            SmallVector<int64_t> dynShape(rank, ShapedType::kDynamic);
            auto outType = MemRefType::get(dynShape, elemType);
            SmallVector<Value> dimSizes;
            for (int64_t d = 0; d < rank; ++d)
                dimSizes.push_back(
                    memref::DimOp::create(*builder, loc, arrayOperand, d).getResult());
            result = memref::AllocOp::create(*builder, loc, outType, ValueRange(dimSizes));

            emitGenericNaryOp(ins, result, naryBody);

            if (node->is_temp)
                tempValues.push_back(result);
            break;
        }

        llvm::errs() << "Unknown opcode in JIT emitOp: " << node->opcode << "\n";
        return;
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
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
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

bool MLIRJitCompiler::lowerToNVIDIAHost() {
    using namespace mlir;

    // 1. Every linalg op becomes one multi-dim scf.parallel (parallel by
    //    construction, so nothing is left to run on the host), tiled so the
    //    innermost tile maps to threads and the rest to blocks. Tile sizes keep
    //    blockDim within limits (x,y <= 1024, z <= 64).
    int64_t rank = 1;
    module->walk([&](linalg::LinalgOp op) {
        rank = std::max<int64_t>(rank, op.getNumLoops());
    });
    SmallVector<int64_t> tiles(rank, 1);
    tiles.back() = (rank >= 3) ? 64 : 256;

    PassManager pm(&context);
    pm.addPass(createConvertLinalgToParallelLoopsPass());
    pm.nest<func::FuncOp>().addPass(createParallelLoopTilingPass(tiles, /*noMinMaxBounds=*/true));
    pm.nest<func::FuncOp>().addPass(createGpuMapParallelLoopsPass());
    pm.addPass(createConvertParallelLoopToGpuPass());
    if (failed(pm.run(*module)))
        return false;

    // 2. Temporaries (memref.alloc / host-side memref.alloca) are read by kernels,
    //    and the host may initialize scalar buffers, so put them in managed
    //    (host_shared) memory. The sync gpu.alloc lowering requires host_shared;
    //    gpu.dealloc lowers only in async form, so wrap it in gpu.wait.
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
    auto emitDealloc = [&](Location loc, Value mem) {
        Type tokTy = gpu::AsyncTokenType::get(&context);
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
        auto gpuAlloc = gpu::AllocOp::create(b, op->getLoc(), type, /*asyncToken=*/Type(),
                                             ValueRange{}, dynSizes, symOps, /*hostShared=*/true);
        op->getResult(0).replaceAllUsesWith(gpuAlloc.getMemref());
        // allocas have no dealloc: free them before the function returns
        if (isa<memref::AllocaOp>(op)) {
            auto func = gpuAlloc->getParentOfType<func::FuncOp>();
            func.walk([&](func::ReturnOp ret) {
                b.setInsertionPoint(ret);
                emitDealloc(ret.getLoc(), gpuAlloc.getMemref());
            });
        }
        op->erase();
    }
    for (memref::DeallocOp op : deallocs) {
        b.setInsertionPoint(op);
        emitDealloc(op.getLoc(), op.getMemref());
        op.erase();
    }

    // 3. Standard GPU pipeline: outline kernels, lower them to NVVM and embed
    //    PTX (the driver JITs it for the actual device), lower the host side to
    //    LLVM with mgpu* runtime calls.
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
    return true;
}

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
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) {
        llvm::errs() << "generateNVIDIAHost: failed to create ExecutionEngine: "
                     << llvm::toString(maybeEngine.takeError()) << "\n";
        return nullptr;
    }
    // Run the global constructors that load the embedded PTX module (only
    // invokePacked() does this implicitly; the dispatcher calls the packed
    // function pointer directly).
    (*maybeEngine)->initialize();
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

/// i1 predicate -> 0/1 in element type `t` (masks: there is no bool dtype).
Value toMask(OpBuilder& b, Location loc, Value pred, mlir::Type t) {
    if (mlir::isa<mlir::FloatType>(t))
        return arith::UIToFPOp::create(b, loc, t, pred).getResult();
    return arith::ExtUIOp::create(b, loc, t, pred).getResult();
}

/// i1: v != 0 (NaN counts as true, like NumPy's bool(nan)).
Value isNonzero(OpBuilder& b, Location loc, Value v) {
    Value zero = constLike(b, loc, v, 0.0);
    if (mlir::isa<mlir::FloatType>(v.getType()))
        return fcmp(b, loc, arith::CmpFPredicate::UNE, v, zero);
    return icmp(b, loc, arith::CmpIPredicate::ne, v, zero);
}

/// Float compare op producing a 0/1 mask in the operand type.
BinaryOpEntry::Fn fcmpMask(arith::CmpFPredicate p) {
    return [p](OpBuilder& b, Location loc, Value x, Value y) {
        return toMask(b, loc, fcmp(b, loc, p, x, y), x.getType());
    };
}

/// Integer compare op producing a 0/1 mask in the operand type.
BinaryOpEntry::Fn icmpMask(arith::CmpIPredicate p) {
    return [p](OpBuilder& b, Location loc, Value x, Value y) {
        return toMask(b, loc, icmp(b, loc, p, x, y), x.getType());
    };
}

/// Integer x**e by square-and-multiply over every exponent bit (branch-free,
/// so it lowers the same on CPU and GPU). Wraps on overflow like NumPy. For
/// e < 0 the result is 1 for x == 1, +-1 for x == -1 and 0 otherwise (NumPy
/// raises instead).
Value intPow(OpBuilder& b, Location loc, Value x, Value e) {
    unsigned bits = mlir::cast<mlir::IntegerType>(x.getType()).getWidth();
    Value one = constLike(b, loc, x, 1.0);
    Value zero = constLike(b, loc, x, 0.0);
    Value r = one, base = x;
    for (unsigned i = 0; i + 1 < bits; ++i) {
        Value sh = arith::ShRSIOp::create(b, loc, e, constLike(b, loc, e, i)).getResult();
        Value bit = arith::TruncIOp::create(b, loc, b.getI1Type(), sh).getResult();
        r = sel(b, loc, bit, arith::MulIOp::create(b, loc, r, base).getResult(), r);
        if (i + 2 < bits)
            base = arith::MulIOp::create(b, loc, base, base).getResult();
    }
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

/// C fmod(x, y), exact. arith.remf cannot be used: the NVVM pipeline lowers
/// arith to LLVM before convert-gpu-to-nvvm, and llvm.frem is illegal there.
/// So call libm's fmod on the host and libdevice's __nv_fmod on NVIDIA GPUs.
/// (The callee is declared afterwards by declareExternalCallees: the body is
/// built before the linalg.generic is attached to the module.)
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

/// Add private declarations for callees (e.g. fmod) that are called but not defined.
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

/// Wraps a math dialect unary op as a registry body.
template <typename OpT>
UnaryOpEntry::Fn mathUnary() {
    return [](OpBuilder& b, Location loc, Value v) { return OpT::create(b, loc, v).getResult(); };
}

/// Wraps an arith binary op as a registry body.
template <typename OpT>
BinaryOpEntry::Fn arithBinary() {
    return [](OpBuilder& b, Location loc, Value x, Value y) {
        return OpT::create(b, loc, x, y).getResult();
    };
}

Value identityUnary(OpBuilder&, Location, Value v) { return v; }
} // namespace

const std::unordered_map<int, BinaryOpEntry>& MLIRJitCompiler::binaryOpRegistry() {
    static const std::unordered_map<int, BinaryOpEntry> registry = {
        {(int)Opcode::ADD,
         {[](mlir::OpBuilder& b, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
              return mlir::arith::AddFOp::create(b, loc, lhs, rhs).getResult();
          },
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
              return mlir::arith::AddIOp::create(b, loc, lhs, rhs).getResult();
          }}},
        {(int)Opcode::SUB,
         {[](mlir::OpBuilder& b, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
              return mlir::arith::SubFOp::create(b, loc, lhs, rhs).getResult();
          },
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
              return mlir::arith::SubIOp::create(b, loc, lhs, rhs).getResult();
          }}},
        {(int)Opcode::MUL,
         {[](mlir::OpBuilder& b, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
              return mlir::arith::MulFOp::create(b, loc, lhs, rhs).getResult();
          },
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
              return mlir::arith::MulIOp::create(b, loc, lhs, rhs).getResult();
          }}},
        {(int)Opcode::DIV,
         {[](mlir::OpBuilder& b, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
              return mlir::arith::DivFOp::create(b, loc, lhs, rhs).getResult();
          },
          [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
              return mlir::arith::DivSIOp::create(b, loc, lhs, rhs).getResult();
          }}},
        {(int)Opcode::POW,
         {arithBinary<math::PowFOp>(), [](OpBuilder& b, Location loc, Value x, Value e) {
              return intPow(b, loc, x, e);
          }}},
        {(int)Opcode::MAXIMUM, {arithBinary<arith::MaximumFOp>(), arithBinary<arith::MaxSIOp>()}},
        {(int)Opcode::MINIMUM, {arithBinary<arith::MinimumFOp>(), arithBinary<arith::MinSIOp>()}},
        {(int)Opcode::FMAX, {arithBinary<arith::MaxNumFOp>(), arithBinary<arith::MaxSIOp>()}},
        {(int)Opcode::FMIN, {arithBinary<arith::MinNumFOp>(), arithBinary<arith::MinSIOp>()}},
        {(int)Opcode::MOD,
         {floatMod,
          [](OpBuilder& b, Location loc, Value x, Value y) { return intDivMod(b, loc, x, y, true); }}},
        {(int)Opcode::FLOOR_DIV,
         {floatFloorDiv, [](OpBuilder& b, Location loc, Value x, Value y) {
              return intDivMod(b, loc, x, y, false);
          }}},
        // Comparisons: ordered for <, <=, == (NaN -> 0), unordered for != (NaN -> 1)
        {(int)Opcode::LESS,
         {fcmpMask(arith::CmpFPredicate::OLT), icmpMask(arith::CmpIPredicate::slt)}},
        {(int)Opcode::LESS_EQUAL,
         {fcmpMask(arith::CmpFPredicate::OLE), icmpMask(arith::CmpIPredicate::sle)}},
        {(int)Opcode::EQUAL,
         {fcmpMask(arith::CmpFPredicate::OEQ), icmpMask(arith::CmpIPredicate::eq)}},
        {(int)Opcode::NOT_EQUAL,
         {fcmpMask(arith::CmpFPredicate::UNE), icmpMask(arith::CmpIPredicate::ne)}},
    };
    return registry;
}

const std::unordered_map<int, UnaryOpEntry>& MLIRJitCompiler::unaryOpRegistry() {
    static const std::unordered_map<int, UnaryOpEntry> registry = {
        {(int)Opcode::TANH,
         {[](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) {
              return mlir::math::TanhOp::create(b, loc, v).getResult();
          },
          nullptr}},
        {(int)Opcode::EXP,
         {[](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) {
              return mlir::math::ExpOp::create(b, loc, v).getResult();
          },
          nullptr}},
        // Float-only math (the frontend rejects integer inputs: NumPy would
        // return float64, and there is no in-kernel dtype conversion).
        {(int)Opcode::SQRT, {mathUnary<math::SqrtOp>(), nullptr}},
        {(int)Opcode::LOG, {mathUnary<math::LogOp>(), nullptr}},
        {(int)Opcode::LOG2, {mathUnary<math::Log2Op>(), nullptr}},
        {(int)Opcode::LOG10, {mathUnary<math::Log10Op>(), nullptr}},
        {(int)Opcode::LOG1P, {log1pAccurate, nullptr}},
        {(int)Opcode::EXP2, {mathUnary<math::Exp2Op>(), nullptr}},
        {(int)Opcode::EXPM1, {expm1Accurate, nullptr}},
        {(int)Opcode::SIN, {mathUnary<math::SinOp>(), nullptr}},
        {(int)Opcode::COS, {mathUnary<math::CosOp>(), nullptr}},
        // Integer inputs keep their dtype (rounding is the identity on ints)
        {(int)Opcode::ABS, {mathUnary<math::AbsFOp>(), mathUnary<math::AbsIOp>()}},
        {(int)Opcode::FLOOR, {mathUnary<math::FloorOp>(), identityUnary}},
        {(int)Opcode::CEIL, {mathUnary<math::CeilOp>(), identityUnary}},
        {(int)Opcode::RINT, {mathUnary<math::RoundEvenOp>(), identityUnary}},
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
          }}},
        // Predicates: 0/1 masks in the input type
        {(int)Opcode::ISNAN,
         {[](OpBuilder& b, Location loc, Value v) {
              return toMask(b, loc, fcmp(b, loc, arith::CmpFPredicate::UNO, v, v), v.getType());
          },
          [](OpBuilder& b, Location loc, Value v) { return constLike(b, loc, v, 0.0); }}},
        {(int)Opcode::ISINF,
         {[](OpBuilder& b, Location loc, Value v) {
              Value a = math::AbsFOp::create(b, loc, v).getResult();
              Value inf = constLike(b, loc, v, std::numeric_limits<double>::infinity());
              return toMask(b, loc, fcmp(b, loc, arith::CmpFPredicate::OEQ, a, inf), v.getType());
          },
          [](OpBuilder& b, Location loc, Value v) { return constLike(b, loc, v, 0.0); }}},
        {(int)Opcode::ISFINITE,
         {[](OpBuilder& b, Location loc, Value v) {
              Value a = math::AbsFOp::create(b, loc, v).getResult();
              Value inf = constLike(b, loc, v, std::numeric_limits<double>::infinity());
              return toMask(b, loc, fcmp(b, loc, arith::CmpFPredicate::OLT, a, inf), v.getType());
          },
          [](OpBuilder& b, Location loc, Value v) { return constLike(b, loc, v, 1.0); }}},
    };
    return registry;
}

const std::unordered_map<int, TernaryOpEntry>& MLIRJitCompiler::ternaryOpRegistry() {
    static const std::unordered_map<int, TernaryOpEntry> registry = {
        // where(cond, x, y): any nonzero cond (including NaN) selects x
        {(int)Opcode::WHERE,
         {[](OpBuilder& b, Location loc, Value c, Value x, Value y) {
              return sel(b, loc, isNonzero(b, loc, c), x, y);
          },
          [](OpBuilder& b, Location loc, Value c, Value x, Value y) {
              return sel(b, loc, isNonzero(b, loc, c), x, y);
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

void* jit_compile_ast(void* ast, void** moduleHandleOut) {
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
    if (!node_jit.verify())
        return nullptr;
    node_jit.optimizeAndFuse();
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
