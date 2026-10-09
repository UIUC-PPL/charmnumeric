#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// LLVM Backend
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

// MLIR Execution Engine
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

// MLIR Translation
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// SPIR-V Serialization
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Target/SPIRV/Serialization.h"

// MLIR Core
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

// Dialects
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

// Passes & Conversions
#include "mlir/Conversion/Passes.h" // umbrella: declares all conversion pass factories
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

// Application AST / Opcodes
#include "array_region.hpp"
#include "charmtyles/core/dag.hpp"
#include "opcodes.hpp"

#if defined(USE_NVIDIA)
#include <cuda.h>
#elif defined(USE_AMD)
#include <hip/hip_runtime.h>
#elif defined(USE_INTEL)
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#endif

/// Registry entry for a binary elementwise op.
struct BinaryOpEntry {
    using Fn = std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>;
    Fn float_body;
    Fn int_body; // nullptr if not supported for integers
};

/// Registry entry for a unary elementwise op.
struct UnaryOpEntry {
    using Fn = std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Value)>;
    Fn float_body;
    Fn int_body; // nullptr if not supported for integers
};

/// Registry entry for a ternary elementwise op (where, clip).
struct TernaryOpEntry {
    using Fn = std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Value,
                                         mlir::Value, mlir::Value)>;
    Fn float_body;
    Fn int_body; // nullptr if not supported for integers
};

struct LeafUse {
    ASTNode* operand;
    Region* region;
};

/**
 * @brief A JIT Compiler to transform Array ASTs into Fused GPU Kernels.
 */
class MLIRJitCompiler {
  public:
    MLIRJitCompiler();

    void buildFromAST(void* astPtr);
    bool optimizeAndFuse();
    std::unique_ptr<mlir::ExecutionEngine> generateCPU();
    bool loadCPU(std::unique_ptr<mlir::ExecutionEngine>& engine, const char* kernelName,
                 void** moduleOut, void** functionOut);
    void dumpMLIR();

#ifdef USE_NVIDIA
    /// Lower the fused function to a host-side launcher (kernels embedded as PTX,
    /// launched through the MLIR CUDA runtime wrappers) and JIT it. The packed
    /// entry point takes the same arguments as the CPU path, with device pointers
    /// in the memref descriptors; load it with loadCPU().
    std::unique_ptr<mlir::ExecutionEngine> generateNVIDIAHost();
#endif
#ifdef JIT_ENABLE_GPU_BACKEND
    std::string generateNVIDIA();
    std::string generateAMD();
    std::string generateIntel();
    bool loadNVIDIA(const std::string& ptx, const char* kernelName, void** moduleOut,
                    void** functionOut);
    bool loadAMD(const std::string& gcn, const char* kernelName, void** moduleOut,
                 void** functionOut);
    bool loadIntel(const std::string& spirv, const char* kernelName, void** moduleOut,
                   void** functionOut);
#endif

    /// Op registries: map opcode → body builder lambdas.
    static const std::unordered_map<int, BinaryOpEntry>& binaryOpRegistry();
    static const std::unordered_map<int, UnaryOpEntry>& unaryOpRegistry();
    static const std::unordered_map<int, TernaryOpEntry>& ternaryOpRegistry();

  private:
#ifdef USE_NVIDIA
    bool lowerToNVIDIAHost();
#endif
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::unique_ptr<mlir::OpBuilder> builder;

    /// Maps AST node result_name → MLIR Value (memref or scalar) produced by that node.
    std::unordered_map<int, mlir::Value> nodeValues;
    /// Distinguish repeated views of the same array by the carried operand region.
    std::unordered_map<std::string, mlir::Value> leafValues;
    std::vector<mlir::Value> tempValues; // deferred deallocs for temporaries

    // AST helpers
    void collectLeaves(AST* ast, std::vector<LeafUse>& memrefLeaves,
                       std::vector<ASTNode*>& scalarLeaves,
                       std::vector<ASTNode*>& broadcastLeaves);
    std::string leafUseKey(ASTNode* operand, Region* region) const;
    mlir::Value lookupOperand(ASTNode* operand, Region* region);
    void emitOp(ASTNode* node, mlir::Location loc);

    // Lowering helpers
    bool lowerToCPU();
#ifdef JIT_ENABLE_GPU_BACKEND
    bool lowerToGPU();
    std::string
    translateToTarget(llvm::StringRef triple, llvm::StringRef cpu, llvm::StringRef features,
                      llvm::CodeGenFileType fileType = llvm::CodeGenFileType::AssemblyFile);
    std::string serializeToPTX();
    std::string serializeToGCN();
    std::string serializeToSPIRV();
#endif

    // Binary element-wise op helper (template must be in header)
    template <typename BodyBuilderFn>
    void emitGenericArrayOp(mlir::Value a, mlir::Value b, mlir::Value out,
                            BodyBuilderFn bodyBuilder) {
        using namespace mlir;
        auto loc = builder->getUnknownLoc();

        // Derive rank from the output memref (always a memref)
        auto outType = mlir::cast<MemRefType>(out.getType());
        int64_t rank = outType.getRank();

        bool aIsScalar = !mlir::isa<MemRefType>(a.getType());
        bool bIsScalar = !mlir::isa<MemRefType>(b.getType());

        // Identity map for memref operands; empty-result map for 0-d scalar memrefs
        auto identityMap = builder->getMultiDimIdentityMap(rank);
        auto scalarMap0 = AffineMap::get(rank, /*symbolCount=*/0, /*results=*/{}, &context);

        // linalg.generic requires shaped types (MemRef/Tensor) for all operands.
        // Wrap bare scalar values in a 0-d memref so they broadcast correctly.
        auto wrapScalar = [&](Value v) -> Value {
            auto zeroD = MemRefType::get({}, v.getType());
            auto mem = memref::AllocaOp::create(*builder, loc, zeroD);
            memref::StoreOp::create(*builder, loc, v, mem, ValueRange{});
            return mem;
        };

        SmallVector<AffineMap, 3> indexingMaps;
        indexingMaps.push_back(aIsScalar ? scalarMap0 : identityMap);
        indexingMaps.push_back(bIsScalar ? scalarMap0 : identityMap);
        indexingMaps.push_back(identityMap); // output is always memref

        SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);

        SmallVector<Value> inputs;
        SmallVector<Value> outputs{out};
        inputs.push_back(aIsScalar ? wrapScalar(a) : a);
        inputs.push_back(bIsScalar ? wrapScalar(b) : b);

        linalg::GenericOp::create(
            *builder,
            loc, TypeRange{}, ValueRange(inputs), ValueRange(outputs), indexingMaps, iterTypes,
            [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
                Value result = bodyBuilder(nestedBuilder, nestedLoc, args[0], args[1]);
                linalg::YieldOp::create(nestedBuilder, nestedLoc, result);
            });
    }

    // N-ary element-wise op helper: any input may be a bare scalar (wrapped in
    // a 0-d memref so it broadcasts); memref inputs use the identity map.
    template <typename BodyBuilderFn>
    void emitGenericNaryOp(mlir::ValueRange ins, mlir::Value out, BodyBuilderFn bodyBuilder) {
        using namespace mlir;
        auto loc = builder->getUnknownLoc();

        auto outType = mlir::cast<MemRefType>(out.getType());
        int64_t rank = outType.getRank();
        auto identityMap = builder->getMultiDimIdentityMap(rank);
        auto scalarMap0 = AffineMap::get(rank, /*symbolCount=*/0, /*results=*/{}, &context);

        SmallVector<AffineMap> indexingMaps;
        SmallVector<Value> inputs;
        for (Value v : ins) {
            if (mlir::isa<MemRefType>(v.getType())) {
                indexingMaps.push_back(identityMap);
                inputs.push_back(v);
            } else {
                auto mem = memref::AllocaOp::create(*builder, loc, MemRefType::get({}, v.getType()));
                memref::StoreOp::create(*builder, loc, v, mem, ValueRange{});
                indexingMaps.push_back(scalarMap0);
                inputs.push_back(mem);
            }
        }
        indexingMaps.push_back(identityMap); // output
        SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);

        linalg::GenericOp::create(
            *builder, loc, TypeRange{}, ValueRange(inputs), ValueRange{out}, indexingMaps,
            iterTypes, [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
                Value result = bodyBuilder(nestedBuilder, nestedLoc, args.drop_back());
                linalg::YieldOp::create(nestedBuilder, nestedLoc, result);
            });
    }

    // Unary element-wise op helper
    template <typename BodyBuilderFn>
    void emitGenericUnaryOp(mlir::Value input, mlir::Value out, BodyBuilderFn bodyBuilder) {
        using namespace mlir;
        auto loc = builder->getUnknownLoc();

        auto outType = mlir::cast<MemRefType>(out.getType());
        int64_t rank = outType.getRank();

        auto identityMap = builder->getMultiDimIdentityMap(rank);
        SmallVector<AffineMap, 2> indexingMaps = {identityMap, identityMap};
        SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);

        linalg::GenericOp::create(
            *builder,
            loc, TypeRange{}, ValueRange{input}, ValueRange{out}, indexingMaps, iterTypes,
            [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
                Value result = bodyBuilder(nestedBuilder, nestedLoc, args[0]);
                linalg::YieldOp::create(nestedBuilder, nestedLoc, result);
            });
    }
};
