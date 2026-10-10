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
#include "mlir/Dialect/Complex/IR/Complex.h"
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

// Registry entries for elementwise ops. Bodies receive operands already
// converted to the op's compute type (float16 math runs in float32; bool is
// i8 0/1) and may return i1 (predicates). Optional bodies: uint_body (default
// int_body), complex_body (complex<f32|f64>), bool_body (default int_body).

/// Registry entry for a binary elementwise op.
struct BinaryOpEntry {
    using Fn = std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>;
    Fn float_body;
    Fn int_body; // nullptr if not supported for integers
    Fn uint_body = nullptr;
    Fn complex_body = nullptr;
    Fn bool_body = nullptr;
};

/// Registry entry for a unary elementwise op.
struct UnaryOpEntry {
    using Fn = std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Value)>;
    Fn float_body;
    Fn int_body; // nullptr if not supported for integers
    Fn uint_body = nullptr;
    Fn complex_body = nullptr;
    Fn bool_body = nullptr;
};

/// Ternary op: where, clip.
struct TernaryOpEntry {
    using Fn = std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Value,
                                         mlir::Value, mlir::Value)>;
    Fn float_body;
    Fn int_body; // nullptr if not supported for integers
    Fn uint_body = nullptr;
    Fn complex_body = nullptr;
    Fn bool_body = nullptr;
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
    bool verify();
    void dumpMLIR();
    void printModule(llvm::raw_ostream& os);

#ifdef USE_NVIDIA
    /// Host-side launcher (PTX embedded, device pointers in memref descriptors);
    /// same packed args as the CPU path. Load with loadCPU().
    std::unique_ptr<mlir::ExecutionEngine> generateNVIDIAHost();
#endif
#ifdef JIT_ENABLE_GPU_BACKEND
    std::string generateAMD();
    std::string generateIntel();
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
    mlir::MLIRContext& context;
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
    mlir::Value allocLike(mlir::Value arrayOperand, DType dt);
    mlir::Value convertArray(mlir::Value v, DType from, DType to, bool is_temp);
    void emitOp(ASTNode* node, mlir::Location loc);

    // Lowering helpers
    bool lowerToCPU();
#ifdef JIT_ENABLE_GPU_BACKEND
    bool lowerToGPU();
    std::string
    translateToTarget(llvm::StringRef triple, llvm::StringRef cpu, llvm::StringRef features,
                      llvm::CodeGenFileType fileType = llvm::CodeGenFileType::AssemblyFile);
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

        // Bare scalars are captured by the body instead of becoming operands.
        auto identityMap = builder->getMultiDimIdentityMap(rank);
        bool aIsScalar = !mlir::isa<MemRefType>(a.getType());
        bool bIsScalar = !mlir::isa<MemRefType>(b.getType());
        SmallVector<Value> inputs;
        if (!aIsScalar)
            inputs.push_back(a);
        if (!bIsScalar)
            inputs.push_back(b);
        SmallVector<AffineMap, 3> indexingMaps(inputs.size() + 1, identityMap);
        SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);

        linalg::GenericOp::create(
            *builder,
            loc, TypeRange{}, ValueRange(inputs), ValueRange{out}, indexingMaps, iterTypes,
            [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
                Value x = aIsScalar ? a : args[0];
                Value y = bIsScalar ? b : args[aIsScalar ? 0 : 1];
                Value result = bodyBuilder(nestedBuilder, nestedLoc, x, y);
                linalg::YieldOp::create(nestedBuilder, nestedLoc, result);
            });
    }

    // N-ary element-wise op helper: bare scalar inputs are captured by the
    // body; memref inputs use the identity map.
    template <typename BodyBuilderFn>
    void emitGenericNaryOp(mlir::ValueRange ins, mlir::Value out, BodyBuilderFn bodyBuilder) {
        using namespace mlir;
        auto loc = builder->getUnknownLoc();

        auto outType = mlir::cast<MemRefType>(out.getType());
        int64_t rank = outType.getRank();
        auto identityMap = builder->getMultiDimIdentityMap(rank);

        SmallVector<Value> inputs;
        for (Value v : ins)
            if (mlir::isa<MemRefType>(v.getType()))
                inputs.push_back(v);
        SmallVector<AffineMap> indexingMaps(inputs.size() + 1, identityMap);
        SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
        SmallVector<Value> scalarsAndArrays(ins.begin(), ins.end());

        linalg::GenericOp::create(
            *builder, loc, TypeRange{}, ValueRange(inputs), ValueRange{out}, indexingMaps,
            iterTypes, [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
                SmallVector<Value> vals;
                unsigned k = 0;
                for (Value v : scalarsAndArrays)
                    vals.push_back(mlir::isa<MemRefType>(v.getType()) ? args[k++] : v);
                Value result = bodyBuilder(nestedBuilder, nestedLoc, ValueRange(vals));
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
