cmake_minimum_required(VERSION 3.20)
project(charmnumeric_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NDEBUG "Enable NDEBUG preprocessor define (-DNDEBUG)" ON)

# =====================================================================
# Charm++ -- located via CHARM_HOME environment variable
# =====================================================================
if(NOT DEFINED ENV{CHARM_HOME} AND NOT DEFINED CHARM_HOME)
    message(FATAL_ERROR "CHARM_HOME is not set. Point it to your Charm++ installation.")
endif()
if(DEFINED ENV{CHARM_HOME} AND NOT DEFINED CHARM_HOME)
    set(CHARM_HOME "$ENV{CHARM_HOME}")
endif()

set(CHARMC "${CHARM_HOME}/bin/charmc")

# Charmtyles root (three levels up from this CMakeLists.txt)
set(CHARMTYLES_HOME "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

# =====================================================================
# MLIR / LLVM
# =====================================================================
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# =====================================================================
# Include paths
# =====================================================================
include_directories(${CHARM_HOME}/include)
include_directories(${CHARMTYLES_HOME}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})          # for local headers
include_directories(${CMAKE_CURRENT_BINARY_DIR})          # for generated decl/def.h

# =====================================================================
# Eigen (header-only)
# =====================================================================
find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# =====================================================================
# Charm++ Interface (.ci) code generation
# =====================================================================
# 1. Core charmtyles CI file  →  charmtyles.decl.h / charmtyles.def.h
set(CHARMTYLES_CI "${CHARMTYLES_HOME}/include/charmtyles/core/charmtyles.ci")
add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/charmtyles.decl.h
            ${CMAKE_CURRENT_BINARY_DIR}/charmtyles.def.h
    COMMAND ${CHARMC} -E -module CommonLBs
            ${CHARMTYLES_CI}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CHARMTYLES_CI}
    COMMENT "Generating charmtyles.decl.h / charmtyles.def.h"
)

# Pass -DUSE_KOKKOS to charmc when GPU backend is enabled (for nocopydevice in .ci)
set(CHARMC_CI_DEFS "")
if(DEFINED GPU_BACKEND)
    set(CHARMC_CI_DEFS "-DUSE_KOKKOS")
endif()

# 2. charmnumeric backend CI file  →  charmnumeric.decl.h / charmnumeric.def.h
set(BACKEND_CI "${CMAKE_CURRENT_SOURCE_DIR}/backend.ci")
add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/charmnumeric.decl.h
            ${CMAKE_CURRENT_BINARY_DIR}/charmnumeric.def.h
    COMMAND ${CHARMC} -E -module CommonLBs
            ${CHARMC_CI_DEFS}
            ${BACKEND_CI}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${BACKEND_CI}
            ${CMAKE_CURRENT_BINARY_DIR}/charmtyles.decl.h
    COMMENT "Generating charmnumeric.decl.h / charmnumeric.def.h"
)

# Convenience target so other targets can depend on the generated headers
add_custom_target(ci_headers
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/charmtyles.decl.h
            ${CMAKE_CURRENT_BINARY_DIR}/charmtyles.def.h
            ${CMAKE_CURRENT_BINARY_DIR}/charmnumeric.decl.h
            ${CMAKE_CURRENT_BINARY_DIR}/charmnumeric.def.h
)

# =====================================================================
# Build the server executable via charmc
#
# charmc is used as the linker because it injects the Charm++ runtime,
# main trampoline, and module registration.  We compile the C++ source
# with CMake (so MLIR/LLVM flags are picked up automatically), then
# link the resulting object with charmc.
# =====================================================================

# Locate pre-built charmtyles_core static library.
# Set CHARMTYLES_CORE_BUILD_DIR to the core build directory.
if(NOT DEFINED CHARMTYLES_CORE_BUILD_DIR)
    set(CHARMTYLES_CORE_BUILD_DIR "${CHARMTYLES_HOME}/src/charmtyles/core/build")
endif()
set(CHARMTYLES_CORE_LIB "${CHARMTYLES_CORE_BUILD_DIR}/libcharmtyles_core.a")
if(NOT EXISTS ${CHARMTYLES_CORE_LIB})
    message(FATAL_ERROR "charmtyles_core not found at ${CHARMTYLES_CORE_LIB}.\n"
        "Build it first: cd ${CHARMTYLES_HOME}/src/charmtyles/core/build && cmake .. && make")
endif()
# Add core's generated headers to include path
include_directories(${CHARMTYLES_CORE_BUILD_DIR})

# -- JIT module (heaviest TU — all MLIR/LLVM codegen) --
add_library(jit_obj OBJECT jit.cpp)
add_dependencies(jit_obj ci_headers)
set(SIMPLE_ARRAY_DEFS)
if(NDEBUG)
    list(APPEND SIMPLE_ARRAY_DEFS -DNDEBUG)
endif()
target_compile_options(jit_obj PRIVATE -O3 -g ${SIMPLE_ARRAY_DEFS})

# -- Backend module (Charm++ classes + JIT integration) --
set(BACKEND_GROUP_SOURCES
    dag_group_runtime.cpp
    dag_group_receive.cpp
    dag_group_compile.cpp
)

set(BACKEND_NODE_SOURCES
    execute_node.cpp
    execute_node_regions.cpp
)

set(BACKEND_EXECUTOR_SOURCES
    executor_reducer.cpp
    executor_core.cpp
    executor_reduce.cpp
    executor_reduction.cpp
    executor_matmul.cpp
    executor_matmatmul.cpp
    executor_transfer.cpp
    executor_incremental.cpp
    executor_ast_visitor.cpp
    executor_creation.cpp
)

set(BACKEND_PARTITION_SOURCES
    partition_lifecycle.cpp
    partition_comm.cpp
    partition_get.cpp
)

add_library(backend_obj OBJECT
    ${BACKEND_GROUP_SOURCES}
    ${BACKEND_NODE_SOURCES}
    ${BACKEND_EXECUTOR_SOURCES}
    ${BACKEND_PARTITION_SOURCES}
)
add_dependencies(backend_obj ci_headers)
target_compile_options(backend_obj PRIVATE -O3 -g ${SIMPLE_ARRAY_DEFS})
# RNG / iota kernels must evaluate the same IEEE operations as the GPU build
# (which uses the _rn intrinsics), so no FMA contraction on the host either.
if(NOT DEFINED GPU_BACKEND)
    set_source_files_properties(executor_creation.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()

# -- Server module (Main class, startup) --
add_library(server_obj OBJECT server.cpp)
add_dependencies(server_obj ci_headers)
target_compile_options(server_obj PRIVATE -O3 -g ${SIMPLE_ARRAY_DEFS})

# GPU backend selection (pass -DGPU_BACKEND=NVIDIA|AMD|INTEL on cmake line)
if(DEFINED GPU_BACKEND)
    string(TOUPPER "${GPU_BACKEND}" GPU_BACKEND_UPPER)

    # Kokkos provides portable memory management for all GPU backends.
    # Pre-build Kokkos with the matching backend (Kokkos_ENABLE_CUDA, etc.)
    find_package(Kokkos REQUIRED)
    find_package(KokkosKernels REQUIRED)
    target_link_libraries(backend_obj PRIVATE Kokkos::kokkos Kokkos::kokkoskernels)
    target_link_libraries(server_obj  PRIVATE Kokkos::kokkos)
    target_compile_definitions(backend_obj PRIVATE USE_KOKKOS)
    target_compile_definitions(server_obj  PRIVATE USE_KOKKOS)

    # JIT dispatch still needs vendor driver APIs for kernel launch
    if(GPU_BACKEND_UPPER STREQUAL "NVIDIA")
        target_compile_definitions(jit_obj     PRIVATE USE_NVIDIA JIT_ENABLE_GPU_BACKEND)
        find_library(MLIR_CUDA_RUNTIME_LIB mlir_cuda_runtime HINTS ${LLVM_LIBRARY_DIR} REQUIRED)
        target_compile_definitions(jit_obj PRIVATE MLIR_CUDA_RUNTIME_LIB="${MLIR_CUDA_RUNTIME_LIB}")
        target_compile_definitions(backend_obj PRIVATE USE_NVIDIA JIT_ENABLE_GPU_BACKEND)
        # server.cpp instantiates the chare classes (charmnumeric.def.h), so it must see
        # the same class layout as the backend TUs.
        target_compile_definitions(server_obj  PRIVATE USE_NVIDIA JIT_ENABLE_GPU_BACKEND)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(jit_obj     PRIVATE CUDA::cuda_driver)
        target_link_libraries(backend_obj PRIVATE CUDA::cuda_driver)
    elseif(GPU_BACKEND_UPPER STREQUAL "AMD")
        target_compile_definitions(jit_obj     PRIVATE USE_AMD JIT_ENABLE_GPU_BACKEND)
        target_compile_definitions(backend_obj PRIVATE USE_AMD JIT_ENABLE_GPU_BACKEND)
        # server.cpp instantiates the chare classes (charmnumeric.def.h), so it must see
        # the same class layout as the backend TUs.
        target_compile_definitions(server_obj  PRIVATE USE_AMD JIT_ENABLE_GPU_BACKEND)
        find_package(hip REQUIRED)
        target_link_libraries(jit_obj     PRIVATE hip::host)
        target_link_libraries(backend_obj PRIVATE hip::host)
    elseif(GPU_BACKEND_UPPER STREQUAL "INTEL")
        target_compile_definitions(jit_obj     PRIVATE USE_INTEL JIT_ENABLE_GPU_BACKEND)
        target_compile_definitions(backend_obj PRIVATE USE_INTEL JIT_ENABLE_GPU_BACKEND)
        # server.cpp instantiates the chare classes (charmnumeric.def.h), so it must see
        # the same class layout as the backend TUs.
        target_compile_definitions(server_obj  PRIVATE USE_INTEL JIT_ENABLE_GPU_BACKEND)
        find_library(LEVEL_ZERO_LIB NAMES ze_loader)
        find_path(LEVEL_ZERO_INCLUDE_DIR NAMES level_zero/ze_api.h)
        if(LEVEL_ZERO_LIB AND LEVEL_ZERO_INCLUDE_DIR)
            target_include_directories(jit_obj     PRIVATE ${LEVEL_ZERO_INCLUDE_DIR})
            target_include_directories(backend_obj PRIVATE ${LEVEL_ZERO_INCLUDE_DIR})
            target_link_libraries(jit_obj     PRIVATE ${LEVEL_ZERO_LIB})
            target_link_libraries(backend_obj PRIVATE ${LEVEL_ZERO_LIB})
        else()
            message(FATAL_ERROR "Level Zero not found. Install oneAPI Level Zero Loader.")
        endif()
    else()
        message(WARNING "Unknown GPU_BACKEND '${GPU_BACKEND}', building CPU-only.")
    endif()

    # The final executable is linked by charmc (custom command below), which
    # does not see the OBJECT libraries' link interfaces, so pass the GPU
    # runtime libraries explicitly (with rpaths for the shared ones).
    set(GPU_LINK_ARGS
        "-Wl,$<TARGET_FILE:Kokkos::kokkoskernels>"
        "-Wl,$<TARGET_FILE:Kokkos::kokkoscontainers>"
        "-Wl,$<TARGET_FILE:Kokkos::kokkoscore>"
        "-Wl,-rpath,$<TARGET_FILE_DIR:Kokkos::kokkoskernels>"
        "-Wl,-rpath,$<TARGET_FILE_DIR:Kokkos::kokkoscore>")
    if(GPU_BACKEND_UPPER STREQUAL "NVIDIA")
        list(APPEND GPU_LINK_ARGS
            "-L${CUDAToolkit_LIBRARY_DIR}" "-L${CUDAToolkit_LIBRARY_DIR}/stubs"
            "-Wl,-rpath,${CUDAToolkit_LIBRARY_DIR}" -lcudart -lcuda)
    endif()
endif()

# MLIR/LLVM link libraries — needed by jit_obj and backend_obj
get_property(mlir_dialect_libs  GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

foreach(tgt jit_obj backend_obj)
    target_link_libraries(${tgt}
        PRIVATE
        ${mlir_dialect_libs}
        ${mlir_conversion_libs}
        MLIRPass
        MLIRTransforms
        MLIRExecutionEngine
        MLIRTargetLLVMIRExport
        MLIRLLVMToLLVMIRTranslation
        MLIRNVVMToLLVMIRTranslation
        MLIRROCDLToLLVMIRTranslation
        MLIRSPIRVSerialization
        MLIRSPIRVTransforms
        MLIRIR
        MLIRSupport
        MLIRParser
        LLVMCore
        LLVMSupport
    )
endforeach()

# =====================================================================
# Link the object file with charmc to produce the final Charm++ executable.
#
# charmc must be used as the linker (it injects the Charm++ runtime).
# MLIR/LLVM have deep transitive dependency chains, so we use
# llvm-config to get the complete set of LLVM libraries, and glob
# all MLIR static libraries from the install directory.
# =====================================================================

# -- Locate llvm-config --
find_program(LLVM_CONFIG_EXE NAMES llvm-config
    HINTS "${LLVM_TOOLS_BINARY_DIR}" "${LLVM_DIR}/../../../bin")
if(NOT LLVM_CONFIG_EXE)
    message(FATAL_ERROR "llvm-config not found. Set LLVM_TOOLS_BINARY_DIR or add it to PATH.")
endif()
message(STATUS "Using llvm-config: ${LLVM_CONFIG_EXE}")

# -- Get ALL LLVM link flags (resolves full transitive closure) --
execute_process(COMMAND ${LLVM_CONFIG_EXE} --link-static --libfiles all
                OUTPUT_VARIABLE LLVM_ALL_LIB_FILES
                OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(LLVM_ALL_LIB_FILES)

execute_process(COMMAND ${LLVM_CONFIG_EXE} --system-libs
                OUTPUT_VARIABLE LLVM_SYSTEM_LIBS
                OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(LLVM_SYSTEM_LIBS)

execute_process(COMMAND ${LLVM_CONFIG_EXE} --ldflags
                OUTPUT_VARIABLE LLVM_LD_FLAGS
                OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(LLVM_LD_FLAGS)

# -- Glob all MLIR static libraries --
file(GLOB MLIR_ALL_LIB_FILES "${LLVM_LIBRARY_DIR}/libMLIR*.a")

# -- Combine into a single list for charmc --
# Use --whole-archive for MLIR libs to avoid registration issues,
# then --no-whole-archive for LLVM libs.
set(ALL_STATIC_LIBS ${MLIR_ALL_LIB_FILES} ${LLVM_ALL_LIB_FILES})

# charmc does not forward absolute .a paths to the underlying linker.
# Write them to a response file and pass via -Wl,@file so they reach ld.
set(LINK_RESPONSE_FILE "${CMAKE_CURRENT_BINARY_DIR}/link_libs.rsp")
set(_rsp_content "--start-group\n${CHARMTYLES_CORE_LIB}\n")
foreach(lib ${ALL_STATIC_LIBS})
    string(APPEND _rsp_content "${lib}\n")
endforeach()
string(APPEND _rsp_content "--end-group\n")
foreach(lib ${LLVM_SYSTEM_LIBS})
    string(APPEND _rsp_content "${lib}\n")
endforeach()
file(WRITE ${LINK_RESPONSE_FILE} "${_rsp_content}")

add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/server.out
    COMMAND ${CHARMC}
            -c++-option -std=c++17
            -O3 -g $<$<BOOL:${NDEBUG}>:-DNDEBUG>
            -language charm++
            -module CommonLBs
            ${LLVM_LD_FLAGS}
            -o ${CMAKE_CURRENT_BINARY_DIR}/server.out
            "$<TARGET_OBJECTS:jit_obj>"
            "$<TARGET_OBJECTS:backend_obj>"
            "$<TARGET_OBJECTS:server_obj>"
            -Wl,@${LINK_RESPONSE_FILE}
            ${GPU_LINK_ARGS}
            -ldl
    DEPENDS jit_obj backend_obj server_obj ci_headers
            ${CHARMTYLES_CORE_LIB}
            $<TARGET_OBJECTS:jit_obj>
            $<TARGET_OBJECTS:backend_obj>
            $<TARGET_OBJECTS:server_obj>
    COMMAND_EXPAND_LISTS
    COMMENT "Linking server.out with charmc"
)

add_custom_target(server ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/server.out
)

# =====================================================================
# Convenience run target
# =====================================================================
add_custom_target(run-server
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/charmrun +p4
            ${CMAKE_CURRENT_BINARY_DIR}/server.out
            ++server ++server-port 10000
    DEPENDS server
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running server with 4 PEs"
)
