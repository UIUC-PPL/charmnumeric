cmake_minimum_required(VERSION 3.20)
project(charmnumeric_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NDEBUG "Enable NDEBUG preprocessor define (-DNDEBUG)" ON)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Charm++ and the charmtyles checkout (see cmake/CnumCharm.cmake for the search order)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CnumCharm.cmake)

# GPU backend: -DGPU_BACKEND=NVIDIA|AMD|INTEL; defaults to NVIDIA with a CUDA Charm++ build.
if(NOT DEFINED GPU_BACKEND AND CHARM_CUDA)
    set(GPU_BACKEND NVIDIA)
endif()
string(TOUPPER "${GPU_BACKEND}" GPU_BACKEND_UPPER)
if(GPU_BACKEND_UPPER MATCHES "^(|OFF|NONE|CPU|0|FALSE)$")
    set(GPU_BACKEND_UPPER "")
endif()
set(GPU_BACKEND "${GPU_BACKEND_UPPER}" CACHE STRING "GPU backend: NVIDIA, AMD, INTEL or empty for CPU" FORCE)
if(GPU_BACKEND_UPPER STREQUAL "NVIDIA" AND NOT CHARM_CUDA)
    message(FATAL_ERROR "GPU_BACKEND=NVIDIA needs a CUDA build of Charm++, but ${CHARM_HOME} is not "
        "one. Build Charm++ with the cuda option (e.g. ./build charm++ netlrts-linux-x86_64 cuda) "
        "and point CHARM_HOME at it, or run scripts/install.sh --gpu.")
elseif(NOT GPU_BACKEND_UPPER AND CHARM_CUDA)
    message(WARNING "CPU build against the CUDA Charm++ ${CHARM_HOME}.")
endif()
message(STATUS "charmnumeric GPU backend: ${GPU_BACKEND_UPPER}")

# =====================================================================
# MLIR / LLVM 22 (MLIR_DIR, LLVM_DIR, llvm-config on PATH, or the install.sh prefix)
# =====================================================================
set(_mlir_hints)
foreach(_v MLIR_DIR LLVM_DIR)
    if(DEFINED ENV{${_v}})
        list(APPEND _mlir_hints "$ENV{${_v}}" "$ENV{${_v}}/../mlir")
    endif()
endforeach()
if(LLVM_DIR)
    list(APPEND _mlir_hints "${LLVM_DIR}/../mlir")
endif()
find_program(_cnum_llvm_config NAMES llvm-config-22 llvm-config NO_CACHE)
if(_cnum_llvm_config)
    execute_process(COMMAND ${_cnum_llvm_config} --version --cmakedir
        OUTPUT_VARIABLE _lc OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    string(REPLACE "\n" ";" _lc "${_lc}")
    list(LENGTH _lc _n)
    if(_n EQUAL 2)
        list(GET _lc 0 _ver)
        list(GET _lc 1 _dir)
        if(_ver MATCHES "^22\\.")
            list(APPEND _mlir_hints "${_dir}/../mlir")
        endif()
    endif()
endif()
list(APPEND _mlir_hints "${CNUM_DEPS_DIR}/mlir/lib/cmake/mlir")
find_package(MLIR CONFIG HINTS ${_mlir_hints})
if(NOT MLIR_FOUND)
    message(FATAL_ERROR "MLIR/LLVM 22 not found.\n"
        "Run scripts/install.sh (fetches conda-forge mlir 22.1.8 into ${CNUM_DEPS_DIR}), or set "
        "MLIR_DIR=<llvm prefix>/lib/cmake/mlir for an existing LLVM 22 build with MLIR.")
endif()
find_package(LLVM CONFIG REQUIRED HINTS "${MLIR_DIR}/../llvm")
if(NOT LLVM_VERSION_MAJOR EQUAL 22)
    message(FATAL_ERROR "charmnumeric needs LLVM/MLIR 22, found ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}. "
        "Set MLIR_DIR to an LLVM 22 install or run scripts/install.sh.")
endif()
if(NOT LLVM_ENABLE_RTTI)
    message(WARNING "LLVM at ${LLVM_DIR} was built without RTTI; linking server.out may fail.")
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR} (${LLVM_PACKAGE_VERSION})")

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
# Eigen 3 (header-only): an installed one, else downloaded
# =====================================================================
find_package(Eigen3 3.3 QUIET NO_MODULE HINTS "${CNUM_DEPS_DIR}/mlir/share/eigen3/cmake")
if(Eigen3_FOUND)
    get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
else()
    include(FetchContent)
    message(STATUS "Eigen3 not found; downloading Eigen 3.4.0")
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        SOURCE_SUBDIR _no_cmake)
    FetchContent_MakeAvailable(eigen)
    set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}")
endif()
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
if(GPU_BACKEND_UPPER)
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

# charmtyles core: built here unless CHARMTYLES_CORE_BUILD_DIR names a prebuilt one.
if(CHARMTYLES_CORE_BUILD_DIR)
    set(CHARMTYLES_CORE_LIB "${CHARMTYLES_CORE_BUILD_DIR}/libcharmtyles_core.a")
    if(NOT EXISTS ${CHARMTYLES_CORE_LIB})
        message(FATAL_ERROR "charmtyles_core not found at ${CHARMTYLES_CORE_LIB}. Build it first, "
            "or leave CHARMTYLES_CORE_BUILD_DIR unset to build it as part of this project.")
    endif()
    set(CHARMTYLES_CORE_DEP ${CHARMTYLES_CORE_LIB})
else()
    set(CHARMTYLES_CORE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/charmtyles_core")
    add_subdirectory(${CHARMTYLES_HOME}/src/charmtyles/core ${CHARMTYLES_CORE_BUILD_DIR})
    set(CHARMTYLES_CORE_LIB "$<TARGET_FILE:charmtyles_core>")
    set(CHARMTYLES_CORE_DEP charmtyles_core)
endif()
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
    executor_transpose.cpp
    executor_incremental.cpp
    executor_ast_visitor.cpp
    executor_creation.cpp
    executor_castscansort.cpp
    executor_linalg.cpp
    executor_signal.cpp
    executor_indexing.cpp
    executor_shape.cpp
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
if(NOT GPU_BACKEND_UPPER)
    set_source_files_properties(executor_creation.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()

# -- Server module (Main class, startup) --
add_library(server_obj OBJECT server.cpp)
add_dependencies(server_obj ci_headers)
target_compile_options(server_obj PRIVATE -O3 -g ${SIMPLE_ARRAY_DEFS})

# GPU backend selection (pass -DGPU_BACKEND=NVIDIA|AMD|INTEL on cmake line)
if(GPU_BACKEND_UPPER)
    # Kokkos provides portable memory management for all GPU backends.
    # Pre-build Kokkos with the matching backend (Kokkos_ENABLE_CUDA, etc.)
    find_package(Kokkos CONFIG HINTS "${CNUM_DEPS_DIR}/kokkos")
    find_package(KokkosKernels CONFIG HINTS "${CNUM_DEPS_DIR}/kokkos")
    if(NOT Kokkos_FOUND OR NOT KokkosKernels_FOUND)
        message(FATAL_ERROR "GPU builds need Kokkos and Kokkos Kernels (4.x, built for ${GPU_BACKEND_UPPER}). "
            "Run scripts/install.sh --gpu (builds both into ${CNUM_DEPS_DIR}/kokkos), or set "
            "Kokkos_ROOT and KokkosKernels_ROOT.")
    endif()
    if(GPU_BACKEND_UPPER STREQUAL "NVIDIA" AND NOT Kokkos_ENABLE_CUDA)
        message(FATAL_ERROR "Kokkos at ${Kokkos_DIR} was built without CUDA.")
    endif()
    target_link_libraries(backend_obj PRIVATE Kokkos::kokkos Kokkos::kokkoskernels)
    target_link_libraries(server_obj  PRIVATE Kokkos::kokkos)
    target_compile_definitions(backend_obj PRIVATE USE_KOKKOS)
    target_compile_definitions(server_obj  PRIVATE USE_KOKKOS)

    # JIT dispatch still needs vendor driver APIs for kernel launch
    if(GPU_BACKEND_UPPER STREQUAL "NVIDIA")
        target_compile_definitions(jit_obj     PRIVATE USE_NVIDIA JIT_ENABLE_GPU_BACKEND)
        find_package(CUDAToolkit REQUIRED)
        if(NOT "NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
            message(FATAL_ERROR "LLVM at ${LLVM_DIR} lacks the NVPTX target needed for GPU JIT.")
        endif()
        # MLIR's CUDA runtime wrappers: from the LLVM build (MLIR_ENABLE_CUDA_RUNNER), else built here.
        find_library(MLIR_CUDA_RUNTIME_LIB mlir_cuda_runtime HINTS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
        if(NOT MLIR_CUDA_RUNTIME_LIB)
            set(_crw "${LLVM_BUILD_MAIN_SRC_DIR}/../mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp")
            if(NOT EXISTS "${_crw}")
                set(_crw "${CMAKE_CURRENT_BINARY_DIR}/CudaRuntimeWrappers.cpp")
                if(NOT EXISTS "${_crw}")
                    message(STATUS "Downloading MLIR CudaRuntimeWrappers.cpp for LLVM ${LLVM_PACKAGE_VERSION}")
                    file(DOWNLOAD "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-${LLVM_PACKAGE_VERSION}/mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp"
                        "${_crw}.tmp" STATUS _st TLS_VERIFY ON)
                    list(GET _st 0 _rc)
                    if(NOT _rc EQUAL 0)
                        message(FATAL_ERROR "Could not download CudaRuntimeWrappers.cpp (${_st}). Build LLVM "
                            "with -DMLIR_ENABLE_CUDA_RUNNER=ON or set MLIR_CUDA_RUNTIME_LIB.")
                    endif()
                    file(RENAME "${_crw}.tmp" "${_crw}")
                endif()
            endif()
            add_library(mlir_cuda_runtime SHARED "${_crw}")
            target_include_directories(mlir_cuda_runtime PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
            target_link_libraries(mlir_cuda_runtime PRIVATE CUDA::cuda_driver)
            set(CNUM_OWN_CUDA_RUNTIME ON)
            set(MLIR_CUDA_RUNTIME_LIB "${CMAKE_CURRENT_BINARY_DIR}/libmlir_cuda_runtime.so")
            set_target_properties(mlir_cuda_runtime PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
            add_dependencies(jit_obj mlir_cuda_runtime)
        endif()
        target_compile_definitions(jit_obj PRIVATE MLIR_CUDA_RUNTIME_LIB="${MLIR_CUDA_RUNTIME_LIB}")
        target_compile_definitions(backend_obj PRIVATE USE_NVIDIA JIT_ENABLE_GPU_BACKEND)
        # server.cpp instantiates the chare classes (charmnumeric.def.h), so it must see
        # the same class layout as the backend TUs.
        target_compile_definitions(server_obj  PRIVATE USE_NVIDIA JIT_ENABLE_GPU_BACKEND)
        target_link_libraries(jit_obj     PRIVATE CUDA::cuda_driver)
        target_link_libraries(backend_obj PRIVATE CUDA::cuda_driver)
        # cuSOLVER/cuBLAS (LINALG) and cuFFT (CONVOLVE): CUDA toolkit or the HPC SDK math_libs
        get_filename_component(_sdk_math
            "${CUDAToolkit_LIBRARY_ROOT}/../../math_libs/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}" ABSOLUTE)
        set(CT_MATH_LIBS_DIR "${_sdk_math}"
            CACHE PATH "Directory with include/ and lib64/ of cuSOLVER and cuBLAS")
        find_path(CUSOLVER_INCLUDE_DIR cusolverDn.h
            HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CT_MATH_LIBS_DIR}/include REQUIRED)
        find_library(CUSOLVER_LIB cusolver
            HINTS ${CUDAToolkit_LIBRARY_DIR} ${CT_MATH_LIBS_DIR}/lib64 REQUIRED)
        find_library(CUBLAS_LIB cublas
            HINTS ${CUDAToolkit_LIBRARY_DIR} ${CT_MATH_LIBS_DIR}/lib64 REQUIRED)
        find_library(CUFFT_LIB cufft
            HINTS ${CUDAToolkit_LIBRARY_DIR} ${CT_MATH_LIBS_DIR}/lib64 REQUIRED)
        get_filename_component(CUSOLVER_LIB_DIR ${CUSOLVER_LIB} DIRECTORY)
        get_filename_component(CUFFT_LIB_DIR ${CUFFT_LIB} DIRECTORY)
        get_filename_component(CUBLAS_LIB_DIR ${CUBLAS_LIB} DIRECTORY)
        target_include_directories(backend_obj PRIVATE ${CUSOLVER_INCLUDE_DIR})
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
            "-Wl,-rpath,${CUDAToolkit_LIBRARY_DIR}" -lcudart -lcuda
            "-L${CUSOLVER_LIB_DIR}" "-Wl,-rpath,${CUSOLVER_LIB_DIR}"
            "-L${CUBLAS_LIB_DIR}" "-Wl,-rpath,${CUBLAS_LIB_DIR}" -lcusolver -lcublas
            "-L${CUFFT_LIB_DIR}" "-Wl,-rpath,${CUFFT_LIB_DIR}" -lcufft)
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

execute_process(COMMAND ${LLVM_CONFIG_EXE} --link-static --system-libs
                OUTPUT_VARIABLE LLVM_SYSTEM_LIBS
                OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(LLVM_SYSTEM_LIBS)
# Resolve -lfoo to files, system copies first: a -L to a conda LLVM prefix would also
# make ld pick that prefix's newer libstdc++, which the system one cannot satisfy at run time.
set(_sys_libs)
foreach(_f ${LLVM_SYSTEM_LIBS})
    string(REGEX REPLACE "^-l" "" _n "${_f}")
    if(NOT _n STREQUAL _f AND NOT _n MATCHES "^(rt|dl|m|pthread)$")
        find_library(_cnum_sys_${_n} ${_n} NO_CACHE PATHS /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH)
        if(NOT _cnum_sys_${_n})
            find_library(_cnum_sys_${_n} ${_n} NO_CACHE HINTS ${LLVM_LIBRARY_DIR})
        endif()
        if(_cnum_sys_${_n})
            set(_f ${_cnum_sys_${_n}})
        endif()
    endif()
    list(APPEND _sys_libs ${_f})
endforeach()
set(LLVM_SYSTEM_LIBS ${_sys_libs})

# -- Glob all MLIR static libraries --
file(GLOB MLIR_ALL_LIB_FILES "${LLVM_LIBRARY_DIR}/libMLIR*.a")

# -- Combine into a single list for charmc --
# Use --whole-archive for MLIR libs to avoid registration issues,
# then --no-whole-archive for LLVM libs.
set(ALL_STATIC_LIBS ${MLIR_ALL_LIB_FILES} ${LLVM_ALL_LIB_FILES})

# --no-relax (below): GNU ld < 2.43 rewrites the GOTPCRELX "push" that newer assemblers
# emit (e.g. in conda-forge's LLVM) into a call, and server.out crashes in MLIRContext.
# charmc does not forward absolute .a paths to the underlying linker.
# Write them to a response file and pass via -Wl,@file so they reach ld.
# (file(GENERATE) because the core library path may be a generator expression.)
set(LINK_RESPONSE_FILE "${CMAKE_CURRENT_BINARY_DIR}/link_libs.rsp")
set(_rsp_content "--start-group\n${CHARMTYLES_CORE_LIB}\n")
foreach(lib ${ALL_STATIC_LIBS})
    string(APPEND _rsp_content "${lib}\n")
endforeach()
string(APPEND _rsp_content "--end-group\n")
foreach(lib ${LLVM_SYSTEM_LIBS})
    string(APPEND _rsp_content "${lib}\n")
endforeach()
file(GENERATE OUTPUT ${LINK_RESPONSE_FILE} CONTENT "${_rsp_content}")

add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/server.out
    COMMAND ${CHARMC}
            -c++-option -std=c++17
            -O3 -g $<$<BOOL:${NDEBUG}>:-DNDEBUG>
            -language charm++
            -module CommonLBs
            -o ${CMAKE_CURRENT_BINARY_DIR}/server.out
            "$<TARGET_OBJECTS:jit_obj>"
            "$<TARGET_OBJECTS:backend_obj>"
            "$<TARGET_OBJECTS:server_obj>"
            -Wl,@${LINK_RESPONSE_FILE}
            -Wl,--no-relax
            ${GPU_LINK_ARGS}
            -ldl
    DEPENDS jit_obj backend_obj server_obj ci_headers
            ${CHARMTYLES_CORE_DEP} ${LINK_RESPONSE_FILE}
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
# Install into the Python package (scikit-build) and record where the server is
# =====================================================================
# charmnumeric/_build_info.py tells Python where server.out and charmrun are: the
# package's _bin/ for pip installs, this build directory otherwise.
option(CNUM_WRITE_BUILD_INFO "Write charmnumeric/_build_info.py into the source tree (non-pip builds)" ON)
function(_cnum_py_bool var value)
    if(${value})
        set(${var} True PARENT_SCOPE)
    else()
        set(${var} False PARENT_SCOPE)
    endif()
endfunction()
_cnum_py_bool(CNUM_BI_RECONVERSE CHARM_RECONVERSE)
_cnum_py_bool(CNUM_BI_SMP CHARM_SMP)
if(GPU_BACKEND_UPPER)
    set(CNUM_BI_GPU "\"${GPU_BACKEND_UPPER}\"")
else()
    set(CNUM_BI_GPU None)
endif()
if(SKBUILD)
    set(CNUM_BI_SERVER "_os.path.join(_here, \"_bin\", \"server.out\")")
    set(CNUM_BI_CHARMRUN "_os.path.join(_here, \"_bin\", \"charmrun\")")
else()
    set(CNUM_BI_SERVER "\"${CMAKE_CURRENT_BINARY_DIR}/server.out\"")
    set(CNUM_BI_CHARMRUN "\"${CMAKE_CURRENT_BINARY_DIR}/charmrun\"")
endif()
if(CHARM_RECONVERSE)
    set(CNUM_BI_CHARMRUN None)
endif()
foreach(_r CNUM_GIT_REV:${CMAKE_CURRENT_SOURCE_DIR} CNUM_CORE_GIT_REV:${CHARMTYLES_HOME})
    string(REPLACE ":" ";" _r "${_r}")
    list(GET _r 0 _var)
    list(GET _r 1 _dir)
    execute_process(COMMAND git -C ${_dir} describe --always --dirty
        OUTPUT_VARIABLE ${_var} OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endforeach()
configure_file(cmake/_build_info.py.in ${CMAKE_CURRENT_BINARY_DIR}/_build_info.py @ONLY)

if(SKBUILD)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/server.out DESTINATION charmnumeric/_bin)
    if(NOT CHARM_RECONVERSE)
        install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/charmrun DESTINATION charmnumeric/_bin)
    endif()
    if(CNUM_OWN_CUDA_RUNTIME)
        install(TARGETS mlir_cuda_runtime LIBRARY DESTINATION charmnumeric/_bin)
    endif()
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_build_info.py DESTINATION charmnumeric)
elseif(CNUM_WRITE_BUILD_INFO)
    file(COPY ${CMAKE_CURRENT_BINARY_DIR}/_build_info.py DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../charmnumeric)
endif()

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
