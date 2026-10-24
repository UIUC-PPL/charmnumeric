#!/usr/bin/env bash
# Installs charmnumeric into the active venv/conda env, building or reusing Charm++,
# MLIR 22, Kokkos (GPU) and PyCCS. Re-running skips finished steps. See --help.
set -euo pipefail

CHARM_REPO=https://github.com/charmplusplus/charm
CHARM_CLASSIC_REF=v8.1.0           CHARM_CLASSIC_SHA=69c48ad848a26455a33ee736f45c3ef7d64e101b
CHARM_RECONVERSE_REF=v9.0.0-alpha2 CHARM_RECONVERSE_SHA=247c38f985a6c754ba2662edd15a117188669aac
PYCCS_REPO=https://github.com/charmplusplus/PyCCS
PYCCS_REF=8b8a54a5cb5c254d50181dfd23420b0f39078923
CORE_REPO=https://github.com/UIUC-PPL/charmtyles
MICROMAMBA_URL=https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64
MICROMAMBA_SHA256=366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3
LLVM_VER=22.1.8
LLVM_SHA256=922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888
KOKKOS_VER=4.7.01
KOKKOS_SHA256=404cf33e76159e83b8b4ad5d86f6899d442b5da4624820ab457412116cdcd201
KK_SHA256=f3e1452db0e182c8e32c61632465e3a829159b9ae0645d9e4cd97b4fa09c36e1

die() { echo "install.sh: error: $*" >&2; exit 1; }
warn() { echo "install.sh: warning: $*" >&2; }

CN=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NOTEBOOK=0 GPU=0 RCV=0 PREFIX=$CN/.deps CHARM="" MLIR=conda CUDA_ARCH="" DEPS_ONLY=0 EDITABLE=0 CORE_REF=""
JOBS=$(nproc 2>/dev/null || echo 4); [ "$JOBS" -gt 16 ] && JOBS=16

usage() {
    cat <<EOF
Usage: scripts/install.sh [options]   (inside an activated venv or conda env)
  --gpu              NVIDIA GPU build (CUDA 12.x: nvcc on PATH or CUDA_HOME)
  --reconverse       Reconverse Charm++ $CHARM_RECONVERSE_REF instead of classic $CHARM_CLASSIC_REF
  --prefix DIR       dependency prefix (default $CN/.deps)
  --charm DIR        existing Charm++ build directory (default: \$CHARM_HOME if it fits, else build)
  --mlir conda|source|DIR
                     LLVM/MLIR $LLVM_VER from conda-forge (default), built from source, or an
                     existing prefix; a valid \$MLIR_DIR is used as is
  --cuda-arch NN     GPU compute capability, e.g. 80 (default: from nvidia-smi)
  --core-ref REF     charmtyles branch to clone if this checkout is not inside one
                     (default: this checkout's branch, else the default branch)
  --jobs N           build jobs (default $JOBS)
  -e, --editable     editable install
  --notebook         add the [notebook] extra (Jupyter)
  --deps-only        stop after the dependencies and write \$prefix/env.sh
Compilers are gcc/g++ (\$CNUM_CC/\$CNUM_CXX override; CC/CXX are ignored).
Paths must not contain spaces (Charm++ limitation).
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in --*=*) set -- "${1%%=*}" "${1#*=}" "${@:2}" ;; esac
    case "$1" in
        --prefix|--charm|--mlir|--cuda-arch|--core-ref|--jobs|-j)
            [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value" ;;
    esac
    case "$1" in
        --gpu) GPU=1 ;;
        --reconverse) RCV=1 ;;
        --prefix) PREFIX=$2; shift ;;
        --charm) CHARM=$2; shift ;;
        --mlir) MLIR=$2; shift ;;
        --cuda-arch) CUDA_ARCH=$2; shift ;;
        --core-ref) CORE_REF=$2; shift ;;
        --jobs|-j) JOBS=$2; shift ;;
        --editable|-e) EDITABLE=1 ;;
        --notebook) NOTEBOOK=1 ;;
        --deps-only) DEPS_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option $1" ;;
    esac
    shift
done

T0=$(date +%s)
step() { echo; echo "==> $*"; }
mkdir -p "$PREFIX"; PREFIX=$(cd "$PREFIX" && pwd); LOG=$PREFIX/logs; mkdir -p "$LOG"
case "$PREFIX$CN" in *" "*) die "paths with spaces are not supported: $PREFIX" ;; esac
run_logged() { local log=$1; shift; "$@" > "$log" 2>&1 || { tail -30 "$log" >&2; die "failed, see $log"; }; }
fetch() {  # url sha256 dest
    [ -f "$3" ] || { curl -fL --retry 3 -o "$3.part" "$1" && mv "$3.part" "$3"; } || die "download failed: $1"
    echo "$2  $3" | sha256sum -c --quiet - || { rm -f "$3"; die "checksum mismatch for $1"; }
}
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# --- preflight -----------------------------------------------------------
step "Checking the environment"
[ "$(uname -sm)" = "Linux x86_64" ] || die "only Linux x86_64 is supported (this is $(uname -sm))"
for t in git curl tar make sha256sum; do command -v $t >/dev/null || die "$t not found"; done
[ "$MLIR" = source ] && { command -v xz >/dev/null || die "xz not found (needed to unpack the LLVM sources)"; }
PY=$(command -v python || command -v python3) || die "python not found"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 9))' || die "Python >= 3.9 required ($PY is $("$PY" -V 2>&1))"
"$PY" -c 'import os, sys; sys.exit(not (sys.prefix != sys.base_prefix or os.environ.get("CONDA_PREFIX")))' ||
    die "activate a virtualenv (python3 -m venv ~/cnum-venv && . ~/cnum-venv/bin/activate) or a conda env first"
"$PY" -c 'import os, sysconfig; assert os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h"))' 2>/dev/null ||
    die "Python headers (Python.h) missing for $PY; install the python3-devel package or use a full Python build"
pipv=$("$PY" -m pip --version 2>/dev/null | awk '{print $2}') || die "pip missing in $PY"
if ! ver_ge "$pipv" 22.1; then
    echo "upgrading pip $pipv (>= 22.1 needed)"; "$PY" -m pip install -q -U pip
fi
export CC=${CNUM_CC:-gcc} CXX=${CNUM_CXX:-g++}
command -v "$CXX" >/dev/null && command -v "$CC" >/dev/null || die "compiler $CC/$CXX not found (set CNUM_CC/CNUM_CXX)"
gccv=$("$CXX" -dumpfullversion 2>/dev/null || "$CXX" -dumpversion)
[ -n "${CNUM_CXX:-}" ] || ver_ge "$gccv" 9 || die "g++ >= 9 needed (found $gccv); load a newer GCC or set CNUM_CC/CNUM_CXX"
echo "compilers: $CC, $CXX $gccv; Python $("$PY" -V 2>&1 | cut -d' ' -f2), pip $("$PY" -m pip --version | awk '{print $2}')"
cmake_ok() { command -v cmake >/dev/null && ver_ge "$(cmake --version | awk 'NR==1{print $3}')" 3.20; }
need=(); cmake_ok || need+=("cmake>=3.20")
[ "$MLIR" = source ] && ! command -v ninja >/dev/null && need+=(ninja)
if [ ${#need[@]} -gt 0 ]; then
    echo "pip installing ${need[*]}"; "$PY" -m pip install -q "${need[@]}"; hash -r
fi
cmake_ok || die "cmake >= 3.20 needed"
if [ $GPU = 1 ]; then
    NVCC=$(command -v nvcc || echo "${CUDA_HOME:-/nonexistent}/bin/nvcc")
    [ -x "$NVCC" ] || die "--gpu needs the CUDA toolkit: put nvcc on PATH or set CUDA_HOME"
    CUDA_HOME=${CUDA_HOME:-$(cd "$(dirname "$NVCC")/.." && pwd)}
    CUDA_DIR=${CUDA_DIR:-$CUDA_HOME}
    export CUDA_HOME CUDA_DIR PATH=$CUDA_HOME/bin:$PATH
    CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "CUDA $CUDA_VER ($CUDA_HOME)"
    case "$CUDA_VER" in 12.*) ;; *) warn "tested with CUDA 12.x only" ;; esac
    if [ -z "$CUDA_ARCH" ] && command -v nvidia-smi >/dev/null; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.') || true
    fi
fi

# --- charmtyles checkout -------------------------------------------------
CORE=${CHARMTYLES_HOME:-$(cd "$CN/../.." && pwd)}
if [ ! -f "$CORE/include/charmtyles/core/charmtyles.ci" ]; then
    CORE=$PREFIX/src/charmtyles
    if [ ! -d "$CORE/.git" ]; then
        CORE_REF=${CORE_REF:-$(git -C "$CN" symbolic-ref --short -q HEAD || true)}
        if [ -n "$CORE_REF" ] && git ls-remote --exit-code --heads "$CORE_REPO" "$CORE_REF" >/dev/null 2>&1; then
            step "Cloning charmtyles ($CORE_REF) into $CORE"; git clone -q -b "$CORE_REF" "$CORE_REPO" "$CORE"
        else
            step "Cloning charmtyles (default branch) into $CORE"; git clone -q "$CORE_REPO" "$CORE"
        fi
    fi
fi
export CHARMTYLES_HOME=$CORE

# --- Charm++ -------------------------------------------------------------
charm_ok() { [ -x "$1/bin/charmc" ] && [ -f "$1/lib/libck.a" ]; }
charm_cuda() { grep -q conv-mach-cuda.h "$1/include/conv-mach-opt.h" 2>/dev/null; }
charm_rcv() { [ -e "$1/lib/libreconverse.so" ] || [ -e "$1/lib/libreconverse.a" ]; }
charm_fits() {
    charm_ok "$1" || return 1
    local c=0 r=0; charm_cuda "$1" && c=1; charm_rcv "$1" && r=1
    [ $c = $GPU ] && [ $r = $RCV ]
}
want=classic; [ $RCV = 0 ] || want=Reconverse; [ $GPU = 0 ] || want="$want CUDA"
if [ -n "$CHARM" ]; then
    charm_ok "$CHARM" || die "--charm $CHARM is not a complete Charm++ build (needs bin/charmc and lib/libck.a)"
    charm_fits "$CHARM" || die "--charm $CHARM does not match the requested $want configuration"
elif [ -n "${CHARM_HOME:-}" ]; then
    if charm_fits "$CHARM_HOME"; then CHARM=$CHARM_HOME
    else warn "ignoring CHARM_HOME=$CHARM_HOME: not a complete $want Charm++ build; building one in $PREFIX"; fi
fi
if [ -z "$CHARM" ]; then
    if [ $RCV = 1 ]; then ref=$CHARM_RECONVERSE_REF sha=$CHARM_RECONVERSE_SHA arch=reconverse-linux-x86_64
    else ref=$CHARM_CLASSIC_REF sha=$CHARM_CLASSIC_SHA arch=netlrts-linux-x86_64; fi
    src=$PREFIX/src/charm-$ref; opts=""; [ $GPU = 1 ] && opts=cuda
    CHARM=$src/$arch${opts:+-$opts}
    if [ ! -f "$CHARM/.cnum-build-ok" ]; then
        step "Building Charm++ $ref $arch $opts (log: $LOG/charm-$arch$opts.log)"
        [ -d "$src/.git" ] || git -c advice.detachedHead=false clone -q --depth 1 -b "$ref" "$CHARM_REPO" "$src"
        [ "$(git -C "$src" rev-parse HEAD)" = "$sha" ] || die "Charm++ $ref in $src is not commit $sha"
        (cd "$src" && run_logged "$LOG/charm-$arch$opts.log" ./build charm++ $arch $opts --with-production --force -j"$JOBS")
        charm_ok "$CHARM" || die "Charm++ build incomplete, see $LOG/charm-$arch$opts.log"
        touch "$CHARM/.cnum-build-ok"
    fi
fi
CHARM=$(cd "$CHARM" && pwd)
echo "Charm++: $CHARM"

# --- MLIR / LLVM 22 ------------------------------------------------------
mlir_ok() { [ -f "$1/lib/cmake/mlir/MLIRConfig.cmake" ] && "$1/bin/llvm-config" --version 2>/dev/null | grep -q '^22\.'; }
MLIR_PREFIX=""
if [ -n "${MLIR_DIR:-}" ]; then
    p=$(cd "$MLIR_DIR/../../.." 2>/dev/null && pwd || true)
    if [ -n "$p" ] && mlir_ok "$p"; then MLIR_PREFIX=$p
    else warn "ignoring MLIR_DIR=$MLIR_DIR: not an LLVM/MLIR 22 install; using --mlir $MLIR"; fi
fi
if [ -n "$MLIR_PREFIX" ]; then :
elif [ "$MLIR" != conda ] && [ "$MLIR" != source ]; then
    MLIR_PREFIX=$(cd "$MLIR" && pwd)
elif [ "$MLIR" = conda ]; then
    MLIR_PREFIX=$PREFIX/mlir
    if ! mlir_ok "$MLIR_PREFIX"; then
        step "Fetching LLVM/MLIR $LLVM_VER and Eigen from conda-forge into $MLIR_PREFIX"
        mkdir -p "$PREFIX/bin"; MM=$PREFIX/bin/micromamba
        fetch "$MICROMAMBA_URL" "$MICROMAMBA_SHA256" "$MM"; chmod +x "$MM"
        rm -rf "$MLIR_PREFIX"
        MAMBA_ROOT_PREFIX=$PREFIX/mamba run_logged "$LOG/mlir-conda.log" \
            "$MM" create -y -p "$MLIR_PREFIX" -c conda-forge --override-channels \
            "mlir=$LLVM_VER" "llvmdev=$LLVM_VER" eigen=3.4.0
        rm -rf "$PREFIX/mamba/pkgs"
    fi
else
    MLIR_PREFIX=$PREFIX/llvm-$LLVM_VER
    if ! mlir_ok "$MLIR_PREFIX"; then
        step "Building LLVM/MLIR $LLVM_VER from source into $MLIR_PREFIX (log: $LOG/llvm.log)"
        mkdir -p "$PREFIX/src"; tb=$PREFIX/src/llvm-project-$LLVM_VER.src.tar.xz
        fetch "https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VER/llvm-project-$LLVM_VER.src.tar.xz" $LLVM_SHA256 "$tb"
        [ -d "$PREFIX/src/llvm-project-$LLVM_VER.src" ] || tar -xf "$tb" -C "$PREFIX/src"
        extra=(); command -v ccache >/dev/null && extra+=(-DLLVM_CCACHE_BUILD=ON)
        [ $GPU = 1 ] && extra+=(-DMLIR_ENABLE_CUDA_RUNNER=ON)
        run_logged "$LOG/llvm.log" cmake -G Ninja -S "$PREFIX/src/llvm-project-$LLVM_VER.src/llvm" -B "$PREFIX/build/llvm" \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$MLIR_PREFIX" -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
            -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_RTTI=ON \
            -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
            -DLLVM_INCLUDE_EXAMPLES=OFF -DMLIR_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
            -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
            -DLLVM_INSTALL_UTILS=ON "${extra[@]}"
        run_logged "$LOG/llvm.log" cmake --build "$PREFIX/build/llvm" --target install -j "$JOBS"
        rm -rf "$PREFIX/build/llvm"
    fi
fi
mlir_ok "$MLIR_PREFIX" || die "no LLVM/MLIR 22 at $MLIR_PREFIX (needs lib/cmake/mlir and bin/llvm-config 22.x)"
export MLIR_DIR=$MLIR_PREFIX/lib/cmake/mlir LLVM_DIR=$MLIR_PREFIX/lib/cmake/llvm
echo "MLIR: $MLIR_PREFIX ($("$MLIR_PREFIX/bin/llvm-config" --version))"

# --- Kokkos + Kokkos Kernels (GPU) ---------------------------------------
KOKKOS=""
if [ $GPU = 1 ]; then
    kk_ok() { [ -d "$1/lib64/cmake/KokkosKernels" ] || [ -d "$1/lib/cmake/KokkosKernels" ]; }
    karch() {
        case "$1" in
            70) echo VOLTA70 ;; 75) echo TURING75 ;; 80) echo AMPERE80 ;; 86) echo AMPERE86 ;;
            89) echo ADA89 ;; 90) echo HOPPER90 ;; 100) echo BLACKWELL100 ;; 120) echo BLACKWELL120 ;;
            *) die "unsupported --cuda-arch $1" ;;
        esac
    }
    KARCH=""
    if [ -n "$CUDA_ARCH" ]; then KARCH=$(karch "$CUDA_ARCH"); fi
    if [ -n "${Kokkos_ROOT:-}" ]; then
        if ! kk_ok "${KokkosKernels_ROOT:-$Kokkos_ROOT}"; then
            warn "ignoring Kokkos_ROOT=$Kokkos_ROOT: no Kokkos Kernels next to it (set KokkosKernels_ROOT); using $PREFIX/kokkos"
        else
            KOKKOS=$Kokkos_ROOT
            [ -z "$KARCH" ] || grep -q "KOKKOS_ARCH_$KARCH\b" "$KOKKOS/include/KokkosCore_config.h" 2>/dev/null ||
                warn "Kokkos at $KOKKOS was not built for $KARCH (--cuda-arch $CUDA_ARCH)"
        fi
    fi
    if [ -z "$KOKKOS" ]; then
        KOKKOS=$PREFIX/kokkos
        built=$(cat "$KOKKOS/.cnum-arch" 2>/dev/null || true)
        if [ -n "$built" ] && [ -n "$KARCH" ] && [ "$built" != "$KARCH" ]; then
            echo "Kokkos in $KOKKOS was built for $built; rebuilding for $KARCH"; rm -rf "$KOKKOS"; built=""
        fi
        if [ -z "$built" ]; then
            [ -n "$KARCH" ] || die "no GPU visible to detect its architecture; pass --cuda-arch (80 = A100, 86 = A40, 90 = H100)"
            step "Building Kokkos and Kokkos Kernels $KOKKOS_VER for $KARCH into $KOKKOS (log: $LOG/kokkos.log)"
            mkdir -p "$PREFIX/src"; rm -rf "$KOKKOS"
            fetch "https://github.com/kokkos/kokkos/releases/download/$KOKKOS_VER/kokkos-$KOKKOS_VER.tar.gz" $KOKKOS_SHA256 "$PREFIX/src/kokkos-$KOKKOS_VER.tar.gz"
            fetch "https://github.com/kokkos/kokkos-kernels/releases/download/$KOKKOS_VER/kokkos-kernels-$KOKKOS_VER.tar.gz" $KK_SHA256 "$PREFIX/src/kokkos-kernels-$KOKKOS_VER.tar.gz"
            tar -xzf "$PREFIX/src/kokkos-$KOKKOS_VER.tar.gz" -C "$PREFIX/src"
            tar -xzf "$PREFIX/src/kokkos-kernels-$KOKKOS_VER.tar.gz" -C "$PREFIX/src"
            export NVCC_WRAPPER_DEFAULT_COMPILER=$CXX
            rm -rf "$PREFIX/build/kokkos" "$PREFIX/build/kokkos-kernels"
            run_logged "$LOG/kokkos.log" cmake -S "$PREFIX/src/kokkos-$KOKKOS_VER" -B "$PREFIX/build/kokkos" \
                -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$KOKKOS" -DCMAKE_CXX_STANDARD=17 \
                -DCMAKE_INSTALL_RPATH='$ORIGIN' -DBUILD_SHARED_LIBS=ON \
                -DCMAKE_CXX_COMPILER="$PREFIX/src/kokkos-$KOKKOS_VER/bin/nvcc_wrapper" \
                -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_$KARCH=ON
            run_logged "$LOG/kokkos.log" cmake --build "$PREFIX/build/kokkos" --target install -j "$JOBS"
            run_logged "$LOG/kokkos-kernels.log" cmake -S "$PREFIX/src/kokkos-kernels-$KOKKOS_VER" -B "$PREFIX/build/kokkos-kernels" \
                -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$KOKKOS" -DCMAKE_CXX_COMPILER="$KOKKOS/bin/nvcc_wrapper" \
                -DCMAKE_INSTALL_RPATH='$ORIGIN' -DKokkos_ROOT="$KOKKOS" -DBUILD_SHARED_LIBS=ON \
                -DKokkosKernels_ENABLE_ALL_COMPONENTS=OFF -DKokkosKernels_ENABLE_COMPONENT_BLAS=ON \
                -DKokkosKernels_INST_DOUBLE=ON -DKokkosKernels_INST_FLOAT=ON -DKokkosKernels_INST_LAYOUTRIGHT=ON \
                -DKokkosKernels_INST_LAYOUTLEFT=OFF -DKokkosKernels_ENABLE_TPL_CUBLAS=OFF \
                -DKokkosKernels_ENABLE_TPL_CUSPARSE=OFF -DKokkosKernels_ENABLE_TPL_CUSOLVER=OFF
            run_logged "$LOG/kokkos-kernels.log" cmake --build "$PREFIX/build/kokkos-kernels" --target install -j "$JOBS"
            rm -rf "$PREFIX/build/kokkos" "$PREFIX/build/kokkos-kernels"
            echo "$KARCH" > "$KOKKOS/.cnum-arch"
        fi
    fi
    export Kokkos_ROOT=$KOKKOS KokkosKernels_ROOT=${KokkosKernels_ROOT:-$KOKKOS}
    echo "Kokkos: $KOKKOS ($(cat "$KOKKOS/.cnum-arch" 2>/dev/null || echo "user install"))"
fi

{
    echo "# Written by scripts/install.sh; source it before configuring charmnumeric/src with cmake."
    printf 'export CHARM_HOME=%q CHARMTYLES_HOME=%q CNUM_DEPS_DIR=%q\n' "$CHARM" "$CORE" "$PREFIX"
    printf 'export MLIR_DIR=%q LLVM_DIR=%q\n' "$MLIR_DIR" "$LLVM_DIR"
    [ -z "$KOKKOS" ] || printf 'export Kokkos_ROOT=%q KokkosKernels_ROOT=%q\n' "$Kokkos_ROOT" "$KokkosKernels_ROOT"
} > "$PREFIX/env.sh"
if [ $DEPS_ONLY = 1 ]; then
    echo; echo "Dependencies ready; settings in $PREFIX/env.sh:"; cat "$PREFIX/env.sh"; exit 0
fi

# --- PyCCS (with scripts/pyccs-find-library.patch: libccs-client.so inside the package) ---
if ! "$PY" -c "import os, pyccs; assert os.path.exists(os.path.join(os.path.dirname(pyccs.__file__), 'libccs-client.so'))" 2>/dev/null; then
    if "$PY" -c "import pyccs" 2>/dev/null; then
        step "Replacing the installed PyCCS with one that ships libccs-client.so (findable from any environment)"
    else
        step "Installing PyCCS"
    fi
    P=$PREFIX/src/PyCCS
    [ -d "$P/.git" ] || git clone -q "$PYCCS_REPO" "$P"
    git -C "$P" -c advice.detachedHead=false checkout -q -f "$PYCCS_REF"
    git -C "$P" apply "$CN/scripts/pyccs-find-library.patch"
    mkdir -p "$P/charm_src/charm/lib_so"
    so=$P/charm_src/charm/lib_so/libccs-client.so; csrc=$(cd "$CHARM/.." && pwd)/src
    if [ -f "$CHARM/lib_so/libccs-client.so" ]; then
        cp "$CHARM/lib_so/libccs-client.so" "$so"
    elif ! "$CXX" -shared -o "$so" -Wl,--whole-archive "$CHARM/lib/libccs-client.a" -Wl,--no-whole-archive 2>/dev/null; then
        # non-PIC libccs-client.a (no --build-shared): compile it from source
        [ -f "$csrc/conv-ccs/ccs-client.C" ] || die "need $CHARM/lib_so/libccs-client.so (Charm++ --build-shared) or the Charm++ sources at $csrc"
        "$CXX" -shared -fPIC -O2 -DCMK_NOT_USE_CONVERSE=1 -I"$csrc/conv-ccs" -I"$csrc/util" -I"$CHARM/include" \
            "$csrc/conv-ccs/ccs-client.C" -o "$so"
    fi
    (cd "$P" && run_logged "$LOG/pyccs.log" "$PY" -m pip install --force-reinstall --no-deps .)
fi

# --- charmtyles + charmnumeric --------------------------------------------
step "pip install charmtyles"
pe=(); [ $EDITABLE = 1 ] && pe=(-e)
run_logged "$LOG/pip-charmtyles.log" "$PY" -m pip install "${pe[@]}" "$CORE"

cfg=cpu; [ $GPU = 1 ] && cfg=gpu; [ $RCV = 1 ] && cfg=$cfg-reconverse
step "pip install charmnumeric ($cfg; builds server.out, several minutes; log: $LOG/pip-charmnumeric.log)"
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
skargs=(-C "build-dir=build/$cfg" -C "cmake.define.CHARM_HOME=$CHARM" -C "cmake.define.CHARMTYLES_HOME=$CORE"
        -C "cmake.define.CNUM_DEPS_DIR=$PREFIX" -C "cmake.define.MLIR_DIR=$MLIR_DIR" -C "cmake.define.LLVM_DIR=$LLVM_DIR"
        -C "cmake.define.GPU_BACKEND=$([ $GPU = 1 ] && echo NVIDIA || echo OFF)")
spec=$CN; [ $NOTEBOOK = 1 ] && spec="$CN[notebook]"
run_logged "$LOG/pip-charmnumeric.log" "$PY" -m pip install -v "${pe[@]}" "${skargs[@]}" "$spec"

step "Self test"
(cd / && "$PY" -m charmnumeric.selftest) || die "self test failed"
echo; echo "Done in $(( $(date +%s) - T0 ))s. Keep $PREFIX: the installed server uses its Kokkos/MLIR CUDA runtime."
