#!/usr/bin/env bash
# One-command install of charmnumeric into the active virtualenv/conda env:
# builds or reuses Charm++, MLIR 22, Kokkos (GPU) and PyCCS, then pip-installs
# charmtyles and charmnumeric (server.out + charmrun inside the package).
# Re-running skips finished steps. See --help.
set -euo pipefail

CHARM_REPO=https://github.com/charmplusplus/charm
CHARM_CLASSIC_REF=v8.1.0          # last classic Converse release (netlrts)
CHARM_RECONVERSE_REF=v9.0.0-alpha2
PYCCS_REPO=https://github.com/charmplusplus/PyCCS
PYCCS_REF=8b8a54a5cb5c254d50181dfd23420b0f39078923
CORE_REPO=https://github.com/UIUC-PPL/charmtyles
LLVM_VER=22.1.8
LLVM_SHA256=922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888
KOKKOS_VER=4.7.01
KOKKOS_SHA256=404cf33e76159e83b8b4ad5d86f6899d442b5da4624820ab457412116cdcd201
KK_SHA256=f3e1452db0e182c8e32c61632465e3a829159b9ae0645d9e4cd97b4fa09c36e1

CN=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NOTEBOOK=0; GPU=0; RCV=0; PREFIX=$CN/.deps; CHARM=""; MLIR=conda; CUDA_ARCH=""; DEPS_ONLY=0; EDITABLE=0
CORE_REF=gpu-backend
JOBS=$(nproc 2>/dev/null || echo 4); [ "$JOBS" -gt 16 ] && JOBS=16

usage() {
    cat <<EOF
Usage: scripts/install.sh [options]      (run inside an activated venv or conda env)
  --gpu               NVIDIA GPU build (needs CUDA 12.x: nvcc on PATH or CUDA_HOME)
  --reconverse        use Reconverse Charm++ ($CHARM_RECONVERSE_REF) instead of classic netlrts ($CHARM_CLASSIC_REF)
  --prefix DIR        where dependencies are built (default: $CN/.deps)
  --charm DIR         use this Charm++ build directory (default: \$CHARM_HOME, else build one)
  --mlir conda|source|DIR
                      MLIR/LLVM $LLVM_VER: conda-forge binaries (default, ~2 min), a source
                      build (~10 min on 64 cores, hours on a laptop), or an existing prefix.
                      An LLVM 22 in \$MLIR_DIR is used as is.
  --cuda-arch NN      GPU compute capability for Kokkos, e.g. 80 (default: from nvidia-smi)
  --core-ref REF      charmtyles branch to clone when this checkout is not inside one ($CORE_REF)
  --jobs N            parallel build jobs (default: $JOBS)
                      Compilers: gcc/g++, or \$CNUM_CC/\$CNUM_CXX (CC/CXX are ignored).
  --editable          pip install -e (Python sources used in place)
  --notebook          also install Jupyter support (the [notebook] extra)
  --deps-only         stop after the dependencies; prints the settings for cmake
EOF
}

while [ $# -gt 0 ]; do
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
        *) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

die() { echo "install.sh: error: $*" >&2; exit 1; }
T0=$(date +%s); TIMES=()
step() { echo; echo "==> $*"; STEP="$*"; TS=$(date +%s); }
done_step() { TIMES+=("$(printf '%5ss  %s' $(( $(date +%s) - TS )) "$STEP")"); }
mkdir -p "$PREFIX"; PREFIX=$(cd "$PREFIX" && pwd); LOG=$PREFIX/logs; mkdir -p "$LOG"
run_logged() { local log=$1; shift; "$@" > "$log" 2>&1 || { tail -30 "$log" >&2; die "failed, see $log"; }; }
fetch() {  # url sha256 dest
    [ -f "$3" ] || curl -fL --retry 3 -o "$3.part" "$1" || die "download failed: $1"
    [ -f "$3" ] || mv "$3.part" "$3"
    echo "$2  $3" | sha256sum -c --quiet - || { rm -f "$3"; die "checksum mismatch for $1"; }
}

step "Checking tools"
PY=$(command -v python || command -v python3) || die "python not found"
$PY - <<'EOF' || die "activate a virtualenv first (python3 -m venv ~/cnum-venv && . ~/cnum-venv/bin/activate) or a conda env; PyCCS cannot find its library in --user or system installs"
import os, sys
assert sys.version_info >= (3, 9), "Python >= 3.9 required"
sys.exit(0 if sys.prefix != sys.base_prefix or os.environ.get("CONDA_PREFIX") else 1)
EOF
for t in git curl tar make; do command -v $t >/dev/null || die "$t not found"; done
# Plain GCC by default: HPC modules often set CC/CXX to wrappers (e.g. Cray's) that add MPI/CUDA libraries.
export CC=${CNUM_CC:-gcc} CXX=${CNUM_CXX:-g++}
command -v "$CXX" >/dev/null || die "C++ compiler $CXX not found (set CNUM_CXX and CNUM_CC)"
echo "compilers: $CC, $CXX ($("$CXX" -dumpfullversion 2>/dev/null))"
$PY -m pip --version >/dev/null || die "pip missing in $PY"
cmake_ok() { command -v cmake >/dev/null && cmake --version | awk 'NR==1{split($3,v,"."); exit !(v[1]>3 || (v[1]==3 && v[2]>=20))}'; }
need=(); cmake_ok || need+=("cmake>=3.20")
[ "$MLIR" = source ] && ! command -v ninja >/dev/null && need+=(ninja)
if [ ${#need[@]} -gt 0 ]; then
    echo "pip installing ${need[*]} into $($PY -c 'import sys; print(sys.prefix)')"
    $PY -m pip install -q "${need[@]}"
    hash -r
fi
cmake_ok || die "cmake >= 3.20 needed"
if [ $GPU = 1 ]; then
    NVCC=$(command -v nvcc || echo "${CUDA_HOME:-/nonexistent}/bin/nvcc")
    [ -x "$NVCC" ] || die "--gpu needs the CUDA toolkit: put nvcc on PATH or set CUDA_HOME"
    export CUDA_HOME=${CUDA_HOME:-$(cd "$(dirname "$NVCC")/.." && pwd)} CUDA_DIR=${CUDA_DIR:-${CUDA_HOME:-}}
    export PATH=$CUDA_HOME/bin:$PATH
    CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "CUDA $CUDA_VER ($CUDA_HOME)"
    case "$CUDA_VER" in 12.*) ;; *) echo "warning: tested with CUDA 12.x only" ;; esac
    if [ -z "$CUDA_ARCH" ] && command -v nvidia-smi >/dev/null; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.') || true
    fi
fi
done_step

# --- charmtyles checkout -------------------------------------------------
CORE=${CHARMTYLES_HOME:-$(cd "$CN/../.." && pwd)}
if [ ! -f "$CORE/include/charmtyles/core/charmtyles.ci" ]; then
    CORE=$PREFIX/src/charmtyles
    step "Cloning charmtyles ($CORE_REF) into $CORE"
    [ -d "$CORE/.git" ] || git clone -q -b "$CORE_REF" "$CORE_REPO" "$CORE"
    done_step
fi
export CHARMTYLES_HOME=$CORE

# --- Charm++ -------------------------------------------------------------
CHARM=${CHARM:-${CHARM_HOME:-}}
if [ -z "$CHARM" ]; then
    if [ $RCV = 1 ]; then ref=$CHARM_RECONVERSE_REF; arch=reconverse-linux-x86_64; else ref=$CHARM_CLASSIC_REF; arch=netlrts-linux-x86_64; fi
    src=$PREFIX/src/charm-$ref
    opts=""; [ $GPU = 1 ] && opts=cuda
    CHARM=$src/$arch${opts:+-$opts}
    if [ ! -x "$CHARM/bin/charmc" ]; then
        step "Building Charm++ $ref $arch $opts (log: $LOG/charm-$arch$opts.log)"
        [ -d "$src/.git" ] || git -c advice.detachedHead=false clone -q --depth 1 -b "$ref" "$CHARM_REPO" "$src"
        (cd "$src" && run_logged "$LOG/charm-$arch$opts.log" ./build charm++ $arch $opts --with-production -j"$JOBS")
        [ -x "$CHARM/bin/charmc" ] || die "Charm++ build produced no $CHARM/bin/charmc"
        done_step
    fi
fi
[ -x "$CHARM/bin/charmc" ] || die "no bin/charmc in Charm++ directory $CHARM"
CHARM=$(cd "$CHARM" && pwd)
is_cuda=0; grep -q conv-mach-cuda.h "$CHARM/include/conv-mach-opt.h" 2>/dev/null && is_cuda=1
[ $GPU = 1 ] && [ $is_cuda = 0 ] && die "$CHARM is not a CUDA build of Charm++; --gpu needs one (./build charm++ netlrts-linux-x86_64 cuda), or drop --charm to build it"
[ $GPU = 0 ] && [ $is_cuda = 1 ] && die "$CHARM is a CUDA build of Charm++; pass --gpu, or a non-CUDA Charm++"
echo "Charm++: $CHARM"

# --- MLIR / LLVM 22 ------------------------------------------------------
mlir_ok() { [ -f "$1/lib/cmake/mlir/MLIRConfig.cmake" ] && "$1/bin/llvm-config" --version 2>/dev/null | grep -q '^22\.'; }
MLIR_PREFIX=""
if [ -n "${MLIR_DIR:-}" ]; then
    MLIR_PREFIX=$(cd "$MLIR_DIR/../../.." && pwd)
elif [ "$MLIR" != conda ] && [ "$MLIR" != source ]; then
    MLIR_PREFIX=$(cd "$MLIR" && pwd)
elif [ "$MLIR" = conda ]; then
    MLIR_PREFIX=$PREFIX/mlir
    if ! mlir_ok "$MLIR_PREFIX"; then
        step "Fetching MLIR/LLVM $LLVM_VER and Eigen from conda-forge into $MLIR_PREFIX"
        MM=$(command -v micromamba || command -v mamba || command -v conda || true)
        if [ -z "$MM" ]; then
            mkdir -p "$PREFIX/bin"
            [ -x "$PREFIX/bin/micromamba" ] || curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$PREFIX" bin/micromamba
            MM=$PREFIX/bin/micromamba
        fi
        rm -rf "$MLIR_PREFIX"
        MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$PREFIX/mamba} run_logged "$LOG/mlir-conda.log" \
            "$MM" create -y -p "$MLIR_PREFIX" -c conda-forge --override-channels \
            "mlir=$LLVM_VER" "llvmdev=$LLVM_VER" eigen=3.4.0
        rm -rf "$PREFIX/mamba/pkgs"
        done_step
    fi
else
    MLIR_PREFIX=$PREFIX/llvm-$LLVM_VER
    if ! mlir_ok "$MLIR_PREFIX"; then
        step "Building LLVM/MLIR $LLVM_VER from source into $MLIR_PREFIX (log: $LOG/llvm.log)"
        mkdir -p "$PREFIX/src"; tb=$PREFIX/src/llvm-project-$LLVM_VER.src.tar.xz
        fetch https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VER/llvm-project-$LLVM_VER.src.tar.xz $LLVM_SHA256 "$tb"
        [ -d "$PREFIX/src/llvm-project-$LLVM_VER.src" ] || tar -xf "$tb" -C "$PREFIX/src"
        extra=""; command -v ccache >/dev/null && extra="-DLLVM_CCACHE_BUILD=ON"
        [ $GPU = 1 ] && extra="$extra -DMLIR_ENABLE_CUDA_RUNNER=ON"
        run_logged "$LOG/llvm.log" cmake -G Ninja -S "$PREFIX/src/llvm-project-$LLVM_VER.src/llvm" -B "$PREFIX/build/llvm" \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$MLIR_PREFIX" -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
            -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_RTTI=ON \
            -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
            -DLLVM_INCLUDE_EXAMPLES=OFF -DMLIR_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
            -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
            -DLLVM_INSTALL_UTILS=ON $extra
        run_logged "$LOG/llvm.log" cmake --build "$PREFIX/build/llvm" --target install -j "$JOBS"
        rm -rf "$PREFIX/build/llvm"
        done_step
    fi
fi
mlir_ok "$MLIR_PREFIX" || die "no LLVM/MLIR 22 at $MLIR_PREFIX (need lib/cmake/mlir and bin/llvm-config 22.x)"
export MLIR_DIR=$MLIR_PREFIX/lib/cmake/mlir LLVM_DIR=$MLIR_PREFIX/lib/cmake/llvm
echo "MLIR: $MLIR_PREFIX ($("$MLIR_PREFIX/bin/llvm-config" --version))"

# --- Kokkos + Kokkos Kernels (GPU) ---------------------------------------
if [ $GPU = 1 ]; then
    KOKKOS=${Kokkos_ROOT:-$PREFIX/kokkos}
    if [ ! -d "$KOKKOS/lib64/cmake/KokkosKernels" ] && [ ! -d "$KOKKOS/lib/cmake/KokkosKernels" ]; then
        [ -n "$CUDA_ARCH" ] || die "no GPU visible to detect its architecture; pass --cuda-arch (80 = A100, 86 = A40, 90 = H100)"
        case "$CUDA_ARCH" in
            70) KARCH=VOLTA70 ;; 75) KARCH=TURING75 ;; 80) KARCH=AMPERE80 ;; 86) KARCH=AMPERE86 ;;
            89) KARCH=ADA89 ;; 90) KARCH=HOPPER90 ;; 100) KARCH=BLACKWELL100 ;; 120) KARCH=BLACKWELL120 ;;
            *) die "unsupported --cuda-arch $CUDA_ARCH" ;;
        esac
        step "Building Kokkos/Kokkos Kernels $KOKKOS_VER for $KARCH into $KOKKOS (log: $LOG/kokkos.log)"
        mkdir -p "$PREFIX/src"
        fetch https://github.com/kokkos/kokkos/releases/download/$KOKKOS_VER/kokkos-$KOKKOS_VER.tar.gz $KOKKOS_SHA256 "$PREFIX/src/kokkos-$KOKKOS_VER.tar.gz"
        fetch https://github.com/kokkos/kokkos-kernels/releases/download/$KOKKOS_VER/kokkos-kernels-$KOKKOS_VER.tar.gz $KK_SHA256 "$PREFIX/src/kokkos-kernels-$KOKKOS_VER.tar.gz"
        tar -xzf "$PREFIX/src/kokkos-$KOKKOS_VER.tar.gz" -C "$PREFIX/src"
        tar -xzf "$PREFIX/src/kokkos-kernels-$KOKKOS_VER.tar.gz" -C "$PREFIX/src"
        export NVCC_WRAPPER_DEFAULT_COMPILER=$CXX
        run_logged "$LOG/kokkos.log" cmake -S "$PREFIX/src/kokkos-$KOKKOS_VER" -B "$PREFIX/build/kokkos" \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$KOKKOS" -DCMAKE_CXX_STANDARD=17 \
            -DCMAKE_INSTALL_RPATH='$ORIGIN' \
            -DCMAKE_CXX_COMPILER="$PREFIX/src/kokkos-$KOKKOS_VER/bin/nvcc_wrapper" -DBUILD_SHARED_LIBS=ON \
            -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_$KARCH=ON
        run_logged "$LOG/kokkos.log" cmake --build "$PREFIX/build/kokkos" --target install -j "$JOBS"
        run_logged "$LOG/kokkos-kernels.log" cmake -S "$PREFIX/src/kokkos-kernels-$KOKKOS_VER" -B "$PREFIX/build/kokkos-kernels" \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$KOKKOS" -DCMAKE_CXX_COMPILER="$KOKKOS/bin/nvcc_wrapper" \
            -DCMAKE_INSTALL_RPATH='$ORIGIN' \
            -DKokkos_ROOT="$KOKKOS" -DBUILD_SHARED_LIBS=ON -DKokkosKernels_ENABLE_ALL_COMPONENTS=OFF \
            -DKokkosKernels_ENABLE_COMPONENT_BLAS=ON -DKokkosKernels_INST_DOUBLE=ON -DKokkosKernels_INST_FLOAT=ON \
            -DKokkosKernels_INST_LAYOUTRIGHT=ON -DKokkosKernels_INST_LAYOUTLEFT=OFF -DKokkosKernels_ENABLE_TPL_CUBLAS=OFF \
            -DKokkosKernels_ENABLE_TPL_CUSPARSE=OFF -DKokkosKernels_ENABLE_TPL_CUSOLVER=OFF
        run_logged "$LOG/kokkos-kernels.log" cmake --build "$PREFIX/build/kokkos-kernels" --target install -j "$JOBS"
        rm -rf "$PREFIX/build/kokkos" "$PREFIX/build/kokkos-kernels"
        done_step
    fi
    export Kokkos_ROOT=$KOKKOS KokkosKernels_ROOT=$KOKKOS
    echo "Kokkos: $KOKKOS"
fi

cat > "$PREFIX/env.sh" <<EOF
# Settings used by scripts/install.sh; source before running cmake on charmnumeric/src by hand.
export CHARM_HOME=$CHARM CHARMTYLES_HOME=$CORE MLIR_DIR=$MLIR_DIR LLVM_DIR=$LLVM_DIR CNUM_DEPS_DIR=$PREFIX
${KOKKOS:+export Kokkos_ROOT=$KOKKOS KokkosKernels_ROOT=$KOKKOS}
EOF
if [ $DEPS_ONLY = 1 ]; then
    echo; echo "Dependencies ready; settings in $PREFIX/env.sh:"; cat "$PREFIX/env.sh"; exit 0
fi

# --- PyCCS ---------------------------------------------------------------
# PyCCS with scripts/pyccs-find-library.patch: libccs-client.so ships inside the pyccs
# package (upstream only looks in sys.prefix/lib, which breaks other venvs and kernels).
if ! $PY -c "import os, pyccs; assert os.path.exists(os.path.join(os.path.dirname(pyccs.__file__), 'libccs-client.so'))" 2>/dev/null; then
    step "Installing PyCCS"
    P=$PREFIX/src/PyCCS
    [ -d "$P/.git" ] || git clone -q "$PYCCS_REPO" "$P"
    git -C "$P" -c advice.detachedHead=false checkout -q -f "$PYCCS_REF"
    git -C "$P" apply "$CN/scripts/pyccs-find-library.patch"
    mkdir -p "$P/charm_src/charm/lib_so"
    so=$P/charm_src/charm/lib_so/libccs-client.so; csrc=$(cd "$CHARM/.." && pwd)/src
    if [ -f "$CHARM/lib_so/libccs-client.so" ]; then
        cp "$CHARM/lib_so/libccs-client.so" "$so"
    elif ! "$CXX" -shared -o "$so" -Wl,--whole-archive "$CHARM/lib/libccs-client.a" -Wl,--no-whole-archive 2>/dev/null; then
        # Without --build-shared, libccs-client.a may not be PIC: compile it from the Charm++ sources.
        [ -f "$csrc/conv-ccs/ccs-client.C" ] || die "need $CHARM/lib_so/libccs-client.so (build Charm++ with --build-shared) or the Charm++ sources at $csrc"
        "$CXX" -shared -fPIC -O2 -DCMK_NOT_USE_CONVERSE=1 -I"$csrc/conv-ccs" -I"$csrc/util" -I"$CHARM/include" \
            "$csrc/conv-ccs/ccs-client.C" -o "$so"
    fi
    (cd "$P" && run_logged "$LOG/pyccs.log" $PY -m pip install --force-reinstall --no-deps .)
    done_step
fi

# --- charmtyles + charmnumeric --------------------------------------------
step "pip install charmtyles ($CORE)"
pe=""; [ $EDITABLE = 1 ] && pe=-e
run_logged "$LOG/pip-charmtyles.log" $PY -m pip install $pe "$CORE"
done_step

cfg=cpu; [ $GPU = 1 ] && cfg=gpu; [ $RCV = 1 ] && cfg=$cfg-reconverse
step "pip install charmnumeric ($cfg): builds the charmtyles core and server.out, several minutes (log: $LOG/pip-charmnumeric.log)"
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
skargs=(-C "build-dir=build/$cfg" -C "cmake.define.CHARM_HOME=$CHARM" -C "cmake.define.CHARMTYLES_HOME=$CORE"
        -C "cmake.define.CNUM_DEPS_DIR=$PREFIX" -C "cmake.define.MLIR_DIR=$MLIR_DIR")
[ $GPU = 1 ] && skargs+=(-C "cmake.define.GPU_BACKEND=NVIDIA")
[ $EDITABLE = 1 ] && skargs+=(-e)
spec=$CN; [ $NOTEBOOK = 1 ] && spec="$CN[notebook]"
run_logged "$LOG/pip-charmnumeric.log" $PY -m pip install -v "${skargs[@]}" "$spec"
done_step

step "Self test"
(cd / && $PY -m charmnumeric.selftest) || die "self test failed"
done_step

echo; echo "Step times:"; printf '%s\n' "${TIMES[@]}"
echo "Total: $(( $(date +%s) - T0 ))s.  Dependencies in $PREFIX (keep it: server.out links Kokkos/CUDA from there)."
