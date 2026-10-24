charmnumeric
============

A distributed NumPy-like array library built on charmtyles and Charm++.
See CODEMAP.md for an overview of the code.

Install
-------

Needs Linux x86_64, git, curl, a C++17 compiler (GCC 9 or newer) and Python 3.9+
in an activated virtualenv or conda env; NVIDIA GPU builds also need CUDA 12.x.
``scripts/install.sh`` builds what is missing (Charm++, Kokkos for GPUs, PyCCS),
fetches MLIR/LLVM 22 and Eigen as conda-forge binaries, then ``pip install``\ s
charmtyles and charmnumeric. The server binary and ``charmrun`` go inside the
package (``charmnumeric/_bin``); ``charmnumeric._build_info`` records their paths.

.. code-block:: bash

    git clone -b gpu-backend https://github.com/UIUC-PPL/charmtyles
    git clone -b gpu-backend https://github.com/UIUC-PPL/charmnumeric charmtyles/example/charmnumeric
    python3 -m venv cnum-venv && . cnum-venv/bin/activate
    charmtyles/example/charmnumeric/scripts/install.sh          # CPU
    charmtyles/example/charmnumeric/scripts/install.sh --gpu    # NVIDIA GPU
    python -m charmnumeric.selftest

Dependencies are built under ``charmnumeric/.deps`` (``--prefix`` to move it; keep
it, the server links Kokkos from there). Re-running the script skips finished
steps and rebuilds charmnumeric incrementally. Useful options: ``--reconverse``
(Reconverse Charm++ instead of classic netlrts), ``--charm DIR`` (an existing
Charm++ build), ``--mlir source|DIR`` (build LLVM instead of downloading it, or use
an install), ``--cuda-arch 80`` (when no GPU is visible, e.g. on a login node),
``--jobs N``, ``-e`` (editable), ``--notebook`` (Jupyter extras). ``CHARM_HOME``, ``MLIR_DIR`` and ``Kokkos_ROOT``
from the environment are used when set, so existing module installs work.

Without the script, ``pip install .`` (``-C cmake.define.GPU_BACKEND=NVIDIA`` for
GPUs) or a plain CMake build of ``src/`` work once the dependencies are findable:

.. code-block:: bash

    . .deps/env.sh   # written by install.sh; or set CHARM_HOME, MLIR_DIR, Kokkos_ROOT yourself
    cmake -G Ninja -S src -B src/build && ninja -C src/build

The CMake build compiles the charmtyles core itself (``-DCHARMTYLES_CORE_BUILD_DIR``
uses a prebuilt one) and downloads Eigen if none is installed. A CUDA Charm++
selects the NVIDIA backend. ``CHARM_HOME`` is fixed when a build directory is first
configured; pass ``-DCHARM_HOME=...`` to change it.

install.sh builds with ``gcc``/``g++`` (``CNUM_CC``/``CNUM_CXX`` to change); it ignores
``CC``/``CXX`` because HPC compiler wrappers such as Cray's add MPI and CUDA libraries
to everything they link. On NCSA Delta: ``module load cuda/12.9`` (the default CUDA 13
is untested) and use ``--jobs 6`` on login nodes.

Multi-node runs
---------------

On Charm++'s UDP layer (``netlrts``) launch the server with ``+eth`` (1400-byte
datagrams) when it spans several nodes, e.g. ``charmrun +p8 server.out ++server
++server-port 10000 +eth``. The default 9000-byte datagrams are IP-fragmented on
1500-MTU Ethernet; after enough multi-MB traffic one is never delivered and the
server hangs with every PE idle. ``examples/ops/test_multinode_traffic.py``
reproduces it.

With several GPUs per node, give each server process exactly one visible GPU (e.g.
``CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`` in a per-task wrapper, or ``srun
--gpus-per-task=1``). In a non-SMP build Charm++ uses device 0 of every process
while Kokkos picks the device by local rank, so with all GPUs visible each process
holds contexts on two devices and GPU 0 hosts one for every process.
