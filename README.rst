charmnumeric
============

A distributed NumPy-like array library built on charmtyles and Charm++.
See CODEMAP.md for an overview of the code.

Install
-------

Needs Linux x86_64, git, curl, GCC 9 or newer and Python 3.9+ (with headers) in an
activated virtualenv or conda env; NVIDIA GPU builds also need CUDA 12.x.
``scripts/install.sh`` builds what is missing (Charm++, Kokkos for GPUs, PyCCS),
fetches LLVM/MLIR 22 and Eigen as conda-forge binaries, then ``pip install``\ s
charmtyles and charmnumeric. ``server.out`` and ``charmrun`` go inside the package
(``charmnumeric/_bin``); ``charmnumeric._build_info`` records their paths.

.. code-block:: bash

    # -b gpu-backend until the gpu-backend PRs are merged; then use the default branches
    git clone -b gpu-backend https://github.com/UIUC-PPL/charmtyles
    git clone -b gpu-backend https://github.com/UIUC-PPL/charmnumeric charmtyles/example/charmnumeric
    python3 -m venv cnum-venv && . cnum-venv/bin/activate
    charmtyles/example/charmnumeric/scripts/install.sh          # CPU
    charmtyles/example/charmnumeric/scripts/install.sh --gpu    # NVIDIA GPU
    python -m charmnumeric.selftest

Dependencies are built under ``charmnumeric/.deps`` (``--prefix`` to move it). Keep
it: the installed server loads Kokkos from there, and GPU builds also need the CUDA
toolkit they were built with. Re-running the script skips finished steps and
rebuilds charmnumeric incrementally; editable installs (``-e``) do not rebuild the
C++ code by themselves, so re-run the script (or pip) after changing it. Options:
``--reconverse`` (Reconverse Charm++ instead of classic netlrts), ``--charm DIR``,
``--mlir source|DIR``, ``--cuda-arch 80`` (when no GPU is visible, e.g. on a login
node), ``--jobs N``, ``--notebook`` (Jupyter extras). ``CHARM_HOME``, ``MLIR_DIR``
and ``Kokkos_ROOT`` from the environment are used when they fit the requested
configuration. The script uses ``gcc``/``g++`` (``CNUM_CC``/``CNUM_CXX`` to change)
and ignores ``CC``/``CXX``: wrappers such as Cray's link MPI and CUDA libraries into
everything. Paths must not contain spaces.

Without the script, ``pip install .`` (``-C cmake.define.GPU_BACKEND=NVIDIA`` for
GPUs) or a plain CMake build of ``src/`` work once the dependencies are findable:

.. code-block:: bash

    . .deps/env.sh   # written by install.sh; or set CHARM_HOME, MLIR_DIR, Kokkos_ROOT yourself
    cmake -G Ninja -S src -B src/build && ninja -C src/build

The CMake build compiles the charmtyles core itself (``-DCHARMTYLES_CORE_BUILD_DIR``
uses a prebuilt one) and downloads Eigen if none is installed. A CUDA Charm++
selects the NVIDIA backend. ``CHARM_HOME`` is fixed when a build directory is first
configured; pass ``-DCHARM_HOME=...`` to change it.

On NCSA Delta: ``module load cuda/12.9`` (the default CUDA 13 is untested) and use
``--jobs 6`` on login nodes.

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
