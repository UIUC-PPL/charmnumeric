charmnumeric
============

A distributed NumPy-like array library built on charmtyles and Charm++.
See CODEMAP.md for an overview of the code.

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
