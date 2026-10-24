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

Jupyter notebooks
-----------------

``examples/notebooks/quickstart.ipynb`` walks through interactive use. Connect once
in the first cell; re-running that cell reconnects to the same server and keeps
your arrays. Notebook behavior:

- Displaying an array (or ``print``) shows NumPy's formatting of its values and
  fetches only what NumPy prints: arrays above ``numpy.get_printoptions()``
  threshold fetch just their corners (``edgeitems`` per side). ``repr()`` never
  contacts the server. ``cnp.set_display(preview=False)`` shows only shape, dtype
  and size; ``threshold`` and ``edgeitems`` override NumPy's.
- Operations are recorded lazily and sent to the server when each cell ends, so it
  computes while you write the next cell (``CNUM_FLUSH_AFTER_CELL=0`` disables
  this). ``a.block_until_ready()`` or ``cnp.synchronize()`` waits for the server,
  e.g. for timing.
- *Interrupt kernel* (Ctrl-C) stops waiting for a result and leaves the session
  usable. Work already sent keeps running on the server; the next fetch waits for it.
- ``np.asarray(a)``, ``a.to_numpy()`` and matplotlib fetch an array once. Slice large
  arrays before plotting.
- A server serves one client process. After restarting the kernel, start a new
  server; connecting a new process to a used server raises ``ConnectionError``.
- If the server exits, the next fetch raises ``ConnectionError`` instead of hanging.

``tests/test_notebook.py`` runs the notebook and an interrupt/reconnect scenario in
real kernels (needs ``nbclient``, ``ipykernel``, ``matplotlib``)::

    CHARMNUMERIC_SERVER=src/build/server.out pytest tests/test_notebook.py
