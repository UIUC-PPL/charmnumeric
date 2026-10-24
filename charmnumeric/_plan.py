"""The server command line for a launch (local, srun or mpiexec; charmrun or Reconverse)."""

from __future__ import annotations

import os
import shlex
import shutil

from ._build import DEVICE_VAR, env_int, visible_devices

# Written into each run directory: charmrun's ++remote-shell takes one program.
LOCALEXEC_SH = r"""#!/bin/bash
# mpiexec stand-in: -n N cmd... runs N local ranks, one device each from CNUM_BIND_DEVICES.
[ "$1" = -n ] || { echo "usage: $0 -n N cmd..." >&2; exit 2; }
n=$2; shift 2
IFS=, read -ra devs <<< "$CNUM_BIND_DEVICES"
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE MPIRUN_RANK MPIRUN_NPROCS PMI_ID PMIX_RANK MP_CHILD MP_PROCS
export PMI_SIZE=$n LCT_PMI_BACKEND=file,local LCT_PMI_FILE_NRANKS=$n
pids=()
trap 'kill "${pids[@]}" 2>/dev/null' TERM INT
for ((i = 0; i < n; i++)); do
  [ ${#devs[@]} -gt 0 ] && export "$CNUM_BIND_VAR=${devs[$((i % ${#devs[@]}))]}"
  PMI_RANK=$i "$@" &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
exit $rc
"""

BIND_SH = r"""#!/bin/bash
# srun task wrapper: one device per task (SLURM_LOCALID mod the node's devices).
v=$CNUM_BIND_VAR
if [ -n "$SLURM_LOCALID" ]; then
  if [ -n "${!v}" ]; then IFS=, read -ra devs <<< "${!v}"
  else mapfile -t devs < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); fi
  [ ${#devs[@]} -gt 0 ] && export "$v=${devs[$((SLURM_LOCALID % ${#devs[@]}))]}"
fi
exec "$@"
"""


def write_script(run_dir, name, text):
    path = run_dir / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return str(path)


def ncpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class Plan:
    def __init__(self, cmd, env, npes, nprocs, nodes, launcher, devices, gpus, local_host, request):
        self.cmd = cmd
        self.env = env
        self.npes = npes
        self.nprocs = nprocs
        self.nodes = nodes
        self.launcher = launcher
        self.devices = devices
        self.gpus = gpus
        self.local_host = local_host
        self.request = request  # start() arguments, for comparing repeated starts


def make_plan(build, run_dir, npes=None, gpus=None, nodes=None, launcher="auto", extra_args=(),
              port=None, keyfile=None, env=None):
    request = dict(npes=npes, gpus=gpus, nodes=nodes, launcher=launcher,
                   extra_args=tuple(extra_args), port=port, env=dict(env) if env else None)
    in_slurm = "SLURM_JOB_ID" in os.environ
    alloc_nodes = env_int("SLURM_JOB_NUM_NODES", 1) if in_slurm else 1
    if launcher == "auto":
        launcher = "srun" if in_slurm and (nodes or alloc_nodes) > 1 else "local"
    if launcher not in ("local", "srun", "mpiexec"):
        raise ValueError(f"launcher must be 'auto', 'local', 'srun' or 'mpiexec', not {launcher!r}")
    if launcher == "srun":
        if not in_slurm or not shutil.which("srun"):
            raise RuntimeError("launcher='srun' needs a Slurm allocation (salloc/sbatch) and srun on PATH")
        nodes = int(nodes or alloc_nodes)
    elif launcher == "local":
        if nodes not in (None, 1):
            raise ValueError(f"nodes={nodes} needs launcher='srun' (inside a Slurm allocation) or 'mpiexec'")
        nodes = 1
    if not npes:
        npes = env_int("CNUM_NPES")

    penv = {}
    devices = []
    if build.gpu:
        var = DEVICE_VAR.get(build.gpu, "CUDA_VISIBLE_DEVICES")
        penv["CNUM_BIND_VAR"] = var
        if launcher == "local":
            visible = visible_devices(build.gpu)
            if gpus is not None and not 1 <= gpus <= len(visible):
                raise ValueError(f"gpus={gpus} requested but {len(visible)} GPU(s) are visible "
                                 f"({var}={os.environ.get(var, '<unset>')})")
            n = gpus or npes or len(visible) or 1
            devices = visible[:min(n, len(visible))]
            npes = npes or n
            if devices:
                penv[var] = ",".join(devices)
                penv["CNUM_BIND_DEVICES"] = ",".join(devices)
        else:
            # One process per GPU; bind.sh picks each task's device.
            if gpus and npes and gpus != npes:
                raise ValueError(f"with launcher={launcher!r} each GPU runs one process: "
                                 f"npes ({npes}) must equal gpus ({gpus})")
            per_node = env_int("SLURM_GPUS_ON_NODE", 0) or len(visible_devices(build.gpu)) or 1
            npes = npes or gpus or (nodes or 1) * per_node
        nprocs = npes
        ngpus = len(devices) if launcher == "local" else npes
    else:
        if gpus:
            raise ValueError(f"{build.server} is a CPU build; gpus= needs a GPU build")
        if not npes:
            if launcher == "local":
                npes = min(ncpus(), 4)
            else:
                ntasks = env_int("SLURM_NTASKS", 0)
                npes = ntasks if ntasks >= (nodes or 1) else (nodes or 1) * 4
        nprocs = npes if not build.reconverse else (nodes if launcher != "local" else 1)
        ngpus = None
    npes = int(npes)
    if npes < 1:
        raise ValueError(f"npes must be positive, not {npes}")
    srv_args = ["++server"] + (["++server-port", str(port)] if port else [])
    if keyfile:
        srv_args += ["++server-auth", str(keyfile)]
    extra = list(extra_args)
    multi_local = launcher == "local" and len(devices) > 1 and nprocs > 1

    def plan(cmd, local_host):
        return Plan(cmd, penv, npes, nprocs, nodes, launcher, devices, ngpus or None, local_host, request)

    if build.charmrun is not None and not build.reconverse:
        cmd = [str(build.charmrun), f"+p{npes}", str(build.server)] + srv_args
        if launcher == "local" and not multi_local:
            cmd += ["++local"]
        elif launcher == "local":
            cmd += ["++mpiexec", "++remote-shell", write_script(run_dir, "localexec.sh", LOCALEXEC_SH)]
        elif launcher == "srun":
            bind = shlex.quote(write_script(run_dir, "bind.sh", BIND_SH)) if build.gpu else ""
            srun = ("#!/bin/bash\nn=(); [ \"$1\" = -n ] && { n=(-n \"$2\"); shift 2; }\n"
                    f"exec srun --overlap --kill-on-bad-exit=1 -N {nodes} \"${{n[@]}}\" {bind} \"$@\"\n")
            cmd += ["++mpiexec", "++remote-shell", write_script(run_dir, "srun.sh", srun)]
            if build.arch.startswith("netlrts"):
                # 1400-byte datagrams: the 9000-byte default is IP-fragmented on 1500-MTU networks.
                cmd += ["+eth"]
        else:
            cmd += ["++mpiexec"]
        return plan(cmd + extra, True)

    if not build.reconverse and npes > 1:
        raise RuntimeError(f"no charmrun found next to {build.server}; set CNUM_CHARMRUN, "
                           "or npes=1 to run server.out directly")
    pe_flag = "+pe" if build.reconverse else "+p"
    # Line-buffered, or the ready line sits in the server's stdout buffer.
    stdbuf = ["stdbuf", "-oL", "-eL"] if shutil.which("stdbuf") else []
    base = stdbuf + [str(build.server), pe_flag, str(npes)] + srv_args + extra
    if launcher == "local":
        cmd = ([write_script(run_dir, "localexec.sh", LOCALEXEC_SH), "-n", str(nprocs)] + base
               if nprocs > 1 else base)
    elif launcher == "srun":
        bind = [write_script(run_dir, "bind.sh", BIND_SH)] if build.gpu else []
        cmd = ["srun", "--overlap", "--kill-on-bad-exit=1", "-N", str(nodes), "-n", str(nprocs),
               "-c", str(max(1, -(-npes // nprocs)))] + bind + base
        # libfabric CXI settings multi-node Reconverse runs needed on Delta.
        penv["FI_CXI_RX_MATCH_MODE"] = os.environ.get("FI_CXI_RX_MATCH_MODE", "hybrid")
        penv["FI_MR_CACHE_MONITOR"] = os.environ.get("FI_MR_CACHE_MONITOR", "disabled")
    else:
        cmd = ["mpiexec", "-n", str(nprocs)] + base
    return plan(cmd, launcher == "local")
