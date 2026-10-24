"""Start, connect to and stop the charmnumeric backend server.

Without an explicit connection, the first operation that needs the backend
starts a local server (``start()`` with defaults); it is stopped at exit.

Environment:
    CNUM_HOST, CNUM_PORT  connect to this running server instead of starting one
    CNUM_AUTOSTART=0      never start a server implicitly
    CNUM_SERVER           server.out to run (CNUM_CHARMRUN: its charmrun)
    CNUM_NPES             default PE count
    CNUM_ODF              over-decomposition factor (default 4)
    CNUM_LOG_DIR          where server run directories go (~/.cache/charmnumeric)
    CNUM_START_TIMEOUT    seconds to wait for the server to come up
    CNUM_QUIET=1          no startup message
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import re
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

__all__ = ["start", "stop", "connect", "server", "find_build", "ServerStartError"]

_PKG_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PKG_DIR.parent / "src"
_READY = "charmtyles> CCS server ready"
_ADDR = re.compile(r"Server IP = ([0-9.]+), Server port = (\d+)")
_DEVICE_VAR = {"NVIDIA": "CUDA_VISIBLE_DEVICES", "AMD": "ROCR_VISIBLE_DEVICES",
               "INTEL": "ZE_AFFINITY_MASK"}
_KEEP_RUN_DIRS = 20


class ServerStartError(RuntimeError):
    """The backend server exited or failed during startup."""


class Build:
    """A server binary and how to launch it."""

    def __init__(self, server, charmrun, arch, gpu, source):
        self.server = server
        self.charmrun = charmrun
        self.arch = arch
        self.gpu = gpu
        self.source = source

    @property
    def reconverse(self):
        return self.arch.startswith("reconverse")

    def __repr__(self):
        return (f"Build(server={str(self.server)!r}, charmrun={self.charmrun and str(self.charmrun)!r}, "
                f"arch={self.arch!r}, gpu={self.gpu!r}, source={self.source!r})")


def _build_info():
    try:
        from . import _build_info as info
    except ImportError:
        return None
    return info


def _infer_arch(build_dir):
    for name in ("build.ninja", "CMakeFiles/server.dir/build.make"):
        try:
            text = (build_dir / name).read_text(errors="replace")
        except OSError:
            continue
        m = re.search(r"([^\s\"';]+)/bin/charmc\b", text)
        if m:
            return Path(m.group(1)).name
    return None


def _needed_libs(path):
    """DT_NEEDED entries of an ELF64 little-endian binary (empty if unreadable)."""
    import struct

    try:
        with open(path, "rb") as f:
            hdr = f.read(64)
            if hdr[:4] != b"\x7fELF" or hdr[4] != 2 or hdr[5] != 1:
                return []
            phoff, = struct.unpack_from("<Q", hdr, 0x20)
            phentsize, phnum = struct.unpack_from("<HH", hdr, 0x36)
            f.seek(phoff)
            phdrs = [struct.unpack_from("<IIQQQQ", f.read(phentsize)) for _ in range(phnum)]
            loads = [(v, o, n) for t, _, o, v, _, n in phdrs if t == 1]
            dyn = next(((o, n) for t, _, o, _, _, n in phdrs if t == 2), None)
            if dyn is None:
                return []
            f.seek(dyn[0])
            raw = f.read(dyn[1])
            tags = [struct.unpack_from("<qQ", raw, i) for i in range(0, len(raw) - 15, 16)]
            strtab = next((v for t, v in tags if t == 5), None)
            off = next((o + strtab - v for v, o, n in loads if v <= strtab < v + n), None)
            names = []
            for t, v in tags:
                if t == 1 and off is not None:
                    f.seek(off + v)
                    names.append(f.read(256).split(b"\0", 1)[0].decode(errors="replace"))
            return names
    except (OSError, struct.error, StopIteration):
        return []


def _infer_gpu(server, arch):
    try:
        text = (server.parent / "CMakeCache.txt").read_text()
        m = re.search(r"^GPU_BACKEND:\w+=(\w*)", text, re.M)
    except OSError:
        m = None
    if m:
        return m.group(1).upper() or None
    libs = " ".join(_needed_libs(server))
    for lib, gpu in (("libcudart", "NVIDIA"), ("libamdhip", "AMD"), ("libze_loader", "INTEL")):
        if lib in libs:
            return gpu
    if arch and "cuda" in arch:
        return "NVIDIA"
    return None


def find_build():
    """Locate server.out: $CNUM_SERVER, then charmnumeric._build_info, then src/build-*."""
    info = _build_info()
    tried = []
    env_server = os.environ.get("CNUM_SERVER")
    if env_server:
        server = Path(env_server).expanduser().resolve()
        if not server.is_file():
            raise FileNotFoundError(f"CNUM_SERVER={env_server} does not exist")
        source = "CNUM_SERVER"
    elif info is not None and getattr(info, "SERVER", None):
        server = Path(info.SERVER)
        if not server.is_file():
            raise FileNotFoundError(f"charmnumeric._build_info.SERVER={info.SERVER} does not exist; "
                                    "rebuild charmnumeric or set CNUM_SERVER")
        source = "_build_info"
    else:
        names = ["build-nc", "build", "build-gpu"]
        if _visible_devices("NVIDIA"):
            names.insert(0, names.pop())
        server = None
        for name in names:
            candidate = _SRC_DIR / name / "server.out"
            tried.append(str(candidate))
            if candidate.is_file():
                server = candidate
                break
        if server is None:
            raise FileNotFoundError(
                "charmnumeric backend (server.out) not found. Build it (see README) or set "
                "CNUM_SERVER=/path/to/server.out, or connect to a running server with "
                "CNUM_HOST/CNUM_PORT or charmnumeric.connect(). Looked in:\n  " + "\n  ".join(tried))
        source = str(server.parent)

    use_info = info is not None and str(server) == str(getattr(info, "SERVER", ""))
    if use_info:
        arch = info.CHARM_ARCH or ""
        gpu = info.GPU_BACKEND
        charmrun = info.CHARMRUN
    else:
        arch = _infer_arch(server.parent) or ""
        gpu = _infer_gpu(server, arch)
        sibling = server.parent / "charmrun"
        charmrun = sibling if sibling.is_file() and not arch.startswith("reconverse") else None
        if not arch:
            # A copied binary: Reconverse builds have no charmrun.
            arch = "unknown" if charmrun is not None else "reconverse"
    env_charmrun = os.environ.get("CNUM_CHARMRUN")
    if env_charmrun:
        charmrun = None if env_charmrun.lower() == "none" else env_charmrun
    return Build(server, Path(charmrun) if charmrun else None, arch, gpu, source)


def _visible_devices(gpu):
    val = os.environ.get(_DEVICE_VAR.get(gpu, "CUDA_VISIBLE_DEVICES"))
    if val is not None:
        return [d.strip() for d in val.split(",") if d.strip() and d.strip() != "-1"]
    if gpu == "NVIDIA" and shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return []
        if out.returncode == 0:
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return []


def _ncpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# Wrapper scripts written into each run directory (charmrun's ++remote-shell takes one program).
_LOCALEXEC_SH = r"""#!/bin/bash
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

_BIND_SH = r"""#!/bin/bash
# srun task wrapper: one device per task (SLURM_LOCALID mod the node's devices).
v=$CNUM_BIND_VAR
if [ -n "$SLURM_LOCALID" ]; then
  if [ -n "${!v}" ]; then IFS=, read -ra devs <<< "${!v}"
  else mapfile -t devs < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); fi
  [ ${#devs[@]} -gt 0 ] && export "$v=${devs[$((SLURM_LOCALID % ${#devs[@]}))]}"
fi
exec "$@"
"""


def _write_script(run_dir, name, text):
    path = run_dir / name
    path.write_text(text)
    path.chmod(0o755)
    return str(path)


class _Plan:
    """The server command line for one launch."""

    def __init__(self, cmd, env, npes, nprocs, nodes, launcher, devices, local_host):
        self.cmd = cmd
        self.env = env
        self.npes = npes
        self.nprocs = nprocs
        self.nodes = nodes
        self.launcher = launcher
        self.devices = devices
        self.local_host = local_host


def _plan(build, run_dir, npes, gpus, nodes, launcher, extra_args, port):
    in_slurm = "SLURM_JOB_ID" in os.environ
    alloc_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) if in_slurm else 1
    if launcher == "auto":
        launcher = "srun" if in_slurm and (nodes or alloc_nodes) > 1 else "local"
    if launcher not in ("local", "srun", "mpiexec"):
        raise ValueError(f"launcher must be 'auto', 'local', 'srun' or 'mpiexec', not {launcher!r}")
    if launcher == "srun":
        if not shutil.which("srun"):
            raise RuntimeError("launcher='srun' needs Slurm's srun on PATH")
        if not in_slurm:
            raise RuntimeError("launcher='srun' must run inside a Slurm allocation (salloc/sbatch)")
        nodes = int(nodes or alloc_nodes)
    elif launcher == "local":
        if nodes not in (None, 1):
            raise ValueError(f"nodes={nodes} needs launcher='srun' (inside a Slurm allocation) or 'mpiexec'")
        nodes = 1

    if not npes and os.environ.get("CNUM_NPES"):
        npes = int(os.environ["CNUM_NPES"])
    env = {}
    devices = []
    if build.gpu:
        var = _DEVICE_VAR.get(build.gpu, "CUDA_VISIBLE_DEVICES")
        env["CNUM_BIND_VAR"] = var
        visible = _visible_devices(build.gpu)
        if launcher == "local":
            if gpus is not None and gpus > len(visible):
                raise RuntimeError(f"gpus={gpus} requested but {len(visible)} GPU(s) are visible "
                                   f"({var}={os.environ.get(var, '<unset>')})")
            n = gpus if gpus is not None else (npes or len(visible) or 1)
            devices = visible[:max(1, min(n, len(visible)))]
            npes = npes or n
        else:
            per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "0") or 0) or len(visible) or 1
            npes = npes or gpus or (nodes or 1) * per_node
        nprocs = npes
    else:
        if gpus:
            raise ValueError(f"{build.server} is a CPU build; gpus= needs a GPU build")
        if not npes:
            if launcher == "local":
                npes = min(_ncpus(), 4)
            else:
                ntasks = int(os.environ.get("SLURM_NTASKS", "0") or 0)
                npes = ntasks if ntasks >= (nodes or 1) else (nodes or 1) * 4
        nprocs = npes if not build.reconverse else (nodes if launcher != "local" else 1)
    npes = int(npes)
    if npes < 1:
        raise ValueError(f"npes must be positive, not {npes}")
    if devices:
        env["CNUM_BIND_DEVICES"] = ",".join(devices)
    srv_args = ["++server"] + (["++server-port", str(port)] if port else [])
    extra = list(extra_args)
    multi_local = launcher == "local" and len(devices) > 1 and nprocs > 1

    if build.charmrun is not None and not build.reconverse:
        cmd = [str(build.charmrun), f"+p{npes}", str(build.server)] + srv_args
        if launcher == "local" and not multi_local:
            cmd += ["++local"]
        elif launcher == "local":
            cmd += ["++mpiexec", "++remote-shell", _write_script(run_dir, "localexec.sh", _LOCALEXEC_SH)]
        elif launcher == "srun":
            bind = _write_script(run_dir, "bind.sh", _BIND_SH) if build.gpu else ""
            srun = (f"#!/bin/bash\nn=(); [ \"$1\" = -n ] && {{ n=(-n \"$2\"); shift 2; }}\n"
                    f"exec srun --overlap --kill-on-bad-exit=1 -N {nodes} \"${{n[@]}}\" {shlex.quote(bind) if bind else ''} \"$@\"\n")
            cmd += ["++mpiexec", "++remote-shell", _write_script(run_dir, "srun.sh", srun)]
            if build.arch.startswith("netlrts"):
                # 1400-byte datagrams: the 9000-byte default is IP-fragmented on 1500-MTU networks.
                cmd += ["+eth"]
        else:
            cmd += ["++mpiexec"]
        return _Plan(cmd + extra, env, npes, nprocs, nodes, launcher, devices, True)

    if not build.reconverse and npes > 1:
        raise RuntimeError(f"no charmrun found next to {build.server}; set CNUM_CHARMRUN, "
                           "or npes=1 to run server.out directly")
    pe_flag = "+pe" if build.reconverse else "+p"
    stdbuf = ["stdbuf", "-oL", "-eL"] if shutil.which("stdbuf") else []
    base = stdbuf + [str(build.server), pe_flag, str(npes)] + srv_args + extra
    if launcher == "local":
        cmd = ([_write_script(run_dir, "localexec.sh", _LOCALEXEC_SH), "-n", str(nprocs)] + base
               if nprocs > 1 else base)
    elif launcher == "srun":
        bind = [_write_script(run_dir, "bind.sh", _BIND_SH)] if build.gpu else []
        cpt = max(1, -(-npes // nprocs))
        cmd = ["srun", "--overlap", "--kill-on-bad-exit=1", "-N", str(nodes), "-n", str(nprocs),
               "-c", str(cpt)] + bind + base
        # LCI over libfabric CXI settings that multi-node Reconverse runs needed on Delta.
        env.setdefault("FI_CXI_RX_MATCH_MODE", os.environ.get("FI_CXI_RX_MATCH_MODE", "hybrid"))
        env.setdefault("FI_MR_CACHE_MONITOR", os.environ.get("FI_MR_CACHE_MONITOR", "disabled"))
    else:
        cmd = ["mpiexec", "-n", str(nprocs)] + base
    return _Plan(cmd, env, npes, nprocs, nodes, launcher, devices, launcher == "local")


def _log_root():
    root = os.environ.get("CNUM_LOG_DIR")
    if root:
        return Path(root).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "charmnumeric"


def _new_run_dir():
    root = _log_root()
    root.mkdir(parents=True, exist_ok=True)
    old = sorted(root.glob("server-*"), key=lambda p: p.stat().st_mtime)[:-_KEEP_RUN_DIRS]
    for path in old:
        if time.time() - path.stat().st_mtime > 86400:
            shutil.rmtree(path, ignore_errors=True)
    run_dir = root / f"server-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{next(_run_ids)}"
    run_dir.mkdir()
    return run_dir


def _counter():
    n = 0
    while True:
        n += 1
        yield n


_run_ids = _counter()


class Server:
    """A backend server started by this process."""

    def __init__(self, build, plan, run_dir, proc, odf):
        self.build = build
        self.plan = plan
        self.run_dir = run_dir
        self.log_path = run_dir / "server.log"
        self.proc = proc  # the _supervisor process
        self.odf = odf
        self.owner = os.getpid()
        self.pid = None  # the launcher's (charmrun/srun/server.out) pid and process group
        self.host = None
        self.port = None

    @property
    def npes(self):
        return self.plan.npes

    def alive(self):
        return self.proc.poll() is None

    def log(self):
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def tail(self, lines=30):
        return "\n".join(self.log().splitlines()[-lines:])

    def describe(self):
        kind = f"{self.build.gpu} GPU" if self.build.gpu else "CPU"
        where = f", {self.plan.nodes} nodes" if self.plan.nodes and self.plan.nodes > 1 else ""
        procs = f" in {self.plan.nprocs} processes" if self.plan.nprocs not in (1, self.npes) else ""
        return f"{self.npes} PE{'s' if self.npes != 1 else ''}{procs} ({kind}, {self.build.arch}{where})"

    def _read_pid(self, timeout):
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        line = self.proc.stdout.readline() if ready else b""
        self.pid = int(line) if line.strip().isdigit() else None

    def kill(self, grace=None):
        """Stop the server: EOF to the supervisor, then signals as a last resort."""
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        wait = (grace if grace is not None else _grace(self.plan)) + 5
        try:
            self.proc.wait(wait)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            if self.pid:
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            self.proc.wait()
        if self.proc.stdout:
            self.proc.stdout.close()

    def __repr__(self):
        state = "running" if self.alive() else f"exited ({self.proc.returncode})"
        return f"<charmnumeric Server {self.host}:{self.port} {self.describe()} {state}, log {self.log_path}>"


def _grace(plan):
    return 10.0 if plan.launcher != "local" else 3.0


def _startup_error(srv, msg):
    text = srv.log()
    hint = ""
    if "Error binding server socket" in text:
        cmd = srv.plan.cmd
        port = cmd[cmd.index("++server-port") + 1] if "++server-port" in cmd else "?"
        hint = f"\nThe CCS port {port} is already in use; pass another port, or port=None for a free one."
    elif srv.build.gpu and not srv.plan.devices and srv.plan.launcher == "local":
        hint = "\nThis is a GPU build but no GPU is visible here; use a CPU build (CNUM_SERVER)."
    elif "not found" in text or "No such file" in text:
        hint = "\nA program or library was not found; see the log."
    return ServerStartError(f"{msg}\nCommand: {shlex.join(srv.plan.cmd)}\nLog: {srv.log_path}{hint}\n"
                            f"--- last lines of the log ---\n{srv.tail()}")


def _launch(npes=None, gpus=None, nodes=None, launcher="auto", extra_args=(), env=None,
            timeout=None, port=None, odf=None, build=None):
    """Start a server and wait until it is ready; does not connect."""
    build = build or find_build()
    run_dir = _new_run_dir()
    plan = _plan(build, run_dir, npes, gpus, nodes, launcher, extra_args, port)
    if timeout is None:
        timeout = float(os.environ.get("CNUM_START_TIMEOUT", "120" if plan.launcher == "local" else "300"))
    child_env = dict(os.environ)
    child_env.update(plan.env)
    child_env.update(env or {})
    (run_dir / "command.txt").write_text(shlex.join(plan.cmd) + "\n")
    log_path = run_dir / "server.log"
    proc = subprocess.Popen(
        [sys.executable, "-I", str(_PKG_DIR / "_supervisor.py"), str(_grace(plan)), str(log_path)]
        + plan.cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        cwd=run_dir, env=child_env, start_new_session=True)
    srv = Server(build, plan, run_dir, proc, odf)
    try:
        srv._read_pid(min(timeout, 30))
        deadline = time.monotonic() + timeout
        while True:
            text = srv.log()
            m = _ADDR.search(text)
            if m and _READY in text:
                ip, srv.port = m.group(1), int(m.group(2))
                srv.host = "127.0.0.1" if plan.local_host else ip
                return srv
            if not srv.alive():
                rc = srv.proc.returncode
                raise _startup_error(srv, f"charmnumeric server exited with code {rc} during startup.")
            if time.monotonic() > deadline:
                raise TimeoutError(str(_startup_error(
                    srv, f"charmnumeric server not ready after {timeout:.0f} s "
                    "(CNUM_START_TIMEOUT / timeout=).")))
            time.sleep(0.05 if time.monotonic() + timeout - deadline < 2 else 0.2)
    except BaseException:
        srv.kill(grace=1.0)
        raise


# Process-wide state: the interface in use and the server this process started (if any).
_lock = threading.RLock()
_server = None
_iface = None
_atexit_registered = False


def _default_odf(odf):
    return int(odf if odf is not None else os.environ.get("CNUM_ODF", "4"))


def _resolve(host):
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host


def _new_interface(host, port, odf):
    from .interface import CharmNumericInterface

    iface = CharmNumericInterface()
    iface.connect(_resolve(host), int(port), _default_odf(odf))
    return iface


def _reset_frontend():
    """Forget pending work of a previous server (names keep counting so stale arrays never alias)."""
    import charmtyles.ast as ast_module
    from . import interface as interface_module

    ast_module.execution_state = ast_module.ExecutionState()
    interface_module._default_interface = None


def _live_iface():
    """The interface in use, if its server is still there."""
    global _iface, _server
    if _server is not None and (_server.owner != os.getpid() or not _server.alive()):
        if _server.owner == os.getpid():
            print(f"charmnumeric: the server exited (code {_server.proc.returncode}); "
                  f"log: {_server.log_path}", file=sys.stderr)
        _server = None
        _iface = None
        _reset_frontend()
    if _iface is None:
        from . import interface as interface_module

        _iface = interface_module._default_interface
    return _iface


def _env_target():
    host, port = os.environ.get("CNUM_HOST"), os.environ.get("CNUM_PORT")
    if host is None and port is None:
        return None
    return host or "127.0.0.1", int(port or 1234)


def start(npes=None, gpus=None, nodes=None, launcher="auto", extra_args=(), env=None,
          timeout=None, port=None, odf=None):
    """Start a charmnumeric server and connect to it; returns the interface.

    npes: PEs (default $CNUM_NPES, else: GPU build, one per visible GPU; CPU build,
        min(cores, 4); srun, the allocation's tasks). gpus: GPUs to use (GPU builds run
        one process per GPU). nodes: nodes for launcher='srun' (default: all of the
        allocation). launcher: 'auto' (srun inside a multi-node Slurm allocation,
        else local), 'local', 'srun' or 'mpiexec'. extra_args: more server
        arguments (e.g. ['+isomalloc_sync']). env: extra environment for the server.
        timeout: seconds to wait for startup. port: CCS port (default: free).
        odf: over-decomposition factor (default $CNUM_ODF or 4).

    Idempotent: if this process already started a server (or is connected) it is
    returned; asking for a different configuration raises, call stop() first.
    With CNUM_HOST/CNUM_PORT set, connects to that server instead. The server
    is stopped at exit, and also when this process dies (no orphans).
    """
    global _server, _iface
    requested = dict(npes=npes, gpus=gpus, nodes=nodes, port=port)
    with _lock:
        iface = _live_iface()
        if iface is not None:
            _check_same(requested, launcher, odf)
            return iface
        target = _env_target()
        if target is not None:
            if any(v is not None for v in requested.values()):
                print(f"charmnumeric: CNUM_HOST/CNUM_PORT set; connecting to {target[0]}:{target[1]} "
                      "instead of starting a server", file=sys.stderr)
            return connect(*target, odf=odf)
        srv = _launch(npes, gpus, nodes, launcher, extra_args, env, timeout, port,
                      _default_odf(odf))
        try:
            iface = _new_interface(srv.host, srv.port, srv.odf)
        except BaseException:
            srv.kill(grace=1.0)
            raise
        iface.local_server = srv
        _server, _iface = srv, iface
        _register_atexit()
        if os.environ.get("CNUM_QUIET", "0") in ("", "0"):
            print(f"charmnumeric: started a server with {srv.describe()} at {srv.host}:{srv.port}; "
                  f"log: {srv.log_path}", file=sys.stderr, flush=True)
        return iface


def _check_same(requested, launcher, odf):
    srv = _server
    if srv is None:
        if any(v is not None for v in requested.values()):
            raise RuntimeError("charmnumeric is connected to a server it did not start; "
                               "call charmnumeric.stop() before start(...) with options")
        return
    have = dict(npes=srv.plan.npes, gpus=len(srv.plan.devices) or None, nodes=srv.plan.nodes,
                port=srv.port)
    diff = {k: v for k, v in requested.items() if v is not None and v != have[k]}
    if launcher not in ("auto", srv.plan.launcher):
        diff["launcher"] = launcher
    if odf is not None and int(odf) != srv.odf:
        diff["odf"] = odf
    if diff:
        raise RuntimeError(f"a charmnumeric server is already running ({srv.describe()}); "
                           f"call charmnumeric.stop() before starting one with {diff}")


def connect(host="127.0.0.1", port=1234, odf=None):
    """Connect to a running server (e.g. one from ``python -m charmnumeric server``).

    A server serves one client session: connect once per server.
    """
    global _iface
    with _lock:
        addr = (_resolve(host), int(port))
        current = _live_iface()
        if current is not None:
            if _iface_addr(current) == addr:
                return current
            stop()
        iface = _new_interface(*addr, odf)
        iface.local_server = None
        _iface = iface
        return iface


def _iface_addr(iface):
    try:
        info = iface.server.conn_info
        return str(info.host_ip), int(info.host_port)
    except AttributeError:
        return None


def server():
    """The Server this process started (log_path, host, port, ...), or None."""
    with _lock:
        _live_iface()
        return _server


def stop():
    """Stop the server this process started (or forget a connected one)."""
    global _server, _iface
    with _lock:
        srv, iface = _server, _iface
        _server = _iface = None
        if srv is not None and srv.owner == os.getpid():
            _shutdown(srv)
        if srv is not None or iface is not None:
            _reset_frontend()


def _shutdown(srv):
    """Ask the server to exit (CkExit), then make sure the whole launch is gone."""
    if srv.alive() and srv.port:
        from charmtyles.interface import CCSInterface, Handlers

        try:
            ccs = CCSInterface()
            ccs._open(srv.host, srv.port)
            ccs.send_command_async(Handlers.disconnection_handler, b"")
            srv.proc.wait(5)
        except Exception:
            pass
    srv.kill()


def auto_interface():
    """The interface for operations that need one: the current one, else
    $CNUM_HOST/$CNUM_PORT, else a newly started local server (unless CNUM_AUTOSTART=0)."""
    with _lock:
        iface = _live_iface()
        if iface is not None:
            return iface
        target = _env_target()
        if target is not None:
            return connect(*target)
        if os.environ.get("CNUM_AUTOSTART", "1") in ("0", "false", "no", "off"):
            raise RuntimeError(
                "no charmnumeric server is connected and CNUM_AUTOSTART=0; call "
                "charmnumeric.start() or charmnumeric.connect(host, port), or set CNUM_HOST/CNUM_PORT")
        return start()


def _atexit():
    try:
        if _server is not None and _server.owner == os.getpid():
            stop()
    except Exception:
        pass


def _register_atexit():
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_atexit)
        _atexit_registered = True
