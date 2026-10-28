"""Start, connect to and stop the charmnumeric backend server.

The first operation that needs the backend starts a local server if nothing is connected;
it is stopped at exit. Environment: CNUM_HOST/CNUM_PORT (connect instead; CNUM_CCS_KEY its
key), CNUM_AUTOSTART=0, CNUM_SERVER/CNUM_CHARMRUN, CNUM_NPES, CNUM_ODF, CNUM_LOG_DIR,
CNUM_START_TIMEOUT, CNUM_QUIET=1.
"""

from __future__ import annotations

import atexit
import fcntl
import os
from pathlib import Path
import re
import secrets
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from ._build import Build, env_int, find_build, visible_devices  # noqa: F401
from ._plan import make_plan

__all__ = ["start", "stop", "connect", "server", "find_build", "ServerStartError"]

_SUPERVISOR = Path(__file__).resolve().parent / "_supervisor.py"
_READY = "charmtyles> CCS server ready"
_ADDR = re.compile(r"Server IP = ([0-9.]+), Server port = (\d+)")
_KEEP_RUN_DIRS = 20


class ServerStartError(RuntimeError):
    """The backend server exited or failed during startup."""


def _log_root():
    root = os.environ.get("CNUM_LOG_DIR")
    if root:
        return Path(root).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "charmnumeric"


def _prune(root, keep=_KEEP_RUN_DIRS, min_age=86400):
    """Delete old run directories beyond the newest `keep`, except those of live servers."""
    dirs = []
    for d in root.glob("server-*"):
        try:
            mtime = max(p.stat().st_mtime for p in (d, d / "server.log") if p.exists())
        except (OSError, ValueError):
            continue
        dirs.append((mtime, d))
    dirs.sort()
    for mtime, d in dirs[:max(0, len(dirs) - keep)]:
        if time.time() - mtime < min_age:
            continue
        try:
            fd = os.open(d / "supervisor.lock", os.O_RDONLY)
        except FileNotFoundError:
            fd = None
        except OSError:
            continue
        try:
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            shutil.rmtree(d, ignore_errors=True)
        except BlockingIOError:
            pass  # a supervisor holds it: the server is running
        finally:
            if fd is not None:
                os.close(fd)


def _new_run_dir():
    root = _log_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_uid != os.getuid():
        raise PermissionError(f"log directory {root} is not owned by you; set CNUM_LOG_DIR")
    _prune(root)
    run_dir = root / f"server-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{secrets.token_hex(3)}"
    run_dir.mkdir(mode=0o700)
    os.chmod(run_dir, 0o700)
    return run_dir


def _write_private(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


# Charm++ parses each key byte as two *decimal* digits, so bytes are 0..99 (~106 bits).
def _new_key(run_dir):
    key = bytes(secrets.randbelow(100) for _ in range(16))
    path = run_dir / "ccs.key"
    _write_private(path, "0 " + "".join(f"{b:02d}" for b in key) + "\n")
    return key, path


def read_key(spec):
    """A CCS key from bytes, a key file path, or its 32-digit text."""
    if spec is None or isinstance(spec, (bytes, bytearray)):
        return spec
    text = str(spec)
    if os.path.isfile(os.path.expanduser(text)):
        text = Path(os.path.expanduser(text)).read_text().split()[-1]
    if not re.fullmatch(r"\d{32}", text):
        raise ValueError("a CCS key is a key file (from `python -m charmnumeric server`) "
                         "or its 32 digits")
    return bytes(int(text[i:i + 2]) for i in range(0, 32, 2))


class Server:
    """A backend server started by this process."""

    def __init__(self, build, plan, run_dir, proc, odf, key, keyfile):
        self.build = build
        self.plan = plan
        self.run_dir = run_dir
        self.log_path = run_dir / "server.log"
        self.proc = proc  # the _supervisor
        self.odf = odf
        self.key = key
        self.keyfile = keyfile
        self.owner = os.getpid()
        self.pid = None  # process group of charmrun / srun / server.out
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

    def _close_pipes(self):
        for f in (self.proc.stdin, self.proc.stdout):
            if f is not None and not f.closed:
                try:
                    f.close()
                except OSError:
                    pass

    def kill(self, grace=None):
        """Stop the server: EOF to the supervisor; signals if that fails or is interrupted."""
        try:
            self._close_pipes()
            self.proc.wait((grace if grace is not None else _grace(self.plan)) + 5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                if self.pid:
                    try:
                        os.killpg(self.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                self.proc.wait()

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
    elif srv.build.gpu and srv.plan.launcher == "local" and not srv.plan.devices:
        hint = "\nThis is a GPU build but no GPU is visible here; use a CPU build (CNUM_SERVER)."
    elif re.search(r"charmnumeric> could not run|error while loading shared libraries|"
                   r"command not found|Couldn't find mpiexec", text):
        hint = "\nA program or shared library was not found; see the log."
    return ServerStartError(f"{msg}\nCommand: {shlex.join(srv.plan.cmd)}\nLog: {srv.log_path}{hint}\n"
                            f"--- last lines of the log ---\n{srv.tail()}")


def _launch(npes=None, gpus=None, nodes=None, launcher="auto", extra_args=(), env=None,
            timeout=None, port=None, odf=None, build=None):
    """Start a server (with a fresh CCS key) and wait until it is ready; does not connect."""
    build = build or find_build()
    run_dir = _new_run_dir()
    key, keyfile = _new_key(run_dir)
    plan = make_plan(build, run_dir, npes, gpus, nodes, launcher, extra_args, port, keyfile, env)
    if timeout is None:
        timeout = env_int("CNUM_START_TIMEOUT", 120 if plan.launcher == "local" else 300)
    child_env = dict(os.environ)
    child_env.update(plan.env)
    child_env.update(env or {})
    _write_private(run_dir / "command.txt", shlex.join(plan.cmd) + "\n")
    proc = subprocess.Popen(
        [sys.executable, "-I", str(_SUPERVISOR), str(_grace(plan)), str(run_dir)] + plan.cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        cwd=run_dir, env=child_env, start_new_session=True)
    os.set_inheritable(proc.stdin.fileno(), False)
    srv = Server(build, plan, run_dir, proc, odf, key, keyfile)
    try:
        srv._read_pid(min(timeout, 30))
        start = time.monotonic()
        while True:
            text = srv.log()
            m = _ADDR.search(text)
            if m and _READY in text:
                srv.host = "127.0.0.1" if plan.local_host else m.group(1)
                srv.port = int(m.group(2))
                return srv
            if not srv.alive():
                raise _startup_error(srv, f"charmnumeric server exited with code "
                                          f"{srv.proc.returncode} during startup.")
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                raise TimeoutError(str(_startup_error(
                    srv, f"charmnumeric server not ready after {timeout:.0f} s "
                    "(CNUM_START_TIMEOUT / timeout=).")))
            time.sleep(0.05 if elapsed < 2 else 0.2)
    except BaseException:
        srv.kill(grace=1.0)
        raise


# Process-wide state: the interface in use and the server this process started (if any).
_lock = threading.RLock()
_server = None
_iface = None
_atexit_registered = False


def _default_odf(odf):
    return int(odf) if odf is not None else env_int("CNUM_ODF", 4)


def _resolve(host):
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host


def _new_interface(host, port, odf, key=None):
    from .interface import CharmNumericInterface

    iface = CharmNumericInterface()
    iface.connect(_resolve(host), int(port), _default_odf(odf), key=key)
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
            _server._close_pipes()
        _server = _iface = None
        _reset_frontend()
    if _iface is None:
        from . import interface as interface_module

        _iface = interface_module._default_interface
    return _iface


def _env_target():
    host, port = os.environ.get("CNUM_HOST"), os.environ.get("CNUM_PORT")
    if host is None and port is None:
        return None
    if host == "":
        raise ValueError("CNUM_HOST is set but empty")
    return host or "127.0.0.1", env_int("CNUM_PORT", 1234)


def start(npes=None, gpus=None, nodes=None, launcher="auto", extra_args=(), env=None,
          timeout=None, port=None, odf=None):
    """Start a server and connect to it; returns the interface.

    Defaults: npes $CNUM_NPES, else one per visible GPU (GPU builds) or min(cores, 4);
    launcher 'auto' = srun inside a multi-node Slurm allocation, else local; a free port.
    Returns the running server's interface on repeat calls (a different configuration
    raises: stop() first). With CNUM_HOST/CNUM_PORT set, connects there instead.
    """
    global _server, _iface
    with _lock:
        iface = _live_iface()
        if iface is not None:
            _check_same(dict(npes=npes, gpus=gpus, nodes=nodes, launcher=launcher,
                             extra_args=tuple(extra_args), env=env, port=port, odf=odf))
            return iface
        target = _env_target()
        if target is not None:
            if any(v not in (None, (), "auto") for v in (npes, gpus, nodes, launcher, tuple(extra_args), env, port)):
                print(f"charmnumeric: CNUM_HOST/CNUM_PORT set; connecting to {target[0]}:{target[1]} "
                      "instead of starting a server", file=sys.stderr)
            return connect(*target, odf=odf)
        srv = _launch(npes, gpus, nodes, launcher, extra_args, env, timeout, port, _default_odf(odf))
        try:
            iface = _new_interface(srv.host, srv.port, srv.odf, srv.key)
        except BaseException:
            srv.kill(grace=1.0)
            raise
        iface.local_server = srv
        _server, _iface = srv, iface
        _register_handlers()
        if os.environ.get("CNUM_QUIET", "0") in ("", "0"):
            print(f"charmnumeric: started a server with {srv.describe()} at {srv.host}:{srv.port}; "
                  f"log: {srv.log_path}", file=sys.stderr, flush=True)
        return iface


def _check_same(req):
    srv = _server
    asked = {k: v for k, v in req.items() if v not in (None, (), "auto")}
    if srv is None:
        if asked:
            raise RuntimeError("charmnumeric is connected to a server it did not start; "
                               "call charmnumeric.stop() before start(...) with options")
        return
    p = srv.plan
    have = dict(npes=p.npes, gpus=p.gpus, nodes=p.nodes, launcher=p.launcher,
                extra_args=p.request["extra_args"], env=p.request["env"], port=srv.port, odf=srv.odf)
    diff = {k: v for k, v in asked.items() if v != have[k] and v != p.request.get(k)}
    if diff:
        raise RuntimeError(f"a charmnumeric server is already running ({srv.describe()}); "
                           f"call charmnumeric.stop() before starting one with {diff}")


def connect(host="127.0.0.1", port=1234, odf=None, key=None):
    """Connect to a running server, e.g. one from ``python -m charmnumeric server``.

    key: its key file (or bytes / digits); default $CNUM_CCS_KEY. A server serves one
    client process.
    """
    global _iface
    with _lock:
        addr = (_resolve(host), int(port))
        key = read_key(key if key is not None else os.environ.get("CNUM_CCS_KEY") or None)
        current = _live_iface()
        if current is not None:
            if _iface_addr(current) == addr:
                return current
            if _server is not None:
                raise RuntimeError(f"this process runs a charmnumeric server ({_server.host}:"
                                   f"{_server.port}); call charmnumeric.stop() before connecting elsewhere")
            stop()
        iface = _new_interface(*addr, odf, key)
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
    """The Server this process started (log_path, host, port, keyfile, ...), or None."""
    with _lock:
        _live_iface()
        return _server


def stop():
    """Stop the server this process started (or forget a connected one)."""
    global _server, _iface
    with _lock:
        srv, iface = _server, _iface
        _server = _iface = None
        try:
            if srv is not None and srv.owner == os.getpid():
                _shutdown(srv)
        finally:
            if srv is not None or iface is not None:
                _reset_frontend()


def _shutdown(srv):
    """Ask the server to exit (CkExit), then make sure the whole launch is gone."""
    try:
        if srv.alive() and srv.port:
            from charmtyles.interface import CCSInterface, Handlers

            try:
                ccs = CCSInterface()
                ccs._open(srv.host, srv.port, srv.key)
                ccs.send_command_async(Handlers.disconnection_handler, b"")
                srv.proc.wait(5)
            except Exception:
                pass
    finally:
        srv.kill()


def auto_interface():
    """The interface for an operation that needs one: the current one, else $CNUM_HOST/$CNUM_PORT,
    else a newly started local server (unless CNUM_AUTOSTART=0)."""
    with _lock:
        iface = _live_iface()
        if iface is not None:
            return iface
        target = _env_target()
        if target is not None:
            return connect(*target)
        if os.environ.get("CNUM_SIM", "") not in ("", "0"):
            raise RuntimeError("CNUM_SIM is set but no simulator interface exists; create it "
                               "(e.g. _cnum_connect.get_interface()) before using arrays")
        if os.environ.get("CNUM_AUTOSTART", "1").lower() in ("0", "false", "no", "off"):
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


def _after_fork_in_child():
    # The child must not keep the supervisor's pipe open (EOF signals the owner's exit).
    if _server is not None:
        _server._close_pipes()


def _register_handlers():
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_atexit)
        os.register_at_fork(after_in_child=_after_fork_in_child)
        _atexit_registered = True
