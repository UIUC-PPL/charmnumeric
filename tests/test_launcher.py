"""Server auto-start, explicit start/stop/connect, startup errors and cleanup (charmnumeric.launcher).

Needs a built server.out (found like charmnumeric.find_build: $CNUM_SERVER, _build_info, src/build-*).
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import charmnumeric as cnp
from charmnumeric import launcher

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]


def _have_build():
    try:
        launcher.find_build()
    except FileNotFoundError:
        return False
    return True


needs_build = pytest.mark.skipif(not _have_build(), reason="no charmnumeric server.out")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in ("CNUM_HOST", "CNUM_PORT", "CNUM_AUTOSTART", "CNUM_QUIET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CNUM_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CNUM_NPES", os.environ.get("CNUM_NPES", "2"))
    yield
    launcher.stop()


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("CNUM_HOST", "CNUM_PORT")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), str(ROOT.parents[1]),
                                                      env.get("PYTHONPATH")]))
    env.update(extra)
    return env


def _py(code):
    return [sys.executable, "-c", textwrap.dedent(code)]


def _gone(pid, timeout=20.0):
    """True once no process of group `pid` exists."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)
    return False


def _fake_server(tmp_path, body):
    path = tmp_path / "fake" / "server.out"
    path.parent.mkdir()
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)
    return path


@needs_build
def test_autostart_and_exit_cleanup():
    code = """
        import charmnumeric as cnp
        a = cnp.ones(10)
        print(a.sum())
        srv = cnp.launcher.server()
        print("PIDS", srv.pid, srv.proc.pid)
    """
    out = subprocess.run(_py(code), env=_env(), capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[0] == "10.0"
    assert "started a server" in out.stderr
    pid, sup = map(int, out.stdout.split("PIDS")[1].split())
    assert _gone(pid) and _gone(sup, 5)


@needs_build
def test_start_stop_restart():
    iface = cnp.start(npes=2)
    assert cnp.start() is iface and cnp.start(npes=2) is iface
    assert float(cnp.arange(5).sum()) == 10.0
    srv = launcher.server()
    assert srv.alive() and srv.npes == 2 and srv.log_path.is_file()
    with pytest.raises(RuntimeError, match="already running"):
        cnp.start(npes=3)
    cnp.stop()
    assert not srv.alive() and _gone(srv.pid)
    assert launcher.server() is None
    cnp.stop()  # no-op
    iface2 = cnp.start(npes=1)
    assert iface2 is not iface
    assert float(cnp.ones((3, 4)).sum()) == 12.0


@needs_build
def test_restart_after_server_crash():
    cnp.start(npes=1)
    srv = launcher.server()
    os.killpg(srv.pid, signal.SIGKILL)
    srv.proc.wait(20)
    with pytest.raises(ConnectionError):
        float(cnp.ones(3).sum())
    cnp.start(npes=1)  # notices the dead server and starts a fresh one
    assert launcher.server() is not srv
    assert float(cnp.full(4, 2.0).sum()) == 8.0


@needs_build
def test_connect_to_cli_server(tmp_path):
    cli = subprocess.Popen([sys.executable, "-m", "charmnumeric", "server", "--npes", "2"],
                           env=_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        host, port = cli.stdout.readline().strip().split(":")
        code = """
            import charmnumeric as cnp
            print(cnp.arange(4).sum())
            print(cnp.launcher.server())
        """
        out = subprocess.run(_py(code), env=_env(CNUM_HOST=host, CNUM_PORT=port),
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["6", "None"]
        assert "started a server" not in out.stderr
    finally:
        cli.send_signal(signal.SIGINT)
        assert cli.wait(30) == 0


@needs_build
def test_explicit_connect():
    cli = subprocess.Popen([sys.executable, "-m", "charmnumeric", "server", "--npes", "1"],
                           env=_env(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        host, port = cli.stdout.readline().strip().split(":")
        iface = cnp.connect(host, int(port))
        assert cnp.connect(host, int(port)) is iface
        assert cnp.start() is iface
        assert float(cnp.ones(7).sum()) == 7.0
        cnp.stop()  # forgets the connection; the server keeps running
        assert cli.poll() is None
    finally:
        cli.terminate()
        cli.wait(30)


@needs_build
def test_cli_run(tmp_path):
    script = tmp_path / "prog.py"
    script.write_text("import sys, charmnumeric as cnp\nprint(cnp.ones(int(sys.argv[1])).sum())\n")
    out = subprocess.run([sys.executable, "-m", "charmnumeric", "run", "--npes", "2", str(script), "6"],
                         env=_env(), capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "6.0"


@needs_build
def test_no_orphans_after_client_sigkill():
    code = """
        import sys, time, charmnumeric as cnp
        cnp.start(npes=2)
        srv = cnp.launcher.server()
        print(srv.pid, srv.proc.pid, flush=True)
        time.sleep(600)
    """
    client = subprocess.Popen(_py(code), env=_env(), stdout=subprocess.PIPE, text=True)
    pid, sup = map(int, client.stdout.readline().split())
    os.killpg(pid, 0)
    client.kill()
    client.wait()
    assert _gone(pid) and _gone(sup, 5)


def test_crash_during_startup(tmp_path, monkeypatch):
    fake = _fake_server(tmp_path, 'echo "loading..."; echo "fatal: no such device" >&2; exit 3')
    monkeypatch.setenv("CNUM_SERVER", str(fake))
    with pytest.raises(launcher.ServerStartError) as err:
        cnp.start(npes=1)
    msg = str(err.value)
    assert "exited with code 3" in msg and "fatal: no such device" in msg and "server.log" in msg


def test_startup_timeout_kills_server(tmp_path, monkeypatch):
    fake = _fake_server(tmp_path, f'echo $$ > {tmp_path}/pid; exec sleep 300')
    monkeypatch.setenv("CNUM_SERVER", str(fake))
    with pytest.raises(TimeoutError, match="not ready after 2 s"):
        cnp.start(npes=1, timeout=2)
    pid = int((tmp_path / "pid").read_text())
    assert _gone(pid, 10)


@needs_build
def test_port_in_use():
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        s.listen()
        with pytest.raises(launcher.ServerStartError, match="already in use"):
            cnp.start(npes=1, port=s.getsockname()[1])


def test_binary_not_found(monkeypatch):
    monkeypatch.setenv("CNUM_SERVER", "/nonexistent/server.out")
    with pytest.raises(FileNotFoundError, match="CNUM_SERVER"):
        cnp.start()


def test_autostart_disabled(monkeypatch):
    monkeypatch.setenv("CNUM_AUTOSTART", "0")
    with pytest.raises(RuntimeError, match="CNUM_AUTOSTART=0"):
        float(cnp.ones(3).sum())


def test_build_inference(tmp_path, monkeypatch):
    d = tmp_path / "b"
    d.mkdir()
    (d / "server.out").write_bytes(b"x")
    (d / "build.ninja").write_text("rule x\n  command = /opt/charm/reconverse-linux-x86_64-cuda/bin/charmc -o s\n")
    (d / "CMakeCache.txt").write_text("GPU_BACKEND:UNINITIALIZED=NVIDIA\n")
    (d / "charmrun").write_text("#!/bin/bash\n")
    monkeypatch.setenv("CNUM_SERVER", str(d / "server.out"))
    b = launcher.find_build()
    assert (b.arch, b.gpu, b.charmrun, b.reconverse) == ("reconverse-linux-x86_64-cuda", "NVIDIA", None, True)
