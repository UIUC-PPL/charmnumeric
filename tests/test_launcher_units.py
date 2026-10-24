"""Launcher pieces that need no backend: planning, keys, run-dir rotation, the supervisor."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from charmnumeric import _plan, launcher
from charmnumeric._build import Build


@pytest.fixture
def env(monkeypatch):
    for var in list(os.environ):
        if var.startswith(("SLURM_", "CNUM_")) or var == "CUDA_VISIBLE_DEVICES":
            monkeypatch.delenv(var)
    monkeypatch.setattr(_plan.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(_plan, "ncpus", lambda: 16)
    return monkeypatch


def _slurm(env, nodes, ntasks=None, gpus=None):
    env.setenv("SLURM_JOB_ID", "1")
    env.setenv("SLURM_JOB_NUM_NODES", str(nodes))
    if ntasks:
        env.setenv("SLURM_NTASKS", str(ntasks))
    if gpus:
        env.setenv("SLURM_GPUS_ON_NODE", str(gpus))


CPU = Build("/b/server.out", "/b/charmrun", "netlrts-linux-x86_64", None, "test")
GPU = Build("/b/server.out", "/b/charmrun", "netlrts-linux-x86_64-cuda", "NVIDIA", "test")
RCV = Build("/r/server.out", None, "reconverse-linux-x86_64", None, "test")


def plan(build, tmp_path, **kw):
    return _plan.make_plan(build, tmp_path, keyfile=tmp_path / "ccs.key", **kw)


def test_cpu_local_defaults(env, tmp_path):
    p = plan(CPU, tmp_path)
    assert (p.launcher, p.npes, p.nprocs, p.nodes) == ("local", 4, 4, 1)
    assert p.cmd[:2] == ["/b/charmrun", "+p4"] and "++local" in p.cmd
    assert p.cmd[p.cmd.index("++server-auth") + 1] == str(tmp_path / "ccs.key")
    env.setenv("CNUM_NPES", "3")
    assert plan(CPU, tmp_path).npes == 3
    assert plan(CPU, tmp_path, npes=7, port=1234).cmd[1:6] == [
        "+p7", "/b/server.out", "++server", "++server-port", "1234"]
    with pytest.raises(ValueError, match="nodes=2"):
        plan(CPU, tmp_path, nodes=2)
    with pytest.raises(ValueError, match="CPU build"):
        plan(CPU, tmp_path, gpus=1)


def test_cpu_srun(env, tmp_path):
    _slurm(env, 2, ntasks=8)
    p = plan(CPU, tmp_path)
    assert (p.launcher, p.npes, p.nodes) == ("srun", 8, 2)
    assert "+eth" in p.cmd and "++mpiexec" in p.cmd
    script = Path(p.cmd[p.cmd.index("++remote-shell") + 1])
    assert "srun --overlap --kill-on-bad-exit=1 -N 2" in script.read_text()
    assert oct(script.stat().st_mode & 0o777) == "0o700"
    assert plan(CPU, tmp_path, nodes=1).launcher == "local"
    with pytest.raises(RuntimeError, match="Slurm"):
        env.delenv("SLURM_JOB_ID")
        plan(CPU, tmp_path, launcher="srun")


def test_gpu_local(env, tmp_path):
    env.setenv("CUDA_VISIBLE_DEVICES", "2,5,6,7")
    p = plan(GPU, tmp_path)
    assert (p.npes, p.gpus, p.devices) == (4, 4, ["2", "5", "6", "7"])
    assert "++mpiexec" in p.cmd and p.env["CNUM_BIND_DEVICES"] == "2,5,6,7"
    p = plan(GPU, tmp_path, gpus=1)
    assert (p.npes, p.gpus, p.env["CUDA_VISIBLE_DEVICES"]) == (1, 1, "2") and "++local" in p.cmd
    p = plan(GPU, tmp_path, npes=4, gpus=2)
    assert (p.npes, p.gpus, p.env["CUDA_VISIBLE_DEVICES"]) == (4, 2, "2,5")
    with pytest.raises(ValueError, match="gpus=5"):
        plan(GPU, tmp_path, gpus=5)


def test_gpu_srun(env, tmp_path):
    _slurm(env, 2, gpus=4)
    env.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    p = plan(GPU, tmp_path)
    assert (p.launcher, p.npes, p.gpus, p.nodes) == ("srun", 8, 8, 2)
    assert "bind.sh" in Path(p.cmd[p.cmd.index("++remote-shell") + 1]).read_text()
    assert plan(GPU, tmp_path, gpus=6).npes == 6
    with pytest.raises(ValueError, match="must equal gpus"):
        plan(GPU, tmp_path, npes=4, gpus=6)


def test_reconverse(env, tmp_path):
    p = plan(RCV, tmp_path, npes=6)
    assert (p.nprocs, p.cmd[-6:-2]) == (1, ["/r/server.out", "+pe", "6", "++server"])
    _slurm(env, 2)
    p = plan(RCV, tmp_path, npes=8)
    assert p.cmd[:9] == ["srun", "--overlap", "--kill-on-bad-exit=1", "-N", "2", "-n", "2", "-c", "4"]
    assert p.local_host is False


def test_read_key(tmp_path):
    key, path = launcher._new_key(tmp_path)
    assert len(key) == 16 and max(key) < 100
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert launcher.read_key(str(path)) == key == launcher.read_key(path.read_text().split()[1])
    with pytest.raises(ValueError):
        launcher.read_key("abc")


def test_run_dirs_private_and_owned(env, tmp_path):
    env.setenv("CNUM_LOG_DIR", str(tmp_path / "logs"))
    d = launcher._new_run_dir()
    assert oct(d.stat().st_mode & 0o777) == "0o700"
    uid = os.getuid() + 1
    env.setattr(launcher.os, "getuid", lambda: uid)
    with pytest.raises(PermissionError):
        launcher._new_run_dir()


def test_prune_keeps_new_and_live(tmp_path):
    old = time.time() - 3 * 86400
    dirs = []
    for i in range(6):
        d = tmp_path / f"server-{i}"
        d.mkdir()
        (d / "server.log").write_text("x")
        os.utime(d / "server.log", (old + i, old + i))
        os.utime(d, (old + i, old + i))
        dirs.append(d)
    lock = os.open(dirs[0] / "supervisor.lock", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(lock, fcntl.LOCK_EX)
    os.utime(dirs[0], (old, old))
    try:
        launcher._prune(tmp_path, keep=2)
    finally:
        os.close(lock)
    assert [d.exists() for d in dirs] == [True, False, False, False, True, True]


def test_supervisor_kills_server_when_client_is_killed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    code = f"""
        import subprocess, sys, time
        p = subprocess.Popen([sys.executable, "-I", {str(launcher._SUPERVISOR)!r}, "1", {str(run)!r},
                              "bash", "-c", "sleep 300 & sleep 300; wait"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True)
        print(p.stdout.readline().decode().strip(), p.pid, flush=True)
        time.sleep(300)
    """
    client = subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)], stdout=subprocess.PIPE, text=True)
    pgid, sup = map(int, client.stdout.readline().split())
    os.killpg(pgid, 0)
    client.send_signal(signal.SIGKILL)
    client.wait()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
            os.kill(sup, 0)
        except ProcessLookupError:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
        time.sleep(0.2)
    else:
        pytest.fail("server process group survived its client")
