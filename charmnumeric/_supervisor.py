"""Run a backend server and stop it when the client goes away.

``python -I _supervisor.py GRACE RUN_DIR CMD...`` (stdlib only; -I keeps charmnumeric/signal.py
off sys.path). The server gets its own process group and logs to RUN_DIR/server.log; its pid
goes to stdout. Client exit or death (stdin EOF, or reparenting) or SIGTERM/SIGINT/SIGHUP:
SIGTERM to the group, SIGKILL after GRACE seconds. RUN_DIR/supervisor.lock stays locked while
this runs, so log rotation skips live servers.
"""

import ctypes
import fcntl
import os
import select
import signal
import subprocess
import sys
import time

_PR_SET_PDEATHSIG = 1


def _killpg(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _stop(proc, grace):
    _killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        pass
    _killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def main():
    grace = float(sys.argv[1])
    run_dir = sys.argv[2]
    cmd = sys.argv[3:]
    parent = os.getppid()
    lock = os.open(os.path.join(run_dir, "supervisor.lock"), os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        prctl = None

    def child_setup():
        # preexec_fn is safe here: this process is single-threaded. The server dies with us.
        if prctl is not None:
            prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)

    stop_requested = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop_requested.append(1))

    log = os.open(os.path.join(run_dir, "server.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, preexec_fn=child_setup)
    except OSError as exc:
        os.write(log, f"charmnumeric> could not run {cmd[0]}: {exc}\n".encode())
        return 127
    finally:
        os.close(log)
    os.write(1, f"{proc.pid}\n".encode())
    os.close(1)

    stdin = sys.stdin.fileno()
    while True:
        if stop_requested or os.getppid() != parent:
            _stop(proc, grace)
            break
        try:
            ready, _, _ = select.select([stdin], [], [], 0.25)
        except InterruptedError:
            continue
        if ready and not os.read(stdin, 4096):
            _stop(proc, grace)
            break
        if proc.poll() is not None:
            time.sleep(0.5)
            _killpg(proc.pid, signal.SIGKILL)  # leftovers of a crashed launcher
            break
    rc = proc.returncode
    return 128 - rc if rc < 0 else rc


if __name__ == "__main__":
    sys.exit(main())
