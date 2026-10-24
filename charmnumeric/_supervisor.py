"""Run a backend server and stop it when the client goes away.

Started by launcher.py as ``python -I _supervisor.py GRACE LOG CMD...`` (stdlib
only, so -I keeps charmnumeric/signal.py etc. off sys.path). The server runs in
its own process group with output appended to LOG; its pid is written to
stdout. EOF on stdin (client exit, crash or SIGKILL) or SIGTERM/SIGINT/SIGHUP
sends SIGTERM to the group, then SIGKILL after GRACE seconds. Exits with the
server's exit code.
"""

import ctypes
import os
import select
import signal
import subprocess
import sys
import time

_PR_SET_PDEATHSIG = 1


def _libc_prctl():
    try:
        return ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None


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
    log = sys.argv[2]
    cmd = sys.argv[3:]
    prctl = _libc_prctl()

    def child_setup():
        # The server dies with this supervisor even if it is SIGKILLed.
        if prctl is not None:
            prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)

    stop_requested = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop_requested.append(1))

    with open(log, "ab", buffering=0) as out:
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    preexec_fn=child_setup)
        except OSError as exc:
            out.write(f"charmnumeric> could not run {cmd[0]}: {exc}\n".encode())
            return 127
    os.write(1, f"{proc.pid}\n".encode())
    os.close(1)

    stdin = sys.stdin.fileno()
    while True:
        if stop_requested:
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
