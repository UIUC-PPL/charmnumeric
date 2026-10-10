from __future__ import annotations

from pathlib import Path
import subprocess
import time

import numpy as np

from charmtyles.interface import CCSInterface


_default_interface = None


def default_interface():
    """The most recently created interface (ops with data-dependent sizes sync on it)."""
    if _default_interface is None:
        raise RuntimeError(
            "this operation has a data-dependent result size and needs a "
            "CharmNumericInterface; create (and connect) one first"
        )
    return _default_interface


_BARRIER_NAME = -1  # kBarrierName in src/backend.hpp


def synchronize(interface=None):
    """Block until the backend has finished every operation recorded so far."""
    from charmtyles.core import execute

    interface = interface or default_interface()
    execute(interface)
    interface.get(1, _BARRIER_NAME, 0, dtype=np.bool_)

class CharmNumericInterface(CCSInterface):
    def __init__(self, *args, **kwargs):
        global _default_interface
        super().__init__(*args, **kwargs)
        _default_interface = self

    def from_bytes(self, payload, **kwargs):
        reply_type = kwargs.pop("reply_type", "array")
        if reply_type == "array":
            dtype = kwargs.pop("dtype", int)
            return np.frombuffer(payload, dtype=dtype)
        if reply_type is None:
            return None
        return super().from_bytes(payload, reply_type=reply_type, **kwargs)


class LocalCluster(CharmNumericInterface):
    """Launch a local charmnumeric backend and connect to it."""

    def __init__(
        self,
        server_binary,
        *,
        server_ip="127.0.0.1",
        server_port=1234,
        odf=4,
        max_pes=1,
        charmrun=None,
        workdir=None,
        startup_timeout=20.0,
    ):
        self.server_binary = Path(server_binary).resolve()
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.odf = int(odf)
        self.max_pes = int(max_pes)
        self.workdir = Path(workdir or self.server_binary.parent).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.nodelist_path = self.workdir / "localnodelist"
        self.log_path = self.workdir / "server.log"
        self.charmrun = self._resolve_charmrun(charmrun)
        self.process = None
        self._log_handle = None

        self._write_nodelist()
        self._run_server()
        super().__init__()
        self._connect_with_retry(timeout=float(startup_timeout))

    def _resolve_charmrun(self, charmrun):
        if charmrun:
            if Path(charmrun).is_absolute() or "/" in charmrun:
                return str(Path(charmrun).resolve())
            return charmrun

        sibling = self.server_binary.parent / "charmrun"
        if sibling.is_file():
            return str(sibling)
        return "charmrun"

    def _write_nodelist(self):
        data = "".join("host localhost\n" for _ in range(self.max_pes))
        self.nodelist_path.write_text(data)

    def _run_server(self):
        cmd = [
            self.charmrun,
            f"+p{self.max_pes}",
            str(self.server_binary),
            "++server",
            "++server-port",
            str(self.server_port),
            "++nodelist",
            str(self.nodelist_path),
        ]
        self._log_handle = self.log_path.open("w")
        self.process = subprocess.Popen(
            cmd,
            cwd=self.workdir,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _connect_with_retry(self, timeout):
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                self.connect(self.server_ip, self.server_port, self.odf)
                return
            except Exception as exc:  # pragma: no cover - depends on local runtime
                last_error = exc
                time.sleep(1.0)

        self.close()
        raise RuntimeError(
            f"Timed out connecting to charmnumeric backend at "
            f"{self.server_ip}:{self.server_port}. See {self.log_path}."
        ) from last_error

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)
        self.process = None

        if self._log_handle is not None and not self._log_handle.closed:
            self._log_handle.close()
        self._log_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
