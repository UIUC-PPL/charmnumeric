"""Run examples/notebooks/quickstart.ipynb and interactive scenarios in real Jupyter kernels.

Each test starts its own server (a server serves one client process). Server binary:
CHARMNUMERIC_SERVER, else src/build/server.out or src/build-nc/server.out.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import time

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NOTEBOOK = ROOT / "examples" / "notebooks" / "quickstart.ipynb"

pytestmark = pytest.mark.integration


def _server_binary():
    candidates = [os.environ.get("CHARMNUMERIC_SERVER"),
                  ROOT / "src" / "build" / "server.out", ROOT / "src" / "build-nc" / "server.out"]
    for path in candidates:
        if path and Path(path).is_file():
            return Path(path).resolve()
    pytest.skip("no server.out; set CHARMNUMERIC_SERVER")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    binary = _server_binary()
    port = _free_port()
    log = tmp_path / "server.log"
    pes = os.environ.get("CHARMNUMERIC_PES", "2")
    with log.open("w") as out:
        proc = subprocess.Popen([str(binary.parent / "charmrun"), f"+p{pes}", str(binary),
                                 "++server", "++server-port", str(port), "++local"],
                                stdout=out, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while "CCS server ready" not in log.read_text():
            if proc.poll() is not None or time.monotonic() > deadline:
                pytest.fail(f"server did not start:\n{log.read_text()}")
            time.sleep(0.2)
        yield {"CNUM_HOST": "127.0.0.1", "CNUM_PORT": str(port)}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture
def kernel_env(server, monkeypatch):
    for key, value in server.items():
        monkeypatch.setenv(key, value)
    return server


def _outputs(cell):
    text = []
    for out in cell.get("outputs", []):
        text.append(out.get("text", ""))
        text.extend(str(v) for k, v in out.get("data", {}).items() if k.startswith("text/"))
    return "\n".join(text)


def test_quickstart_notebook(kernel_env):
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert not [o for c in code for o in c.outputs if o.output_type == "error"]
    tagged = {t: c for c in code for t in c.metadata.get("tags", [])}
    display = _outputs(tagged["display"])
    assert "Array([" in display and "shape=(1000001,)" in display and "..." in display
    assert "Array(shape=(1000001,)" in _outputs(tagged["repr"])
    assert "values not fetched" in _outputs(tagged["no-preview"])
    assert any("image/png" in o.get("data", {}) for c in code for o in c.outputs)


def _cell(source):
    return nbformat.v4.new_code_cell(source)


STARTUP = ("import os, time\nimport numpy as np\nimport charmnumeric as cnp\n"
           "cnp.CharmNumericInterface().connect(os.environ['CNUM_HOST'], "
           "int(os.environ['CNUM_PORT']), 4)")
# "Interrupt kernel" sends SIGINT to the kernel process; sent from inside, so it lands
# while the cell runs however loaded the machine is.
SLOW = ("import signal, threading\n"
        "a = cnp.ones((1500, 1500)) * 1e-3\n"
        "for _ in range(10):\n    a = (a @ a) * 1e-3\n"
        "t0 = time.monotonic()\n"
        "threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGINT)).start()\n"
        "float(cnp.sum(a))")


def test_interrupt_rerun_and_restart(kernel_env):
    nb = nbformat.v4.new_notebook(cells=[
        _cell(STARTUP), _cell(SLOW), _cell("v = float(cnp.sum(cnp.ones(10)))\nprint(time.monotonic() - t0, v)"),
        _cell(STARTUP), _cell("b = cnp.arange(6.0) * 2\nb"), _cell("float(cnp.sum(a))"),
    ])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3")
    with client.setup_kernel():
        client.execute_cell(nb.cells[0], 0)
        start = time.monotonic()
        try:
            client.execute_cell(nb.cells[1], 1)
            interrupted_at = None
        except nbclient.exceptions.CellExecutionError as exc:
            assert "KeyboardInterrupt" in str(exc)
            interrupted_at = time.monotonic() - start
        client.execute_cell(nb.cells[2], 2)
        ready, value = _outputs(nb.cells[2]).split()[-2:]
        assert value == "10.0"
        # The interrupt returned well before the abandoned product was done (unless it was
        # done within the 1 s before the interrupt).
        if interrupted_at is None:
            assert float(ready) < 2.5
        else:
            assert interrupted_at < float(ready) - 0.5
        for i in (3, 4, 5):
            client.execute_cell(nb.cells[i], i)
        assert "Array([ 0.,  2.,  4.,  6.,  8., 10.], dtype=float64)" in _outputs(nb.cells[4])
        assert _outputs(nb.cells[5]).strip() == "0.0"

    # A new kernel (as after "Restart kernel") cannot take over the server.
    fresh = nbformat.v4.new_notebook(cells=[_cell(STARTUP)])
    fresh.metadata["kernelspec"] = nb.metadata["kernelspec"]
    with pytest.raises(nbclient.exceptions.CellExecutionError, match="another client process"):
        nbclient.NotebookClient(fresh, timeout=120, kernel_name="python3").execute()
