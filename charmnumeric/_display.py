"""Display of arrays in notebooks and by print(), and the IPython cell hook.

Showing values fetches at most the corners NumPy would print: arrays up to the
print threshold come back whole, larger ones as ``edgeitems`` per side of each axis.
``repr()`` never contacts the server.
"""

import html
import itertools
import os
import sys
import warnings

import numpy as np

_options = {"preview": True, "threshold": None, "edgeitems": None}


def set_display(preview=None, threshold=None, edgeitems=None):
    """Set how notebooks and print() show arrays; returns the previous settings.

    preview: fetch values to show them (False shows only shape, dtype and size).
    threshold, edgeitems: like numpy.set_printoptions (default: NumPy's settings),
    and they also bound how many elements are fetched.
    """
    previous = dict(_options)
    for key, value in (("preview", preview), ("threshold", threshold),
                       ("edgeitems", edgeitems)):
        if value is not None:
            _options[key] = value
    return previous


def _print_limits():
    numpy_options = np.get_printoptions()
    threshold = _options["threshold"]
    edgeitems = _options["edgeitems"]
    return (numpy_options["threshold"] if threshold is None else int(threshold),
            numpy_options["edgeitems"] if edgeitems is None else int(edgeitems))


def _fetch(x, interface):
    """x, or for large x a stand-in that holds only the corners NumPy prints."""
    threshold, edge = _print_limits()
    shape = tuple(x.shape)
    if int(x.size) <= threshold:
        return np.asarray(x.get(interface)).reshape(shape), False
    # Axes longer than 2*edge get a filler element in the middle, which NumPy elides.
    spans = [((0, n, slice(0, n)),) if n <= 2 * edge else
             ((0, edge, slice(0, edge)), (n - edge, n, slice(edge + 1, 2 * edge + 1)))
             for n in shape]
    out = np.zeros([n if n <= 2 * edge else 2 * edge + 1 for n in shape], dtype=x.dtype)
    for corner in itertools.product(*spans):
        block = x[tuple(slice(lo, hi) for lo, hi, _ in corner)]
        out[tuple(dst for _, _, dst in corner)] = np.asarray(block.get(interface)).reshape(
            [hi - lo for lo, hi, _ in corner])
    return out, True


def _host_values(x):
    from .interface import default_interface

    return _fetch(x, default_interface())


def _text(values, summarized, **kwargs):
    threshold, edge = _print_limits()
    with np.printoptions(threshold=0 if summarized else threshold, edgeitems=edge):
        return np.array2string(values, **kwargs) if kwargs else str(values)


def values_str(x):
    """str(x): NumPy's str of x, fetching only what it prints."""
    return _text(*_host_values(x))


def _nbytes(x):
    n = int(x.size) * np.dtype(x.dtype).itemsize
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _pending(x):
    """True if operations writing x are recorded but not yet sent to the server."""
    from charmtyles.ast import get_execution_state

    name = x._base.name if hasattr(x, "_base") else x.name
    return any(access[0] == name
               for node in get_execution_state()._snapshot_tasks() for access in node.writes)


def _header(x):
    return (f"charmnumeric.{type(x).__name__} shape={tuple(x.shape)} "
            f"dtype={np.dtype(x.dtype).name} ({_nbytes(x)})")


def mimebundle(x):
    """Plain-text and HTML display of x for IPython (one fetch for both)."""
    header = _header(x)
    kind = type(x).__name__
    host, note = None, None
    if _options["preview"]:
        try:
            host = _host_values(x)
        except Exception as exc:
            note = f"values unavailable: {exc}"
    else:
        note = "values not fetched; pending" if _pending(x) else "values not fetched"
    if host is not None:
        body = _text(*host, separator=", ", prefix=f"{kind}(")
        tail = f", shape={tuple(x.shape)}" if host[1] else ""
        text = f"{kind}({body}{tail}, dtype={np.dtype(x.dtype).name})"
        shown = f"<pre style='margin:0.3em 0 0 0'>{html.escape(_text(*host))}</pre>"
    else:
        text = f"{header}\n<{note}>"
        shown = f"<div><i>{html.escape(note)}</i></div>"
    return {
        "text/plain": text,
        "text/html": f"<div><code>{html.escape(header)}</code>{shown}</div>",
    }


# IPython: send each cell's pending operations when the cell ends, so the server
# computes while the next cell is written (CNUM_FLUSH_AFTER_CELL=0 disables).
_FLUSH_AFTER_CELL = os.environ.get("CNUM_FLUSH_AFTER_CELL", "1") not in ("", "0")


def _flush_after_cell(result=None):
    from charmtyles.ast import get_execution_state
    from charmtyles.core import execute

    from . import interface

    iface = interface._default_interface
    if iface is None or getattr(iface, "server", None) is None:
        return
    if not get_execution_state()._has_pending_tasks():
        return
    try:
        execute(iface)
    except Exception as exc:
        warnings.warn(f"charmnumeric: could not send this cell's operations: {exc}")


def register_ipython():
    IPython = sys.modules.get("IPython")
    shell = IPython.get_ipython() if IPython is not None else None
    if shell is None or not _FLUSH_AFTER_CELL:
        return
    callbacks = shell.events.callbacks.get("post_run_cell", [])
    if not any(getattr(cb, "__qualname__", None) == "_flush_after_cell" for cb in callbacks):
        shell.events.register("post_run_cell", _flush_after_cell)
