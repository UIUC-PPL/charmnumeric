"""Array display for notebooks and print(), and the IPython after-cell flush.

Shown values come from one fetch of the corners NumPy prints; ``__repr__`` stays
metadata only and never contacts the server.
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
    threshold = numpy_options["threshold"] if threshold is None else threshold
    edgeitems = numpy_options["edgeitems"] if edgeitems is None else int(edgeitems)
    return threshold, max(1, edgeitems)  # NumPy's edgeitems=0 output shows no real values


def _fetch(x, interface):
    """(host values, summarized): x itself, or the corners NumPy prints with one filler
    element in the middle of each elided axis (NumPy never prints it)."""
    from .charmnumeric import empty

    threshold, edge = _print_limits()
    shape = tuple(x.shape)
    summarized = int(x.size) > threshold
    elided = [summarized and n > 2 * edge for n in shape]
    if not any(elided):
        return np.asarray(x.get(interface)).reshape(shape), summarized
    spans = [((slice(0, edge), slice(0, edge)), (slice(n - edge, n), slice(edge, 2 * edge)))
             if cut else ((slice(0, n), slice(0, n)),) for n, cut in zip(shape, elided)]
    corners = empty(tuple(2 * edge if cut else n for n, cut in zip(shape, elided)),
                    dtype=x.dtype)
    for corner in itertools.product(*spans):
        corners[tuple(dst for _, dst in corner)] = x[tuple(src for src, _ in corner)]
    host = np.asarray(corners.get(interface)).reshape(corners.shape)
    for axis, cut in enumerate(elided):
        if cut:
            host = np.insert(host, edge, host.take(0, axis=axis), axis=axis)
    return host, True


def _host_values(x):
    from .interface import default_interface

    return _fetch(x, default_interface())


def _text(values, summarized, **kwargs):
    threshold, edge = _print_limits()
    with np.printoptions(threshold=0 if summarized else threshold, edgeitems=edge):
        return np.array2string(values, **kwargs) if kwargs else str(values)


def _repr_text(kind, values, summarized, shape, dtype):
    """NumPy's array_repr layout (shape shown when summarized), always naming the dtype."""
    prefix = f"{kind}("
    body = prefix + _text(values, summarized, separator=", ", prefix=prefix, suffix=")") + ","
    extras = ", ".join(([f"shape={tuple(shape)}"] if summarized else [])
                       + [f"dtype={np.dtype(dtype).name}"]) + ")"
    last_line = len(body) - (body.rfind("\n") + 1)
    wraps = last_line + len(extras) + 1 > np.get_printoptions()["linewidth"]
    return body + ("\n" + " " * len(prefix) if wraps else " ") + extras


def values_str(x):
    """str(x): NumPy's str of x, fetching only what it prints (raises if the server is gone)."""
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
        text = _repr_text(kind, *host, x.shape, x.dtype)
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


def _flush_after_cell(_result=None):
    from charmtyles.ast import get_execution_state
    from charmtyles.core import execute

    from . import interface

    iface = interface._default_interface
    if iface is None or not iface.connected:
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
