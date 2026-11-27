"""Display helpers that need no server."""

import numpy as np

from charmnumeric import _display


def test_set_display_restores_defaults():
    original = dict(_display._options)
    try:
        prev = _display.set_display(threshold=10, edgeitems=2)
        assert _display._options["threshold"] == 10
        _display.set_display(**prev)
        assert _display._options == original
        _display.set_display(threshold=10)
        _display.set_display(preview=False)
        assert _display._options["threshold"] == 10 and _display._options["preview"] is False
    finally:
        _display._options.update(original)


def test_empty_repr_keeps_shape():
    empty = np.zeros((0, 3))
    text = _display._repr_text("Array", empty, False, empty.shape, empty.dtype)
    assert text == "Array([], shape=(0, 3), dtype=float64)"
    flat = np.zeros(0)
    assert _display._repr_text("Array", flat, False, flat.shape, flat.dtype) == \
        "Array([], dtype=float64)"
