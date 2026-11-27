"""charmtyles.interrupt.defer_interrupt (no server needed)."""

import os
import signal
import threading

import pytest

from charmtyles.interrupt import defer_interrupt


def _sigint():
    os.kill(os.getpid(), signal.SIGINT)


def test_interrupt_is_raised_after_the_block():
    handler = signal.getsignal(signal.SIGINT)
    done = []
    with pytest.raises(KeyboardInterrupt):
        with defer_interrupt():
            _sigint()
            done.append(1)
    assert done == [1]
    assert signal.getsignal(signal.SIGINT) is handler


def test_nested_blocks_hold_until_the_outer_one_ends():
    done = []
    with pytest.raises(KeyboardInterrupt):
        with defer_interrupt():
            with defer_interrupt():
                _sigint()
            done.append("inner")
        done.append("outer")
    assert done == ["inner"]


def test_exception_leaving_the_block_wins():
    handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(ValueError):
        with defer_interrupt():
            _sigint()
            raise ValueError
    assert signal.getsignal(signal.SIGINT) is handler


def test_no_interrupt_no_effect():
    with defer_interrupt():
        pass


def test_other_threads_and_foreign_handlers_are_left_alone(monkeypatch):
    errors = []

    def worker():
        try:
            with defer_interrupt():
                pass
        except Exception as exc:  # signal.signal would raise outside the main thread
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert not errors
    monkeypatch.setattr(signal, "getsignal", lambda sig: None)
    with defer_interrupt():
        pass
