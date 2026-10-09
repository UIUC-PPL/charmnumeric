"""Connection helper shared by the cuPyNumeric -> charmnumeric ports.

Environment variables:
    CNUM_HOST        backend host            (default 127.0.0.1)
    CNUM_PORT        backend CCS port        (default 1234)
    CNUM_ODF         over-decomposition fac. (default 4)
    CNUM_AUTO_FLUSH  if set, call charmtyles set_auto_flush(interface, N)
    CNUM_SIM         if set to a non-empty value other than "0", do not
                     connect; return the host-side reference simulator from
                     _cnum_sim.py instead (static/semantic dry run).
"""

import os

import charmnumeric as cnp

_SIM = os.environ.get("CNUM_SIM", "") not in ("", "0")
if _SIM:
    import _cnum_sim  # noqa: F401  (installs the simulator hooks before any array exists)


def get_interface():
    if _SIM:
        from _cnum_sim import SimInterface

        return SimInterface()

    host = os.environ.get("CNUM_HOST", "127.0.0.1")
    port = int(os.environ.get("CNUM_PORT", "1234"))
    odf = int(os.environ.get("CNUM_ODF", "4"))

    interface = cnp.CharmNumericInterface()
    interface.connect(host, port, odf)

    flush = os.environ.get("CNUM_AUTO_FLUSH")
    if flush:
        from charmtyles.core import set_auto_flush

        set_auto_flush(interface, int(flush))
    return interface
