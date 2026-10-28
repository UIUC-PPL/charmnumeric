"""Checks a charmnumeric install without starting a server: python -m charmnumeric.selftest"""

import importlib
import os
import shutil
import subprocess
import sys

_failed = []


def _report(ok, what, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {what}" + (f": {detail}" if detail else ""))
    if not ok:
        _failed.append(what)


def _try_import(name, hint):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        _report(False, f"import {name}", f"{exc}. {hint}")
        return None
    _report(True, f"import {name}", getattr(mod, "__file__", "") or "")
    return mod


def _check_binary(label, path):
    if path is None:
        return False
    ok = os.path.isfile(path) and os.access(path, os.X_OK)
    _report(ok, label, path if ok else f"{path} is missing or not executable")
    return ok


def _check_loads(path):
    # loader trace mode: resolves all libraries and symbol versions, runs no Charm++ code
    env = dict(os.environ, LD_TRACE_LOADED_OBJECTS="1")
    try:
        out = subprocess.run([path], env=env, capture_output=True, text=True, timeout=60)
    except OSError as exc:
        _report(False, "server loads", str(exc))
        return
    text = out.stdout + out.stderr
    bad = sorted({line.strip() for line in text.splitlines() if "not found" in line or "error" in line.lower()})
    if bad and all(line.startswith("libcuda.so") for line in bad):
        # no NVIDIA driver here (e.g. a login node); the GPU nodes provide libcuda
        print(f"[warn] server loads: {'; '.join(bad)} (fine without a GPU driver; run on a GPU node)")
        return
    _report(out.returncode == 0 and not bad, "server loads (libraries and symbol versions)",
            "; ".join(bad) if bad else ("" if out.returncode == 0 else f"exit {out.returncode}"))


def main():
    reinstall = "rerun scripts/install.sh (or pip install the package again)"
    np = _try_import("numpy", "pip install numpy")
    if np is not None:
        print(f"     numpy {np.__version__}")
    pyccs = _try_import("pyccs.server", "PyCCS is installed by scripts/install.sh")
    if pyccs is not None:
        try:
            _report(True, "libccs-client", pyccs._find_library())
        except Exception as exc:
            _report(False, "libccs-client", f"{exc} (scripts/install.sh installs a PyCCS that "
                    "ships it; or set PYCCS_LIBRARY)")
    for mod in ("charmtyles", "charmtyles._frontier", "charmtyles._native_region"):
        _try_import(mod, "install charmtyles: pip install <charmtyles checkout>")
    _try_import("charmnumeric", reinstall)
    _try_import("charmnumeric._native_region", reinstall)

    try:
        from charmnumeric import _build_info as bi
    except ImportError:
        _report(False, "charmnumeric._build_info",
                f"not generated; the server was never built for this install ({reinstall})")
        bi = None
    if bi is not None:
        print(f"     Charm++ {bi.CHARM_ARCH} ({bi.CHARM_HOME}), GPU backend {bi.GPU_BACKEND}, "
              f"Reconverse {bi.RECONVERSE}, LLVM {bi.LLVM_VERSION}, "
              f"charmnumeric {bi.GIT_REVISION}, charmtyles {bi.CORE_GIT_REVISION}")
        if _check_binary("server", bi.SERVER):
            _check_loads(bi.SERVER)
        if bi.CHARMRUN is not None:
            _check_binary("charmrun", bi.CHARMRUN)
        if bi.GPU_BACKEND == "NVIDIA":
            smi = shutil.which("nvidia-smi")
            gpus = subprocess.run([smi, "-L"], capture_output=True, text=True).stdout if smi else ""
            n = gpus.count("GPU ")
            print(f"     {n} NVIDIA GPU(s) visible" + ("" if n else " (fine on a login node; run on a GPU node)"))

    if _failed:
        print(f"\n{len(_failed)} check(s) failed.")
        return 1
    print("\ncharmnumeric install looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
