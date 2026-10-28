"""Locate the backend binary and infer how it was built."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
DEVICE_VAR = {"NVIDIA": "CUDA_VISIBLE_DEVICES", "AMD": "ROCR_VISIBLE_DEVICES",
              "INTEL": "ZE_AFFINITY_MASK"}


def env_int(name, default=None):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None


class Build:
    """A server binary and how to launch it."""

    def __init__(self, server, charmrun, arch, gpu, source):
        self.server = Path(server)
        self.charmrun = Path(charmrun) if charmrun else None
        self.arch = arch
        self.gpu = gpu
        self.source = source

    @property
    def reconverse(self):
        return self.arch.startswith("reconverse")

    def __repr__(self):
        return (f"Build(server={str(self.server)!r}, charmrun={self.charmrun and str(self.charmrun)!r}, "
                f"arch={self.arch!r}, gpu={self.gpu!r}, source={self.source!r})")


def _build_info():
    try:
        from . import _build_info as info
    except ImportError:
        return None
    return info


def _infer_arch(build_dir):
    for name in ("build.ninja", "CMakeFiles/server.dir/build.make"):
        try:
            text = (build_dir / name).read_text(errors="replace")
        except OSError:
            continue
        m = re.search(r"([^\s\"';]+)/bin/charmc\b", text)
        if m:
            return Path(m.group(1)).name
    return None


def needed_libs(path):
    """DT_NEEDED entries of an ELF64 little-endian binary ([] if unreadable)."""
    try:
        with open(path, "rb") as f:
            hdr = f.read(64)
            if hdr[:4] != b"\x7fELF" or hdr[4] != 2 or hdr[5] != 1:
                return []
            phoff, = struct.unpack_from("<Q", hdr, 0x20)
            phentsize, phnum = struct.unpack_from("<HH", hdr, 0x36)
            f.seek(phoff)
            phdrs = [struct.unpack_from("<IIQQQQ", f.read(phentsize)) for _ in range(phnum)]
            loads = [(v, o, n) for t, _, o, v, _, n in phdrs if t == 1]
            dyn = next(((o, n) for t, _, o, _, _, n in phdrs if t == 2), None)
            if dyn is None:
                return []
            f.seek(dyn[0])
            raw = f.read(dyn[1])
            tags = [struct.unpack_from("<qQ", raw, i) for i in range(0, len(raw) - 15, 16)]
            strtab = next((v for t, v in tags if t == 5), None)
            off = next((o + strtab - v for v, o, n in loads if v <= strtab < v + n), None)
            names = []
            for t, v in tags:
                if t == 1 and off is not None:
                    f.seek(off + v)
                    names.append(f.read(256).split(b"\0", 1)[0].decode(errors="replace"))
            return names
    except (OSError, struct.error, TypeError):
        return []


def _infer_gpu(server, arch):
    try:
        m = re.search(r"^GPU_BACKEND:\w+=(\w*)", (server.parent / "CMakeCache.txt").read_text(), re.M)
    except OSError:
        m = None
    if m:
        return m.group(1).upper() or None
    libs = " ".join(needed_libs(server))
    for lib, gpu in (("libcudart", "NVIDIA"), ("libamdhip", "AMD"), ("libze_loader", "INTEL")):
        if lib in libs:
            return gpu
    return "NVIDIA" if arch and "cuda" in arch else None


def visible_devices(gpu):
    val = os.environ.get(DEVICE_VAR.get(gpu, "CUDA_VISIBLE_DEVICES"))
    if val is not None:
        return [d.strip() for d in val.split(",") if d.strip() and d.strip() != "-1"]
    if gpu == "NVIDIA" and shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return []
        if out.returncode == 0:
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return []


def find_build():
    """server.out from $CNUM_SERVER, then charmnumeric._build_info, then src/build-*."""
    info = _build_info()
    env_server = os.environ.get("CNUM_SERVER")
    if env_server:
        server = Path(env_server).expanduser().resolve()
        if not server.is_file():
            raise FileNotFoundError(f"CNUM_SERVER={env_server} does not exist")
        source = "CNUM_SERVER"
    elif info is not None and getattr(info, "SERVER", None):
        server = Path(info.SERVER)
        if not server.is_file():
            raise FileNotFoundError(f"charmnumeric._build_info.SERVER={info.SERVER} does not exist; "
                                    "rebuild charmnumeric or set CNUM_SERVER")
        source = "_build_info"
    else:
        names = ["build-nc", "build", "build-gpu"]
        if visible_devices("NVIDIA"):
            names.insert(0, names.pop())
        tried = [SRC_DIR / name / "server.out" for name in names]
        server = next((p for p in tried if p.is_file()), None)
        if server is None:
            raise FileNotFoundError(
                "charmnumeric backend (server.out) not found. Build it or set "
                "CNUM_SERVER=/path/to/server.out, or connect to a running server with "
                "CNUM_HOST/CNUM_PORT or charmnumeric.connect(). Looked in:\n  "
                + "\n  ".join(map(str, tried)))
        source = str(server.parent)

    if info is not None and str(server) == str(getattr(info, "SERVER", "")):
        arch, gpu, charmrun = info.CHARM_ARCH or "", info.GPU_BACKEND, info.CHARMRUN
    else:
        arch = _infer_arch(server.parent) or ""
        gpu = _infer_gpu(server, arch)
        sibling = server.parent / "charmrun"
        charmrun = sibling if sibling.is_file() and not arch.startswith("reconverse") else None
        if not arch:
            # A copied binary: Reconverse builds have no charmrun.
            arch = "unknown" if charmrun is not None else "reconverse"
    env_charmrun = os.environ.get("CNUM_CHARMRUN")
    if env_charmrun:
        charmrun = None if env_charmrun.lower() == "none" else env_charmrun
    return Build(server, charmrun, arch, gpu, source)
