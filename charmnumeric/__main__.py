"""Command line: run a script with a fresh server, start a standalone server, or show the build.

    python -m charmnumeric run [options] script.py [args...]
    python -m charmnumeric server [options]    # prints host:port; Ctrl-C stops it
    python -m charmnumeric info
"""

import argparse
import os
import runpy
import signal
import sys


def _launch_options(p):
    p.add_argument("--npes", "-p", type=int, help="PEs (default: one per GPU, or min(cores, 4))")
    p.add_argument("--gpus", type=int, help="GPUs to use (GPU builds)")
    p.add_argument("--nodes", "-N", type=int, help="nodes (launcher srun)")
    p.add_argument("--launcher", default="auto", choices=["auto", "local", "srun", "mpiexec"])
    p.add_argument("--port", type=int, help="CCS port (default: a free one)")
    p.add_argument("--odf", type=int, help="over-decomposition factor (default $CNUM_ODF or 4)")
    p.add_argument("--timeout", type=float, help="startup timeout in seconds")
    p.add_argument("--server-arg", action="append", default=[], metavar="ARG",
                   help="extra server argument (repeatable), e.g. --server-arg=+isomalloc_sync")


def _kwargs(a):
    return dict(npes=a.npes, gpus=a.gpus, nodes=a.nodes, launcher=a.launcher,
                extra_args=a.server_arg, timeout=a.timeout, port=a.port, odf=a.odf)


def _sigterm_exits():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))


def _run(a):
    import charmnumeric

    _sigterm_exits()
    charmnumeric.start(**_kwargs(a))
    sys.argv = [a.script] + a.args
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.script)))
    runpy.run_path(a.script, run_name="__main__")


def _server(a):
    from charmnumeric import launcher

    _sigterm_exits()
    kw = _kwargs(a)
    kw.pop("odf")
    srv = launcher._launch(**kw)
    try:
        print(f"{srv.host}:{srv.port}", flush=True)
        print(f"charmnumeric: server with {srv.describe()}; log: {srv.log_path}\n"
              f"  connect with CNUM_HOST={srv.host} CNUM_PORT={srv.port} "
              f"or charmnumeric.connect({srv.host!r}, {srv.port}); Ctrl-C stops it",
              file=sys.stderr, flush=True)
        rc = srv.proc.wait()
        print(f"charmnumeric: server exited with code {rc}", file=sys.stderr)
        return rc
    except KeyboardInterrupt:
        return 0
    finally:
        launcher._shutdown(srv)


def _info(a):
    from charmnumeric import launcher

    try:
        build = launcher.find_build()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"server:   {build.server}\ncharmrun: {build.charmrun}\narch:     {build.arch}\n"
          f"gpu:      {build.gpu}\nfrom:     {build.source}")
    if build.gpu:
        print(f"visible GPUs: {launcher._visible_devices(build.gpu)}")
    print(f"logs:     {launcher._log_root()}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m charmnumeric")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run a script with a freshly started server")
    _launch_options(p)
    p.add_argument("script")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=_run)
    p = sub.add_parser("server", help="start a server for other clients; prints host:port")
    _launch_options(p)
    p.set_defaults(fn=_server)
    p = sub.add_parser("info", help="show the server build that would be used")
    p.set_defaults(fn=_info)
    a = parser.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
