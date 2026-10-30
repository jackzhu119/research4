#!/usr/bin/env python3
"""Launch the user-supplied GaussView viewer in the private Wine64 runtime.

This is a viewer launcher, not a Gaussian quantum-chemistry engine installer.
Software payloads and the Wine prefix are intentionally outside Git tracking.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--headless", action="store_true", help="Use private Xvfb display :99")
    parser.add_argument("--status", action="store_true", help="Report installed paths without launching")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    tools = repo / ".runtime/gaussview-tools"
    root = tools / "root"
    app = repo / ".runtime/gaussian-app/GaussView6"
    prefix = repo / ".runtime/gaussview-wine"
    wine = root / "usr/lib/wine/wine64"
    required = [app / "gview.exe", wine, prefix / "drive_c/windows/system32/kernel32.dll", prefix / "drive_c/windows/system32/zlib1.dll"]
    status = {
        "viewer": "GaussView 6.0.16, Windows x86-64, user-uploaded package",
        "viewer_payload_present": (app / "gview.exe").is_file(),
        "wine64_present": wine.is_file(),
        "private_prefix_ready": all(p.is_file() for p in required[2:]),
        "application_directory": str(app),
        "wine_prefix": str(prefix),
        "gaussian_engine_on_path": shutil.which("g16") or shutil.which("g09"),
        "gaussian_calculation_validated": False,
    }
    if args.status:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error("Private installation is incomplete; see docs/GAUSSIAN_INSTALLATION.md. Missing: " + ", ".join(missing))
    for path in args.files:
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    env = os.environ.copy()
    libraries = [root / "usr/lib/x86_64-linux-gnu", root / "usr/lib/x86_64-linux-gnu/wine/x86_64-unix"]
    dlls = [root / "usr/lib/x86_64-linux-gnu/wine", root / "usr/lib/x86_64-linux-gnu/wine/x86_64-windows"]
    env.update({
        "WINEPREFIX": str(prefix), "WINEARCH": "win64",
        "WINESERVER": str(root / "usr/lib/wine/wineserver64"),
        "WINELOADER": str(wine),
        "WINEDLLPATH": ":".join(map(str, dlls)),
        "WINEDATADIR": str(root / "usr/share/wine"),
        "LD_LIBRARY_PATH": ":".join(map(str, libraries)) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
        "PATH": str(root / "usr/bin") + ":" + env.get("PATH", "/usr/bin:/bin"),
        "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe",
        "WINEDEBUG": "-all,err+all",
        "WINEDLLOVERRIDES": "mscoree,mshtml=;winemenubuilder.exe=d;zlib1=n",
    })
    logs = tools / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    children = {}
    if args.headless:
        env["DISPLAY"] = ":99"
        if not Path("/tmp/.X11-unix/X99").exists():
            with (logs / "launcher-xvfb.log").open("a") as stream:
                process = subprocess.Popen([str(root / "usr/bin/Xvfb"), ":99", "-screen", "0", "1280x800x24", "-nolisten", "tcp", "-noreset"], env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children["xvfb_pid"] = process.pid
            for _ in range(40):
                if Path("/tmp/.X11-unix/X99").exists():
                    break
                if process.poll() is not None:
                    raise RuntimeError("Xvfb failed; inspect launcher-xvfb.log")
                time.sleep(0.1)
            else:
                raise RuntimeError("Xvfb display did not become ready")
        cp = subprocess.run(["xprop", "-root", "_NET_SUPPORTING_WM_CHECK"], env=env, capture_output=True, text=True, timeout=3)
        has_window_manager = "window id #" in cp.stdout
        if not has_window_manager and shutil.which("xfwm4"):
            with (logs / "launcher-xfwm4.log").open("a") as stream:
                process = subprocess.Popen([shutil.which("xfwm4"), "--compositor=off", "--replace"], env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children["window_manager_pid"] = process.pid
    elif not env.get("DISPLAY"):
        parser.error("No graphical display is selected. Use --headless for virtual display :99, or launch from an existing graphical desktop.")
    winfiles = ["Z:" + str(p.resolve()).replace("/", "\\") for p in args.files]
    with (logs / "gaussview-launcher.log").open("a") as stream:
        process = subprocess.Popen([str(wine), str(app / "gview.exe"), *winfiles], cwd=app, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": process.pid, "display": env["DISPLAY"], "files": [str(p.resolve()) for p in args.files], "children_started": children, "gaussian_engine_installed_in_this_task": False}
    (tools / "last-launch.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("Viewer launched. Missing-Gaussian warnings are expected; no quantum-chemistry engine has been installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
