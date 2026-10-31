"""Build/reuse an unprivileged, pinned Octave/Python cloud environment.

No sudo, system package installation, proxy replacement, HOME override or
MATLAB license installation is performed. Runtime assets live under .runtime/.
"""
import argparse
from datetime import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo
from xml.sax.saxutils import escape


def run(command, log, *, env=None, timeout=900):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        cp = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    if cp.returncode:
        raise RuntimeError(f"Command failed ({cp.returncode}); inspect {log}")


def version_probe(executable, names):
    code = "import importlib.metadata as m,json;d={};\nfor n in " + repr(list(names)) + ":\n try:d[n]=m.version(n)\n except m.PackageNotFoundError:d[n]=None\nprint(json.dumps(d))"
    cp = subprocess.run([str(executable), "-c", code], capture_output=True, text=True, timeout=45)
    if cp.returncode:
        return {}
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return {}


def validate_platform(spec):
    release = {}
    p = Path("/etc/os-release")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1); release[k] = v.strip('"')
    expected = spec["platform"]
    if (platform.system(), platform.machine(), release.get("ID"), release.get("VERSION_CODENAME")) != (
            expected["system"], expected["machine"], expected["distribution"], expected["codename"]):
        raise RuntimeError("Private Debian package extraction supports Debian 13/trixie x86_64 only; provide an existing compatible Octave on other systems")


def octave_wrapper(prefix, runtime):
    usr = prefix / "usr"
    bin_dir = runtime / "octave-bin"; bin_dir.mkdir(exist_ok=True)
    gnuplot = bin_dir / "gnuplot"
    if not gnuplot.exists():
        gnuplot.symlink_to(usr / "bin/gnuplot-nox")
    fonts = runtime / "fonts.conf"
    fonts.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><dir>' + escape(str(usr / "share/fonts")) + '</dir><dir>/usr/share/fonts</dir><cachedir>' + escape(str(runtime / "font-cache")) + '</cachedir></fontconfig>')
    libraries = [usr / "lib", usr / "lib/x86_64-linux-gnu", usr / "lib/x86_64-linux-gnu/octave/9.4.0"]
    target = runtime / "bin/octave"
    target.write_text("#!/bin/sh\n" +
        "export OCTAVE_HOME=" + shlex.quote(str(usr)) + "\n" +
        "export LD_LIBRARY_PATH=" + shlex.quote(":".join(map(str, libraries))) + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' +
        "export PATH=" + shlex.quote(str(bin_dir) + ":" + str(usr / "bin")) + ':"$PATH"\n' +
        "export FONTCONFIG_FILE=" + shlex.quote(str(fonts)) + "\n" +
        "export OCTAVE_FONTS_DIR=" + shlex.quote(str(usr / "share/fonts/opentype/freefont")) + "\n" +
        "export OPENBLAS_NUM_THREADS=1\nexport OMP_NUM_THREADS=1\nexport BROWSER=/bin/false\n" +
        "exec " + shlex.quote(str(usr / "bin/octave-cli")) + ' --no-gui --quiet "$@"\n')
    target.chmod(0o755)
    return target


def setup_octave(runtime, spec, *, existing=None, offline=False, rebuild=False):
    target = runtime / "bin/octave"
    if target.exists() and not rebuild:
        cp = subprocess.run([str(target), "--eval", "disp(version);"], capture_output=True, text=True, timeout=30)
        if cp.returncode == 0 and spec["octave_version"] in cp.stdout:
            return {"mode": "existing_local", "executable": str(target)}
    candidates = ([str(existing)] if existing else []) + spec.get("octave_reuse_candidates", [])
    system = shutil.which("octave-cli") or shutil.which("octave")
    if system:
        candidates.append(system)
    if not rebuild:
        for candidate in candidates:
            p = Path(candidate).resolve()
            if not p.is_file() or p == target:
                continue
            cp = subprocess.run([str(p), "--eval", "disp(version);"], capture_output=True, text=True, timeout=30)
            if cp.returncode == 0 and spec["octave_version"] in cp.stdout:
                target.write_text("#!/bin/sh\nexec " + shlex.quote(str(p)) + ' "$@"\n'); target.chmod(0o755)
                return {"mode": "reused_external", "executable": str(target), "reused_from": str(p)}
    validate_platform(spec)
    apt = runtime / "apt"; archives = apt / "archives"
    for folder in [archives / "partial", apt / "lists/partial"]:
        folder.mkdir(parents=True, exist_ok=True)
    for candidate in spec.get("apt_cache_reuse_candidates", []):
        if Path(candidate).is_dir():
            for package in Path(candidate).glob("*.deb"):
                destination = archives / package.name
                if not destination.exists(): shutil.copyfile(package, destination)
    package_versions = {}
    for package in archives.glob("*.deb"):
        cp = subprocess.run(["dpkg-deb", "--field", str(package), "Package", "Version"], capture_output=True, text=True, check=True)
        fields = dict(line.split(": ", 1) for line in cp.stdout.splitlines() if ": " in line)
        package_versions[fields.get("Package")] = fields.get("Version")
    cached = all(package_versions.get(k) == v for k, v in spec["apt_packages"].items())
    if not offline or not cached:
        if offline:
            raise RuntimeError("Pinned Octave packages are not cached; offline setup cannot download them")
        sources = apt / "sources.list"
        sources.write_text("deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://deb.debian.org/debian trixie main\n")
        config = apt / "apt.conf"
        config.write_text('Dir::Etc::parts "-";\nDir::Etc::main "-";\nDir::Etc::sourcelist "' + str(sources) + '";\nDir::Etc::sourceparts "-";\nDir::State::lists "' + str(apt / "lists") + '";\nDir::Cache::archives "' + str(archives) + '";\nAPT::Sandbox::User "' + getpass.getuser() + '";\nDebug::NoLocking "true";\nAcquire::Retries "2";\n')
        env = dict(os.environ, APT_CONFIG=str(config))
        run(["/usr/bin/apt-get", "update"], runtime / "logs/apt-update.log", env=env)
        packages = [f"{k}={v}" for k, v in spec["apt_packages"].items()]
        run(["/usr/bin/apt-get", "--no-install-recommends", "--download-only", "-y", "install", *packages], runtime / "logs/apt-download.log", env=env)
    prefix = runtime / "octave-root"; prefix.mkdir(exist_ok=True)
    artifacts = []
    for package in sorted(archives.glob("*.deb")):
        subprocess.run(["dpkg-deb", "-x", str(package), str(prefix)], check=True)
        artifacts.append({"file": package.name, "sha256": hashlib.sha256(package.read_bytes()).hexdigest()})
    executable = octave_wrapper(prefix, runtime)
    run([str(executable), "--eval", "pkg rebuild -global; pkg list;"], runtime / "logs/octave-register.log")
    return {"mode": "private_debian_extract", "executable": str(executable), "prefix": str(prefix), "packages": artifacts}


def setup_python(runtime, repo, spec, *, offline=False, no_torch=False):
    venv = runtime / "python"
    executable = venv / "bin/python"
    if not executable.exists():
        run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], runtime / "logs/python-venv.log")
    site = subprocess.check_output([str(executable), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], text=True).strip()
    extra = None
    for candidate in spec.get("python_extra_reuse_candidates", []):
        p = Path(candidate).resolve()
        if (p / "torch/__init__.py").is_file():
            extra = str(p); break
    pth = Path(site) / "research4-runtime.pth"
    pth.write_text(str(repo / "src") + "\n" + (extra + "\n" if extra else ""))
    required = dict(spec["python_packages"])
    if not no_torch:
        required.update(spec["torch_packages"])
    observed = version_probe(executable, required)
    missing = {k: v for k, v in required.items() if observed.get(k) != v}
    installed = sorted(missing)
    if missing and offline:
        raise RuntimeError("Offline setup is missing pinned Python packages: " + ", ".join(sorted(missing)))
    if "torch" in missing:
        run([str(executable), "-m", "pip", "install", "--quiet", "--no-deps", "torch==" + missing.pop("torch"), "--index-url", spec["torch_index"]], runtime / "logs/python-torch.log")
    if missing:
        run([str(executable), "-m", "pip", "install", "--quiet", "--index-url", spec["python_index"], *[f"{k}=={v}" for k, v in missing.items()]], runtime / "logs/python-packages.log")
    wrapper = runtime / "bin/python"
    wrapper.write_text("#!/bin/sh\nexport OPENBLAS_NUM_THREADS=1\nexport OMP_NUM_THREADS=1\nexec " + shlex.quote(str(executable)) + ' "$@"\n')
    wrapper.chmod(0o755)
    return {"mode": "venv_with_reviewed_system_packages", "executable": str(wrapper), "venv": str(venv),
            "reused_extra_directory": extra, "locked_packages": required, "installed_packages": installed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--runtime-dir", type=Path, default=repo / ".runtime")
    parser.add_argument("--spec", type=Path, default=repo / "environment/runtime-spec.json")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--existing-octave", type=Path)
    parser.add_argument("--rebuild-octave", action="store_true")
    parser.add_argument("--no-torch", action="store_true", help="Install core-only; doctor will report CNN readiness separately")
    args = parser.parse_args();runtime = args.runtime_dir.resolve();spec = json.loads(args.spec.read_text())
    if args.check_only:
        return subprocess.run([sys.executable, str(repo / "scripts/environment_doctor.py"), "--runtime-dir", str(runtime), "--spec", str(args.spec)]).returncode
    runtime.mkdir(parents=True, exist_ok=True);(runtime / "bin").mkdir(exist_ok=True)
    (runtime / ".gitignore").write_text("*\n")
    result = {"schema_version": 1, "created_at_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "runtime_dir": str(runtime), "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
              "offline": args.offline, "matlab": {"installed": False, "reason": "Requires a valid user/institution license; not installed by this bootstrap"}}
    result["octave"] = setup_octave(runtime, spec, existing=args.existing_octave, offline=args.offline, rebuild=args.rebuild_octave)
    result["python"] = setup_python(runtime, repo, spec, offline=args.offline, no_torch=args.no_torch)
    (runtime / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    doctor = runtime / "doctor.json"
    cp = subprocess.run([sys.executable, str(repo / "scripts/environment_doctor.py"), "--runtime-dir", str(runtime), "--spec", str(args.spec), "--output", str(doctor)])
    print("Runtime manifest: " + str(runtime / "manifest.json"))
    if args.no_torch:
        record = json.loads(doctor.read_text())
        return 0 if record["python"]["core_ready"] and record["octave"]["ready"] else 2
    return cp.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print("Environment setup failed: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
