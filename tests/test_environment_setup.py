"""Exercise runtime setup contracts without downloading or installing packages."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_script("setup_research_environment")
doctor = load_script("environment_doctor")


def runtime_spec():
    """Use the committed pins, but never inspect ambient package caches."""
    spec = json.loads((REPO / "environment/runtime-spec.json").read_text())
    for key in ("octave_reuse_candidates", "python_extra_reuse_candidates", "apt_cache_reuse_candidates"):
        spec[key] = []
    return spec


def python_probe(spec):
    return {
        "available": True,
        "version": "3.12.0",
        "packages": {**spec["python_packages"], **spec["torch_packages"]},
        "imports": {name: True for name in [*spec["python_imports"], "torch"]},
        "torch_cpu": True,
        "libarchive": True,
    }


def octave_probe(spec):
    return {
        "available": True,
        "version": spec["octave_version"],
        "packages": dict(spec["octave_packages"]),
        "loaded": {name: True for name in spec["octave_packages"]},
        "apis": {"pca": True, "trainNetwork": False, "trainlssvm": False},
    }


class RuntimeSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / "runtime space ' $literal `tick`"
        (self.runtime / "bin").mkdir(parents=True)
        self.spec = runtime_spec()
        self.inherited = dict(os.environ)
        self.inherited.update({
            "HOME": str(self.root / "inherited home"),
            "HTTP_PROXY": "http://proxy.invalid:8080/private",
            "HTTPS_PROXY": "https://secure.invalid:8443/private",
            "ALL_PROXY": "socks5://socks.invalid:1080",
            "NO_PROXY": "localhost,127.0.0.1,example.invalid",
            "http_proxy": "http://lowercase.invalid:8080",
            "https_proxy": "https://lowercase.invalid:8443",
            "no_proxy": "example.invalid",
            "LD_LIBRARY_PATH": "/inherited libraries:/other",
        })
        self.arguments = ["one argument", "apostrophe ' and double \"", "$HOME;`echo unsafe`", ""]

    def make_recorder(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, sys\n"
            "print(json.dumps({'arguments': sys.argv[1:], 'environment': dict(os.environ)}))\n"
        )
        path.chmod(0o755)

    def assert_inherited_environment(self, observed):
        for key in ("HOME", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                    "http_proxy", "https_proxy", "no_proxy"):
            self.assertEqual(observed[key], self.inherited[key], key)

    def test_private_octave_wrapper_preserves_environment_and_literal_arguments(self):
        prefix = self.runtime / "private octave ' $prefix `tick`"
        self.make_recorder(prefix / "usr/bin/octave-cli")
        wrapper = setup.octave_wrapper(prefix, self.runtime)
        cp = subprocess.run([str(wrapper), *self.arguments], env=self.inherited,
                            text=True, capture_output=True, check=True)
        record = json.loads(cp.stdout)
        self.assertEqual(record["arguments"], ["--no-gui", "--quiet", *self.arguments])
        self.assert_inherited_environment(record["environment"])
        self.assertTrue(record["environment"]["LD_LIBRARY_PATH"].endswith(
            ":" + self.inherited["LD_LIBRARY_PATH"]))
        self.assertEqual(record["environment"]["OCTAVE_HOME"], str(prefix / "usr"))
        self.assertEqual(record["environment"]["OPENBLAS_NUM_THREADS"], "1")

    def test_reused_octave_wrapper_quotes_external_path_and_arguments(self):
        external = self.root / "existing octave ' $executable `tick`"
        self.make_recorder(external)
        with mock.patch.object(setup.shutil, "which", return_value=None), \
                mock.patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout=self.spec["octave_version"] + "\n")), \
                mock.patch.object(setup, "validate_platform") as validate:
            result = setup.setup_octave(self.runtime, self.spec, existing=external, offline=True)
        validate.assert_not_called()
        self.assertEqual(result["mode"], "reused_external")
        cp = subprocess.run([result["executable"], *self.arguments], env=self.inherited,
                            text=True, capture_output=True, check=True)
        record = json.loads(cp.stdout)
        self.assertEqual(record["arguments"], self.arguments)
        self.assert_inherited_environment(record["environment"])

    def test_python_wrapper_preserves_environment_and_literal_arguments(self):
        executable = self.runtime / "python/bin/python"
        self.make_recorder(executable)
        site = self.runtime / "python/lib/site-packages"
        site.mkdir(parents=True)
        with mock.patch.object(setup.subprocess, "check_output", return_value=str(site) + "\n"), \
                mock.patch.object(setup, "version_probe", return_value=self.spec["python_packages"]), \
                mock.patch.object(setup, "run") as install:
            result = setup.setup_python(self.runtime, REPO, self.spec, offline=True, no_torch=True)
        install.assert_not_called()
        cp = subprocess.run([result["executable"], *self.arguments], env=self.inherited,
                            text=True, capture_output=True, check=True)
        record = json.loads(cp.stdout)
        self.assertEqual(record["arguments"], self.arguments)
        self.assert_inherited_environment(record["environment"])
        self.assertEqual((site / "research4-runtime.pth").read_text(), str(REPO / "src") + "\n")

    def test_offline_python_refuses_missing_pins_before_installing(self):
        self.make_recorder(self.runtime / "python/bin/python")
        site = self.runtime / "python/lib/site-packages"
        site.mkdir(parents=True)
        with mock.patch.object(setup.subprocess, "check_output", return_value=str(site) + "\n"), \
                mock.patch.object(setup, "version_probe", return_value={}), \
                mock.patch.object(setup, "run") as install:
            with self.assertRaisesRegex(RuntimeError, "Offline setup is missing pinned Python packages"):
                setup.setup_python(self.runtime, REPO, self.spec, offline=True)
        install.assert_not_called()
        self.assertFalse((self.runtime / "bin/python").exists())

    def test_python_install_manifest_includes_torch_from_separate_cpu_index(self):
        self.make_recorder(self.runtime / "python/bin/python")
        site = self.runtime / "python/lib/site-packages"
        site.mkdir(parents=True)
        observed = {**self.spec["python_packages"], **self.spec["torch_packages"]}
        observed["torch"] = None
        observed["numpy"] = None
        with mock.patch.object(setup.subprocess, "check_output", return_value=str(site) + "\n"), \
                mock.patch.object(setup, "version_probe", return_value=observed), \
                mock.patch.object(setup, "run") as install:
            result = setup.setup_python(self.runtime, REPO, self.spec)
        self.assertEqual(set(result["installed_packages"]), {"torch", "numpy"})
        self.assertEqual(install.call_count, 2)
        torch_command = install.call_args_list[0].args[0]
        self.assertIn("torch==" + self.spec["torch_packages"]["torch"], torch_command)
        self.assertIn(self.spec["torch_index"], torch_command)
        self.assertIn("--no-deps", torch_command)
        core_command = install.call_args_list[1].args[0]
        self.assertIn("numpy==" + self.spec["python_packages"]["numpy"], core_command)
        self.assertIn(self.spec["python_index"], core_command)

    def test_offline_octave_refuses_uncached_packages_without_network(self):
        with mock.patch.object(setup, "validate_platform"), \
                mock.patch.object(setup.shutil, "which", return_value=None), \
                mock.patch.object(setup.subprocess, "run") as command, \
                mock.patch.object(setup, "run") as install:
            with self.assertRaisesRegex(RuntimeError, "offline setup cannot download"):
                setup.setup_octave(self.runtime, self.spec, offline=True)
        command.assert_not_called()
        install.assert_not_called()
        self.assertFalse((self.runtime / "apt/sources.list").exists())

    def test_unsupported_distribution_stops_before_installation(self):
        with mock.patch.object(setup.platform, "system", return_value="Linux"), \
                mock.patch.object(setup.platform, "machine", return_value="x86_64"), \
                mock.patch.object(setup.Path, "read_text", return_value='ID=ubuntu\nVERSION_CODENAME=noble\n'), \
                mock.patch.object(setup.shutil, "which", return_value=None), \
                mock.patch.object(setup.subprocess, "run") as command, \
                mock.patch.object(setup, "run") as install:
            with self.assertRaisesRegex(RuntimeError, "supports Debian 13/trixie x86_64 only"):
                setup.setup_octave(self.runtime, self.spec)
        command.assert_not_called()
        install.assert_not_called()
        self.assertFalse((self.runtime / "apt").exists())

    def test_package_download_commands_preserve_home_and_proxy_environment(self):
        with mock.patch.dict(os.environ, self.inherited, clear=True), \
                mock.patch.object(setup, "validate_platform"), \
                mock.patch.object(setup.shutil, "which", return_value=None), \
                mock.patch.object(setup, "octave_wrapper", return_value=self.runtime / "bin/octave"), \
                mock.patch.object(setup, "run") as command:
            setup.setup_octave(self.runtime, self.spec)
        self.assertEqual(len(command.call_args_list), 3)
        for invocation in command.call_args_list[:2]:
            environment = invocation.kwargs["env"]
            self.assert_inherited_environment(environment)
            self.assertEqual(environment["APT_CONFIG"], str(self.runtime / "apt/apt.conf"))
            self.assertIsInstance(invocation.args[0], list)
        packages = command.call_args_list[1].args[0]
        self.assertIn("--download-only", packages)
        self.assertIn("--no-install-recommends", packages)
        for name, version in self.spec["apt_packages"].items():
            self.assertIn(name + "=" + version, packages)


class EnvironmentDoctorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Path(self.temp.name) / "runtime"
        self.spec = runtime_spec()

    def inspect(self, py=None, oct=None, matlab=None, probe_matlab=False, matlab_probe=None):
        candidates = {"python": "/runtime/python", "octave": "/runtime/octave", "matlab": matlab}
        results = [copy.deepcopy(py if py is not None else python_probe(self.spec)),
                   copy.deepcopy(oct if oct is not None else octave_probe(self.spec))]
        if matlab and probe_matlab:
            results.append(copy.deepcopy(matlab_probe))
        with mock.patch.object(doctor, "_candidate", side_effect=lambda _, name, *args: candidates[name]), \
                mock.patch.object(doctor, "_run_json", side_effect=results) as run_probe:
            result = doctor.inspect_environment(self.runtime, self.spec, probe_matlab=probe_matlab)
        return result, run_probe

    def test_ready_open_source_runtime_never_claims_missing_matlab_readiness(self):
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "https://private:credential@proxy.invalid"}):
            result, probes = self.inspect()
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["runtime_dir"], str(self.runtime.resolve()))
        self.assertTrue(result["checked_at_shanghai"].endswith("+08:00"))
        self.assertTrue(result["open_source_ready"])
        self.assertTrue(result["python"]["locked_versions_match"])
        self.assertTrue(result["octave"]["ready"])
        self.assertFalse(result["matlab"]["executable_found"])
        self.assertFalse(result["matlab"]["native_execution_verified"])
        self.assertFalse(result["matlab"]["installed_by_this_bootstrap"])
        self.assertFalse(result["all_original_code_ready"])
        self.assertEqual(probes.call_count, 2)
        self.assertNotIn("private:credential", json.dumps(result))

    def test_unprobed_matlab_executable_does_not_verify_native_execution(self):
        result, probes = self.inspect(matlab="/licensed/matlab")
        self.assertTrue(result["matlab"]["executable_found"])
        self.assertFalse(result["matlab"]["native_execution_verified"])
        self.assertEqual(result["matlab"]["license_status"], "not_configured_or_not_verified")
        self.assertFalse(result["all_original_code_ready"])
        self.assertEqual(probes.call_count, 2)

    def test_failed_explicit_matlab_probe_is_reported_without_native_readiness(self):
        result, probes = self.inspect(matlab="/licensed/matlab", probe_matlab=True,
                                      matlab_probe={"available": False, "error": "probe_failed", "returncode": 1})
        self.assertFalse(result["matlab"]["native_execution_verified"])
        self.assertEqual(result["matlab"]["license_status"], "probe_failed")
        self.assertEqual(probes.call_args.args[0][0:2], ["/licensed/matlab", "-batch"])

    def test_missing_or_wrong_version_core_and_octave_packages_prevent_readiness(self):
        py = python_probe(self.spec)
        core_name = next(iter(self.spec["python_packages"]))
        py["packages"][core_name] = None
        oct = octave_probe(self.spec)
        octave_name = next(iter(self.spec["octave_packages"]))
        oct["packages"][octave_name] = "0.0.0"
        result, _ = self.inspect(py=py, oct=oct)
        self.assertFalse(result["python"]["locked_versions_match"])
        self.assertFalse(result["python"]["core_ready"])
        self.assertFalse(result["octave"]["ready"])
        self.assertFalse(result["open_source_ready"])

    def test_import_or_libarchive_failures_prevent_readiness(self):
        for failure in ("core_import", "torch_import", "libarchive"):
            with self.subTest(failure=failure):
                py = python_probe(self.spec)
                if failure == "libarchive":
                    py["libarchive"] = False
                else:
                    name = next(iter(self.spec["python_packages"])) if failure == "core_import" else "torch"
                    py["imports"][name] = "ImportError"
                result, _ = self.inspect(py=py)
                self.assertFalse(result["open_source_ready"])

    def test_missing_or_wrong_torch_dependency_prevents_pinned_runtime_readiness(self):
        dependency = next(name for name in self.spec["torch_packages"] if name != "torch")
        for version in (None, "0.0.0"):
            with self.subTest(dependency=dependency, observed=version):
                py = python_probe(self.spec)
                py["packages"][dependency] = version
                result, _ = self.inspect(py=py)
                self.assertTrue(result["python"]["core_ready"])
                self.assertFalse(result["python"]["locked_versions_match"])
                self.assertFalse(result["open_source_ready"])

    def test_probe_failures_produce_structured_unready_result(self):
        failed = {"available": False, "error": "probe_failed", "returncode": 1}
        result, _ = self.inspect(py=failed, oct=failed)
        self.assertFalse(result["open_source_ready"])
        self.assertFalse(result["python"]["core_ready"])
        self.assertFalse(result["python"]["cnn_ready"])
        self.assertFalse(result["octave"]["ready"])
        self.assertEqual(result["python"]["error"], "probe_failed")

    def test_probe_parser_uses_last_marker_and_reports_failures(self):
        good = subprocess.CompletedProcess([], 0, stdout='startup noise\nDOCTOR_JSON={"available":false}\n'
                                            'DOCTOR_JSON={"available":true,"version":"9.4.0"}\n')
        with mock.patch.object(doctor.subprocess, "run", return_value=good):
            self.assertEqual(doctor._run_json(["octave"], "DOCTOR_JSON="),
                             {"available": True, "version": "9.4.0"})
        cases = [
            (subprocess.CompletedProcess([], 1, stdout="private license diagnostic"), "probe_failed"),
            (subprocess.CompletedProcess([], 0, stdout="no marker"), "probe_failed"),
            (subprocess.CompletedProcess([], 0, stdout="DOCTOR_JSON=not-json\n"), "JSONDecodeError"),
            (FileNotFoundError("not installed"), "FileNotFoundError"),
            (subprocess.TimeoutExpired(["matlab"], 1), "TimeoutExpired"),
        ]
        for response, error in cases:
            with self.subTest(error=error, response=repr(response)):
                options = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
                with mock.patch.object(doctor.subprocess, "run", **options):
                    result = doctor._run_json(["matlab"], "DOCTOR_JSON=")
                self.assertFalse(result["available"])
                self.assertEqual(result["error"], error)
                self.assertNotIn("private license diagnostic", json.dumps(result))

    def test_cli_json_matches_output_file_and_exit_status(self):
        spec_file = Path(self.temp.name) / "spec.json"
        spec_file.write_text(json.dumps(self.spec))
        output = Path(self.temp.name) / "reports/doctor.json"
        for ready, status in ((True, 0), (False, 2)):
            with self.subTest(ready=ready):
                result = {"schema_version": 1, "open_source_ready": ready,
                          "matlab": {"native_execution_verified": False}}
                stdout = io.StringIO()
                with mock.patch.object(sys, "argv", ["environment_doctor.py", "--runtime-dir", str(self.runtime),
                                                       "--spec", str(spec_file), "--output", str(output)]), \
                        mock.patch.object(doctor, "inspect_environment", return_value=result), \
                        contextlib.redirect_stdout(stdout):
                    self.assertEqual(doctor.main(), status)
                self.assertEqual(json.loads(stdout.getvalue()), result)
                self.assertEqual(json.loads(output.read_text()), result)


if __name__ == "__main__":
    unittest.main()
