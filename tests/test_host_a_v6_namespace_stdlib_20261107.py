"""A006 metadata guards: temporary files/mocked processes only, no Git or Q."""
from __future__ import annotations

import ast
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
UPLOADER=ROOT/"scripts/research4_host_a_git_return_20261107_v6.py"
WRAPPER=ROOT/"scripts/run_research4_host_a_20261107_v6.py"
WORK_RELATIVE="reports/fixed-quadrature-convergence-20261107-v2/queue-host-a-006"
PROTOCOL_RELATIVE="reports/fixed-quadrature-convergence-20261107-v2/resource-protocol"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


U=load("host_a_v6_uploader_test",UPLOADER)
W=load("host_a_v6_wrapper_test",WRAPPER)


def manifest():
    return {"schema":"research4-root-host-a-result-return-v1","role":"A","run_id":"host-a-006",
        "work_directory":WORK_RELATIVE,"ledger":U.A_LEDGER_RELATIVE,
        "upload":{"enabled":True,"branch":U.BRANCH_A,"allowlist":["*.json","*.log"]}}


class A006Namespaces(unittest.TestCase):
    def test_only_exact_A006_manifest_and_research_branch(self):
        original=manifest()
        self.assertEqual(U.expected_branch(original),U.BRANCH_A)
        for changes in ({"run_id":"host-a-005"},{"run_id":"host-a-007"},{"role":"B"},
                        {"schema":"research4-remote-b-manifest-v1"},
                        {"upload":{"enabled":True,"branch":"main"}},
                        {"upload":{"enabled":True,"branch":"research/host-b"}},
                        {"upload":{"enabled":False,"branch":U.BRANCH_A}}):
            with self.subTest(changes=changes),self.assertRaises(U.SafetyError):
                U.expected_branch({**original,**changes})
        self.assertNotEqual(U.BRANCH_A,"main")
        self.assertFalse(hasattr(U,"claim_result_branch"))

    def test_fixed_A006_work_prefix_rejects_A005_B_absolute_and_parent_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertEqual(U.work_path(root,manifest()),root/WORK_RELATIVE)
            for value in ("reports/fixed-quadrature-20261107-v1/queue-host-a-005",
                          "reports/fixed-quadrature-convergence-20261107-v2/queue-host-a-005",
                          "reports/fixed-quadrature-convergence-20261107-v2/queue-host-b-006",
                          "/reports/fixed-quadrature-convergence-20261107-v2/queue-host-a-006",
                          "reports/fixed-quadrature-convergence-20261107-v2/../queue-host-a-006"):
                with self.subTest(value=value),self.assertRaises(U.SafetyError):
                    U.work_path(root,{**manifest(),"work_directory":value})

    def test_symlink_namespace_and_checkpoint_files_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);work=root/WORK_RELATIVE;work.mkdir(parents=True)
            target=work/"native_result.json";target.write_text("{}\n")
            symlink=work/"escaped.json";symlink.symlink_to(target)
            with self.assertRaises(U.SafetyError):U.validate_file(root,work,symlink)
            for name in ("scf.chk","state.h5","density.npz","weights.pt","private_key.json","cache/file.json"):
                file=work/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b"local evidence")
                with self.subTest(name=name),self.assertRaises(U.SafetyError):U.validate_file(root,work,file)
            work_link=root/"reports/symlink-work";work_link.symlink_to(work,target_is_directory=True)
            with self.assertRaises(U.SafetyError):U.work_path(root,{**manifest(),"work_directory":"reports/symlink-work"})

    def test_exact_v2_resource_protocol_prefix_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);work=root/WORK_RELATIVE;work.mkdir(parents=True)
            (work/"result.json").write_text("{}\n")
            protocol=root/PROTOCOL_RELATIVE;protocol.mkdir(parents=True)
            review=protocol/"source_review.json";review.write_text('{"review":"PASS"}\n')
            value=manifest();value["upload"]["resource_protocol_files"]=[str(review.relative_to(root))]
            self.assertEqual(set(U.collect_return_files(root,value)),{work/"result.json",review})
            for relative in ("reports/fixed-quadrature-20261107-v1/resource-protocol/source_review.json",
                             "reports/fixed-quadrature-convergence-20261107-v2/other/source_review.json",
                             "reports/fixed-quadrature-convergence-20261107-v2/resource-protocol/scf.chk"):
                file=root/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_text("{}\n")
                wrong=deepcopy(value);wrong["upload"]["resource_protocol_files"]=[relative]
                with self.subTest(relative=relative),self.assertRaises(U.SafetyError):U.collect_return_files(root,wrong)

    def test_unchanged_A_ledger_and_frozen_B_source_constants(self):
        self.assertEqual(U.A_LEDGER_RELATIVE,"reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json")
        self.assertEqual(U.V8_GUARD_SHA256,"7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65")
        self.assertEqual(U.sha(ROOT/"scripts/research4_parallel_b_git_return_20261106.py"),U.FORKED_B_HELPER_SHA256)
        self.assertEqual(U.KNOWN_PREEXEC_ROW_SHA256,"03a2bdade1ba879cb0e6c4ec0167c9b921432599016004f3b996c144a3e66517")


class A006ProcessClosure(unittest.TestCase):
    def processes(self,script,state="S"):
        class Node:
            def __init__(self,name):self.name=name
            def read_text(self):return "999999 (python3) "+state+" "+"0 "*30
            def read_bytes(self):return ("python3\0"+script+"\0--execute\0").encode()
        class Proc:
            name="999999"
            def __truediv__(self,name):return Node(name)
        def fake_path(value):
            return SimpleNamespace(iterdir=lambda:[Proc()]) if value=="/proc" else Path(value)
        return patch.object(U,"Path",side_effect=fake_path)

    def test_both_new_quantum_workers_block_upload_while_live(self):
        for script in ("research4_fixed_grid_water_convergence_20261107.py",
                       "research4_fixed_grid_energy_curvature_20261107.py",
                       "research4_fixed_quadrature_worker_20261107.py"):
            with self.subTest(script=script),self.processes(script),self.assertRaises(U.SafetyError):
                U.require_no_live_quantum_processes()

    def test_terminal_states_and_metadata_analysis_are_not_live_quantum(self):
        for script,state in (("research4_fixed_grid_energy_curvature_20261107.py","Z"),
                             ("research4_fixed_grid_water_convergence_20261107.py","X"),
                             ("analyze_fixed_grid_energy_curvature.py","S")):
            with self.subTest(script=script,state=state),self.processes(script,state):
                U.require_no_live_quantum_processes()


class A006RawEvidenceReturn(unittest.TestCase):
    def fixture(self,root):
        work=root/WORK_RELATIVE;work.mkdir(parents=True)
        native=work/"pyscf.log";native.write_bytes(b"radial grids: \n    \nSCF cycle 1  \n")
        metadata=work/"result.json";metadata.write_text('{"status":"PARTIAL_RESOURCE_STOP"}\n')
        value=manifest();path=work/"return_manifest.json";path.write_text(json.dumps(value)+"\n")
        # Keep the fixture manifest out of the selected scientific files.
        value["upload"]["allowlist"]=["pyscf.log","result.json"]
        path.write_text(json.dumps(value)+"\n")
        return work,native,metadata,value,path

    def test_native_log_trailing_spaces_collected_without_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);work,native,metadata,value,path=self.fixture(root)
            before=native.read_bytes()
            self.assertEqual(U.collect_return_files(root,value),sorted([native,metadata]))
            self.assertEqual(native.read_bytes(),before)

    def test_mocked_return_lints_metadata_only_and_checks_every_committed_blob(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);work,native,metadata,value,path=self.fixture(root)
            raw_before=native.read_bytes();files=sorted([native,metadata])
            relatives=[str(p.relative_to(root)) for p in files]
            receipt_relative=str((work/"git-return/mock-only/return_receipt.json").relative_to(root))
            calls=[];counts={"diff":0,"revision":0};heads=["a"*40,"b"*40]
            def fake_git(repo,*args):
                self.assertEqual(repo,root);calls.append(args)
                if args==("branch","--show-current"):return U.BRANCH_A
                if args==("diff","--cached","--name-only"):
                    counts["diff"]+=1
                    return ("","\n".join(relatives),receipt_relative)[counts["diff"]-1]
                if args==("rev-parse","HEAD"):
                    current=heads[counts["revision"]];counts["revision"]+=1;return current
                if args[:3]==("ls-remote","--heads","origin"):
                    return heads[counts["revision"]-1]+"\trefs/heads/"+U.BRANCH_A
                return ""
            blob_calls=[]
            def fake_blob(command,**kwargs):
                self.assertEqual(command[:5],["git","-C",str(root),"cat-file","blob"])
                relative=command[-1].split(":",1)[1];blob_calls.append(relative)
                return SimpleNamespace(returncode=0,stdout=(root/relative).read_bytes())
            with patch.object(U,"git",side_effect=fake_git),patch.object(U,"validate_origin"),\
                    patch.object(U,"require_no_active_quantum",return_value={"no_active_quantum_observed":True}),\
                    patch.object(U.subprocess,"run",side_effect=fake_blob):
                result=U.execute_return(root,value,path,U.sha(path),attempt="mock-only")
            self.assertEqual(result["status"],"DONE_ACTUAL_PUSH_AND_REMOTE_REF_VERIFIED")
            self.assertTrue(result["raw_native_logs_preserved_verbatim_not_whitespace_normalized"])
            self.assertTrue(result["all_committed_scientific_blobs_match"])
            self.assertEqual(result["source_helper_sha256"],U.sha(UPLOADER))
            self.assertEqual(set(blob_calls),set(relatives));self.assertEqual(native.read_bytes(),raw_before)
            lints=[args for args in calls if args and args[0]=="-c"]
            self.assertEqual(len(lints),1)
            self.assertEqual(lints[0][-1],str(metadata.relative_to(root)))
            self.assertNotIn(str(native.relative_to(root)),lints[0])
            pushes=[args for args in calls if args and args[0]=="push"]
            self.assertEqual(pushes,[("push","origin","HEAD:refs/heads/"+U.BRANCH_A)]*2)
            self.assertFalse(any("--force" in args for args in calls))


class A006WrapperGuards(unittest.TestCase):
    def test_frozen_placeholder_hashes_fail_before_files_or_subprocess(self):
        with patch.object(W,"CONFIG_SHA","ROOT_FILL_CONFIG_SHA256"),\
                patch.object(W,"GRANT_SHA","ROOT_FILL_GRANT_SHA256"),\
                patch.object(W.subprocess,"run",side_effect=AssertionError("No subprocess permitted")),\
                patch.object(W.subprocess,"check_output",side_effect=AssertionError("No Git permitted")),\
                patch.object(W,"sha",side_effect=AssertionError("No file I/O before frozen-binding gate")):
            with self.assertRaises(RuntimeError):W.require_frozen_bindings()

    def test_placeholder_bindings_block_all_actual_entry_modes_before_io(self):
        for flags in ([],["--execute"],["--execute","--detach"]):
            with self.subTest(flags=flags),patch.object(W,"CONFIG_SHA","ROOT_FILL_CONFIG_SHA256"),\
                    patch.object(W,"GRANT_SHA","ROOT_FILL_GRANT_SHA256"),\
                    patch.object(W,"sha",side_effect=AssertionError("No file I/O before frozen-binding gate")) as hash_read,\
                    patch.object(W.subprocess,"run",side_effect=AssertionError("No process permitted")),\
                    patch.object(W.subprocess,"check_output",side_effect=AssertionError("No Git permitted")),\
                    patch.object(W.subprocess,"Popen",side_effect=AssertionError("No launch permitted")):
                with self.assertRaisesRegex(RuntimeError,"freeze and fill"):
                    W.main([*flags,"--attempt","meta-only","--uploader-sha256","a"*64,
                        "--return-manifest","manifest.json","--return-manifest-sha256","b"*64])
                hash_read.assert_not_called()

    def test_namespace_keeps_same_registered_ledger_cap_and_exact_stricter_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);work=root/WORK_RELATIVE
            value={"work_directory":WORK_RELATIVE,"ledger":U.A_LEDGER_RELATIVE,
                "quantum_cpu_cap_seconds":390160.8308359999,"deadline_utc":"2026-11-09T08:25:17.730193+00:00"}
            with patch.object(W,"ROOT",root),patch.object(W,"WORK_DIRECTORY",work):
                self.assertEqual(W.validate_queue_namespace(value),work)
                self.assertEqual(W.validate_queue_namespace({**value,"work_directory":str(work),
                    "ledger":str(root/U.A_LEDGER_RELATIVE)}),work)
                for changes in ({"work_directory":"reports/fixed-quadrature-20261107-v1/queue-host-a-005"},
                                {"ledger":"reports/copied-cpu-budget.json"},
                                {"quantum_cpu_cap_seconds":390161.},
                                {"deadline_utc":"2026-11-09T08:25:18.730193+00:00"},
                                {"deadline_utc":"2026-11-09T08:25:17.730193"}):
                    with self.subTest(changes=changes),self.assertRaises(RuntimeError):
                        W.validate_queue_namespace({**value,**changes})

    def test_check_mode_reads_canonical_ledger_even_when_config_uses_relative_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);work=root/WORK_RELATIVE
            config=root/"config.json";grant=root/"grant.json";driver=root/"driver.py";uploader=root/"uploader.py"
            ledger=root/U.A_LEDGER_RELATIVE;ledger.parent.mkdir(parents=True)
            ledger.write_text('{"phases":[]}\n')
            config.write_text(json.dumps({"work_directory":WORK_RELATIVE,"ledger":U.A_LEDGER_RELATIVE,
                "quantum_cpu_cap_seconds":390160.8308359999,"deadline_utc":"2026-11-09T08:25:17.730193+00:00"})+"\n")
            grant.write_text("{}\n");driver.write_text("# metadata fixture\n");uploader.write_text("# metadata fixture\n")
            return_manifest=root/"return_manifest.json";return_manifest.write_text(json.dumps(manifest())+"\n")
            original_read=Path.read_text;reads=[]
            def guarded_read(path,*args,**kwargs):
                self.assertTrue(path.resolve().is_relative_to(root),"Wrapper must not read a cwd-relative live ledger")
                reads.append(path.resolve());return original_read(path,*args,**kwargs)
            with patch.multiple(W,ROOT=root,WORK_DIRECTORY=work,CONFIG=config,GRANT=grant,DRIVER=driver,
                    UPLOADER=uploader,CONFIG_SHA=U.sha(config),GRANT_SHA=U.sha(grant),DRIVER_SHA=U.sha(driver)),\
                    patch.object(W.subprocess,"check_output",return_value=W.BRANCH+"\n") as git_read,\
                    patch.object(W.subprocess,"run",return_value=SimpleNamespace(returncode=0)) as driver_check,\
                    patch.object(W.subprocess,"Popen",side_effect=AssertionError("No launch permitted")),\
                    patch.object(Path,"read_text",guarded_read),redirect_stdout(io.StringIO()) as output:
                W.main(["--attempt","check-only","--uploader-sha256",U.sha(uploader),
                    "--return-manifest",str(return_manifest),"--return-manifest-sha256",U.sha(return_manifest)])
            self.assertIn(ledger.resolve(),reads)
            self.assertIn("CHECK_ONLY_NOT_RUN",output.getvalue())
            git_read.assert_called_once();driver_check.assert_called_once()
            command=driver_check.call_args.args[0]
            self.assertNotIn("--execute",command)
            self.assertEqual(command[command.index("--config")+1],str(config))
            self.assertEqual(command[command.index("--config-sha256")+1],U.sha(config))
            self.assertEqual(command[command.index("--grant")+1],str(grant))
            self.assertEqual(command[command.index("--grant-sha256")+1],U.sha(grant))

    def test_wrapper_uses_separate_A006_config_uploader_and_namespace(self):
        self.assertEqual(W.CONFIG,ROOT/"reports/fixed-quadrature-convergence-20261107-v2/queue-configs/host-a-006/config_ROOT_APPROVED.json")
        self.assertEqual(W.GRANT,W.CONFIG.with_name("root_grant_ROOT_APPROVED.json"))
        self.assertEqual(W.UPLOADER,UPLOADER)
        self.assertEqual(W.WORK_DIRECTORY,ROOT/WORK_RELATIVE)
        self.assertEqual(W.DRIVER_SHA,"06a3c5f22854f78bf8f868bf85b3956506b8514110147b977550ba838eb3780f")
        tree=ast.parse(WRAPPER.read_text())
        finite_loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call)
            and isinstance(n.iter.func,ast.Name) and n.iter.func.id=="range"
            and [x.value for x in n.iter.args if isinstance(x,ast.Constant)]==[4]]
        self.assertEqual(len(finite_loops),1)

    def test_detach_without_execute_rejected_before_subprocess(self):
        with patch.object(W,"require_frozen_bindings",return_value=None),\
                patch.object(W.subprocess,"Popen",side_effect=AssertionError("No launch permitted")),\
                patch.object(W.subprocess,"run",side_effect=AssertionError("No command permitted")),\
                patch.object(W.subprocess,"check_output",side_effect=AssertionError("No Git permitted")):
            with self.assertRaises(RuntimeError):
                W.main(["--detach","--attempt","meta-only","--uploader-sha256","a"*64,
                    "--return-manifest","manifest.json","--return-manifest-sha256","b"*64])

    def test_fresh_attempt_rejects_prior_record_console_launch_and_phase_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder);attempt="fresh-only"
            W.validate_fresh_attempt(work,attempt,detached=True)
            for relative in ("host-a-wrapper-fresh-only.json","host-a-wrapper-fresh-only.console.log",
                             "host-a-wrapper-fresh-only.launch.json","host-a-queue-fresh-only-q1.console.log",
                             "host-a-queue-fresh-only-q4.console.log","closed-budget-snapshots/fresh-only-q1.json"):
                artifact=work/relative;artifact.parent.mkdir(parents=True,exist_ok=True);artifact.write_text("preserved\n")
                with self.subTest(relative=relative),self.assertRaises(RuntimeError):
                    W.validate_fresh_attempt(work,attempt,detached=True)
                self.assertEqual(artifact.read_text(),"preserved\n");artifact.unlink()


if __name__=="__main__":unittest.main()
