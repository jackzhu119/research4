"""Pure stdlib synthetic protocols: no Q/real ledgers/arrays/Git are used."""
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research4_post_xc_pipeline_20261105_v1.py"
SPEC = importlib.util.spec_from_file_location("finite_post_XC_fixture", SOURCE)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "reports/pyscf-spinosin-grid-checkpoint-20261105-v7/fixture"
        self.work.mkdir(parents=True)
        item = patch.object(P, "ROOT", self.root); item.start(); self.addCleanup(item.stop)
        def read(file, expected=None):
            if expected is not None:
                self.assertEqual(P.sha(file), expected)
            return json.loads(Path(file).read_text())
        def checked(record):
            self.assertEqual(P.sha(record["path"]), record["sha256"])
            return Path(record["path"])
        self.q = SimpleNamespace(path=lambda p: Path(p).resolve(), read=read, sha=P.sha, checked=checked,
            artifact=lambda p: {"path": str(p), "sha256": P.sha(p), "bytes": Path(p).stat().st_size},
            option=lambda a,k: a[a.index(k)+1], process_observation=Mock(return_value={"conflicts": []}),
            utc=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            proc_identity=Mock(return_value={"pid": 123, "start_ticks": 44, "uid": 1000, "boot_id": "fixture"}),
            verify_existing=Mock(), verify_stage=Mock(return_value=[{"path": "synthetic", "sha256": "a"*64}]),
            resource_stop_sample=Mock(return_value={"reasons": []}), signal_owned_guardian=Mock(),
            own_rss=lambda: 1000, queue_lock=nullcontext)
        self.config = {"work_directory": str(self.work), "phase_prefix": "v7-synthetic",
            "python_executable": ".runtime/bin/python", "pythonpath": [".runtime/quantum-pilot-site", ".runtime/quantum-upgrade-site"],
            "snapshot": str(self.root/"source/snapshot"), "gradient_result": {"path": "gradient.json", "sha256": P.GRADIENT},
            "control_release": {"path": "release.json", "sha256": P.RELEASE},
            "phase_requests": {s: {"cpu_seconds": 600, "wall_seconds": 900} for s in P.STAGES},
            "ledger": str(self.root/"fixture-ledger.json")}
        self.deps = {k: {"path": str(self.root/(k+".json")), "sha256": k[0]*64}
                     for k in ("components", "partial", "h1", "responses")}
        self.grant = {"conflict_wait_seconds": 10, "opaque_process_identity_exceptions": [],
            "launcher_cpu_seconds_per_invocation": 600, "launcher_rss_stop_bytes": 268435456}
        self.guard = SimpleNamespace(validate_ledger=Mock(), healthy_closed_row=lambda r: r["status"]=="EXITED" and r["exit_code"]==0)
        self.write(Path(self.config["ledger"]), {"phases": [], "quantum_cpu_cap_seconds": 216000, "run_deadline_at_utc": P.DEADLINE})

    def write(self, p, value):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value)+"\n")
        return {"path": str(p), "sha256": P.sha(p)}

    def test_thirteen_bounded_quantum_stages_and_ten_global_response_blocks(self):
        self.assertEqual(len(P.STAGES), 13)
        self.assertEqual([s for s in P.STAGES if s.startswith("response-")], [f"response-{i:02d}" for i in range(10)])
        self.assertEqual(P.STAGES[-1], "global-assemble")

    def test_actual_A_argv_component_flags_and_unchanged_target_source(self):
        job=P.stage_job(self.q,self.config,"rks-partial",self.deps); av=job["payload_argv"]
        self.assertEqual(job["worker"]["sha256"],P.XC_SHA)
        self.assertEqual(self.q.option(av,"--component-index"),self.deps["components"]["path"])
        self.assertEqual(self.q.option(av,"--snapshot-manifest-sha256"),P.SNAPSHOT)
        self.assertEqual(self.q.option(av,"--xc-max-memory-mb"),"2000")
        self.assertNotIn("--control-validation",av)

    def test_H1_uses_real_partial_and_response_uses_same_H1_and_global_block_index(self):
        for i in range(10):
            av=P.stage_job(self.q,self.config,f"response-{i:02d}",self.deps)["payload_argv"]
            self.assertEqual(self.q.option(av,"--stage"),"response-block")
            self.assertEqual(self.q.option(av,"--block-index"),str(i))
            self.assertEqual(self.q.option(av,"--partial-result-sha256"),self.deps["partial"]["sha256"])
            self.assertEqual(self.q.option(av,"--h1-result-sha256"),self.deps["h1"]["sha256"])

    def test_global_uses_ordered_actual_response_index_and_frequency_status_only(self):
        j=P.stage_job(self.q,self.config,"global-assemble",self.deps)
        self.assertEqual(j["success_status"],"FREQUENCIES_COMPUTED")
        self.assertEqual(self.q.option(j["payload_argv"],"--response-index-sha256"),self.deps["responses"]["sha256"])
        self.assertNotIn("--block-index",j["payload_argv"])

    def test_immutable_index_rejects_mutation_and_does_not_overwrite(self):
        p=self.work/"meta.json";one={"schema":"index","results":[]}
        P.immutable_json(self.q,p,one);old=p.read_bytes()
        self.assertEqual(P.immutable_json(self.q,p,one)["sha256"],P.sha(p))
        with self.assertRaises(ValueError):P.immutable_json(self.q,p,{"schema":"changed"})
        self.assertEqual(p.read_bytes(),old)

    def component_fixture(self):
        plan=self.write(self.root/"plan.json",{"physical_binding":{"exact":"same"}})
        plan["bytes"]=99  # Science dependencies must strip optional artifact metadata.
        rhf={"path":str(self.root/"rhf.json"),"sha256":"b"*64,"bytes":22}
        jobs=[]
        for kind,n in (("xc-diag-chunk",20),("xc-deriv2-chunk",11)):
            for i in range(n):
                record=self.write(self.root/f"{kind}-{i}.json",{"status":"XC_CHUNK_COMPUTED"})
                jobs.append({"stage":kind,"result":record["path"],"payload_argv":["--start-block",str(i),"--block-count","1"]})
        master=self.write(self.root/"master.json",{"jobs":jobs})
        cfg=dict(self.config,plan_result=plan,RHF_result=rhf,master_XC_config=master)
        xc=SimpleNamespace(qualify_component=Mock(return_value=(True,)),_assembly_sources=Mock(return_value={"verified":True}))
        return cfg,xc

    def test_component_index_requires_all31_real_closed_refs_and_exact_two_key_dependencies(self):
        cfg,xc=self.component_fixture();state={}
        record,physical=P.component_index(self.q,self.guard,xc,cfg,state)
        value=self.q.read(record["path"])
        self.assertEqual(set(value),{"schema","RHF_partial","XC_plan","XC_chunks"})
        self.assertEqual(set(value["XC_plan"]),{"path","sha256"})
        self.assertEqual(set(value["RHF_partial"]),{"path","sha256"})
        self.assertEqual(len(value["XC_chunks"]),31);self.assertEqual(xc.qualify_component.call_count,31)
        self.assertEqual(xc._assembly_sources.call_count,1)
        self.assertEqual(physical,{"exact":"same"})

    def test_incomplete_or_duplicate_grid_coverage_cannot_materialize_A(self):
        cfg,xc=self.component_fixture();m=self.q.read(cfg["master_XC_config"]["path"])
        m["jobs"][-1]["payload_argv"]=["--start-block","9","--block-count","1"]
        cfg["master_XC_config"]=self.write(Path(cfg["master_XC_config"]["path"]),m)
        with self.assertRaises(ValueError):P.component_index(self.q,self.guard,xc,cfg,{})
        xc._assembly_sources.assert_not_called()

    def imported_fixture(self, stage="rks-partial"):
        physical={"same":"MF"};xc=SimpleNamespace(qualify_partial=Mock());con=SimpleNamespace(qualify_stage=Mock(),qualify_response=Mock())
        actual_stage="response-block" if stage.startswith("response-") else stage
        worker,source=(P.XC,P.XC_SHA) if stage=="rks-partial" else (P.CONSUMER,P.CONSUMER_SHA)
        value={"stage":actual_stage,"component_index":self.deps["components"],"partial_result_dependency":self.deps["partial"],
            "block_index":int(stage.split("-")[1]) if stage.startswith("response-") else None,
            "source_worker":{"path":str(worker),"sha256":source},"resource_supervision":{"proof":{"ledger_path":self.config["ledger"],"phase":"actual-root-phase"}}}
        record=self.write(self.root/"actual-result.json",value)
        self.write(Path(self.config["ledger"]),{"phases":[{"phase":"actual-root-phase","status":"EXITED","exit_code":0,
            "command":[".runtime/bin/python","scripts/worker.py","--execute"],"log":"actual.log"}]})
        return record,value,physical,xc,con

    def test_actual_root_A_import_preserves_relative_guardian_argv(self):
        record,_,physical,xc,con=self.imported_fixture()
        j=P.imported_healthy_job(self.q,self.guard,self.config,"rks-partial",record,physical,xc,con,self.deps)
        self.assertEqual(j["payload_argv"][0],".runtime/bin/python")
        self.assertEqual(j["phase"],"actual-root-phase");xc.qualify_partial.assert_called_once()

    def test_imported_A_H1_or_response_cannot_use_different_dependencies_or_block(self):
        for stage,key,bad in (("rks-partial","component_index",{}),("h1ao","partial_result_dependency",{}),("response-03","block_index",4)):
            record,v,physical,xc,con=self.imported_fixture(stage);v[key]=bad
            record=self.write(Path(record["path"]),v)
            with self.subTest(stage=stage),self.assertRaises(ValueError):
                P.imported_healthy_job(self.q,self.guard,self.config,stage,record,physical,xc,con,self.deps)

    def test_imported_A_from_failed_or_different_cumulative_pool_is_rejected(self):
        record,v,physical,xc,con=self.imported_fixture();v["resource_supervision"]["proof"]["ledger_path"]="different.json"
        record=self.write(Path(record["path"]),v)
        with self.assertRaises(ValueError):P.imported_healthy_job(self.q,self.guard,self.config,"rks-partial",record,physical,xc,con,self.deps)

    def test_transient_conflicts_wait_finitely_then_continue_without_cancelling_any_child(self):
        self.q.process_observation.side_effect=[{"conflicts":[{"category":"GIT_WRITE_OR_UNKNOWN"}]},{"conflicts":[]}]
        events=[]
        with patch.object(P.time,"sleep") as sleep:
            P.wait_for_clear(self.q,self.grant,P.time.time()+1000,P.time.monotonic()+1000,lambda e,d:events.append(e),[])
        self.assertEqual(events,["WAITING_FOR_EXTERNAL_CONFLICT_CLEAR"]);sleep.assert_called_once()
        self.q.signal_owned_guardian.assert_not_called()

    def test_persistent_conflicts_exhaust_finite_wait_before_new_phase(self):
        self.grant["conflict_wait_seconds"]=0;self.q.process_observation.return_value={"conflicts":[{"category":"ML"}]}
        with patch.object(P.subprocess,"Popen") as spawn,self.assertRaises(ValueError):
            P.wait_for_clear(self.q,self.grant,P.time.time()+1000,P.time.monotonic()+1000,lambda *_:None,[])
        spawn.assert_not_called()

    def test_existing_healthy_seal_is_verified_without_new_quantum_subprocess(self):
        job=P.stage_job(self.q,self.config,"h1ao",self.deps);Path(job["result"]).parent.mkdir(parents=True)
        self.write(Path(job["result"]),{"status":"STAGE_COMPLETE"})
        state={"stages":{"h1ao":{"status":"HEALTHY_CLOSED","seal":[{"marker":"real-seal"}]}}}
        with patch.object(P.subprocess,"Popen") as spawn:
            result=P.launch_quantum(self.q,self.guard,self.config,self.grant,job,state,P.time.time()+1000,P.time.monotonic()+1000,[],lambda *_:None)
        spawn.assert_not_called();self.q.verify_existing.assert_called_once();self.assertEqual(result["path"],job["result"])

    def test_failed_or_open_checkpoint_never_retries_or_refunds(self):
        job=P.stage_job(self.q,self.config,"h1ao",self.deps)
        for status in ("RUNNING","BLOCKED_REQUIRES_ROOT_REVIEW","STARTING"):
            with patch.object(P.subprocess,"Popen") as spawn,self.assertRaises(ValueError):
                P.launch_quantum(self.q,self.guard,self.config,self.grant,job,{"stages":{"h1ao":{"status":status}}},P.time.time()+1000,P.time.monotonic()+1000,[],lambda *_:None)
            spawn.assert_not_called()

    def test_new_guardian_is_frozen_same_ledger_detached_with_stdin_DEVNULL(self):
        job=P.stage_job(self.q,self.config,"response-00",self.deps);state={"stages":{}}
        child=Mock(pid=123);child.poll.return_value=0;child.wait.return_value=0
        def launch(*args,**kwargs):
            self.write(Path(job["result"]),{"status":"STAGE_COMPLETE"});return child
        with patch.object(P.subprocess,"Popen",side_effect=launch) as spawn:
            P.launch_quantum(self.q,self.guard,self.config,self.grant,job,state,P.time.time()+1000,P.time.monotonic()+1000,[],lambda *_:None)
        args,kwargs=spawn.call_args
        self.assertEqual(args[0][1],str(P.GUARD));self.assertEqual(self.q.option(args[0],"--ledger"),self.config["ledger"])
        self.assertTrue(kwargs["start_new_session"]);self.assertEqual(kwargs["stdin"],P.subprocess.DEVNULL)
        self.assertEqual(state["stages"]["response-00"]["status"],"HEALTHY_CLOSED")

    def test_nonzero_guardian_stops_and_records_unknown_external_origin(self):
        job=P.stage_job(self.q,self.config,"response-00",self.deps);state={"stages":{}}
        child=Mock(pid=123);child.poll.return_value=7;child.wait.return_value=7
        with patch.object(P.subprocess,"Popen",return_value=child) as spawn,self.assertRaises(ValueError):
            P.launch_quantum(self.q,self.guard,self.config,self.grant,job,state,P.time.time()+1000,P.time.monotonic()+1000,[],lambda *_:None)
        self.assertEqual(spawn.call_count,1);self.assertEqual(state["stages"]["response-00"]["guardian_exit_origin"],"UNKNOWN_EXTERNAL_TO_LAUNCHER")
        self.assertEqual(state["stages"]["response-00"]["status"],"BLOCKED_REQUIRES_ROOT_REVIEW")

    def test_negative_modes_do_not_get_relabelled_as_missing_completion_or_IR(self):
        state={};global_ref={"path":"global.json","sha256":"e"*64};output=self.work/"independent-modes"
        checks={"schema":"research4-V7-independent-staged-modes-analysis-v1","status":"DONE_COMPUTED_MODES_AUDITED","normal_modes":219,
            "normal_mode_count_complete":True,"new_quantum_evaluations":0,"source_result_sha256":"e"*64,"IR_intensities_status":"NOT_COMPUTED",
            "qualified_for_mechanism":False,"mechanism_pairing_status":"BLOCKED_NONPOSITIVE_MINIMUM"}
        child=Mock(pid=123);child.poll.return_value=0;child.wait.return_value=0
        def launch(*args,**kwargs):
            item=self.write(output/"checks.json",checks)
            self.write(output/"artifact_index.json",{"analysis_source_sha256":P.ANALYZER_SHA,"V7_worker_sha256":P.CONSUMER_SHA,
                "files":[{"path":"checks.json","sha256":item["sha256"],"bytes":(output/"checks.json").stat().st_size}]})
            return child
        with patch.object(P.subprocess,"Popen",side_effect=launch) as spawn:
            P.run_analysis(self.q,self.config,self.grant,global_ref,state,P.time.time()+1000,P.time.monotonic()+1000,[],lambda *_:None)
        self.assertEqual(state["analysis"]["status"],"HEALTHY_CLOSED");self.assertFalse(state["analysis"]["qualified_for_mechanism"])
        self.assertNotIn(str(P.GUARD),spawn.call_args.args[0]);self.assertEqual(spawn.call_args.kwargs["stdin"],P.subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
