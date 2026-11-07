"""Synthetic first-point/event fixtures only; no molecule or quantum execution."""
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
SPEC=spec_from_file_location("first_gradient_comparator",ROOT/"scripts/compare_pyscf_spinosin_first_gradient_20261031.py")
comparison=module_from_spec(SPEC);SPEC.loader.exec_module(comparison)


def save(path,value):
    path.write_text(json.dumps(value)+"\n")


def fixture(folder,threads,gradient_change=0):
    folder.mkdir();(folder/"geometry_steps").mkdir()
    symbols=["C"]*28+["H"]*32+["O"]*15
    coords=np.arange(225,dtype=float).reshape(75,3)/100
    gradient=np.arange(225,dtype=float).reshape(75,3)*1e-6+gradient_change
    xyz="75\nSYNTHETIC_NONPHYSICAL_UNIT_TEST_ONLY\n"+"".join(f"{s} {x:.12f} {y:.12f} {z:.12f}\n" for s,(x,y,z) in zip(symbols,coords))
    (folder/"input.xyz").write_text(xyz)
    np.save(folder/"input_coordinates_angstrom.npy",coords)
    settings={**comparison.PHYSICS,"threads":threads}
    if threads==4:settings["native_direct_driver_threads"]=4
    save(folder/"input.json",{"phase":"spinosin-optimize","identity":"spinosin","settings":settings,
        "synthetic_nonphysical_fixture":True,"input_xyz_sha256":comparison.sha(folder/"input.xyz"),
        "input_coordinates_angstrom_npy_sha256":comparison.sha(folder/"input_coordinates_angstrom.npy"),
        "input_provenance":{"identity_verified":True,"stereochemistry_verified":True,
            "source_inchikey":comparison.INCHIKEY,"source_url":"https://example.invalid/unit-test-only","electron_count":320}})
    for suffix,array in (("coordinates_angstrom",coords),("coordinates_bohr",coords/comparison.BOHR),("gradient",gradient)):
        np.save(folder/"geometry_steps"/f"step_0001.{suffix}.npy",array)
    metadata={"cycle":1,"scf_converged":True,"energy_hartree":-2000.,**comparison.statistics(gradient)}
    for suffix in ("coordinates_angstrom","coordinates_bohr","gradient"):
        metadata[suffix+"_npy_sha256"]=comparison.sha(folder/"geometry_steps"/f"step_0001.{suffix}.npy")
    save(folder/"geometry_steps/step_0001.json",metadata)
    policy={"density_fitting":False}
    start={"event":"direct_gradient_started","utc":"2020-01-01T00:00:00+00:00","process_cpu_seconds":10.,"wall_seconds":10.,"scf_converged":True,"SCF_energy_hartree":-2000.,"direct_integral_policy":policy}
    complete={"event":"direct_gradient_kernel_complete","utc":"2020-01-01T00:00:05+00:00" if threads==1 else "2020-01-01T00:00:02+00:00","process_cpu_seconds":14.,"wall_seconds":15. if threads==1 else 12.,"gradient_cpu_seconds":3.9,"gradient_wall_seconds":4.9 if threads==1 else 1.9,"direct_integral_policy":policy}
    cycle={"event":"geometry_cycle","utc":"2020-01-01T00:00:06+00:00" if threads==1 else "2020-01-01T00:00:03+00:00",**metadata}
    (folder/"events.jsonl").write_text("".join(json.dumps(event)+"\n" for event in (start,complete,cycle)))
    return metadata


class FirstGradientTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="first-gradient-pure-");self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.old=self.root/"old";self.new=self.root/"new"
        fixture(self.old,1);fixture(self.new,4)

    def events(self,folder):
        return [json.loads(line) for line in (folder/"events.jsonl").read_text().splitlines()]

    def write_events(self,folder,events):
        (folder/"events.jsonl").write_text("".join(json.dumps(event)+"\n" for event in events))

    def release(self):
        path=self.root/"release.json"
        save(path,{"allow_first_gradient_comparison":True,"created_at_utc":datetime.now(timezone.utc).isoformat(),
            "comparator_sha256":comparison.sha(comparison.__file__),
            "old_step_json_sha256":comparison.sha(self.old/"geometry_steps/step_0001.json"),
            "new_step_json_sha256":comparison.sha(self.new/"geometry_steps/step_0001.json")})
        return path

    def test_fixed_tolerances_and_complete_synthetic_comparison(self):
        self.assertEqual(comparison.TOLERANCES,{"coordinates_angstrom":5e-12,"energy_hartree":1e-8,
            "gradient_components_hartree_per_bohr":1e-8,"atom_gradient_norms_hartree_per_bohr":1e-8})
        out=self.root/"audit"
        result=comparison.run(SimpleNamespace(old_output=self.old,new_output=self.new,release=self.release(),output=out))
        self.assertEqual(result["status"],"PASS");self.assertEqual(result["checkpoint_reads"],0)
        self.assertEqual(result["SCF_or_gradient_or_integral_or_Hessian_evaluations"],0)
        self.assertAlmostEqual(result["timing"]["observed_wall_ratio_native1_over_native4"],4.9/1.9)
        self.assertEqual(len((out/"all75_atom_gradient_comparison.csv").read_text().splitlines()),76)
        with self.assertRaises(FileExistsError):
            comparison.run(SimpleNamespace(old_output=self.old,new_output=self.new,release=self.release(),output=out))

    def test_bad_or_unfinished_step_cannot_be_loaded(self):
        point=self.new/"geometry_steps/step_0001.json";record=json.loads(point.read_text())
        record["scf_converged"]=False;save(point,record)
        with self.assertRaisesRegex(ValueError,"SCF-converged"):comparison.load_point(self.new,4)
        point.unlink()
        with self.assertRaises(FileNotFoundError):comparison.load_point(self.new,4)

    def test_changed_array_or_source_sha_is_rejected(self):
        gradient=self.new/"geometry_steps/step_0001.gradient.npy"
        values=np.load(gradient);values[0,0]+=1e-9;np.save(gradient,values)
        with self.assertRaisesRegex(ValueError,"NPY SHA"):comparison.load_point(self.new,4)

    def test_different_method_or_atom_order_is_rejected(self):
        path=self.new/"input.json";value=json.loads(path.read_text());value["settings"]["density_fitting"]=True;save(path,value)
        with self.assertRaisesRegex(ValueError,"Physical method"):comparison.load_point(self.new,4)
        old,new=comparison.load_point(self.old,1),comparison.load_point(self.old,1)
        new["symbols"]=list(new["symbols"]);new["symbols"][0],new["symbols"][28]=new["symbols"][28],new["symbols"][0]
        with self.assertRaisesRegex(ValueError,"Atom order"):comparison.compare_points(old,new)

    def test_all_components_checked_not_only_norm_or_average(self):
        old,new=comparison.load_point(self.old,1),comparison.load_point(self.new,4)
        new["arrays"]["gradient"]=new["arrays"]["gradient"].copy()
        # Negating one component preserves that atom's norm, but fails raw comparison.
        new["arrays"]["gradient"][1,0]*=-1
        result=comparison.compare_points(old,new)
        self.assertFalse(result["checks"]["all225_gradient_components"])
        self.assertTrue(result["checks"]["all75_atom_gradient_norms"])

    def test_complete_prefix_ignores_later_partial_append(self):
        point=comparison.load_point(self.new,4);before=(self.new/"events.jsonl").read_bytes()
        with (self.new/"events.jsonl").open("ab") as handle:handle.write(b'{"event":"still_writing')
        actual=comparison.completed_gradient_prefix(self.new/"events.jsonl",point)
        self.assertEqual(actual["prefix"],before)
        self.assertEqual(actual["timing"]["completed_prefix_line_count"],3)

    def test_no_eager_events_read_or_iteration_after_completed_cycle(self):
        point=comparison.load_point(self.new,4)
        events_path=self.new/"events.jsonl"; lines=events_path.read_bytes().splitlines(keepends=True)
        real_open=Path.open; real_read_bytes=Path.read_bytes
        class LimitedPrefix:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def __iter__(self):
                yield from lines
                raise AssertionError("Comparator attempted to read beyond completed cycle1")
        def scoped_open(path,*args,**kwargs):
            return LimitedPrefix() if path==events_path else real_open(path,*args,**kwargs)
        def scoped_read(path):
            if path==events_path:raise AssertionError("Eager whole-events read is forbidden")
            return real_read_bytes(path)
        with patch.object(Path,"open",scoped_open),patch.object(Path,"read_bytes",scoped_read):
            result=comparison.completed_gradient_prefix(events_path,point)
        self.assertEqual(result["prefix"],b"".join(lines))

    def test_unpaired_events_or_cycle_hash_mismatch_fail(self):
        events=self.events(self.new);point=comparison.load_point(self.new,4)
        self.write_events(self.new,[events[0],events[0],events[1],events[2]])
        with self.assertRaisesRegex(ValueError,"Ambiguous"):comparison.completed_gradient_prefix(self.new/"events.jsonl",point)
        events[2]["gradient_npy_sha256"]="0"*64;self.write_events(self.new,events)
        with self.assertRaisesRegex(ValueError,"hashes disagree"):comparison.completed_gradient_prefix(self.new/"events.jsonl",point)

    def test_inner_timer_not_forced_equal_to_outer_but_invalid_timing_fails(self):
        point=comparison.load_point(self.old,1)
        result=comparison.completed_gradient_prefix(self.old/"events.jsonl",point)
        self.assertGreater(result["timing"]["outer_event_CPU_delta_seconds"],result["timing"]["gradient_cpu_seconds"])
        events=self.events(self.old);events[1]["gradient_wall_seconds"]=20;self.write_events(self.old,events)
        with self.assertRaisesRegex(ValueError,"timing"):comparison.completed_gradient_prefix(self.old/"events.jsonl",point)

    def test_anomalously_small_inner_timer_cannot_create_huge_benchmark(self):
        point=comparison.load_point(self.new,4);events=self.events(self.new)
        events[1]["gradient_wall_seconds"]=1e-9;self.write_events(self.new,events)
        with self.assertRaisesRegex(ValueError,"overhead"):
            comparison.completed_gradient_prefix(self.new/"events.jsonl",point)

    def test_missing_or_wrong_dated_release_prevents_output_creation(self):
        out=self.root/"audit";release=self.release();value=json.loads(release.read_text());value["new_step_json_sha256"]="0"*64;save(release,value)
        with self.assertRaisesRegex(ValueError,"Released"):comparison.run(SimpleNamespace(old_output=self.old,new_output=self.new,release=release,output=out))
        self.assertFalse(out.exists())
        value["comparator_sha256"]="0"*64;save(release,value)
        with self.assertRaisesRegex(ValueError,"frozen comparator"):comparison.run(SimpleNamespace(old_output=self.old,new_output=self.new,release=release,output=out))


if __name__=="__main__":unittest.main()
