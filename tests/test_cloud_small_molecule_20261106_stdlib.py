"""No numerical imports or quantum evaluations: provenance and execution gates."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PATH = Path(__file__).resolve().parents[1] / 'scripts/research4_cloud_small_molecule_20261106.py'
spec = importlib.util.spec_from_file_location('small_molecule_test_subject', PATH)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class ProvenanceAndExecutionGates(unittest.TestCase):
    def authorized_fixture(self, directory):
        main = Path(directory) / 'main.json'
        args = argparse.Namespace(threads=2, authorization=Path(directory) / 'auth.json',
            authorization_sha256='a' * 64, prepared_sha256='b' * 64,
            output=Path(directory) / 'new-output', maxsteps=25,
            main_pipeline_state=main, main_pipeline_state_sha256='c' * 64)
        auth = {'schema': w.AUTH_SCHEMA, 'enable_execution': True, 'worker_sha256': w.sha(PATH),
            'supervisor_sha256': w.GUARD_SHA, 'prepared_manifest_sha256': args.prepared_sha256,
            'output_directory': str(args.output.resolve()), 'selected_threads': 2, 'optimizer_maxsteps': 25,
            'no_budget_reset': True, 'no_deadline_extension': True,
            'main_pipeline_state': str(main.resolve()), 'main_pipeline_state_sha256': 'c' * 64,
            'maximum_cpu_seconds': 7200, 'maximum_wall_seconds': 7200,
            'not_after_utc': '2026-11-09T08:25:17.730193+00:00'}
        ledger = {'quantum_cpu_cap_seconds': w.CAP, 'run_deadline_at_utc': w.DEADLINE,
            'phases': [{'phase': 'actual-pilot', 'status': 'RUNNING', 'command': ['python', str(PATH)]}]}
        proof = {'ledger_path': str(w.LEDGER), 'cpu_hard_limit_seconds': 7200,
            'wall_limit_seconds': 7200, 'phase': 'actual-pilot'}
        class Guard:
            @staticmethod
            def verify_resource_supervision(**kwargs):
                return {'proof': proof}
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 11, 6, 9, 0, 0, tzinfo=timezone.utc)
        def loader(path, expected=None):
            return ledger if Path(path) == w.LEDGER else auth
        return args, auth, ledger, Guard, Clock, loader

    def test_registered_neutral_graph_and_formula(self):
        value = w.source_identity()
        self.assertEqual(value['formula'], 'C7H6O4')
        self.assertEqual(value['full_inchikey'], 'YQUVCSBJEUQKSH-UHFFFAOYSA-N')
        self.assertEqual((value['charge'], value['spin_2S'], value['electrons']), (0, 0, 80))
        self.assertFalse(value['downloaded_experimental_3D'])

    def test_changed_graph_registry_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'registry.csv'; p.write_text('tampered\n')
            with mock.patch.object(w, 'REGISTRY', p):
                with self.assertRaisesRegex(ValueError, 'graph sources changed'):
                    w.source_identity()

    def test_changed_metadata_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'metadata.json'; p.write_text('{"stage":"NOT_RUN"}')
            with self.assertRaisesRegex(ValueError, 'Changed metadata'):
                w.load(p, '0' * 64)

    def test_exclusive_result_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'result.json'
            w.save(p, {'status': 'PARTIAL'}, exclusive=True)
            before = p.read_bytes()
            with self.assertRaises(FileExistsError):
                w.save(p, {'status': 'FREQUENCIES_COMPUTED'}, exclusive=True)
            self.assertEqual(p.read_bytes(), before)

    def test_output_outside_reports_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'unapproved'
            with self.assertRaisesRegex(ValueError, 'Fresh reports output required'):
                w.fresh_output(p)
            self.assertFalse(p.exists())

    def test_missing_actual_supervision_blocks_before_coordinates(self):
        class AbsentGuard:
            @staticmethod
            def verify_resource_supervision(**kwargs):
                raise ValueError('No actual guardian')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'uncreated'
            args = argparse.Namespace(threads=2, output=output)
            with mock.patch.object(w, 'frozen', return_value=AbsentGuard):
                with self.assertRaisesRegex(ValueError, 'No actual guardian'):
                    w.compute(args)
            self.assertFalse(output.exists())

    def test_missing_root_authorization_blocks_before_coordinates(self):
        class Guard:
            @staticmethod
            def verify_resource_supervision(**kwargs):
                return {'proof': {}}
        args = argparse.Namespace(threads=2, authorization=None, authorization_sha256=None)
        with mock.patch.object(w, 'frozen', return_value=Guard):
            with self.assertRaisesRegex(ValueError, 'Root source-bound authorization required'):
                w.compute(args)

    def test_frozen_helper_hash_mismatch_is_rejected_before_import(self):
        with self.assertRaisesRegex(ValueError, 'Frozen helper changed'):
            w.frozen(PATH, '0' * 64, 'must_not_import')

    def test_source_bound_authorization_metadata_only_happy_path(self):
        with tempfile.TemporaryDirectory() as directory:
            args, auth, ledger, guard, clock, loader = self.authorized_fixture(directory)
            with mock.patch.object(w, 'frozen', return_value=guard), mock.patch.object(w, 'load', side_effect=loader), mock.patch.object(w, 'datetime', clock):
                self.assertEqual(w.authorize(args)['proof']['phase'], 'actual-pilot')
            self.assertFalse(args.output.exists())

    def test_expired_followup_authorization_cannot_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            args, auth, ledger, guard, clock, loader = self.authorized_fixture(directory)
            auth['not_after_utc'] = '2026-11-06T08:00:00+00:00'
            with mock.patch.object(w, 'frozen', return_value=guard), mock.patch.object(w, 'load', side_effect=loader), mock.patch.object(w, 'datetime', clock):
                with self.assertRaisesRegex(ValueError, 'stricter follow-up authorization deadline'):
                    w.authorize(args)

    def test_wrong_main_state_sha_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, auth, ledger, guard, clock, loader = self.authorized_fixture(directory)
            auth['main_pipeline_state_sha256'] = 'd' * 64
            with mock.patch.object(w, 'frozen', return_value=guard), mock.patch.object(w, 'load', side_effect=loader), mock.patch.object(w, 'datetime', clock):
                with self.assertRaisesRegex(ValueError, 'closed Spinosin main-state binding'):
                    w.authorize(args)

    def test_unrelated_live_worker_cannot_be_relabelled_as_this_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            args, auth, ledger, guard, clock, loader = self.authorized_fixture(directory)
            ledger['phases'][0]['command'] = ['python', 'unrelated.py']
            with mock.patch.object(w, 'frozen', return_value=guard), mock.patch.object(w, 'load', side_effect=loader), mock.patch.object(w, 'datetime', clock):
                with self.assertRaisesRegex(ValueError, 'exact source-bound live quantum phase'):
                    w.authorize(args)


if __name__ == '__main__':
    unittest.main()
