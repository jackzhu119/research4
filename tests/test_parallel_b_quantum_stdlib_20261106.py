"""Portable B provenance/execution gates, no RDKit/NumPy/PySCF imports."""
import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PATH = Path(__file__).resolve().parents[1] / 'scripts/research4_parallel_b_quantum_20261106.py'
spec = importlib.util.spec_from_file_location('portable_b_quantum_subject', PATH)
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


class PortableBGates(unittest.TestCase):
    def test_actual_control_result_flags_release_only_full_positive_qualified_modes(self):
        valid = w.signed_mode_flags([1500.0, 3500.0, 3600.0], 3, True)
        self.assertTrue(valid['all_frequencies_real_positive'])
        self.assertTrue(valid['stationary_minimum_verified'])
        self.assertEqual(valid['imaginary_frequency_count'], 0)
        self.assertFalse(w.signed_mode_flags([1500., 3500., 3600.], 3, False)['stationary_minimum_verified'])
        failed = w.signed_mode_flags([-100., 3500., 3600.], 3, True)
        self.assertEqual(failed['imaginary_frequency_count'], 1)
        self.assertFalse(failed['frequency_validation_passed'])
        zero = w.signed_mode_flags([0., 3500., 3600.], 3, True)
        self.assertEqual(zero['zero_frequency_count'], 1)
        self.assertFalse(zero['all_frequencies_real_positive'])
        with self.assertRaisesRegex(ValueError, 'All finite signed modes required'):
            w.signed_mode_flags([1500., 3500.], 3, True)

    def test_artifact_paths_are_checkout_relative_and_round_trip(self):
        record = w.artifact(PATH)
        self.assertEqual(record['path'], 'scripts/research4_parallel_b_quantum_20261106.py')
        self.assertFalse(Path(record['path']).is_absolute())
        self.assertEqual(w.checked_record(record), PATH.resolve())

    def test_default_is_not_run_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(case='water', operation='frequency', conformers=8,
                execute=False, output=Path(directory) / 'absent')
            result = w.run(args)
            self.assertEqual(result['status'], 'NOT_RUN')
            self.assertEqual(result['new_quantum_evaluations'], 0)
            self.assertFalse(args.output.exists())

    def test_only_registered_cases(self):
        with self.assertRaisesRegex(ValueError, 'three explicitly registered B cases'):
            w.input_case('spinosin')

    def test_portable_neutral_case_sizes_and_meaning(self):
        expected = {'water': (3, 10, 24, 3), 'protocatechuic_acid': (17, 80, 184, 45),
                    'calycosin': (33, 148, 354, 93)}
        for name, counts in expected.items():
            _, case = w.input_case(name)
            self.assertEqual(tuple(case[k] for k in ('atoms', 'electrons', 'basis_functions', 'expected_normal_modes')), counts)
            self.assertEqual((case['charge'], case['spin_2S'], case['multiplicity']), (0, 0, 1))

    def test_changed_case_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.json'; path.write_text('{}')
            with mock.patch.object(w, 'INPUT', path):
                with self.assertRaisesRegex(ValueError, 'Metadata source hash differs'):
                    w.input_case('water')

    def test_source_artifact_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'geometry.json'; path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                w.checked_record({'path': str(path), 'sha256': '0' * 64})

    def test_existing_attempt_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'; w.save(path, {'status': 'PARTIAL'}, exclusive=True)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                w.save(path, {'status': 'FREQUENCIES_COMPUTED'}, exclusive=True)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'Fresh independent attempt'):
                w.new_output(Path(directory))

    def test_invalid_or_truncated_coordinate_payload_rejected(self):
        _, case = w.input_case('water')
        with self.assertRaisesRegex(ValueError, 'composition/shape'):
            w.validate_coordinates(['O', 'H'], [[0, 0, 0], [0, 0, 1]], case)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            w.validate_coordinates(['O', 'H', 'H'], [[0, 0, 0], [0, 0, 1], [float('nan'), 1, 0]], case)

    def test_quantum_execution_cannot_bypass_live_B_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(case='water', operation='prepare', conformers=8,
                execute=True, output=Path(directory) / 'absent')
            with mock.patch.object(w, 'supervise', side_effect=ValueError('No actual B resource proof')):
                with self.assertRaisesRegex(ValueError, 'No actual B resource proof'):
                    w.run(args)
            self.assertFalse(args.output.exists())

    def test_unqualified_opt_result_cannot_release_frequency(self):
        _, case = w.input_case('water')
        args = argparse.Namespace(source_result=Path('not-read.json'))
        incomplete = {'optimizer_converged': True, 'scf_converged': True,
            'fresh_gradient_thresholds_passed': False}
        with mock.patch.object(w, 'source_result', return_value=incomplete):
            with self.assertRaisesRegex(ValueError, 'fresh-gradient-qualified optimization'):
                w.optimized_geometry(args, case)

    def test_source_different_case_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json';w.save(path, {'schema': w.SCHEMA,
                'case': 'calycosin', 'operation': 'prepare', 'status': 'PREPARED_FORCE_FIELD_MODELS',
                'input_manifest_sha256': w.INPUT_SHA}, exclusive=True)
            args = argparse.Namespace(source_result=path, source_result_sha256=w.sha(path), case='water')
            with self.assertRaisesRegex(ValueError, 'Preceding case/operation/result status differs'):
                w.source_result(args, 'prepare', ('PREPARED_FORCE_FIELD_MODELS',))

    def test_foreign_source_or_changed_method_cannot_be_used(self):
        manifest, case = w.input_case('water')
        original = {'schema': w.SCHEMA, 'case': 'water', 'operation': 'prepare',
            'status': 'PREPARED_FORCE_FIELD_MODELS', 'input_manifest_sha256': w.INPUT_SHA,
            'source_worker': {'sha256': w.sha(PATH)}, 'settings': manifest['method']}
        for mutation, error in [('source', 'computation source differs'), ('method', 'physical method differs')]:
            value = deepcopy(original)
            if mutation == 'source':value['source_worker']['sha256'] = '0' * 64
            else:value['settings']['dispersion'] = 'UNDECLARED_D3'
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'result.json'; w.save(path, value, exclusive=True)
                args = argparse.Namespace(source_result=path, source_result_sha256=w.sha(path), case='water')
                with self.assertRaisesRegex(ValueError, error):
                    w.source_result(args, 'prepare', ('PREPARED_FORCE_FIELD_MODELS',))


if __name__ == '__main__':
    unittest.main()
