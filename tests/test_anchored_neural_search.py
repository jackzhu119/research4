"""Small synthetic invariance checks, not medicinal-plant accuracy evidence."""
from copy import deepcopy
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import torch
from torch import nn

from herb_spectra.anchored_neural_search import (adapt_paths, candidate_grid,
    choose_candidate, index_labels, predict_scores, source_class_weights,
    validate_candidate, verify_context)
from herb_spectra.paper_fewshot_classification import state_fingerprint


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(8, 10)
        self.head = nn.Linear(10, 7)

    def forward(self, values):
        return self.head(torch.relu(self.embedding(values)))


def fixture():
    rng = np.random.default_rng(45)
    x = rng.normal(size=(28, 8)).astype(np.float32)
    y = np.tile(np.arange(7), 4)
    torch.manual_seed(12)
    return TinyClassifier().eval(), x, y


class AnchoredNeuralSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_grid_is_exact_unique24_and_rejects_unlocked_configuration(self):
        grid = candidate_grid()
        self.assertEqual(len(grid), 24)
        self.assertEqual(len({r['id'] for r in grid}), 24)
        for c in grid:
            validate_candidate(c)
        wrong = {**grid[0], 'learning_rate': .5}
        with self.assertRaisesRegex(ValueError, 'frozen24'):
            validate_candidate(wrong)

    def test_source_weighting_is_globally_normalized_without_double_sampling(self):
        y = np.repeat(np.arange(7), [20, 1, 4, 4, 4, 4, 4])
        weights = source_class_weights(y, 7, 'balanced').numpy()
        assert_allclose(weights[y].sum(), len(y), rtol=1e-7)
        for c in range(7):
            assert_allclose(weights[y][y == c].sum(), len(y)/7, rtol=1e-7)
        assert_array_equal(source_class_weights(y, 7, 'none').numpy(), np.ones(7))
        with self.assertRaises(ValueError):
            source_class_weights(y[y != 1], 7, 'balanced')

    def test_no_target_no_replay_control_is_exact_identity_and_preserves_source(self):
        initial, x, y = fixture()
        before = state_fingerprint(initial)
        zero = predict_scores(initial, x[:6])
        configs = [c for c in candidate_grid() if c['source_replay_weight'] == 0]
        scores, traces, models = adapt_paths(initial, x, y, x[:6], configs,
            capture_models=True)
        for c in configs:
            assert_array_equal(scores[c['id']], zero)
            self.assertEqual(traces[c['id']]['updates'], 0)
            self.assertEqual(state_fingerprint(models[c['id']]), before)
        self.assertEqual(state_fingerprint(initial), before)

    def test_source_only_replay_changes_only_a_copy_and_support_has_extra_effect(self):
        initial, x, y = fixture()
        before = state_fingerprint(initial)
        config = [candidate_grid()[-1]]
        matched, _, _ = adapt_paths(initial, x, y, x[:10], config)
        targeted, _, models = adapt_paths(initial, x, y, x[:10], config,
            support_x=x[:7], support_y=y[:7], capture_models=True)
        key = config[0]['id']
        self.assertGreater(float(np.abs(matched[key]-predict_scores(initial, x[:10])).max()), 1e-5)
        self.assertGreater(float(np.abs(targeted[key]-matched[key]).max()), 1e-5)
        self.assertNotEqual(state_fingerprint(models[key]), before)
        self.assertEqual(state_fingerprint(initial), before)

    def test_support_labels_change_gradients_but_query_membership_does_not(self):
        initial, x, y = fixture()
        config = [c for c in candidate_grid() if c['learning_rate'] == .001
            and c['steps'] == 20 and c['source_replay_weight'] == 0 and c['proximal_lambda'] == 0]
        a, _, _ = adapt_paths(initial, x, y, x[:6], config, support_x=x[:7], support_y=y[:7])
        b, _, _ = adapt_paths(initial, x, y, x[:1], config, support_x=x[:7], support_y=y[:7])
        changed, _, _ = adapt_paths(initial, x, y, x[:6], config,
            support_x=x[:7], support_y=(y[:7]+1) % 7)
        key = config[0]['id']
        assert_allclose(a[key][:1], b[key], atol=1e-7)
        self.assertGreater(float(np.abs(changed[key]-a[key]).max()), 1e-5)

    def test_anchoring_penalty_has_an_effect_and_invalid_support_is_rejected(self):
        initial, x, y = fixture()
        configs = [c for c in candidate_grid() if c['learning_rate'] == .001
            and c['steps'] == 20 and c['source_replay_weight'] == 0]
        scores, _, _ = adapt_paths(initial, x, y, x[:4], configs, support_x=x[:7], support_y=y[:7])
        self.assertGreater(float(np.abs(scores[configs[0]['id']]-scores[configs[1]['id']]).max()), 0.)
        with self.assertRaisesRegex(ValueError, 'every known class'):
            adapt_paths(initial, x, y, x[:4], configs, support_x=x[:6], support_y=y[:6])

    def test_physical_role_overlap_and_final_truth_leak_are_rejected(self):
        context = {name: {'record_ids': np.asarray([name+'1', name+'2']),
            'X': np.ones((2, 8))} for name in ('source', 'support_pool', 'prediction')}
        self.assertEqual(verify_context(context, final=True)['source'], 2)
        wrong = deepcopy(context)
        wrong['prediction']['record_ids'][0] = 'source1'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            verify_context(wrong)
        wrong = deepcopy(context); wrong['prediction']['y'] = np.arange(2)
        with self.assertRaisesRegex(ValueError, 'truth'):
            verify_context(wrong, final=True)

    def test_selection_requires_gains_over_both_controls_and_independent_gate(self):
        config = candidate_grid()[0]
        policy = {'minimum_accuracy_gain': .001, 'max_macro_f1_loss': .005,
                  'max_balanced_accuracy_loss': .005}
        def rows(actual, source, matched, gate=None):
            result = []
            for stage in ('selection', 'gate'):
                target = actual if stage == 'selection' or gate is None else gate
                result.append({'candidate_id': config['id'], 'stage': stage,
                    **{k: target for k in ('accuracy', 'balanced_accuracy', 'macro_f1')},
                    **{'source_'+k: source for k in ('accuracy', 'balanced_accuracy', 'macro_f1')},
                    **{'matched_'+k: matched for k in ('accuracy', 'balanced_accuracy', 'macro_f1')}})
            return result
        self.assertIsNone(choose_candidate(rows(.75, .8, .7), [config], policy)['selected'])
        self.assertIsNone(choose_candidate(rows(.82, .8, .81, gate=.80), [config], policy)['selected'])
        self.assertEqual(choose_candidate(rows(.82, .8, .81), [config], policy)['selected'], config)

    def test_encoded_and_named_labels_share_the_fixed_vocabulary(self):
        classes = np.asarray(['ZY01', 'ZY02', 'ZY03'])
        assert_array_equal(index_labels(classes, classes), np.arange(3))
        assert_array_equal(index_labels(np.arange(3), classes), np.arange(3))
        with self.assertRaises(ValueError):
            index_labels(np.asarray([3]), classes)


if __name__ == '__main__':
    unittest.main()
