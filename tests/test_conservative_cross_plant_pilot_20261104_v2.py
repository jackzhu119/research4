"""Software-only checks; fixtures are not additional measured plant samples."""
import importlib.util
import json
import os
import signal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('conservative_pilot_v2',ROOT/'scripts/run_conservative_cross_plant_pilot_20261104_v2.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


class ConservativePilotTests(unittest.TestCase):
    def test_epoch_selection_uses_source_macro_then_guards_and_earliest(self):
        rows=[{'epoch':e,'macro_f1':.4,'balanced_accuracy':.5,'accuracy':.9} for e in range(1,97)]
        rows[1].update(macro_f1=.5,balanced_accuracy=.6,accuracy=.8)
        rows[2].update(macro_f1=.5,balanced_accuracy=.6,accuracy=.8)
        self.assertEqual(pilot.select_epoch(rows)['epoch'],2)
        rows[2]['balanced_accuracy']=.61
        self.assertEqual(pilot.select_epoch(rows)['epoch'],3)

    def test_missing_epoch_candidates_rejected(self):
        with self.assertRaises(ValueError):pilot.select_epoch([])

    def test_duplicate_or_shared_physical_record_identity_rejected(self):
        for source,query in [(['s','s'],['q']),(['s'],['s'])]:
            with self.assertRaises(ValueError):pilot.assert_membership(source,query)
        pilot.assert_membership(['s'],['q'])

    def test_protocol_registration_is_prior_to_training_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'new'
            pilot.register(out)
            self.assertEqual(pilot.read(out/'registration.json')['training_fits'],0)
            with self.assertRaises(FileExistsError):pilot.register(out)
            changed=pilot.read(out/'protocol.json');changed['target_N']=900
            (out/'protocol.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):pilot.supervise(out)

    def test_supervisor_forbids_unregistered_attempt_before_spawning(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):pilot.supervise(Path(directory)/'missing')

    def test_encoder_anchor_has_real_gradient_but_no_gradient_into_anchor_or_bn_buffer(self):
        import torch
        torch.set_num_threads(1)
        model=torch.nn.Sequential(torch.nn.Linear(2,2,bias=False),torch.nn.BatchNorm1d(2))
        anchor={n:v.detach().clone().requires_grad_(True) for n,v in model.named_parameters()}
        self.assertEqual(float(pilot.encoder_l2_sp(model,anchor).detach()),0.)
        with torch.no_grad():
            model[0].weight.add_(.1)
            model[1].running_mean.add_(100)
        loss=pilot.encoder_l2_sp(model,anchor);loss.backward()
        self.assertTrue(torch.allclose(model[0].weight.grad,torch.full_like(model[0].weight,.2)))
        self.assertTrue(all(v.grad is None for v in anchor.values()))
        self.assertAlmostEqual(float(loss.detach()),.04,places=6)
        with self.assertRaises(ValueError):pilot.encoder_l2_sp(model,{})

    def test_process_exit_race_reaps_without_fabricated_monitor_failure(self):
        child=SimpleNamespace(pid=123,returncode=None);now=[0.];calls=[0];signals=[]
        usage=SimpleNamespace(ru_utime=.2,ru_stime=.1)
        def waiter(*_):
            calls[0]+=1
            return (123,0,usage) if calls[0]==5 else (0,0,None)
        def missing():raise FileNotFoundError('owned child exiting')
        def sleep(seconds):now[0]+=seconds
        found=pilot.monitor_owned_child(child,0.,0.,waiter=waiter,reader=missing,clock=lambda:now[0],sleeper=sleep,sender=signals.append)
        self.assertEqual(found['exit_code'],0);self.assertIsNone(found['stop_reason']);self.assertEqual(signals,[])
        self.assertIs(found['usage'],usage)

    def test_unreadable_hung_child_stops_bounded_and_retains_unknown_cost(self):
        now=[0.];signals=[]
        def denied():raise PermissionError('simulated monitor denial')
        def sleep(seconds):now[0]+=seconds
        found=pilot.monitor_owned_child(SimpleNamespace(pid=123),0.,17.,waiter=lambda *_:(0,0,None),reader=denied,
            clock=lambda:now[0],sleeper=sleep,sender=signals.append)
        self.assertIsNone(found['usage']);self.assertIsNone(found['exit_code'])
        self.assertEqual(found['stop_reason'],'UNRESOLVED_WAIT4_AFTER_BOUNDED_KILL')
        self.assertGreaterEqual(found['actual_CPU_seconds_lower_bound'],17.)
        self.assertLess(now[0],3.5);self.assertEqual(signals,[signal.SIGTERM,signal.SIGKILL])


if __name__=='__main__':unittest.main()
