"""Synthetic unit fixtures for input derivatives, never real study samples."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from herb_spectra.leakage_guard import FitScope,LeakageError


SPEC=importlib.util.spec_from_file_location("source_attribution_script",Path(__file__).resolve().parents[1]/"scripts/analyze_upgrade_source_attributions.py")
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class SourceAttributionTests(unittest.TestCase):
    def test_known_linear_input_derivative_and_existing_parameter_grad_are_preserved(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layer=torch.nn.Linear(3,2,bias=False)
                with torch.no_grad():self.layer.weight.copy_(torch.tensor([[1.,2.,3.],[-1.,-2.,-3.]]))
                self.layer.weight.grad=torch.ones_like(self.layer.weight)*7
            def forward(self,x):return self.layer(x["alpha"])
        model=Linear();X=np.array([[1.,2.,3.],[-1.,-2.,-3.]],dtype=np.float32)
        result=MODULE.source_alpha_attribution(model,{"alpha":X},["s0","s1"],FitScope(["s0","s1"],"source","fixture"))
        expected_gradient=np.array([[1.,2.,3.],[-1.,-2.,-3.]],dtype=np.float32)
        np.testing.assert_array_equal(result["input_gradient"],expected_gradient)
        np.testing.assert_array_equal(result["absolute_input_times_gradient"],np.abs(X*expected_gradient))
        self.assertTrue(torch.equal(model.layer.weight.grad,torch.ones_like(model.layer.weight)*7))
        self.assertEqual(result["state_sha256_before"],result["state_sha256_after"])

    def test_batchnorm_buffers_and_none_parameter_grads_unchanged(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bn=torch.nn.BatchNorm1d(3);self.head=torch.nn.Linear(3,2)
            def forward(self,x):return self.head(self.bn(x["alpha"]))
        torch.manual_seed(42);model=Net();prior={k:v.clone() for k,v in model.state_dict().items()}
        MODULE.source_alpha_attribution(model,{"alpha":np.ones((2,3),dtype=np.float32)},["s0","s1"],FitScope(["s0","s1"],"source","fixture"))
        self.assertTrue(all(torch.equal(v,prior[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(model.training)

    def test_query_identity_rejected_before_any_model_forward(self):
        class Spy(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,x):self.calls+=1;return x["alpha"]
        model=Spy()
        with self.assertRaises(LeakageError):
            MODULE.source_alpha_attribution(model,{"alpha":np.ones((2,3))},["s0","query"],FitScope(["s0","s1"],"source","fixture"))
        self.assertEqual(model.calls,0)


if __name__=="__main__":
    unittest.main()
