"""Finite-difference checks of pulse-optimization adjoints."""
import unittest
import numpy as np
from ..models.grape import grape_eval
from ..models.phase_control import propagate
from ..scripts.stateprep import objective

class PulseGradientTests(unittest.TestCase):
    def check_gradient(self,fun,x):
        _,g=fun(x)
        for k in (0,len(x)//2,len(x)-1):
            delta=np.zeros_like(x);delta[k]=1e-6
            fd=(fun(x+delta)[0]-fun(x-delta)[0])/2e-6
            self.assertAlmostEqual(fd,g[k],delta=2e-7)

    def test_coherent_and_measurement_grape(self):
        x=np.random.default_rng(23).normal(0,.1,12)
        for measurement in (False,True):
            with self.subTest(measurement=measurement):
                self.check_gradient(lambda x:grape_eval(x,.17,3,8,measurement=measurement),x)

    def test_preparation_grape(self):
        x=np.random.default_rng(24).normal(0,.1,12)
        self.check_gradient(lambda x:objective(x,.17,8),x)

    def test_joint_phase_grape(self):
        x=np.random.default_rng(25).normal(0,.1,8)
        self.check_gradient(lambda x:propagate(x,.12,np.array([-.4,1.2,3.3]),True,np.array([.3,-.1,.5])),x)
