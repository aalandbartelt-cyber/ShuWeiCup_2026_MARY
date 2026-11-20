from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from shuwei_c.q2_learning import Gate
from shuwei_c.q2_live import SparseGate


def test_numpy_gate_matches_pytorch_mixture_and_residual():
    rng=np.random.default_rng(4)
    pack=dict(features=rng.normal(size=(9,4)).astype(np.float32),
              feature_names=['a','b','c','d'],experts=rng.normal(size=(9,3,16)).astype(np.float32))
    for kind in ('mixture','residual'):
        net=Gate(4,16,width=8,kind=kind)
        with torch.no_grad():
            net.net[-1].weight.uniform_(-.2,.2)
        model=dict(kind=kind,state_dict=net.state_dict(),feature_names=pack['feature_names'],
                   mean=np.zeros(4),std=np.ones(4),noise={'EgoDemo':{'hand_scale':.04}})
        actual=SparseGate([model],'EgoDemo').displacement(pack)
        with torch.inference_mode():
            expected,_=net(torch.tensor(pack['features']),torch.tensor(pack['experts']/.04))
        np.testing.assert_allclose(actual,expected.numpy()*.04,atol=5e-7,rtol=2e-6)


def test_zero_displacement_experts_remain_exactly_zero():
    net=Gate(3,16,kind='mixture');model=dict(kind='mixture',state_dict=net.state_dict(),
        feature_names=['a','b','c'],mean=np.zeros(3),std=np.ones(3),noise={'EgoDemo':{'hand_scale':.04}})
    pack=dict(feature_names=['a','b','c'],features=np.ones((2,3),np.float32),experts=np.zeros((2,3,16),np.float32))
    np.testing.assert_array_equal(SparseGate([model],'EgoDemo').displacement(pack),np.zeros((2,3)))
