"""Streaming ledger equivalence and delayed/hidden-reference isolation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from shuwei_c.q1v3_online import StreamingBallState
from shuwei_c.q1v3_feedback import adapt_radii
from shuwei_c.q1v3_calibration import mature_residual_features


def test_online_matches_batch_issued_radii_and_matured_features():
    time=np.arange(24)/30;errors=np.c_[1+np.sin(time*7),.7+np.cos(time*9)**2]
    frame=pd.DataFrame([dict(episode='x',hand=h,origin_s=t,target_s=t+.1,gate_normalized_error=errors[i,h]) for i,t in enumerate(time) for h in range(2)])
    base=np.tile([.4,.8,1.2,1.7],(len(frame),1));radii,_=adapt_radii(frame,base,[.5,.8,.9,.95],.01);mature=mature_residual_features(frame)
    online=StreamingBallState(1.);pending=[]
    for i,t in enumerate(time):
        arrived=[item for item in pending if item[1]<=t+1e-9]
        for key,target,j in arrived:
            position=np.c_[errors[j],np.zeros((2,2))];online.observe(key,target,position,[True,True]);pending.remove((key,target,j))
        features=online.features(t)
        for field,values in features.items():np.testing.assert_allclose(values,mature.loc[[2*i,2*i+1],field],atol=1e-12,equal_nan=True)
        key,r=online.issue(t,t+.1,np.zeros((2,3)),base[[2*i,2*i+1]],[False,False]);np.testing.assert_allclose(r,radii[[2*i,2*i+1]],atol=1e-12)
        pending.append((key,t+.1,i))


def test_delayed_or_unqualified_reference_cannot_change_an_earlier_prediction():
    state=StreamingBallState(1.);key,_=state.issue(0.,.1,np.zeros((2,3)),np.ones((2,4)),[False,False])
    assert not state.features(.15)['past_residual_known'].any()
    state.observe(key,.1,np.ones((2,3)),[False,True],arrived_at=.2)
    np.testing.assert_array_equal(state.features(.2)['past_residual_known'],[0,1])
    with pytest.raises(ValueError,match='earlier time'):state.features(.15)
