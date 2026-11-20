"""Delayed feedback must use only arrived residuals and immutable issued balls."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q1v3_feedback import adapt_radii


def frame():
    t=np.arange(20)/30
    return pd.DataFrame(dict(episode='one',hand=0,origin_s=t,target_s=t+.1,
                             gate_normalized_error=np.full(len(t),2.)))


def test_feedback_cannot_see_unarrived_errors_or_future_sampling_intervals():
    f=frame();base=np.ones((len(f),2));levels=[.5,.9]
    a,_=adapt_radii(f,base,levels,.1)
    other=f.copy();other.loc[other.target_s>.3,'gate_normalized_error']=1000
    other.loc[other.origin_s>.3,'origin_s']+=10
    other.loc[other.origin_s>10,'target_s']+=10
    b,_=adapt_radii(other,base,levels,.1)
    np.testing.assert_array_equal(a[f.origin_s<=.3],b[f.origin_s<=.3])
    np.testing.assert_array_equal(a[f.origin_s<.1],base[f.origin_s<.1])


def test_feedback_uses_the_radius_actually_issued_and_preserves_nesting():
    f=frame();f=f.iloc[:6].copy();f['gate_normalized_error']=1.05
    base=np.ones((6,2));r,_=adapt_radii(f,base,[.5,.9],1.)
    # First three forecasts were issued at radius 1 and all miss, even when
    # their feedback arrives after the scale has already increased.
    np.testing.assert_allclose(r[3:,1],np.exp(np.minimum([.9,1.8,2.7],np.log(3))))
    assert np.all(np.diff(r,axis=1)>=0)
    crossed=np.tile([2.,1.],(6,1));r,_=adapt_radii(f,crossed,[.5,.9],0.)
    np.testing.assert_array_equal(r,np.full((6,2),2.))
