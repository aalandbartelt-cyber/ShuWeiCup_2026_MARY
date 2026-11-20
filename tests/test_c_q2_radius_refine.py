from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q2_radius_refine import AgeRatioCorrection


def test_age_correction_uses_qualified_support_and_keeps_zero_age():
    meta=pd.DataFrame(dict(environment=np.repeat(['a','b','c'],4),episode=np.repeat(['A','B','C'],4),
                           window_id=np.repeat(['w1','w2','w3'],4),hand=np.tile([0,1,0,1],3)))
    age=np.tile([0.,.1,.3,.8],3);error=np.tile([0.,2.,3.,4.],3);radius=np.ones((12,2))
    cal=AgeRatioCorrection(levels=(.8,.9)).fit(age,error,radius,meta)
    out=cal.predict(age,radius)
    assert not out[age==0].any()
    np.testing.assert_array_equal(out[age==.1],np.full((3,2),2.))
    np.testing.assert_array_equal(out[age==.3],np.full((3,2),3.))
    assert np.all(np.diff(out,axis=1)>=0)
    assert cal.counts[0]['fallback']


def test_positive_age_zero_base_radius_can_be_corrected():
    meta=pd.DataFrame(dict(environment=['a','b','c'],episode=['A','B','C'],
                           window_id=['w1','w2','w3'],hand=[0,0,0]))
    age=np.full(3,.1);error=np.ones(3);radius=np.zeros((3,1))
    cal=AgeRatioCorrection(levels=(.9,)).fit(age,error,radius,meta)
    np.testing.assert_allclose(cal.predict(age,radius),np.ones((3,1)))
