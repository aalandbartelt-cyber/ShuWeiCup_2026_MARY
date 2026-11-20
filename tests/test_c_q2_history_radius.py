from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q2_history_radius import HistoryBinnedRadius,scale_path_radius


def test_equal_age_different_history_gaps_have_distinct_radii():
    meta=pd.DataFrame(dict(environment=np.repeat(['a','b','c'],4),episode=np.repeat(['A','B','C'],4),
                           window_id=np.repeat(['w1','w2','w3'],4),hand=np.tile([0,1,0,1],3)))
    age=np.full(12,.1);gap=np.tile([.03,.03,.3,.3],3);speed=np.ones(12);error=np.where(gap<.1,1.,4.)
    for rate in (False,True):
        model=HistoryBinnedRadius(levels=(.8,.9),rate=rate).fit(age,gap,speed,error,meta)
        out=model.predict(age,gap,speed)
        np.testing.assert_allclose(out,np.repeat(error[:,None],2,axis=1))
        assert not model.predict(np.zeros(12),gap,speed).any()


def test_unsupported_history_cell_backs_off_to_age_cell():
    meta=pd.DataFrame(dict(environment=['a','b','c'],episode=['A','B','C'],window_id=['w1','w2','w3'],hand=[0,0,0]))
    age=np.full(3,.1);gap=np.array([.03,.03,.3]);speed=np.ones(3);error=np.array([1.,1.,4.])
    model=HistoryBinnedRadius(levels=(.9,)).fit(age,gap,speed,error,meta)
    np.testing.assert_array_equal(model.predict(age,gap,speed),np.full((3,1),4.))


def test_joint_path_floor_matches_score_and_retains_direct_observations():
    # Positive error with a zero base must retain the scale used to fit its score.
    radii=np.zeros((2,2));age=np.array([0.,.2]);factors=np.array([2e5,3e5])
    np.testing.assert_allclose(scale_path_radius(radii,factors,age),[[0.,0.],[2.,3.]])


def test_history_radius_serialization_keeps_predictions(tmp_path):
    meta=pd.DataFrame(dict(environment=['a','b','c'],episode=['A','B','C'],window_id=['w1','w2','w3'],hand=[0,0,0]))
    a=np.full(3,.1);g=np.array([.03,.03,.3]);s=np.ones(3);e=np.array([1.,1.,4.])
    model=HistoryBinnedRadius(rate=True,motion=True).fit(a,g,s,e,meta)
    file=tmp_path/'model.json';model.save(file)
    np.testing.assert_array_equal(model.predict(a,g,s),HistoryBinnedRadius.load(file).predict(a,g,s))
