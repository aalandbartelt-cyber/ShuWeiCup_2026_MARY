from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from shuwei_c.q2_replay import History
from shuwei_c.q2_state import StateTracker


def test_velocity_uses_archived_forecasts_until_qualified_sample_arrives():
    centres=np.zeros((2,2,3));centres[1,:,0]=.1
    points=np.repeat(centres[:,:,None,:],21,axis=2)
    history=History(np.array([0.,.1]),points,np.zeros((2,2),bool))
    predictor=lambda h,t:np.tile([2*t-.1,0.,0.],(2,1))
    tracker=StateTracker(predictor,lag=.1)
    tracker(history,.2);tracker(history,.3)
    np.testing.assert_allclose(tracker.states[-1]['state'],[.5,0,0,2,0,0,.5,0,0,2,0,0])
    assert tracker.states[-1]['velocity_support']==('archived_forecast','archived_forecast')
    received=np.concatenate([points,np.full((1,2,21,3),0.)]);received[-1,:,:,0]=.45
    new=History(np.array([0.,.1,.3]),received,np.zeros((3,2),bool))
    tracker(new,.4)
    np.testing.assert_allclose(tracker.states[-1]['state'][3:6],[2.5,0,0])
    assert tracker.states[-1]['velocity_support']==('received','received')


def test_quality_fallback_uses_each_hands_actual_support_time():
    centres=np.zeros((3,2,3));centres[1,:,0]=.5;centres[2,:,0]=.9
    points=np.repeat(centres[:,:,None,:],21,axis=2)
    history=History(np.array([0.,.5,.9]),points,np.array([[False,False],[True,False],[True,True]]))
    tracker=StateTracker(lambda h,t:np.tile([1.,0.,0.],(2,1)),lag=.1)
    tracker(history,1.)
    state=tracker.states[-1]
    np.testing.assert_array_equal(state['velocity_support_time_s'],[0.,.5])
    np.testing.assert_allclose(state['state'][[3,9]],[1.,1.])
    assert state['velocity_support']==('earlier_received_fallback','earlier_received_fallback')


def test_all_flagged_prefix_is_reported_as_flagged_fallback():
    history=History(np.array([0.,.5]),np.zeros((2,2,21,3)),np.ones((2,2),bool))
    tracker=StateTracker(lambda h,t:np.tile([1.,0.,0.],(2,1)),lag=.1)
    tracker(history,1.)
    np.testing.assert_allclose(tracker.states[-1]['state'][[3,9]],[2.,2.])
    assert tracker.states[-1]['velocity_support']==('flagged_received_fallback','flagged_received_fallback')
