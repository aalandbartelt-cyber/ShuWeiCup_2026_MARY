"""Protocol tests using synthetic records; no competition data or GPU required.

AI: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
"""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from shuwei_c.q3 import Job,Replay,metrics_from_events,sample_sources
from shuwei_c.q2 import replay_window
from shuwei_c.data import Recording,balanced_weights


@pytest.fixture
def rules():
    return dict(initial_zone='Z0',max_actions=200,observation=dict(prefix_seconds=[.2,.6,1.],delay_s=1.,cost=1.,max_per_job=2),
        service=dict(reserve_min_s=.2),movement=dict(speed=1.,changeover_s=.5),
        fees=dict(service_fixed=2.,reserve_per_second=.1,distance_per_unit=.2))


def make_replay(rules,duration=2.,jobs=None,horizon=10.,budget=10.):
    jobs=jobs or [Job('J','P',0,5,1,'Z0')]
    source={j.job_id:dict(duration_s=duration,unit_id='SECRET',source_segment_index=99) for j in jobs}
    return Replay(jobs,source,lambda u,h:{'length':h},dict(Z0=(0,0),Z1=(1,0)),
        dict(horizon_s=horizon,cost_budget=budget,observation_limit=3),rules)


def test_minimum_reservation_overrun_and_ledger(rules):
    replay=make_replay(rules)
    replay.step(dict(type='execute',job='J',reserve=.2));replay.step(dict(type='stop'))
    m=replay.metrics()
    assert m['elapsed_s']==3. # .5 changeover + (2+.5) service
    assert m['observed_overrun_s']==pytest.approx(2.3)
    assert m['total_cost']==pytest.approx(2.02)
    assert m['weighted_on_time_rate']==1
    assert m==metrics_from_events(tuple(replay.jobs.values()),10,replay.events)


def test_early_completion_does_not_release_reservation(rules):
    replay=make_replay(rules,duration=1.)
    replay.step(dict(type='execute',job='J',reserve=4.))
    assert replay.time==4.5
    assert replay.completed['J']==2.
    assert replay.metrics()['reserved_idle_s']==2.5


def test_hidden_overrun_is_accepted_then_censored(rules):
    jobs=[Job('J','P',0,10,2,'Z0'),Job('K','A',0,4,1,'Z0')]
    replay=make_replay(rules,100,jobs)
    replay.step(dict(type='execute',job='J',reserve=.2))
    assert replay.ended and replay.invalid==0
    m=replay.metrics()
    assert m['censored_units']==1 and m['weighted_on_time_rate']==0
    assert m['weighted_tardiness_s']==6 # Unattempted K, not normalized by W.
    assert m['observed_overrun_s']==pytest.approx(9.3)


def test_review_delay_and_predecessor_not_required(rules):
    jobs=[Job('J','P',0,5,1,'Z0'),Job('K','A',0,7,1,'Z1',('J',))]
    replay=make_replay(rules,jobs=jobs)
    assert replay.snapshot()['prefixes']['K']['length']==.2
    replay.step(dict(type='review',job='K'))
    assert replay.time==1 and replay.cost==1
    assert replay.snapshot()['prefixes']['K']['length']==.6
    replay.step(dict(type='execute',job='K',reserve=.2))
    assert replay.invalid==1 and replay.time==1 and replay.cost==1


def test_releases_and_nonpositive_wait_rejected(rules):
    replay=make_replay(rules,jobs=[Job('J','P',4,9,1,'Z0')])
    assert replay.snapshot()['prefixes']=={}
    replay.step(dict(type='review',job='J'))
    assert replay.invalid==1
    for seconds in [0,-1,float('nan'),11]:
        r=make_replay(rules);r.step(dict(type='wait',seconds=seconds));assert r.invalid==1


def test_snapshot_independent_of_hidden_future_and_ids(rules):
    a=make_replay(rules,duration=2.);b=make_replay(rules,duration=200.)
    assert a.snapshot()==b.snapshot()
    assert 'SECRET' not in str(a.snapshot())
    with pytest.raises(RuntimeError):a.source_audit()
    snap=a.snapshot();snap['rules']['fees']['service_fixed']=0
    assert a.rules['fees']['service_fixed']==2


def test_boundary_completion_and_review_budget(rules):
    replay=make_replay(rules,duration=9.,jobs=[Job('J','P',0,10,1,'Z0')])
    replay.step(dict(type='execute',job='J',reserve=.2))
    assert replay.completed['J']==10 and replay.metrics()['weighted_on_time_rate']==1
    r=make_replay(rules,budget=.9);r.step(dict(type='review',job='J'))
    assert r.invalid==1 and r.cost==0


def test_raw_prefix_no_interpolation_and_future_invariance():
    def rec(values):
        return Recording('private','EgoDemo','private','private','train',Path('.'),
                         {'left':dict(time=np.array([0.,.15,.3]),position=np.array(values))})
    a=rec([1.,2.,3.]).prefix(0,.2);b=rec([1.,2.,900.]).prefix(0,.2)
    np.testing.assert_array_equal(a['left']['position'],[1,2])
    np.testing.assert_array_equal(a['left']['position'],b['left']['position'])
    assert not a['left']['position'].flags.writeable


def test_q2_hidden_values_invariant_until_response():
    t=np.arange(0,1.1,.1);y=np.zeros((len(t),2,3));bad=np.zeros((len(t),2),bool)
    initial=np.zeros((2,2,3));cov=np.tile(np.eye(2),(2,1,1))
    noise=dict(r=.1,acceleration=1.,hand_scale=.1)
    a=replay_window(t,y,bad,2,10,initial,cov,noise,1,.3)
    changed=y.copy();changed[2:]=999
    b=replay_window(t,changed,bad,2,10,initial,cov,noise,1,.3)
    response=a['request_log'][0]['return_time']
    before=t[2:10]<response-1e-9
    np.testing.assert_array_equal(a['predictions'][before],b['predictions'][before])
    assert not np.array_equal(a['predictions'],b['predictions'])


def test_q2_late_return_never_updates_and_cap_is_respected():
    t=np.arange(0,1.1,.1);y=np.ones((len(t),2,3));bad=np.zeros((len(t),2),bool)
    state=np.zeros((2,2,3));cov=np.tile(np.eye(2),(2,1,1));noise=dict(r=.1,acceleration=1.,hand_scale=.1)
    a=replay_window(t,y,bad,2,6,state,cov,noise,4,2.)
    assert a['responses_within_window']==0 and a['requests']<=4
    np.testing.assert_array_equal(a['predictions'],0)


def test_balancing_not_frame_count():
    w=balanced_weights(['a','a','a','b'],['one','one','two','three'])
    np.testing.assert_allclose(w,[.125,.125,.25,.5])


def test_sampling_uses_csv_order_and_python_random():
    import random
    class Data:
        def eligible(self,split,family):return [dict(id=5),dict(id=1),dict(id=8)]
    jobs=[Job('b','P',0,1,1,'Z0'),Job('a','P',0,1,1,'Z0')]
    result=sample_sources(Data(),jobs,'validation',123)
    rng=random.Random(123)
    assert result=={j:rng.choice(Data().eligible('validation','P')) for j in ['a','b']}
