from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from types import SimpleNamespace
import numpy as np
import pytest
from shuwei_c.q2_age_schedule import equal_age_times,continuous_age_cost,EqualAgeSchedule


def test_fixed_delay_equal_endpoint_age_solution():
    np.testing.assert_allclose(equal_age_times(0,1,2,.1),[.3,.6])
    np.testing.assert_allclose(equal_age_times(0,.5,2,.1),[.4/3,.8/3])
    np.testing.assert_allclose(equal_age_times(5,6,1,0),[5.5])
    assert not len(equal_age_times(0,1,0,.1))
    assert not len(equal_age_times(0,.1,4,.1))
    assert not len(equal_age_times(0,.1,4,.2))


@pytest.mark.parametrize('primitive',[lambda t:t,lambda t:t*t/2,lambda t:t**3/3,lambda t:max(t-.3,0.)])
def test_convex_age_integrals_do_not_improve_by_random_rescheduling(primitive):
    rng=np.random.default_rng(20261120)
    for b in (1,2,4):
        for delay in (0.,.1,.4):
            optimum=continuous_age_cost(equal_age_times(0,1,b,delay),1,delay,primitive)
            for _ in range(100):
                schedule=np.sort(rng.uniform(0,1-delay,b))
                assert continuous_age_cost(schedule,1,delay,primitive)>=optimum-1e-12


def simulate(ticks,cap,delay,last=1.,single=False):
    p=EqualAgeSchedule(cap,single_pending=single);issued=[];answers=[]
    for t in ticks:
        pending=tuple(x for x in issued if x+delay>t+1e-9)
        c=SimpleNamespace(now=t,origin=0.,last_score_time=last,delay=delay,remaining=cap-len(issued),pending=pending)
        answer=p(c);answers.append(answer)
        if answer:issued.append(t)
        assert not p(c)  # Repeated policy query at one tick cannot duplicate a request.
    return issued


def test_irregular_ticks_round_forward_merge_duplicates_and_return_timely():
    ticks=[.09,.31,.32,.61,.91,1.]
    assert simulate(ticks,2,.1)==[.31,.61]
    # Nominal targets .02,.04,.06,.08 collapse onto only two available ticks.
    assert simulate([.07,.1,.16,.2],4,.1,last=.2)==[.07,.1]
    assert simulate([.2,.4,.6,.8,1.],2,.9)==[]


def test_single_pending_mapping_waits_for_real_channel():
    times=np.arange(1,11)/10
    requests=simulate(times,4,.3,single=True)
    assert len(requests)<=4
    assert all(b-a>=.3-1e-9 for a,b in zip(requests,requests[1:]))
    assert all(t+.3<=1+1e-9 for t in requests)


def test_invalid_parameters_rejected():
    for args in [(0,1,-1,.1),(0,1,1.2,.1),(0,1,1,-.1),(1,0,1,.1)]:
        with pytest.raises(ValueError):equal_age_times(*args)
    with pytest.raises(ValueError):continuous_age_cost([.2,.2],1,.1,lambda t:t*t/2)
