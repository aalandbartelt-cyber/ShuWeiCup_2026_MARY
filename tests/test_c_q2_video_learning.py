from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from shuwei_c.q2_video_provider import AsOfVideoFeatures
from shuwei_c.q2_video_learning import snapshots, paired_features, permute_content


def provider(future=3.):
    return AsOfVideoFeatures({'head_left': dict(source_time=[0., .1, .2],
        available_time=[.05, .15, .25], features=[[1.,2.],[2.,4.],[future,6.]])},
        names=('one','two'), views=('head_left','head_right'))


def test_vectorized_asof_matches_scalar_including_boundaries():
    p = provider(); times = np.array([-.1,.049999,.05,.15,.2499,.25,200.])
    actual = snapshots(p,times).reshape(len(times),-1)
    expected = np.stack([p.at(t)['features'] for t in times])
    np.testing.assert_array_equal(actual, expected)


def test_paired_changes_require_both_available_and_ignore_future():
    p=provider(); times=np.array([.1,.2,.24]); old=np.array([-.1,.1,.2])
    a,names=paired_features(p,times,old);b,_=paired_features(provider(999),times,old)
    np.testing.assert_array_equal(a,b)
    assert a[0,names.index('video_head_left_pair_available_value')]==0
    assert a[0,names.index('video_head_left_change_one_value')]==0
    assert a[1,names.index('video_head_left_change_one_value')]==1
    assert a[2,names.index('video_head_left_change_one_value')]==0
    assert a[2,names.index('video_head_left_source_gap_value')]==0
    assert not a[:,len(names)//2:].any()
    with pytest.raises(ValueError,match='future'):
        paired_features(p,[.1],[.2])


def test_tiny_future_rounding_is_clipped_and_control_preserves_masks():
    p=provider()
    a,_=paired_features(p,[.15-2e-12],[.15])
    b,_=paired_features(p,[.15-2e-12],[.15-2e-12])
    np.testing.assert_array_equal(a,b)
    now=np.array([.01,.1,.1,.2,.24,.3,.35]);old=np.array([0.,0.,.1,.1,.2,.2,.3])
    x,names=paired_features(p,now,old);y=permute_content(x,p)
    for j,name in enumerate(names):
        if any(tag in name for tag in ('available','source_gap','frame_age','processing_delay')):
            np.testing.assert_array_equal(x[:,j],y[:,j])
    pairs=y[:,names.index('video_head_left_pair_available_value')]>0
    gaps=y[:,names.index('video_head_left_source_gap_value')]>0
    assert not y[~(pairs&gaps),names.index('video_head_left_change_one_value')].any()
    assert not y[:,len(names)//2:].any()
