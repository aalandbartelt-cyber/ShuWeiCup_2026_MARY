"""Causal and permutation-invariance tests for Q2 image-only features."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from shuwei_c.q2_video import ImageFeatures, VideoProvider, FEATURE_NAMES


def detection(x=.4, y=.4, label='Left'):
    points = np.c_[x+np.linspace(-.05, .05, 21), y+np.sin(np.arange(21))*.05, np.zeros(21)]
    return dict(landmarks=points.tolist(), handedness_score=.7, label=label)


def frame(seed):
    return np.random.default_rng(seed).integers(0, 256, (64, 96, 3), np.uint8)


def test_feature_schema_and_missing_detection():
    extractor = ImageFeatures(64)
    x = extractor.step(frame(1), 0., [])
    assert x.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(x).all()
    assert x[FEATURE_NAMES.index('flow_known')] == 0
    y = extractor.step(frame(2), .1, [])
    assert y[FEATURE_NAMES.index('flow_known')] == 1
    assert y[FEATURE_NAMES.index('hand_region_known')] == 0


def test_detection_order_and_labels_do_not_change_unordered_features():
    a, b = ImageFeatures(64), ImageFeatures(64)
    for i in range(3):
        detections = [detection(.2+i*.01), detection(.7-i*.01, label='Right')]
        shuffled = [dict(x, label='arbitrary') for x in reversed(detections)]
        np.testing.assert_allclose(a.step(frame(i), i*.1, detections),
                                   b.step(frame(i), i*.1, shuffled), rtol=0, atol=1e-7)


def test_future_frames_cannot_revise_already_emitted_features():
    a, b = ImageFeatures(64), ImageFeatures(64)
    prefix_a, prefix_b = [], []
    for i in range(5):
        prefix_a.append(a.step(frame(i), i*.1, [detection(.3+i*.01)]))
        prefix_b.append(b.step(frame(i if i < 3 else i+100), i*.1,
                               [detection(.3+i*.01)] if i < 3 else []))
    np.testing.assert_array_equal(prefix_a[:3], prefix_b[:3])


def stream():
    return dict(source_time=np.array([0., .1, .2]), available_time=np.array([.05, .15, .25]),
                features=np.arange(3*len(FEATURE_NAMES), dtype=np.float32).reshape(3, -1))


def test_provider_honours_availability_not_source_time():
    provider = VideoProvider({'head_left': stream()}, views=('head_left',))
    assert provider.at(.04)['features'][0] == 0
    x = provider.at(.12)
    assert x['views']['head_left']['source_time'] == 0
    assert x['views']['head_left']['available_time'] == .05
    assert x['features'][1] == pytest.approx(.12)
    assert not x['features'].flags.writeable
    assert 'episode' not in x and 'source' not in x
    np.testing.assert_array_equal(provider.at(.15)['features'][3:], stream()['features'][1])


def test_provider_future_cache_perturbation_and_lagged_features():
    a, b = stream(), stream()
    b['features'][1:] = -999
    p, q = VideoProvider({'head_left': a}), VideoProvider({'head_left': b})
    np.testing.assert_array_equal(p.vector(.14, (0., .1)), q.vector(.14, (0., .1)))
    assert not np.array_equal(p.vector(.16), q.vector(.16))


def test_missing_views_are_zero_and_latency_sensitivity_is_explicit():
    provider = VideoProvider({'head_left': stream()}, processing_delay_s=.1)
    x = provider.at(.14)['features']
    assert not x.any()
    x = provider.at(.16)
    assert x['features'][0] == 1
    assert x['views']['head_left']['available_time'] == pytest.approx(.15)
    assert not x['features'][3+len(FEATURE_NAMES):].any()


def test_provider_rejects_noncausal_and_nonfinite_cache():
    a = stream(); a['available_time'][1] = -.1
    with pytest.raises(ValueError, match='causal'):
        VideoProvider({'head_left': a})
    b = stream(); b['features'][0, 0] = np.nan
    with pytest.raises(ValueError, match='causal'):
        VideoProvider({'head_left': b})
    with pytest.raises(ValueError):
        VideoProvider({}, processing_delay_s=-.1)


def test_repeated_or_backward_source_times_rejected():
    x = ImageFeatures(64); x.step(frame(1), .1, [])
    with pytest.raises(ValueError, match='increasing'):
        x.step(frame(2), .1, [])


def write_verified_fixture(tmp_path, kind='image'):
    import json, hashlib
    from shuwei_c.q2_video_provider import TRACK_FEATURE_NAMES
    root=tmp_path/'cache'; root.mkdir()
    names=FEATURE_NAMES if kind=='image' else TRACK_FEATURE_NAMES
    source=[]; metadata=[]
    for view in ('head_left','head_right'):
        item=dict(episode='one',view=view,split='train',sampled_frames=24 if kind=='tracker' else 2)
        source.append(item)
        path=root/f'one__{view}.npz'
        service_key='inference_s' if kind=='tracker' else 'feature_compute_s'
        np.savez(path,source_time=[1.,2.],available_time=[1.,2.],features=np.ones((2,len(names))),
                 feature_names=np.array(names),**{service_key:np.array([.75,.75])})
        metadata.append(dict(episode='one',view=view,frames=item['sampled_frames'],
                             inputs=dict(source_manifest=item),output_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (tmp_path/'source.json').write_text(json.dumps(source))
    (root/'manifest.json').write_text(json.dumps(dict(complete_development_set=True,metadata=metadata)))
    return root,tmp_path/'source.json'


def test_verified_cache_distinguishes_missing_physical_views_from_incomplete_cache(tmp_path):
    from shuwei_c.q2_video_provider import VerifiedVideoCache
    root,source=write_verified_fixture(tmp_path)
    cache=VerifiedVideoCache(root,source_manifest=source)
    provider=cache.for_episode('one')
    assert set(provider.at(1.1)['views'])=={'head_left','head_right'}
    (root/'one__head_right.npz').unlink()
    with pytest.raises(ValueError,match='Expected physical view'):
        cache.for_episode('one')


def test_verified_cache_rejects_false_complete_inventory_and_hash_changes(tmp_path):
    import json
    from shuwei_c.q2_video_provider import VerifiedVideoCache
    root,source=write_verified_fixture(tmp_path)
    cache=VerifiedVideoCache(root,source_manifest=source)
    (root/'one__head_right.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        cache.for_episode('one')
    manifest=json.loads((root/'manifest.json').read_text()); manifest['metadata'].pop()
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='inventory disagree'):
        VerifiedVideoCache(root,source_manifest=source)


def test_tracker_views_share_one_gpu_queue_when_requested(tmp_path):
    from shuwei_c.q2_video_provider import VerifiedVideoCache
    root,source=write_verified_fixture(tmp_path,'tracker')
    cache=VerifiedVideoCache(root,kind='tracker',source_manifest=source)
    separate=cache.for_episode('one',time_mode='measured')
    shared=cache.for_episode('one',time_mode='serial_gpu')
    assert len(separate.at(1.8)['views'])==2
    assert len(shared.at(1.8)['views'])==1
    assert len(shared.at(2.6)['views'])==2
    assert shared.at(3.3)['views']['head_left']['source_time']==2.
    assert shared.at(3.3)['views']['head_right']['source_time']==1.


def test_verified_loader_requires_finished_manifest(tmp_path):
    from shuwei_c.q2_video_provider import VerifiedVideoCache
    with pytest.raises(ValueError,match='manifest is absent'):
        VerifiedVideoCache(tmp_path)
