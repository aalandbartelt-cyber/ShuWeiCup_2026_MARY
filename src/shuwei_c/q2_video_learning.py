"""Image-only changes since the latest disclosed numerical observation.

The camera views are not anatomical hand labels. Each numerical hand uses the
same unordered image descriptors, referenced to its own disclosed source time.
No pose projection, future frame, or numerical reference enters this adapter.
"""
import numpy as np


def snapshots(provider, query_times):
    """Vectorized equivalent of the verified provider's scalar as-of lookup."""
    times = np.asarray(query_times, float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError('Expected finite one-dimensional query times')
    width = 3+len(provider.names)
    result = np.zeros((len(times), len(provider.views), width), np.float32)
    for column, view in enumerate(provider.views):
        if view not in provider.streams:
            continue
        stream = provider.streams[view]
        index = np.searchsorted(stream['available_time'], times, side='right')-1
        valid = index >= 0
        ix = index[valid]
        result[valid, column, 0] = 1
        result[valid, column, 1] = times[valid]-stream['source_time'][ix]
        result[valid, column, 2] = stream['available_time'][ix]-stream['source_time'][ix]
        result[valid, column, 3:] = stream['features'][ix]
    return result


def paired_features(provider, now, observed_at):
    now, observed_at = np.broadcast_arrays(np.asarray(now, float), np.asarray(observed_at, float))
    if np.any(observed_at > now+1e-10):
        raise ValueError('Numerical observation source cannot be in the future')
    observed_at = np.minimum(observed_at, now)
    current, previous = snapshots(provider, now), snapshots(provider, observed_at)
    paired = current[:, :, 0]*previous[:, :, 0]
    # Compute source times directly in float64; float32 frame ages lose precision
    # on long recordings and would introduce a false image motion signal.
    source_gap = np.zeros_like(paired)
    for j, view in enumerate(provider.views):
        if view not in provider.streams:
            continue
        stream = provider.streams[view]
        a = np.searchsorted(stream['available_time'], now, side='right')-1
        b = np.searchsorted(stream['available_time'], observed_at, side='right')-1
        known = (a >= 0) & (b >= 0)
        source_gap[known, j] = stream['source_time'][a[known]]-stream['source_time'][b[known]]
    delta = (current[:, :, 3:]-previous[:, :, 3:])*paired[:, :, None]
    value = np.concatenate([current, paired[:, :, None], source_gap[:, :, None], delta], axis=2)
    labels = ('available', 'frame_age', 'processing_delay')+tuple(provider.names)
    names = [f'video_{view}_{label}_value' for view in provider.views
             for label in tuple('current_'+x for x in labels)+('pair_available', 'source_gap')+
             tuple('change_'+x for x in provider.names)]
    return value.reshape(len(now), -1), names


def permute_content(features, provider, seed=20261120):
    """Shuffle content only within physically compatible availability groups."""
    result = features.copy()
    width = 5+2*len(provider.names)
    rng = np.random.default_rng(seed)
    for j in range(len(provider.views)):
        block = result[:,j*width:(j+1)*width]
        current = block[:,0]>0
        paired = block[:,3+len(provider.names)]>0
        different = block[:,4+len(provider.names)]>0
        key = current.astype(int)+2*paired+4*different
        columns = list(range(3,3+len(provider.names)))+list(range(5+len(provider.names),width))
        for value in np.unique(key):
            indices = np.flatnonzero(key==value)
            order = rng.permutation(indices)
            block[np.ix_(indices,columns)] = block[np.ix_(order,columns)]
    return result


def augment(pack, cache, time_mode='nominal', extra_delay=0., control='aligned', views=None):
    """Add legal video features without changing numerical rows or targets.

    Permutation is an offline negative control only: it is never deployed as a
    causal feature provider. It preserves availability/age bookkeeping while
    breaking the connection between image content and the current state.
    """
    if control not in ('aligned', 'permuted'):
        raise ValueError('Unknown video alignment control')
    matrix, names = None, None
    for episode, group in pack['meta'].groupby('episode', sort=False):
        options = {} if views is None else {'views': tuple(views)}
        provider = cache.for_episode(episode, time_mode=time_mode, processing_delay_s=extra_delay, **options)
        now = group.time_s.to_numpy()
        features, names = paired_features(provider, now, now-group.age_s.to_numpy())
        if control == 'permuted':
            features = permute_content(features, provider)
        if matrix is None:
            matrix = np.empty((len(pack['meta']), features.shape[1]), np.float32)
        matrix[group.index.to_numpy()] = features
    if matrix is None:
        raise ValueError('Cannot augment an empty pack')
    result = dict(pack)
    result['features'] = np.concatenate([pack['features'], matrix], axis=1)
    result['feature_names'] = list(pack['feature_names'])+names
    return result
