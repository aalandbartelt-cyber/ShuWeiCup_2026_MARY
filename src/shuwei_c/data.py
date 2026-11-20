"""Canonical numerical records; official splits and raw-prefix disclosure.

AI: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
No anatomical index assumptions and no interpolation in disclosure.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd

DEFAULT_DATA = Path('00_赛题与规则/正式发布/C_Problem_Data/Restocking_Problem_and_Data_EN')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def array_column(frame, name):
    return np.stack([np.asarray(list(v), dtype=float) for v in frame[name]])


@dataclass
class Recording:
    episode: str
    dataset: str
    environment: str
    operator: str
    split: str
    folder: Path
    streams: dict

    def hands(self):
        left, right = self.streams['left'], self.streams['right']
        if not np.array_equal(left['time'], right['time']):
            raise ValueError('Hand streams need explicit alignment; refusing implicit interpolation')
        time = left['time']
        points = np.stack([left['position'], right['position']], axis=1)
        quality = self.streams['quality']
        ix = np.searchsorted(quality['time'], time, side='right') - 1
        known = ix >= 0
        bad = np.ones((len(time), 2), dtype=bool)
        bad[known] = quality['bad'][ix[known]]
        bad |= ~np.isfinite(points).all(axis=(2, 3))
        return time, points, bad

    def prefix(self, start, length):
        """Return only existing samples; independent, immutable arrays, no source IDs."""
        out = {}
        for name, stream in self.streams.items():
            take = (stream['time'] >= start - 1e-9) & (stream['time'] <= start + length + 1e-9)
            out[name] = {}
            for key, value in stream.items():
                a = np.array(value[take], copy=True)
                if key == 'time':
                    a -= start
                a.setflags(write=False)
                out[name][key] = a
        return out


class Dataset:
    def __init__(self, root=DEFAULT_DATA):
        self.root = Path(root)
        self.task_root = self.root / 'Restocking_Tasks_and_Constraints'
        self.split_rows = csv_rows(self.task_root / 'record_split.csv')
        self.units = csv_rows(self.task_root / 'source_segment_index.csv')
        self.conditions = json.loads((self.task_root / 'common_conditions.json').read_text())
        self.jobs = csv_rows(self.task_root / 'tasks.csv')
        self.zones = {r['zone_id']: (float(r['x']), float(r['y']))
                      for r in csv_rows(self.task_root / 'service_zones.csv')}
        self._split = {r['episode_uuid']: r for r in self.split_rows}
        self._folders, self._cache = {}, {}
        # Metadata is used only here, never supplied to a Q3 policy.
        for ds in ['EgoDemo', 'EgoPro', 'EgoStandard']:
            for folder in sorted((self.root / 'Raw_Data' / ds).iterdir()):
                if not folder.is_dir():
                    continue
                if ds == 'EgoDemo':
                    uid = str(pd.read_parquet(folder / 'meta/episodes/chunk-000/file-000.parquet',
                                              columns=['episode_uuid']).episode_uuid.iloc[0])
                else:
                    meta = {r['field']: r['value'] for r in csv_rows(folder / 'session_metadata.csv')}
                    uid = meta['task_info.episode_uuid']
                if uid in self._folders:
                    raise ValueError('Duplicate episode metadata')
                self._folders[uid] = folder
        if set(self._folders) != set(self._split):
            raise ValueError('Metadata and official split disagree')

    def records(self, split):
        return [self.record(r['episode_uuid']) for r in self.split_rows if r['split'] == split]

    def record(self, episode):
        if episode in self._cache:
            return self._cache[episode]
        r, folder = self._split[episode], self._folders[episode]
        streams = {}
        if r['dataset'] == 'EgoDemo':
            f = pd.read_parquet(folder / 'data/chunk-000/file-000.parquet')
            time = f.timestamp.to_numpy(float)
            for side in ['left', 'right']:
                streams[side] = dict(time=time.copy(),
                    position=array_column(f, f'observation.state.hand_{side}_world'),
                    quaternion=array_column(f, f'observation.state.hand_{side}_world_rotation'))
            for name, p, q in [
                ('head', 'head_world', 'head_world_rotation'),
                ('head_left_camera', 'head_left_camera_position', 'head_left_camera_rotation'),
                ('head_right_camera', 'head_right_camera_position', 'head_right_camera_rotation')]:
                streams[name] = dict(time=time.copy(), position=array_column(f, 'observation.state.' + p),
                                    quaternion=array_column(f, 'observation.state.' + q))
            flags = json.loads((folder / 'bad_frame_ratio.json').read_text())
            bad, reasons = np.zeros((len(f), 2), bool), np.full((len(f), 2), '', dtype=object)
            for row in flags['hand_bad_frame_details']:
                for j, side in enumerate(['left', 'right']):
                    reason = row.get(side + '_bad_reason') or ''
                    bad[int(row['frame']), j] = bool(reason)
                    reasons[int(row['frame']), j] = str(reason)
            streams['quality'] = dict(time=time.copy(), bad=bad, reasons=reasons)
        else:
            for name, file, n in [('left', 'pose_left_hand', 21), ('right', 'pose_right_hand', 21),
                                 ('head', 'pose_head', 1), ('head_left_camera', 'pose_headcam', 1),
                                 ('head_right_camera', 'pose_right_eye_cam', 1)]:
                f = pd.read_csv(folder / 'tables' / (file + '.csv'))
                streams[name] = dict(time=f['_offset_from_first_log_s'].to_numpy(float),
                    position=f[[f'transforms[{k}].pos.{c}' for k in range(n) for c in 'xyz']].to_numpy().reshape(-1,n,3),
                    quaternion=f[[f'transforms[{k}].quat.{c}' for k in range(n) for c in 'wxyz']].to_numpy().reshape(-1,n,4))
            f = pd.read_csv(folder / 'tables/annotation_bad_frame_pose_hand.csv')
            reasons = f[[c for c in f if c.startswith('problem_type[')]].fillna('').astype(str).agg('|'.join, axis=1)
            bad = np.zeros((len(f), 2), bool)
            raw_bad = f.is_bad.astype(str).str.lower().isin(['true','1']).to_numpy()
            for j, side in enumerate(['left', 'right']):
                bad[:, j] = raw_bad & reasons.str.contains(side, case=False).to_numpy()
            bad[raw_bad & ~bad.any(axis=1)] = True  # Unassigned bad flag: conservative both-hand flag.
            streams['quality'] = dict(time=f['_offset_from_first_log_s'].to_numpy(float), bad=bad,
                is_bad=raw_bad, reasons=reasons.to_numpy())
        for stream in streams.values():
            if not np.all(np.diff(stream['time']) > 0):
                raise ValueError('Non-increasing time requires an explicit audited repair')
        result = Recording(episode, r['dataset'], r['environment_id'], r['operator_id'], r['split'], folder, streams)
        self._cache[episode] = result
        return result

    def eligible(self, split, family=None):
        # Keep official CSV order and official flag, including floating threshold edge cases.
        return [r for r in self.units if r['split'] == split and r['eligible'] == '1'
                and (family is None or r['family_id'] == family)]

    def unit_prefix(self, unit, length):
        return self.record(unit['episode_uuid']).prefix(float(unit['start_s']), length)


def balanced_weights(environments, episodes):
    """Equal environments, equal episodes within environment, equal rows within episode."""
    env, ep = np.asarray(environments), np.asarray(episodes)
    result = np.empty(len(env))
    for e in np.unique(env):
        episodes_e = np.unique(ep[env == e])
        for u in episodes_e:
            mask = (env == e) & (ep == u)
            result[mask] = 1 / len(np.unique(env)) / len(episodes_e) / mask.sum()
    return result


def prefix_features(prefix, use_quality=True):
    """Low dimensional rigid-translation-invariant summaries; no source matching."""
    features, last = [], []
    for name, j in [('left', 0), ('right', 1)]:
        s = prefix[name]
        p, t = s['position'], s['time']
        good = np.isfinite(p).all(axis=(1,2))
        quality = prefix['quality']
        if len(quality['time']):
            ix = np.searchsorted(quality['time'], t, side='right') - 1
            bad = np.ones(len(t), bool)
            bad[ix >= 0] = quality['bad'][ix[ix >= 0], j]
        else:
            bad = np.ones(len(t), bool)
        good &= ~bad if use_quality else True
        c = p.mean(axis=1)
        # Cleaning is performed AFTER raw disclosure; no future interpolation.
        ix = np.flatnonzero(good)
        speed = np.linalg.norm(np.diff(c[ix], axis=0), axis=1) / np.maximum(np.diff(t[ix]), 1e-6) if len(ix)>1 else np.array([0.])
        radius = np.linalg.norm(p[ix] - c[ix,None,:], axis=-1) if len(ix) else np.array([0.])
        features += [float(np.mean(speed)), float(np.std(speed)), float(np.mean(radius)),
                     float(np.linalg.norm(c[ix[-1]]-c[ix[0]])) if len(ix)>1 else 0.,
                     float(np.mean(bad)) if use_quality and len(bad) else 0.]
        last.append(c[ix[-1]] if len(ix) else np.zeros(3))
    features += [float(np.linalg.norm(last[1]-last[0]))]
    return np.nan_to_num(features, nan=0., posinf=0., neginf=0.)
