#!/usr/bin/env python3
"""Retrospective semantic/stage analysis with within-record contrasts.

AI-assisted by Codex, 2026-11-20. Text labels and stage endpoints are used for
explanation only, never made available to the online predictors.
"""
from pathlib import Path
import hashlib,json,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.data import Dataset,balanced_weights

RULES={
 'open_close':r'\b(open|close|pull out|push in)\b',
 'pick':r'\b(pick|take|grab|grasp|remove)\b',
 'place':r'\b(put|place|set down|insert)\b',
 'transport':r'\b(move|carry|transfer|bring)\b',
 'arrange':r'\b(arrange|rearrange|adjust|flip|rotate|align|organize|organization|arrangement|neatly)\b',
 'reach':r'\b(reach|extend|approach)\b'}


def segments(rec):
    if rec.dataset=='EgoDemo':
        raw=json.loads((rec.folder/'annotation.json').read_text());task=raw['episode_description']
        return [dict(start=float(s['start_time']),end=float(s['end_time']),description=s['description'],task=task) for s in raw['subtask_annotations']]
    frame=pd.read_csv(rec.folder/'tables/annotation_semantic_segments.csv')
    return [dict(start=float(s['segment.start_time']),end=float(s['segment.end_time']),description=s['segment.subtask_description'],task=s['task_description']) for s in frame.to_dict('records')]


def main():
    out=Path('artifacts/c_q1_v2');data=Dataset();rows=[];audit=[]
    for rec in data.records('train')+data.records('validation'):
        t,p,bad=rec.hands();c=p.mean(axis=2);v=np.diff(c,axis=0)/np.diff(t)[:,None,None]
        v=np.concatenate([v[:1],v]);speed=np.linalg.norm(v,axis=-1).mean(axis=1)
        valid=~bad.any(axis=1);valid[1:] &= ~bad[:-1].any(axis=1)
        assigned=np.full(len(t),-1,int);segs=segments(rec);overlaps=0
        for n,s in enumerate(segs):
            mask=(t>=s['start']-1e-9)&(t<s['end']-1e-9)
            overlaps+=int(np.sum(mask&(assigned>=0)));assigned[mask]=n
        audit.append(dict(episode=rec.episode,environment=rec.environment,dataset=rec.dataset,split=rec.split,
            frames=len(t),covered=int((assigned>=0).sum()),overlap_assignments=overlaps,segments=len(segs)))
        for n,s in enumerate(segs):
            labels={k:float(bool(re.search(pattern,str(s['description']).lower()))) for k,pattern in RULES.items()}
            for phase in range(3):
                a=s['start']+(s['end']-s['start'])*phase/3;b=s['start']+(s['end']-s['start'])*(phase+1)/3
                take=(assigned==n)&(t>=a-1e-9)&(t<b-1e-9)
                if not take.any():continue
                qualified=take&valid
                rows.append(dict(episode=rec.episode,environment=rec.environment,operator=rec.operator,
                    dataset=rec.dataset,split=rec.split,segment=n,start=a,end=b,phase=phase,frames=int(take.sum()),
                    description=s['description'],task=s['task'],task_hash=hashlib.sha256(s['task'].encode()).hexdigest()[:12],
                    bad_fraction=float(bad[take].any(axis=1).mean()),
                    mean_log_speed=float(np.log1p(speed[qualified]).mean()) if qualified.any() else np.nan,
                    qualified_frames=int(qualified.sum()),interhand_distance=float(np.linalg.norm(c[take,0]-c[take,1],axis=-1).mean()),
                    unknown=float(not any(labels.values())),**labels))
    frame=pd.DataFrame(rows);frame.to_csv(out/'semantic_segments.csv',index=False);pd.DataFrame(audit).to_csv(out/'semantic_coverage.csv',index=False)
    # Multi-label weak descriptions are not micro-action ground truth. Within-
    # record centering removes constant operator/environment/task nuisance terms.
    frame['phase_middle']=(frame.phase==1).astype(float);frame['phase_late']=(frame.phase==2).astype(float)
    names=[*RULES,'phase_middle','phase_late'];effects=[];rng=np.random.default_rng(20261120)
    for target in ['bad_fraction','mean_log_speed']:
        f=frame[np.isfinite(frame[target])].copy();x=f[names].to_numpy();y=f[target].to_numpy()
        exposure=f.frames.to_numpy() if target=='bad_fraction' else f.qualified_frames.to_numpy()
        for episode in f.episode.unique():
            m=(f.episode==episode).to_numpy();w=exposure[m]/exposure[m].sum()
            x[m]-=np.average(x[m],weights=w,axis=0);y[m]-=np.average(y[m],weights=w)
        envs=sorted(f.environment.unique());grams=[];bs=[]
        for env in envs:
            gs=[];vs=[]
            for episode in f[f.environment==env].episode.unique():
                m=(f.episode==episode).to_numpy();w=exposure[m]/exposure[m].sum()
                gs.append(x[m].T@(w[:,None]*x[m]));vs.append(x[m].T@(w*y[m]))
            grams.append(np.mean(gs,axis=0));bs.append(np.mean(vs,axis=0))
        grams=np.array(grams);bs=np.array(bs);g=grams.mean(axis=0);coef=np.linalg.pinv(g)@bs.mean(axis=0)
        samples=[]
        for _ in range(2000):
            ix=rng.integers(len(envs),size=len(envs));samples.append(np.linalg.pinv(grams[ix].mean(axis=0))@bs[ix].mean(axis=0))
        for j,name in enumerate(names):
            effects.append(dict(target=target,term=name,coefficient=coef[j],low=np.quantile(np.array(samples)[:,j],.025),
                high=np.quantile(np.array(samples)[:,j],.975),environments=len(envs),rank=int(np.linalg.matrix_rank(g)),terms=len(names)))
    pd.DataFrame(effects).to_csv(out/'semantic_within_record_effects.csv',index=False)
    tables=[]
    for name in [*RULES,'unknown']:
        for ds,g in frame[frame[name]>0].groupby('dataset'):
            tables.append(dict(label=name,dataset=ds,frames=int(g.frames.sum()),records=g.episode.nunique(),environments=g.environment.nunique(),
                bad_fraction=float(np.average(g.bad_fraction,weights=g.frames)),
                mean_log_speed=float(np.average(g.loc[g.mean_log_speed.notna(),'mean_log_speed'],weights=g.loc[g.mean_log_speed.notna(),'qualified_frames']))))
    pd.DataFrame(tables).to_csv(out/'semantic_summary.csv',index=False)
    record=frame.drop_duplicates('episode');cross=record.groupby(['environment','operator','dataset']).episode.nunique().reset_index(name='records');cross.to_csv(out/'context_confounding.csv',index=False)
    tasks=record.groupby('task_hash').agg(environments=('environment','nunique'),operators=('operator','nunique'),records=('episode','nunique'));tasks.to_csv(out/'task_support.csv')
    (out/'semantic_protocol.json').write_text(json.dumps(dict(rules=RULES,multilabel=True,stage='three retrospective thirds of supplied semantic interval',
        estimation='within-record weighted least squares; 2000 environment cluster bootstrap; descriptive associations, not causal effects',
        limitations=['description labels may combine actions','endpoints are retrospective only','quality flags are not physical error','environment/operator effects cannot be independently attributed when collinear'],
        test_used=False,task_hashes=len(tasks),tasks_in_multiple_environments=int((tasks.environments>1).sum()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(records=len(audit),segments=len(frame),covered=sum(a['covered'] for a in audit),frames=sum(a['frames'] for a in audit),task_hashes=len(tasks),tasks_multi_env=int((tasks.environments>1).sum()))))


if __name__=='__main__':main()
