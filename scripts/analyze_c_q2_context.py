#!/usr/bin/env python3
"""Post-hoc Q2 semantic/history associations on exactly matched development rows.

This script does not produce online inputs, choose model weights, or read test
records. Interval endpoints/stages and target errors remain evaluation-only.
Q1 weak-action rules/parser are reproduced explicitly with source hashes.
"""
from pathlib import Path
from collections import Counter
import argparse
import ast
import hashlib
import json
import re
import shutil
import time

import numpy as np
import pandas as pd

DATA = Path('00_赛题与规则/正式发布/C_Problem_Data/Restocking_Problem_and_Data_EN')
Q1_SOURCE = Path('scripts/analyze_c_q1_v2_context.py')
VIDEO_MANIFEST = Path('artifacts/c_video_v2_15hz/video_manifest.json')
LEARNING = Path('artifacts/c_q2/learning_data')
RULES = {
    'open_close': r'\b(open|close|pull out|push in)\b',
    'pick': r'\b(pick|take|grab|grasp|remove)\b',
    'place': r'\b(put|place|set down|insert)\b',
    'transport': r'\b(move|carry|transfer|bring)\b',
    'arrange': r'\b(arrange|rearrange|adjust|flip|rotate|align|organize|organization|arrangement|neatly)\b',
    'reach': r'\b(reach|extend|approach)\b',
}
KEYS = ['split','episode','environment','window_id','horizon_s','cap','delay_s',
        'phase','scenario','time_s','hand']
CONDITION = ['split','horizon_s','cap','delay_s']
UNIT = ['environment','episode','window_id','scenario','phase']
HISTORY = ['age_s','latest_gap_s','received_bad','past_bad_fraction','speed_0',
           'received_in_last_second','qualified_history_count','no_qualified_history']
METHODS = ('static','residual','equal_blend','fast_lag_expert8')
ERRORS = tuple('error_'+name for name in METHODS)
GAINS = ('gain_vs_static','gain_vs_residual','gain_vs_fast_lag_expert8')
METRICS = ERRORS+GAINS


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):
            value.update(block)
    return value.hexdigest()


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=json_default)+'\n')


def json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def selected(frame):
    return (frame.horizon_s.isin([.2,.5,1.]) & frame.cap.isin([0,2]) & frame.delay_s.eq(.1)).to_numpy()


def source_inventory():
    rows=json.loads(VIDEO_MANIFEST.read_text()); result={}
    for row in rows:
        if row['split'] not in ('train','validation'):
            raise ValueError('Context analysis refuses test video inventory')
        parts=Path(row['source']).parts
        if parts[0]!='Raw_Data' or parts[1]!=row['dataset']:
            raise ValueError('Unexpected development source layout')
        value=dict(dataset=row['dataset'],split=row['split'],folder=DATA/Path(*parts[:3]))
        if row['episode'] in result and result[row['episode']]!=value:
            raise ValueError('Inconsistent episode source folders')
        result[row['episode']]=value
    return result


def annotation_path(item):
    return item['folder']/('annotation.json' if item['dataset']=='EgoDemo'
                           else 'tables/annotation_semantic_segments.csv')


def segments(item,episode):
    path=annotation_path(item)
    if item['dataset']=='EgoDemo':
        raw=json.loads(path.read_text())['subtask_annotations']
        source=[(s['start_time'],s['end_time'],s['description']) for s in raw]
    else:
        raw=pd.read_csv(path)
        source=list(zip(raw['segment.start_time'],raw['segment.end_time'],raw['segment.subtask_description']))
    result=[]
    for number,(start,end,description) in enumerate(source):
        start,end=float(start),float(end); description=str(description)
        valid=bool(np.isfinite([start,end]).all() and end>start)
        result.append(dict(interval_id=f'{episode}:{number}',episode=episode,split=item['split'],
                           dataset=item['dataset'],start=start,end=end,valid=valid,
                           description_sha256=hashlib.sha256(description.encode()).hexdigest(),
                           **{name:bool(re.search(pattern,description.lower())) for name,pattern in RULES.items()}))
    return result


def semantic_rows(times,intervals):
    """Half-open intervals; overlap is ambiguous regardless of input ordering."""
    times=np.asarray(times,float); counts=np.zeros(len(times),int)
    ids=[[] for _ in times]; phase=np.full(len(times),'uncovered',object)
    start=np.full(len(times),np.nan);end=start.copy();fraction=start.copy()
    actions={name:np.zeros(len(times),bool) for name in RULES}
    for interval in intervals:
        if not interval['valid']:
            continue
        hit=(times>=interval['start'])&(times<interval['end'])
        counts+=hit
        for i in np.flatnonzero(hit):ids[i].append(interval['interval_id'])
        f=(times[hit]-interval['start'])/(interval['end']-interval['start'])
        phase[hit]=np.array(['early','middle','late'])[np.minimum((3*f).astype(int),2)]
        start[hit]=interval['start'];end[hit]=interval['end'];fraction[hit]=f
        for name in RULES:actions[name]|=hit&interval[name]
    ambiguous=counts>1;phase[ambiguous]='ambiguous'
    for values in (start,end,fraction):values[ambiguous]=np.nan
    status=np.where(counts==0,'uncovered',np.where(ambiguous,'ambiguous','unique'))
    result=pd.DataFrame(dict(posthoc_semantic_status=status,posthoc_stage=phase,
                              posthoc_active_interval_count=counts,
                              posthoc_interval_ids=['|'.join(sorted(value)) for value in ids],
                              posthoc_interval_start_s=start,posthoc_interval_end_s=end,
                              posthoc_interval_fraction=fraction))
    for name,values in actions.items():result['posthoc_action_'+name]=values
    result['posthoc_unknown_action']=~np.any(np.stack(list(actions.values())),axis=0)
    return result


def row_weights(frame):
    """Equal environment/episode/window experiment/hand/time within this group."""
    nenv=frame.environment.nunique()
    episodes=frame.groupby('environment',sort=False).episode.transform('nunique').to_numpy()
    unit=pd.factorize(pd.MultiIndex.from_frame(frame[UNIT]))[0]
    g=frame[['environment','episode','hand']].copy();g['_unit']=unit
    windows=g.groupby(['environment','episode'],sort=False)._unit.transform('nunique').to_numpy()
    hands=g.groupby('_unit',sort=False).hand.transform('nunique').to_numpy()
    rows=g.groupby(['_unit','hand'],sort=False).hand.transform('size').to_numpy()
    result=1/(nenv*episodes*windows*hands*rows)
    if not np.isclose(result.sum(),1.,atol=1e-10):raise ValueError('Incorrect hierarchical weights')
    return result


def weighted_quantile(values,weights,levels):
    order=np.argsort(values,kind='stable');cdf=np.cumsum(weights[order])/weights.sum()
    return np.asarray(values)[order[np.minimum(np.searchsorted(cdf,levels,side='left'),len(order)-1)]]


def speed_thresholds(train):
    if set(train.split)!={'train'}:raise ValueError('Speed thresholds must use training rows only')
    return weighted_quantile(train.history_speed_per_scale_s.to_numpy(),row_weights(train),[1/3,2/3])


def history_groups(frame,thresholds):
    f=frame.copy();age=f.history_age_s.to_numpy();gap=f.history_latest_gap_s.to_numpy()
    f['history_age_group']=np.select([age<=1e-9,age<=.1,age<=.3,age<=1.],
                                    ['current','(0,0.1]','(0.1,0.3]','(0.3,1]'],default='>1')
    f['history_gap_group']=np.select([gap<=1e-9,gap<=.05,gap<=.2],
                                    ['single_or_zero','(0,0.05]','(0.05,0.2]'],default='>0.2')
    f['history_received_bad_group']=np.where(f.history_received_bad>0,'flagged','unflagged')
    f['history_past_bad_group']=np.where(f.history_past_bad_fraction>0,'some_flagged','none_flagged')
    f['history_speed_group']=np.select([f.history_speed_per_scale_s<=thresholds[0],
                                      f.history_speed_per_scale_s<=thresholds[1]],['low','middle'],default='high')
    return f


def load_rows(split,inventory,noise):
    folder=LEARNING/split;manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['split']!=split or manifest.get('test_used') or manifest['sample_stride']!=1:
        raise ValueError('Only complete non-test learning rows are supported')
    if manifest['expert_names'][8]!='damped_fast_0.2':raise ValueError('Expert 8 identity changed')
    names=manifest['feature_names'];frames=[];references=[];fast=[];intervals=[];offset=0
    for path in sorted(folder.glob('*.parquet')):
        meta=pd.read_parquet(path)
        if set(meta.split)!={split} or set(meta.episode)!={path.stem} or inventory[path.stem]['split']!=split:
            raise ValueError('Learning split or episode differs from development inventory')
        mask=selected(meta);f=meta.loc[mask].copy();f['cache_row']=offset+np.flatnonzero(mask)
        f['episode_file_row']=np.flatnonzero(mask);offset+=len(meta)
        if f.empty:continue
        item=inventory[path.stem];sem=segments(item,path.stem);intervals.extend(sem)
        unique,inverse=np.unique(f.time_s.to_numpy(),return_inverse=True)
        labels=semantic_rows(unique,sem).iloc[inverse].reset_index(drop=True)
        f=f.reset_index(drop=True);f=pd.concat([f,labels],axis=1)
        scale=float(noise[item['dataset']]['hand_scale'])
        if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid training spatial scale')
        f['analysis_scale']=scale
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as raw:
            if len(raw['features'])!=len(meta):raise ValueError('Learning arrays do not align')
            x=raw['features'][mask]
            for name in HISTORY:f['history_'+name]=x[:,names.index(name)].astype(float)
            # Keep the exact disclosed-age ledger, not its float32 feature copy.
            np.testing.assert_allclose(f.history_age_s,f.age_s,rtol=1e-6,atol=1e-6)
            f['history_age_s']=f.age_s
            f['history_speed_per_scale_s']=f.history_speed_0/scale
            anchor=raw['anchor'][mask].astype(float)
            references.append(anchor+raw['target'][mask].astype(float))
            fast.append(anchor+raw['experts'][mask,:,8].astype(float))
        if not np.isfinite(f[['history_'+n for n in HISTORY]].to_numpy()).all():
            raise ValueError('Nonfinite disclosed history descriptor')
        frames.append(f)
    if offset!=manifest['rows']:raise ValueError('Learning source row accounting differs')
    result=pd.concat(frames,ignore_index=True)
    if result.duplicated(KEYS).any():raise ValueError('Duplicate atomic context rows')
    return result,np.concatenate(references),np.concatenate(fast),intervals


def prediction_paths(split,method):
    folders=[f'fold{i}' for i in range(4)] if split=='train' else ['full']
    root=Path('artifacts/c_q2/static_gate/nested' if method=='static' else 'artifacts/c_q2/gate_oof_main/residual')
    name='ensemble_predictions.parquet' if method=='static' else 'predictions.parquet'
    return [root/folder/name for folder in folders]


def load_predictions(split,method,metadata):
    columns=KEYS+['prediction_'+c for c in 'xyz'];parts=[]
    for path in prediction_paths(split,method):
        part=pd.read_parquet(path,columns=columns)
        expected='train_oof' if split=='train' else 'validation'
        if set(part.split)!={expected}:raise ValueError('Prediction file contains an unexpected split')
        # This is only a ledger join: preserve the OOF provenance in manifest.
        part['split']=split
        parts.append(part.loc[selected(part)])
    frame=pd.concat(parts,ignore_index=True)
    if frame.duplicated(KEYS).any():raise ValueError('Duplicate component prediction keys')
    indexed=frame.set_index(KEYS);wanted=pd.MultiIndex.from_frame(metadata[KEYS])
    if len(indexed)!=len(wanted) or not wanted.isin(indexed.index).all():
        raise ValueError('Component predictions are not exactly the selected original rows')
    return indexed.loc[wanted,['prediction_'+c for c in 'xyz']].to_numpy(float)


def attach_errors(frame,reference,fast):
    static=load_predictions(frame.split.iloc[0],'static',frame)
    residual=load_predictions(frame.split.iloc[0],'residual',frame)
    predictions=dict(static=static,residual=residual,equal_blend=(static+residual)/2,fast_lag_expert8=fast)
    out=frame[KEYS+['cache_row','episode_file_row']].copy();scale=frame.analysis_scale.to_numpy()
    for name,prediction in predictions.items():
        if prediction.shape!=reference.shape or not np.isfinite(prediction).all():raise ValueError('Invalid point prediction')
        error=np.linalg.norm(prediction-reference,axis=1)
        out['raw_error_'+name]=error;out['error_'+name]=error/scale
    if np.any(out.error_equal_blend>(out.error_static+out.error_residual)/2+1e-10):
        raise ValueError('Equal-blend Euclidean convexity check failed')
    out['component_disagreement']=np.linalg.norm(static-residual,axis=1)/scale
    for base in ('static','residual','fast_lag_expert8'):
        out['gain_vs_'+base]=out['error_'+base]-out.error_equal_blend
    return out


def unit_means(frame,columns):
    hand=frame.groupby(UNIT+['hand'],sort=False,observed=True)[list(columns)].mean()
    window=hand.groupby(level=UNIT,sort=False).mean()
    episode=window.groupby(level=['environment','episode'],sort=False).mean()
    environment=episode.groupby(level='environment',sort=False).mean()
    return window,episode,environment


def groups(frame):
    yield 'overall','all',np.ones(len(frame),bool)
    for value in ('unique','ambiguous','uncovered'):
        yield 'semantic_status',value,frame.posthoc_semantic_status.eq(value).to_numpy()
    unique=frame.posthoc_semantic_status.eq('unique').to_numpy()
    for value in ('early','middle','late'):
        yield 'retrospective_stage',value,unique&frame.posthoc_stage.eq(value).to_numpy()
    for value in RULES:
        yield 'weak_action',value,unique&frame['posthoc_action_'+value].to_numpy(bool)
    yield 'weak_action','unknown',unique&frame.posthoc_unknown_action.to_numpy(bool)
    for name,values in [('age',('current','(0,0.1]','(0.1,0.3]','(0.3,1]','>1')),
                        ('gap',('single_or_zero','(0,0.05]','(0.05,0.2]','>0.2')),
                        ('received_bad',('flagged','unflagged')),('past_bad',('none_flagged','some_flagged')),
                        ('speed',('low','middle','high'))]:
        for value in values:yield 'history_'+name,value,frame['history_'+name+'_group'].eq(value).to_numpy()


def summarize(frame,args):
    summaries=[];paired=[];environments=[];supports=[]
    for condition,part in frame.groupby(CONDITION,sort=True,observed=True):
        base=dict(zip(CONDITION,condition));base['analysis_priority']='main' if condition[1]==1. else 'additional'
        for dataset in ['ALL']+sorted(part.dataset.unique()):
            data=part if dataset=='ALL' else part.loc[part.dataset==dataset]
            for grouping,label,mask in groups(data):
                subset=data.loc[mask];key=dict(**base,dataset=dataset,grouping=grouping,group=label)
                support=dict(**key,hand_time_rows=len(subset),
                             unique_record_times=len(subset[['episode','time_s']].drop_duplicates()),
                             windows=len(subset[UNIT].drop_duplicates()),episodes=subset.episode.nunique(),
                             environments=subset.environment.nunique())
                supports.append(support)
                if subset.empty:
                    mean=low=high=np.full(len(METRICS),np.nan);status='no_support';loenv=hienv=mean
                else:
                    _,_,env=unit_means(subset,METRICS);value=env.to_numpy();mean=value.mean(axis=0)
                    np.testing.assert_allclose(mean,np.average(subset[list(METRICS)],axis=0,weights=row_weights(subset)),atol=1e-10)
                    loenv=value.min(axis=0);hienv=value.max(axis=0)
                    for name,row in env.iterrows():environments.append(dict(**key,environment=name,**row.to_dict()))
                    if len(env)>=args.min_environments:
                        seed=int(hashlib.sha256(json.dumps(key,sort_keys=True,default=json_default).encode()).hexdigest()[:8],16)^args.seed
                        indices=np.random.default_rng(seed).integers(len(env),size=(args.bootstrap,len(env)))
                        bootstrap=value[indices].mean(axis=1);low,high=np.quantile(bootstrap,[.025,.975],axis=0)
                        status='exploratory_environment_cluster_bootstrap'
                    else:
                        low=high=np.full(len(METRICS),np.nan);status='descriptive_few_environments'
                for j,method in enumerate(METHODS):
                    summaries.append(dict(**support,method=method,mean_normalized_error=mean[j],
                                          ci_low=low[j],ci_high=high[j],environment_min=loenv[j],environment_max=hienv[j],
                                          interval_status=status))
                for j,method in enumerate(('static','residual','fast_lag_expert8')):
                    k=4+j;denominator=mean[METHODS.index(method)]
                    paired.append(dict(**support,comparison='equal_blend_vs_'+method,mean_gain=mean[k],
                                       ci_low=low[k],ci_high=high[k],
                                       relative_gain_percent=100*mean[k]/denominator if denominator>0 else np.nan,
                                       interval_status=status,positive_gain_means='equal_blend_lower_error'))
    return map(pd.DataFrame,(summaries,paired,environments,supports))


def failure_cases(frame,out):
    rows=[];trace_root=out/'case_traces';trace_root.mkdir(exist_ok=True)
    for condition,data in frame.groupby(CONDITION,sort=True,observed=True):
        columns=list(ERRORS)+['component_disagreement','history_age_s','history_latest_gap_s',
                            'history_received_bad','history_past_bad_fraction','history_speed_per_scale_s']
        windows,_,_=unit_means(data,columns);windows=windows.reset_index()
        windows['blend_deterioration_vs_fast']=windows.error_equal_blend-windows.error_fast_lag_expert8
        selectors={'largest_blend_error':'error_equal_blend',
                   'largest_deterioration_vs_fast':'blend_deterioration_vs_fast',
                   'largest_component_disagreement':'component_disagreement'}
        for selector,column in selectors.items():
            chosen=windows.sort_values([column,*UNIT],ascending=[False]+[True]*len(UNIT),kind='stable').iloc[0]
            mask=np.ones(len(data),bool)
            for key in UNIT:mask&=data[key].eq(chosen[key]).to_numpy()
            trace=data.loc[mask].sort_values(['time_s','hand']).copy()
            identity=dict(zip(CONDITION,condition))|{k:chosen[k] for k in UNIT}|dict(selector=selector)
            name=hashlib.sha256(json.dumps(identity,sort_keys=True,default=json_default).encode()).hexdigest()[:16]
            trace.to_parquet(trace_root/(name+'.parquet'),index=False)
            peak=trace.loc[trace.error_equal_blend.idxmax()]
            diagnostics=[]
            if trace.history_age_s.max()>.3:diagnostics.append('observation_age_exceeds_0.3s')
            if trace.history_latest_gap_s.max()>.2:diagnostics.append('past_qualified_gap_exceeds_0.2s')
            if trace.history_past_bad_fraction.max()>0:diagnostics.append('known_past_flags_present')
            if trace.history_received_bad.max()>0:diagnostics.append('last_received_packet_flagged')
            if trace.history_speed_group.eq('high').any():diagnostics.append('training_defined_high_past_speed_present')
            rows.append(dict(case_id=name,**identity,**{k:float(chosen[k]) for k in columns},
                             blend_deterioration_vs_fast=float(chosen.blend_deterioration_vs_fast),
                             peak_time_s=float(peak.time_s),peak_hand=int(peak.hand),peak_error=float(peak.error_equal_blend),
                             peak_semantic_status=peak.posthoc_semantic_status,peak_stage=peak.posthoc_stage,
                             peak_intervals=peak.posthoc_interval_ids,diagnostic_observations='|'.join(diagnostics),
                             trace_file=str(Path('case_traces')/(name+'.parquet')),
                             interpretation='post-hoc selected example; diagnostic associations, not causal mechanisms or failure rate'))
    return pd.DataFrame(rows)


def self_checks():
    a=dict(interval_id='a',start=0.,end=3.,valid=True,**{k:k=='pick' for k in RULES})
    b=dict(interval_id='b',start=1.,end=2.,valid=True,**{k:k=='place' for k in RULES})
    query=[-.1,0.,.9,1.,1.5,2.,2.9,3.]
    first=semantic_rows(query,[a,b]);second=semantic_rows(query,[b,a])
    pd.testing.assert_frame_equal(first,second)
    assert first.posthoc_semantic_status.tolist()==['uncovered','unique','unique','ambiguous','ambiguous','unique','unique','uncovered']
    assert first.posthoc_stage.iloc[5]=='late' and np.isnan(first.posthoc_interval_fraction.iloc[3])
    assert first.posthoc_action_pick.iloc[3] and first.posthoc_action_place.iloc[3]
    rows=[]
    for env,count,value in [('a',20,1.),('b',2,3.)]:
        for i in range(count):rows.append(dict(environment=env,episode=env,window_id=env,scenario='s',phase=.5,hand=i%2,v=value))
    f=pd.DataFrame(rows);_,_,env=unit_means(f,['v'])
    assert np.isclose(np.dot(row_weights(f),f.v),2.) and env.v.mean()==2.
    invalid=f.copy();invalid['split']='validation';invalid['history_speed_per_scale_s']=1.
    try:speed_thresholds(invalid)
    except ValueError:pass
    else:raise AssertionError('Validation threshold fitting was not rejected')
    q1=ast.parse(Q1_SOURCE.read_text())
    original=next(ast.literal_eval(n.value) for n in q1.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='RULES' for t in n.targets))
    assert original==RULES
    return dict(overlap_order_invariant=True,half_open_boundaries=True,ambiguous_stage_never_selected=True,
                multilabel_union_preserved=True,hierarchical_weighting=True,validation_threshold_fit_rejected=True,
                q1_rule_definition_exact_match=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('artifacts/c_q2/context/retrospective'))
    parser.add_argument('--bootstrap',type=int,default=2000);parser.add_argument('--min-environments',type=int,default=5)
    parser.add_argument('--seed',type=int,default=20261120);parser.add_argument('--self-check-only',action='store_true')
    args=parser.parse_args();checks=self_checks()
    if args.self_check_only:print(json.dumps(checks));return
    if args.bootstrap<100 or args.min_environments<2:raise ValueError('Invalid cluster interval settings')
    inventory=source_inventory();noise_path=Path('artifacts/c_q2/protocol/training_noise.json')
    noise=json.loads(noise_path.read_text())
    inputs=[Path(__file__),Q1_SOURCE,VIDEO_MANIFEST,noise_path]
    for split in ('train','validation'):
        inputs.extend(sorted((LEARNING/split).glob('*.npz')));inputs.extend(sorted((LEARNING/split).glob('*.parquet')))
        inputs.append(LEARNING/split/'manifest.json')
        for method in ('static','residual'):inputs.extend(prediction_paths(split,method))
    inputs.extend(annotation_path(value) for value in inventory.values())
    identity=dict(input_hashes={str(p):digest(p) for p in sorted(set(inputs))},horizons=[.2,.5,1.],caps=[0,2],delay_s=.1,
                  main_horizon_s=1.,rules=RULES,interval_rule='[start,end); overlap is ambiguous; no selected phase for overlaps',
                  action_summary='unique interval only; multilabel within interval; ambiguous unions saved separately',
                  bootstrap=args.bootstrap,minimum_environments_for_interval=args.min_environments,seed=args.seed,
                  threshold_fit='training selected rows; hierarchical weighted speed tertiles; validation excluded',
                  methods=list(METHODS),equal_blend_weights=[.5,.5],expert8='damped_fast_0.2',
                  component_file_split_labels=dict(train='train_oof',validation='validation'),
                  weighting='equal environment/episode/window-experiment/hand/time within each reported group',
                  test_used=False,online_input_approved=False)
    args.out.mkdir(parents=True,exist_ok=True);ip=args.out/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Context inputs changed; use a new output directory')
    if (args.out/'manifest.json').exists():raise ValueError('Completed context analysis already exists; retain its identity')
    json_write(ip,identity);snapshot=args.out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in (Path(__file__),Q1_SOURCE):shutil.copy2(path,snapshot/path.name)
    json_write(args.out/'self_checks.json',checks);start=time.perf_counter();packs={};intervals=[]
    for split in ('train','validation'):
        frame,reference,fast,sem=load_rows(split,inventory,noise);errors=attach_errors(frame,reference,fast)
        packs[split]=(frame,errors);intervals.extend(sem)
        print(json.dumps(dict(stage='matched',split=split,rows=len(frame),records=frame.episode.nunique()),default=json_default),flush=True)
    thresholds=speed_thresholds(packs['train'][0]);json_write(args.out/'history_thresholds.json',
        dict(speed_tertiles_per_training_hand_scale_per_second=thresholds.tolist(),fit_split='train',
             fit_rows=len(packs['train'][0]),fit_environments=int(packs['train'][0].environment.nunique()),
             age_bounds_s=[0.,.1,.3,1.],gap_bounds_s=[0.,.05,.2],past_bad_group_rule='zero versus positive'))
    pd.DataFrame(intervals).drop_duplicates('interval_id').to_parquet(args.out/'semantic_intervals.parquet',index=False)
    joined=[]
    for split,(frame,errors) in packs.items():
        frame=history_groups(frame,thresholds)
        frame.to_parquet(args.out/f'group_metadata_{split}.parquet',index=False)
        errors.to_parquet(args.out/f'matched_errors_{split}.parquet',index=False)
        additions=errors.drop(columns=KEYS+['cache_row','episode_file_row'])
        joined.append(pd.concat([frame,additions],axis=1))
    frame=pd.concat(joined,ignore_index=True)
    summary,paired,environment,support=summarize(frame,args)
    summary.to_csv(args.out/'group_error_summary.csv',index=False)
    paired.to_csv(args.out/'paired_gain_summary.csv',index=False)
    environment.to_parquet(args.out/'environment_metrics.parquet',index=False)
    support.to_csv(args.out/'group_support.csv',index=False)
    cases=failure_cases(frame,args.out);cases.to_csv(args.out/'failure_cases.csv',index=False)
    semantic_audit=[]
    for (split,episode),f in frame.groupby(['split','episode'],sort=True):
        u=f.drop_duplicates('time_s');counts=Counter(u.posthoc_semantic_status)
        semantic_audit.append(dict(split=split,episode=episode,dataset=f.dataset.iloc[0],
                                   selected_hand_time_rows=len(f),unique_selected_times=len(u),
                                   unique_interval_times=counts['unique'],ambiguous_times=counts['ambiguous'],
                                   uncovered_times=counts['uncovered']))
    pd.DataFrame(semantic_audit).to_csv(args.out/'semantic_assignment_audit.csv',index=False)
    for path,expected in identity['input_hashes'].items():
        if digest(path)!=expected:raise ValueError('An input changed during context analysis')
    outputs={str(p.relative_to(args.out)):digest(p) for p in sorted(args.out.rglob('*')) if p.is_file()}
    json_write(args.out/'manifest.json',dict(**identity,complete=True,elapsed_s=time.perf_counter()-start,
        rows_by_split={split:len(pack[0]) for split,pack in packs.items()},
        summary_rows=len(summary),paired_summary_rows=len(paired),case_count=len(cases),output_hashes=outputs,
        numpy=np.__version__,pandas=pd.__version__,
        interpretation='Post-hoc descriptive associations; cluster intervals condition on fixed point forecasts, not refitting. '
        'Small environment support gets no inferential interval. No causal or final-calibration claim. '
        'All windows were preselected to have qualified reference targets; this is not natural-gap performance.'))
    print(json.dumps(dict(complete=True,out=str(args.out),elapsed_s=time.perf_counter()-start,
                         rows=len(frame),summary_rows=len(summary),cases=len(cases))),flush=True)


if __name__=='__main__':main()
