#!/usr/bin/env python3
"""Evaluate frozen Q1 point transfer on fixed official-validation Q2 windows.

This is a development comparison, never training OOF or a test evaluation.
The imported Q1 interface's dense uncertainty path is explicitly disabled.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.data import Dataset
from shuwei_c.q1v3_interface import Q1V3
from shuwei_c.q2_live import Q1Transferred
from shuwei_c.q2_replay import Window,Periodic,replay
from shuwei_c.q2_metrics import summarize_windows,prediction_metrics

FREEZE=Path('artifacts/c_q1_v3/frozen_numeric/freeze.json')
PROTOCOL=Path('artifacts/c_q2/protocol')
KEYS=['window_id','cap','delay_s','phase']
GROUPS=['split','horizon_s','cap','delay_s','phase','method','policy','protocol']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def disabled_radius(*args,**kwargs):
    raise AssertionError('Q1 dense radius is prohibited in this point-only transfer evaluation')


class WindowMemo:
    """Memoize fixed-window immutable disclosed histories across fixed schedules.

    Packets at one source time have identical contents for this one replay record;
    the cache never crosses a window, record, or counterfactual reference variant.
    """
    def __init__(self,predictor,prefix_length):
        self.predictor=predictor;self.prefix_length=prefix_length;self.cache={};self.calls=0;self.hits=0

    def __call__(self,history,now):
        self.calls+=1
        key=(tuple(map(float,history.time[self.prefix_length:])),float(now))
        if key in self.cache:self.hits+=1;return self.cache[key].copy()
        pred=np.array(self.predictor(history,now),copy=True)
        self.cache[key]=pred
        return pred.copy()


def make_window(row):
    return Window(row.window_id,row.episode,row.dataset,row.environment,row.split,
                  int(row.origin_index),int(row.stop),float(row.origin_s),float(row.horizon_s))


def counterfactual_check(data,frame,adapters,out):
    checks=[]
    for ds,g in frame[frame.horizon_s.eq(1.)].groupby('dataset',sort=False):
        row=g.iloc[len(g)//2];w=make_window(row);rec=data.record(row.episode);t,p,b=rec.hands()
        pred=adapters[ds]
        a=replay(rec,w,pred,Periodic(1,.5),1,.3,numerical=(t,p,b))
        altered=p.copy();altered[w.origin_index+1:]+=np.array([.123,-.456,.789])
        z=replay(rec,w,pred,Periodic(1,.5),1,.3,numerical=(t,altered,b))
        first=min((r['arrival_time'] for r in a['requests']),default=w.end+1.)
        mask=a['times']<first-1e-9
        maximum=float(np.max(abs(a['predictions'][mask]-z['predictions'][mask])))
        assert mask.any() and maximum==0.,(ds,maximum)
        assert a['requests']==z['requests']
        checks.append(dict(dataset=ds,window_id=w.window_id,first_arrival_s=first,
            undisclosed_scoring_times=int(mask.sum()),prediction_max_difference_before_return=maximum,
            action_ledger_identical=True,altered_future_points=True))
    assert len(checks)==3
    (out/'counterfactual_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('artifacts/c_q2/q1_transfer'))
    parser.add_argument('--smoke',action='store_true',help='One predetermined middle window per dataset/horizon')
    parser.add_argument('--delays',nargs='+',type=float,default=[.1,0.,.3])
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    config=json.loads(Path('configs/c_q2.json').read_text());freeze=json.loads(FREEZE.read_text())
    noise=json.loads((PROTOCOL/'training_noise.json').read_text())
    frame=pd.read_csv(PROTOCOL/'development_windows.csv');frame=frame[frame.split.eq('validation')].copy()
    assert set(frame.reason)=={'included'} and frame.episode.nunique()==7 and frame.environment.nunique()==4
    if args.smoke:
        frame=pd.concat([g.iloc[[len(g)//2]] for _,g in frame.groupby(['dataset','horizon_s'],sort=False)],ignore_index=True)
    protected={**freeze['source_hashes'],**freeze['inherited_v2_hashes']}
    assert all(sha(p)==h for p,h in protected.items())
    sources=[Path(__file__).resolve().relative_to(Path.cwd()),Path('configs/c_q2.json'),Path('src/shuwei_c/q2_live.py'),
        Path('src/shuwei_c/q2_replay.py'),Path('src/shuwei_c/q2_metrics.py'),
        Path('src/shuwei_c/q1v3_interface.py'),Path('src/shuwei_c/q1v3_prefix.py'),
        Path('src/shuwei_c/q1v3_state.py'),Path('src/shuwei_c/q1v3_learning.py'),
        Path('src/shuwei_c/q1v2_state.py'),Path('src/shuwei_c/q1v2_quality.py'),
        PROTOCOL/'development_windows.csv',PROTOCOL/'training_noise.json',FREEZE,
        Path('artifacts/c_q2/baselines/validation_windows.csv')]
    source_hashes={str(p):sha(p) for p in sources}
    identity=dict(split='official_validation_only',smoke=args.smoke,phases=[.5],delays=args.delays,
        windows=frame.window_id.tolist(),source_hashes=source_hashes,
        checkpoints={p:sha(p) for values in freeze['point_checkpoints'].values() for p in values})
    identity_path=args.out/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('Run identity differs; use a fresh output directory')
    identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    for p in sources:
        target=args.out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    started=time.perf_counter();data=Dataset();interfaces={ds:Q1V3(FREEZE,ds) for ds in sorted(frame.dataset.unique())}
    for interface in interfaces.values():interface.forecast=disabled_radius;interface.observe=disabled_radius
    adapters={ds:Q1Transferred(interface) for ds,interface in interfaces.items()}
    counterfactual=counterfactual_check(data,frame,adapters,args.out)
    print(json.dumps(dict(stage='loaded_and_counterfactual_checked',interfaces=len(interfaces),
                         windows=len(frame),elapsed_s=time.perf_counter()-started)),flush=True)
    records=args.out/'validation_records';records.mkdir(exist_ok=True);results=[];call_counts=[]
    for episode,group in frame.groupby('episode',sort=False):
        target=records/f'{episode}.csv';marker=target.with_suffix('.json')
        expected=len(group)*len(config['request_caps'])*len(args.delays)
        if target.exists() and marker.exists():
            meta=json.loads(marker.read_text());cached=pd.read_csv(target)
            assert sha(target)==meta['sha256'] and len(cached)==expected and not cached.duplicated(KEYS).any()
            results.extend(cached.to_dict('records'));call_counts.append(meta);continue
        rec=data.record(episode);t,p,bad=rec.hands();c=p.mean(axis=2);local=[];calls=hits=0
        for row in group.itertuples(index=False):
            window=make_window(row);memo=WindowMemo(adapters[rec.dataset],window.origin_index+1)
            for cap in config['request_caps']:
                for delay in args.delays:
                    output=replay(rec,window,memo,Periodic(cap,.5),cap,delay,
                        cost=config['cost_per_request'],multiple_pending=config['multiple_pending'],numerical=(t,p,bad))
                    assert output['valid'].all()
                    direct=[r['current_direct'] for r in output['steps']]
                    metrics=dict(output['metrics']);metrics.update(prediction_metrics(output['predictions'],
                        c[window.origin_index+1:window.stop],direct,noise[rec.dataset]['hand_scale']))
                    ages=np.array([r['age_s'] for r in output['steps']])
                    local.append(dict(window_id=window.window_id,split='validation',dataset=rec.dataset,
                        environment=rec.environment,episode=episode,origin_s=window.origin,horizon_s=window.horizon,
                        cap=cap,delay_s=delay,phase=.5,method='q1_transferred',policy='periodic',
                        protocol='all_modalities_hidden',age_over_fitted_horizon_fraction=float(np.mean(ages>1.+1e-9)),
                        maximum_disclosed_age_s=float(ages.max()),**metrics))
            calls+=memo.calls;hits+=memo.hits
        assert len(local)==expected and not pd.DataFrame(local).duplicated(KEYS).any()
        temp=target.with_suffix('.tmp');pd.DataFrame(local).to_csv(temp,index=False);temp.replace(target)
        meta=dict(episode=episode,windows=len(group),rows=len(local),sha256=sha(target),calls=calls,memo_hits=hits)
        marker.write_text(json.dumps(meta,indent=2)+'\n');call_counts.append(meta);results.extend(local)
        print(json.dumps(dict(stage='record_complete',**meta,elapsed_s=time.perf_counter()-started)),flush=True)
    result=pd.DataFrame(results);result.to_csv(args.out/'validation_windows.csv',index=False)
    baseline=pd.read_csv('artifacts/c_q2/baselines/validation_windows.csv')
    baseline=baseline[baseline.phase.eq(.5)&baseline.delay_s.isin(args.delays)&baseline.window_id.isin(frame.window_id)]
    matched=result.merge(baseline[baseline.method.eq('constant')],on=KEYS,suffixes=('_transfer','_baseline'),validate='one_to_one')
    assert len(matched)==len(result)
    ledger_columns=['requests','total_cost','waiting_union_s','latency_sum_s','responses_within_window',
                    'scored_hand_times','unobserved_hand_times','current_direct_fraction']
    differences={name:float(np.max(abs(matched[name+'_transfer']-matched[name+'_baseline']))) for name in ledger_columns}
    assert all(v<1e-10 for v in differences.values()),differences
    summary=summarize_windows(result,GROUPS);summary.to_csv(args.out/'validation_summary.csv',index=False)
    joint=summarize_windows(pd.concat([result,baseline],ignore_index=True),GROUPS)
    joint.to_csv(args.out/'matched_comparison.csv',index=False)
    group=[k for k in GROUPS if k!='method']+['dataset']
    comparable=joint[joint.method.ne('q1_transferred')].merge(summary[group+['mean_error','normalized_error','unobserved_error']],
        on=group,suffixes=('_baseline','_transfer'),validate='many_to_one')
    comparable['transfer_error_reduction_pct']=100*(1-comparable.mean_error_transfer/comparable.mean_error_baseline)
    comparable.to_csv(args.out/'paired_gains.csv',index=False)
    assert all(sha(p)==h for p,h in protected.items()),'Frozen Q1 scientific input changed'
    changed=[p for p,h in source_hashes.items() if sha(p)!=h]
    # Loaded Python code is fixed for this process; retain its start-of-run snapshot
    # and explicitly report concurrent working-tree changes, never conceal them.
    manifest=dict(status='PASS',scope='Official validation development comparison only; not training OOF or fresh test',
        test_used=False,train_used_for_scoring=False,models_fitted=False,dense_radius_used=False,
        windows=len(frame),rows=len(result),interfaces_loaded=len(interfaces),delays=args.delays,phase=.5,
        elapsed_s=time.perf_counter()-started,source_hashes=source_hashes,concurrent_source_changes=changed,
        matched_baseline_ledger_max_differences=differences,counterfactual_checks=counterfactual,
        calls=sum(r.get('calls',0) for r in call_counts),memo_hits=sum(r.get('memo_hits',0) for r in call_counts),
        extrapolation='Disclosed ages above 1 second use fixed 1-second Q1 weights with actual elapsed propagation; explicit transfer stress only',
        weighting=config['weighting'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
