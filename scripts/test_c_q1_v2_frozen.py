#!/usr/bin/env python3
"""Evaluate frozen Q1 once on the official held-out test environments.

AI-assisted by Codex, 2026-11-20. No refitting, retuning or outcome-based selection.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import pandas as pd
from shuwei_c.data import Dataset
from shuwei_c.q1v2_state import grouped_summary,evaluate_sets
from shuwei_c import q1v2_quality as q
from run_c_q1_v2_state import evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--test-video-root');args=parser.parse_args()
    out=Path('artifacts/c_q1_v2');freeze=json.loads((out/'freeze.json').read_text());config=json.loads(Path('configs/c_q1_v2.json').read_text())
    if (out/'test_run_manifest.json').exists():raise RuntimeError('Frozen test evaluation already completed')
    for name,hash_value in freeze['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==hash_value,f'Source changed after freeze: {name}'
    started=datetime.now(timezone.utc).isoformat();records=Dataset().records('test');state=freeze['state'];summaries=[];sets=[];quality=[];events=[];cache={}
    for h,entry in state['horizons'].items():
        h=float(h);frames=[]
        for name,spec in entry['family_models'].items():
            f=evaluate(records,h,config,spec,state['noise'],cache);frames.append(f);summaries.append(grouped_summary(f))
            if name==entry['selected']['method']:
                for row in evaluate_sets(f,entry['calibration']):row.update(horizon=h,method=name);sets.append(row)
        pd.concat(frames).to_parquet(out/f'state_test_{h:.6f}.parquet',index=False)
    for h,selected in freeze['quality'].items():
        h=float(h);video=args.test_video_root if selected['tag'].startswith('video') else None
        if selected['tag'].startswith('video') and not video:raise ValueError('Frozen selected visual model needs test video features')
        frame=q.dataset_frame(records,h,config,video)
        # Same frozen baselines remain visible even when the selected model wins.
        comparisons=[r for r in freeze['quality_candidates'][str(h)] if r['tag']=='numerical']
        if selected['tag']!='numerical':comparisons.append(selected)
        for candidate in comparisons:
            name=candidate['tag']+'_'+candidate['method'];spec=candidate['spec'];model=joblib.load(candidate['model_path'])
            probability=q.predict_baseline(model,frame) if spec['kind']=='baseline' else q.predict_classifier(model,frame,spec['columns'])
            probability=q.temperature(probability,spec['temperature'])
            for row in q.metrics(frame,probability,name):row.update(horizon=h,selected=(candidate['model_path']==selected['model_path']));quality.append(row)
            saved=frame[['episode','environment','dataset','origin','target','time_s','target_time_s','current_state','target_state']].copy()
            for j in range(4):saved[f'p{j}']=probability[:,j]
            saved.to_parquet(out/f'quality_test_{h:.1f}_{name}.parquet',index=False)
            for mode,thresholds in spec['thresholds'].items():
                for level,threshold in thresholds.items():
                    values,details=q.event_metrics(records,frame,probability,threshold,h,mode)
                    events.append(dict(horizon=h,method=name,mode=mode,nominal_false_alarm=float(level),**values))
    pd.concat(summaries).to_csv(out/'state_test_summary.csv',index=False);pd.DataFrame(sets).to_csv(out/'state_set_test.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'quality_test_metrics.csv',index=False);pd.DataFrame(events).to_csv(out/'quality_test_events.csv',index=False)
    manifest=dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),freeze_timestamp=freeze['timestamp_utc'],
        freeze_sha256=hashlib.sha256((out/'freeze.json').read_bytes()).hexdigest(),test_records=len(records),environments=len({r.environment for r in records}),
        status='evaluated once; all model choices fixed before reading test outcomes')
    (out/'test_run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
