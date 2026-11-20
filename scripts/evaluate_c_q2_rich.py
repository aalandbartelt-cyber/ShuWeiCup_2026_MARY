#!/usr/bin/env python3
"""Evaluate previously frozen numerical candidates on official validation."""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from shuwei_c.q2_learning import load_pack,score_predictions,row_weights
from shuwei_c.q2_rich_learning import predict


def main():
    out=Path('artifacts/c_q2/rich_state/validation');out.mkdir(parents=True,exist_ok=True)
    roots={name:Path('artifacts/c_q2/rich_state')/name/'full' for name in ('nested','main_horizons')}
    checkpoints={};sources={}
    for name,root in roots.items():
        freeze=root/'freeze.json';spec=json.loads(freeze.read_text())
        sources[str(freeze)]=hashlib.sha256(freeze.read_bytes()).hexdigest();checkpoints[name]=[]
        for file,digest in spec['checkpoint_hashes'].items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen checkpoint changed')
            sources[file]=digest;checkpoints[name].append(file)
    # All fitted model identities are checked before opening validation targets.
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    data=Path('artifacts/c_q2/rich_state/data/validation');pack=load_pack(data)
    for file in list(data.glob('*.npz'))+list(data.glob('*.parquet'))+[data/'manifest.json',Path(__file__),Path('src/shuwei_c/q2_rich_learning.py'),Path('src/shuwei_c/q2_learning.py')]:
        sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    identity=dict(source_hashes=sources,role='official validation development comparison after candidate training freeze',test_used=False)
    ip=out/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Evaluation identity changed')
    ip.write_text(json.dumps(identity,indent=2)+'\n');started=time.perf_counter();summaries=[];scopes=[]
    for name,paths in checkpoints.items():
        models=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        result=np.mean([predict(pack,m) for m in models],axis=0)
        frame,summary=score_predictions(pack,result,models[0]['noise'])
        for j,c in enumerate('xyz'):frame['prediction_'+c]=result[:,j]+pack['anchor'][:,j]
        frame['method']=name;frame.to_parquet(out/f'{name}_predictions.parquet',index=False)
        summary['method']=name;summaries.append(summary)
        for scope,mask in [('main_all_delays',frame.horizon_s.le(1)),('main_primary_delay',frame.horizon_s.le(1)&frame.delay_s.eq(.1)),('stress_all_delays',frame.horizon_s.gt(1))]:
            f=frame.loc[mask];w=row_weights(f);scopes.append(dict(method=name,scope=scope,error=float(w@f.normalized_error.to_numpy())))
    pd.concat(summaries,ignore_index=True).to_csv(out/'conditions.csv',index=False)
    pd.DataFrame(scopes).to_csv(out/'scopes.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(**identity,elapsed_s=time.perf_counter()-started,rows=len(pack['meta'])),indent=2)+'\n')
    print(pd.DataFrame(scopes).to_string(index=False),flush=True)


if __name__=='__main__':main()
