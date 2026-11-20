"""Equal endpoint-age requests under a fixed-delay continuous-age benchmark.

For a reliable age-zero initial state, B fixed requests, constant delay d,
continuous scoring on [0,T], and nondecreasing age loss F, request times
j*(T-d)/(B+1) minimize the integrated age loss (T>d). This is a benchmark
assumption, not a claim about every real hand trajectory or waiting penalties.
The online policy rounds targets forward to actual disclosed decision ticks,
merges duplicate ticks, and refuses returns after the final scored tick.
"""
from pathlib import Path
import argparse,hashlib,json,math,shutil,time
import numpy as np
from .q2_replay import EPS


def equal_age_times(origin,last_score,cap,delay):
    if int(cap)!=cap or cap<0 or not all(math.isfinite(float(x)) for x in (origin,last_score,delay)) or delay<0 or last_score<origin:
        raise ValueError('Invalid time horizon, integer budget or delay')
    useful=last_score-origin-delay
    if cap==0 or useful<=EPS:return np.empty(0,float)
    return origin+np.arange(1,int(cap)+1)*useful/(int(cap)+1)


def continuous_age_cost(requests,horizon,delay,primitive):
    """Integrated F(age), where primitive is any antiderivative of F."""
    requests=np.asarray(requests,float)
    if horizon<0 or delay<0 or np.any(np.diff(requests)<=0) or (len(requests) and (requests[0]<0 or requests[-1]+delay>horizon+EPS)):
        raise ValueError('Requests must be distinct, ordered and timely')
    if not len(requests):return float(primitive(horizon)-primitive(0.))
    lengths=np.r_[requests[0]+delay,np.diff(requests),horizon-requests[-1]-delay]
    ends=np.r_[lengths[0],delay+lengths[1:]]
    return float(sum(primitive(float(x)) for x in ends)-primitive(0.)-len(requests)*primitive(delay))


class EqualAgeSchedule:
    """A causal, fixed schedule. A fresh instance is required for each window."""
    def __init__(self,cap,single_pending=False):
        if int(cap)!=cap or cap<0:raise ValueError('Nonnegative integer budget required')
        self.cap=int(cap);self.single_pending=bool(single_pending)
        self.targets=None;self.cursor=0;self.last_request=-math.inf

    def __call__(self,context):
        if self.targets is None:
            self.targets=equal_age_times(context.origin,context.last_score_time,self.cap,context.delay)
        if context.remaining<=0 or self.cursor>=len(self.targets):return False
        if context.now+context.delay>context.last_score_time+EPS:return False
        if self.single_pending and context.pending:return False
        if context.now<=self.last_request+EPS:return False
        if context.now+EPS<self.targets[self.cursor]:return False
        self.last_request=context.now
        # Several nominal targets may share the same next real tick. Issue once.
        while self.cursor<len(self.targets) and self.targets[self.cursor]<=context.now+EPS:
            self.cursor+=1
        return True


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    temporary=Path(str(path)+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    import pandas as pd
    from .data import Dataset
    from .q2_replay import Window,Periodic,NoRequests,replay
    from .q2_predictors import Kinematic
    from .q2_metrics import summarize_windows
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['train','validation'],required=True)
    parser.add_argument('--out',type=Path,default=Path('artifacts/c_q2/equal_age_baseline'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    config_path=Path('configs/c_q2.json');window_path=Path('artifacts/c_q2/protocol/development_windows.csv');noise_path=Path('artifacts/c_q2/protocol/training_noise.json')
    config=json.loads(config_path.read_text());noise=json.loads(noise_path.read_text())
    windows=pd.read_csv(window_path);windows=windows[windows.split==args.split]
    caps=config['request_caps'];delays=sorted(set([config['main_delay_s']]+config['sensitivity_delays_s']));phases=config['periodic_phases']
    files=[Path(__file__),Path('src/shuwei_c/q2_replay.py'),Path('src/shuwei_c/q2_predictors.py'),Path('src/shuwei_c/q2_metrics.py'),Path('src/shuwei_c/data.py'),config_path,window_path,noise_path]
    identity=dict(split=args.split,caps=caps,delays=delays,phases=phases,equal_age='j*(last_score-origin-delay)/(B+1)',tick_mapping='first actual tick at/after target; duplicate ticks merged; timely returns only',point_predictor='Kinematic last-two-qualified velocity tau=.2',multiple_pending=True,phase_selection='all four retained; no validation-window oracle',source_hashes={str(p):sha(p) for p in files},test_used=False)
    ip=args.out/f'{args.split}_identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Baseline identity changed')
    save_json(ip,identity)
    for p in files:
        relative=p.relative_to(Path.cwd()) if p.is_absolute() else p;destination=args.out/f'{args.split}_source_snapshot'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,destination)
    folder=args.out/f'{args.split}_records';folder.mkdir(exist_ok=True)
    data=Dataset();all_rows=[];started=time.perf_counter()
    for episode,group in windows.groupby('episode',sort=False):
        target=folder/f'{episode}.csv';request_path=folder/f'{episode}_requests.parquet';marker=folder/f'{episode}.json';expected=len(group)*len(caps)*len(delays)*(len(phases)+1)
        if marker.exists():
            saved=json.loads(marker.read_text())
            if sha(target)!=saved['sha256'] or sha(request_path)!=saved['requests_sha256']:raise ValueError('Completed baseline cache changed')
            frame=pd.read_csv(target)
            if len(frame)!=expected or frame.duplicated(['window_id','cap','delay_s','policy','phase']).any():raise ValueError('Baseline cache incomplete')
            all_rows.append(frame);continue
        rec=data.record(episode);numerical=rec.hands();rows=[];request_rows=[]
        for w in group.itertuples(index=False):
            window=Window(w.window_id,episode,rec.dataset,rec.environment,rec.split,int(w.origin_index),int(w.stop),float(w.origin_s),float(w.horizon_s))
            origin_bad=int(numerical[2][window.origin_index].sum())
            for delay in delays:
                no_requests=None
                for cap in caps:
                    specs=[('equal_age',-1.,EqualAgeSchedule(cap))]+[('periodic',phase,Periodic(cap,phase)) for phase in phases]
                    for policy_name,phase,policy in specs:
                        if cap==0:
                            if no_requests is None:no_requests=replay(rec,window,Kinematic('damped',damping=.2),NoRequests(),0,delay,numerical=numerical)
                            output=no_requests
                        else:output=replay(rec,window,Kinematic('damped',damping=.2),policy,cap,delay,numerical=numerical)
                        metrics=dict(output['metrics']);scale=noise[rec.dataset]['hand_scale']
                        metrics['normalized_error']=metrics['mean_error']/scale;metrics['unobserved_normalized_error']=metrics['unobserved_error']/scale
                        if metrics['requests']>cap or metrics['late_responses'] or any(not r['within_scoring_period'] for r in output['requests']):raise ValueError('Illegal request budget/deadline')
                        rows.append(dict(window_id=w.window_id,episode=episode,environment=rec.environment,dataset=rec.dataset,split=rec.split,horizon_s=w.horizon_s,origin_s=window.origin,last_score_s=float(output['times'][-1]),cap=cap,delay_s=delay,policy=policy_name,phase=phase,method='last_two_damped_tau0.2',origin_flagged_hands=origin_bad,**metrics))
                        for request in output['requests']:
                            request_rows.append(dict(window_id=w.window_id,episode=episode,horizon_s=w.horizon_s,origin_s=window.origin,cap=cap,delay_s=delay,policy=policy_name,phase=phase,**request))
        frame=pd.DataFrame(rows)
        if len(frame)!=expected:raise ValueError('Wrong baseline row count')
        temp=target.with_suffix('.tmp.csv');frame.to_csv(temp,index=False);temp.replace(target)
        temp=request_path.with_suffix('.tmp.parquet');pd.DataFrame(request_rows).to_parquet(temp,index=False);temp.replace(request_path)
        info=dict(episode=episode,rows=len(frame),windows=len(group),requests=len(request_rows),sha256=sha(target),requests_sha256=sha(request_path),elapsed_s=time.perf_counter()-started)
        save_json(marker,info);all_rows.append(frame);print(json.dumps(info),flush=True)
    result=pd.concat(all_rows,ignore_index=True);target=args.out/f'{args.split}_windows.csv';temp=target.with_suffix('.tmp.csv');result.to_csv(temp,index=False);temp.replace(target)
    summarize_windows(result,['split','horizon_s','cap','delay_s','policy','phase','method']).to_csv(args.out/f'{args.split}_summary.csv',index=False)
    save_json(args.out/f'{args.split}_manifest.json',dict(**identity,rows=len(result),windows=result.window_id.nunique(),records=result.episode.nunique(),elapsed_s=time.perf_counter()-started,output_sha256=sha(target),origin_flagged_windows=int(result.loc[result.origin_flagged_hands>0,'window_id'].nunique())))


if __name__=='__main__':main()
