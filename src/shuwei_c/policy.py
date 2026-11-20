"""Q3 public-information policies and approximate decision-value reviews.

AI: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
Two-stage, deterministic-sample approximate lookahead; no global optimality claim.
"""
import math
import numpy as np
from .duration import service_samples


class SchedulingPolicy:
    def __init__(self, model, rule='lookahead', review=False, reservation='minimum', review_samples=8):
        self.model,self.rule,self.review,self.reservation=model,rule,review,reservation
        self.review_samples=review_samples

    def __call__(self,s):
        jobs={j.job_id:j for j in s['jobs']}
        pending=[j for j in s['jobs'] if j.job_id not in s['attempted']]
        if not pending:
            return {'type':'stop'}
        dist={j.job_id:self.model.distribution(j.family,s['prefixes'].get(j.job_id,{}),s['reviews'][j.job_id])
              if j.job_id in s['prefixes'] else self._prior(j.family) for j in pending}
        samples={k:service_samples(v,s['alpha'],s['beta']) for k,v in dist.items()}
        feasible=self.feasible(s,samples)
        if self.rule in ['edd','weighted']:
            if feasible:
                key=min(feasible,key=lambda k:(jobs[k].due,k)) if self.rule=='edd' else min(feasible,key=lambda k:(-jobs[k].weight,jobs[k].due,k))
                return dict(type='execute',job=key,reserve=self.reserve(samples[key],s))
            return self.wait_or_stop(s)
        score,action=self.best_action(s,samples)
        if self.review and s['observations']<s['case']['observation_limit']:
            r=s['rules']['observation']; delayed=s.copy()
            delayed.update(time=s['time']+r['delay_s'],cost=s['cost']+r['cost'])
            if delayed['time']<=s['case']['horizon_s'] and delayed['cost']<=s['case']['cost_budget']:
                eligible=[j for j in pending if j.job_id in s['prefixes'] and s['reviews'][j.job_id]<2]
                # Explicit computational scope: top three weighted uncertainty candidates,
                # including predecessor-blocked jobs; this is approximate VoI.
                eligible=sorted(eligible,key=lambda j:-j.weight*dist[j.job_id][1])[:3]
                for j in eligible:
                    hypot=self.model.hypothetical(j.family,s['prefixes'][j.job_id],s['reviews'][j.job_id],self.review_samples)
                    values=[]
                    for d in hypot:
                        revised=samples.copy();revised[j.job_id]=service_samples(d,s['alpha'],s['beta'])
                        v,_=self.best_action(delayed,revised)
                        values.append(v)
                    # Review consumes real time and money in all counterfactual schedules.
                    gain=float(np.mean(values))-score if values else -math.inf
                    if gain>1e-6:
                        score+=gain;action=dict(type='review',job=j.job_id)
        return action

    def _prior(self,family):
        from .duration import DurationModel
        return DurationModel(self.model.params,use_prefix=False).distribution(family,{},0)

    def reserve(self,samples,s):
        if self.reservation=='minimum':
            return s['rules']['service']['reserve_min_s']
        return max(s['rules']['service']['reserve_min_s'],float(np.quantile(samples,.5 if self.reservation=='duration_q50' else .8)))

    def feasible(self,s,samples):
        r=s['rules'];out={}
        for j in s['jobs']:
            if j.job_id in s['attempted'] or j.release>s['time']+1e-9 or any(p not in s['completed'] for p in j.predecessors):
                continue
            distance=math.dist(s['zones'][s['zone']],s['zones'][j.zone])
            start=s['time']+distance/r['movement']['speed']+r['movement']['changeover_s']
            reserve=self.reserve(samples[j.job_id],s)
            fee=r['fees']['service_fixed']+r['fees']['reserve_per_second']*reserve+r['fees']['distance_per_unit']*distance
            if start+reserve<=s['case']['horizon_s']+1e-9 and s['cost']+fee<=s['case']['cost_budget']+1e-9:
                out[j.job_id]=(start,reserve,fee)
        return out

    def best_action(self,s,samples):
        feasible=self.feasible(s,samples)
        options=[]
        for key in feasible:
            v=self.rollout(s,samples,first=key)
            options.append((v,key))
        # Waiting is a real decision even when current jobs are available.
        wait=self.wait_or_stop(s)
        wait_value=0.
        if wait['type']=='wait':
            later=s.copy();later['time']=s['time']+wait['seconds']
            wait_value=self.rollout(later,samples)
        if not options or wait_value>max(v for v,k in options)+1e-9:
            return wait_value,wait
        v,key=max(options,key=lambda pair:(pair[0],pair[1]))
        return v,dict(type='execute',job=key,reserve=self.reserve(samples[key],s))

    def rollout(self,state,samples,first=None):
        s=state.copy();s['attempted']=set(s['attempted']);s['completed']=dict(s['completed'])
        total=0.;jobs={j.job_id:j for j in s['jobs']}
        for iteration in range(len(jobs)+1):
            feasible=self.feasible(s,samples)
            if not feasible:
                a=self.wait_or_stop(s)
                if a['type']=='stop':break
                s['time']+=a['seconds'];continue
            def gain(k):
                start,_,_=feasible[k];j=jobs[k]
                return j.weight*float(np.mean(start+samples[k]<=min(j.due,s['case']['horizon_s'])))
            if iteration==0 and first in feasible:
                k=first
            else:
                k=max(feasible,key=lambda k:(gain(k)/max(np.mean(samples[k])+feasible[k][0]-s['time'],.1),-jobs[k].due,k))
            start,reserve,fee=feasible[k]
            total+=gain(k)
            # Small completion tie-break; remaining-cost opportunity already constrains rollout.
            total+=1e-4*jobs[k].weight*float(np.mean(start+samples[k]<=s['case']['horizon_s']))-1e-6*fee
            finish=start+float(np.mean(samples[k]))
            s['time']=start+max(float(np.mean(samples[k])),reserve)
            s['cost']+=fee;s['zone']=jobs[k].zone;s['attempted'].add(k)
            if finish<=s['case']['horizon_s']:s['completed'][k]=finish
            if s['time']>=s['case']['horizon_s']:break
        return total

    @staticmethod
    def wait_or_stop(s):
        releases=[j.release for j in s['jobs'] if j.job_id not in s['attempted'] and s['time']<j.release<=s['case']['horizon_s']]
        return dict(type='wait',seconds=min(releases)-s['time']) if releases else dict(type='stop')
