"""Q3 event simulator. Policies receive a detached public snapshot only.

AI: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
The simulator is the trusted boundary, not a sandbox for malicious Python code.
"""
from dataclasses import dataclass
import copy
import math
import random
import numpy as np


@dataclass(frozen=True)
class Job:
    job_id: str
    family: str
    release: float
    due: float
    weight: float
    zone: str
    predecessors: tuple = ()


def jobs_from_rows(rows, case):
    return sorted([Job(r['job_id'], r['family_id'], float(r['release_s']), float(r['due_s']),
                       float(r['priority_weight']), r['zone_id'],
                       tuple(r['predecessors'].split('|')) if r['predecessors'] else ())
                   for r in rows if r['case_id'] == case], key=lambda j:j.job_id)


def sample_sources(data, jobs, split, seed):
    rng = random.Random(int(seed))
    return {j.job_id: rng.choice(data.eligible(split, j.family)) for j in sorted(jobs, key=lambda j:j.job_id)}


class Replay:
    def __init__(self, jobs, sources, prefix_provider, zones, case, conditions, alpha=1., beta=.5):
        self.jobs = {j.job_id:j for j in jobs}
        self._sources, self._prefix_provider = sources, prefix_provider
        self.zones, self.case, self.rules = zones, case, conditions
        self.alpha, self.beta = alpha, beta
        self.time, self.zone, self.cost = 0., conditions['initial_zone'], 0.
        self.reviews = {j:0 for j in self.jobs}
        self.attempted, self.completed = set(), {}
        self.observations, self.invalid, self.ended = 0, 0, False
        self.events = []

    def distance(self, a, b):
        return math.dist(self.zones[a], self.zones[b])

    def snapshot(self):
        prefixes = {k:self._prefix_provider(self._sources[k], self.rules['observation']['prefix_seconds'][self.reviews[k]])
                    for k,j in self.jobs.items() if j.release<=self.time and k not in self.attempted}
        # Neither the seed nor a source handle is reachable from this dictionary.
        return dict(time=self.time, zone=self.zone, cost=self.cost, observations=self.observations,
                    reviews=self.reviews.copy(), attempted=frozenset(self.attempted), completed=self.completed.copy(),
                    jobs=tuple(self.jobs.values()), prefixes=prefixes, zones=copy.deepcopy(self.zones),
                    case=copy.deepcopy(self.case), rules=copy.deepcopy(self.rules), alpha=self.alpha, beta=self.beta)

    def step(self, action):
        if self.ended:
            raise RuntimeError('Replay already terminated')
        h, budget = self.case['horizon_s'], self.case['cost_budget']
        r, obs = self.rules, self.rules['observation']
        event = dict(index=len(self.events), time=self.time, action=copy.deepcopy(action), cost_before=self.cost)
        error = None
        kind = action.get('type') if isinstance(action, dict) else None
        if len(self.events) >= r['max_actions']:
            error = 'action_limit'
        elif kind == 'stop':
            self.ended = True
        elif kind == 'wait':
            dt = action.get('seconds', float('nan'))
            if not isinstance(dt, (float,int)) or not math.isfinite(dt) or dt<=0 or self.time+dt>h+1e-9:
                error = 'invalid_wait'
            else:
                self.time += dt
        elif kind in ['review', 'execute']:
            key = action.get('job')
            j = self.jobs.get(key)
            if j is None or j.release>self.time+1e-9 or key in self.attempted:
                error = 'unavailable_job'
            elif kind == 'review':
                if (self.reviews[key]>=obs['max_per_job'] or self.observations>=self.case['observation_limit']
                    or self.cost+obs['cost']>budget+1e-9 or self.time+obs['delay_s']>h+1e-9):
                    error = 'review_limit'
                else:
                    self.cost += obs['cost']
                    self.time += obs['delay_s']
                    self.reviews[key] += 1
                    self.observations += 1
                    event.update(return_time=self.time, prefix_seconds=obs['prefix_seconds'][self.reviews[key]])
            else:
                b = action.get('reserve', float('nan'))
                distance = self.distance(self.zone, j.zone)
                start = self.time + distance/r['movement']['speed'] + r['movement']['changeover_s']
                fee = r['fees']['service_fixed'] + r['fees']['reserve_per_second']*b + r['fees']['distance_per_unit']*distance if isinstance(b,(float,int)) else float('nan')
                if any(p not in self.completed for p in j.predecessors):
                    error = 'predecessor_incomplete'
                elif not isinstance(b,(float,int)) or not math.isfinite(b) or b<r['service']['reserve_min_s']-1e-9:
                    error = 'invalid_reservation'
                elif start+b>h+1e-9 or self.cost+fee>budget+1e-9:
                    error = 'known_resource_limit'
                else:
                    # Hidden duration is read only AFTER acceptance using public conditions.
                    duration = self.alpha*float(self._sources[key]['duration_s']) + self.beta
                    finish = start + duration
                    self.cost += fee
                    self.zone = j.zone
                    self.attempted.add(key)
                    completed = finish<=h+1e-9
                    if completed:
                        self.completed[key] = finish
                    self.time = min(h, start+max(duration,b))
                    event.update(start=start, finish=finish if completed else None, completed=completed,
                        censored=not completed, observed_service=min(duration,h-start), reserve=b, distance=distance,
                        fee=fee, overrun=max(min(duration,h-start)-b,0),
                        reserved_idle=max(b-duration,0) if completed else 0.)
        else:
            error = 'unknown_action'
        if error:
            self.invalid += 1
            self.ended = True
            event['invalid_reason'] = error
        if self.time>=h-1e-9:
            self.ended = True
        event.update(time_after=self.time, cost_after=self.cost)
        self.events.append(event)
        return event

    def run(self, policy):
        while not self.ended:
            self.step(policy(self.snapshot()))
        return self.metrics()

    def metrics(self):
        return metrics_from_events(tuple(self.jobs.values()), self.case['horizon_s'], self.events)

    def source_audit(self):
        if not self.ended:
            raise RuntimeError('Source audit is unavailable before termination')
        return copy.deepcopy(self._sources)


def metrics_from_events(jobs, horizon, events):
    """Independent ledger recomputation; does not consult replay's hidden state."""
    executed = [e for e in events if e.get('action',{}).get('type')=='execute' and 'start' in e]
    completed = {e['action']['job']:e['finish'] for e in executed if e['completed']}
    total = sum(j.weight for j in jobs)
    return dict(weighted_completion_rate=sum(j.weight for j in jobs if j.job_id in completed)/total,
        weighted_on_time_rate=sum(j.weight for j in jobs if j.job_id in completed and completed[j.job_id]<=j.due+1e-9)/total,
        weighted_tardiness_s=sum(j.weight*max(completed.get(j.job_id,horizon)-j.due,0) for j in jobs),
        total_cost=events[-1]['cost_after'] if events else 0.,
        observations=sum(e.get('action',{}).get('type')=='review' and 'return_time' in e for e in events),
        elapsed_s=events[-1]['time_after'] if events else 0.,
        observed_overrun_s=sum(e['overrun'] for e in executed),
        reserved_idle_s=sum(e['reserved_idle'] for e in executed),
        censored_units=sum(e['censored'] for e in executed),
        invalid_actions=sum('invalid_reason' in e for e in events),
        valid_protocol=all('invalid_reason' not in e for e in events))
