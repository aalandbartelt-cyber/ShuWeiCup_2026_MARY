"""Causal observation value features and delayed counterfactual development.

Policies see only Context. `counterfactual_pair` is an offline training helper:
its private replay references produce labels, never deployment feature columns.
"""
import numpy as np
from .q2_features import CausalExperts, transform_features
from .q2_replay import EPS, Periodic, replay
from .q2_predictors import Kinematic


class ValueFeatures:
    def __init__(self, dataset, scale):
        self.extractor = CausalExperts(dataset)
        self.scale = float(scale)

    def __call__(self, context, cost_weight=0.):
        pack = self.extractor.at(context.history, context.now)
        f = transform_features(pack['features'], pack['feature_names'], np.full(2, self.scale))
        names = [f'hand{hand}_{name}' for hand in range(2) for name in pack['feature_names']]
        extra = dict(elapsed_s=context.now-context.origin,
                     remaining_s=max(context.end-context.now, 0.),
                     remaining_scoring_s=max(context.last_score_time-context.now, 0.),
                     horizon_s=context.end-context.origin,
                     return_delay_s=context.delay,
                     budget_remaining=context.remaining,
                     pending_count=len(context.pending), cost_weight=float(cost_weight))
        pending = sorted(context.pending, key=lambda x: x.arrival_time)
        for j in range(4):
            present = j < len(pending)
            extra[f'pending_{j}_known'] = float(present)
            extra[f'pending_{j}_eta'] = max(pending[j].arrival_time-context.now, 0.) if present else 0.
            extra[f'pending_{j}_sample_age'] = context.now-pending[j].sample_time if present else 0.
        return np.r_[f.ravel(), list(extra.values())].astype(np.float32), names+list(extra)


class ValuePolicy:
    def __init__(self, model, dataset, scale, cost_weight=0., single_pending=False):
        self.model = model
        self.features = ValueFeatures(dataset, scale)
        self.cost_weight = cost_weight
        self.single_pending = single_pending

    def __call__(self, context):
        if context.remaining <= 0 or context.now+context.delay > context.last_score_time+EPS:
            return False
        if self.single_pending and context.pending:
            return False
        x, names = self.features(context, self.cost_weight)
        # Single-row online inference must not create an OpenMP team per tick.
        # Loaded LightGBM Boosters otherwise fall back to all host CPU cores.
        options = {'num_threads': 1} if type(self.model).__module__.startswith('lightgbm') else {}
        gain = float(np.asarray(self.model.predict(x[None], **options)).reshape(-1)[0])
        return gain > 0.


class BranchPolicy:
    """Replay identical issued requests, force one action, then continue legally."""
    def __init__(self, prefix, decision_time, request_now, continuation=None, phase=.5):
        self.prefix = tuple(prefix)
        self.decision_time, self.request_now = decision_time, request_now
        self.continuation, self.phase = continuation, phase
        self.schedule = None
        self.cursor = 0

    def __call__(self, context):
        if context.now < self.decision_time-EPS:
            return any(abs(context.now-t) <= EPS for t in self.prefix)
        if abs(context.now-self.decision_time) <= EPS:
            return bool(self.request_now)
        if self.continuation is not None:
            return bool(self.continuation(context))
        if context.remaining <= 0 or context.now+context.delay > context.last_score_time+EPS:
            return False
        if self.schedule is None:
            span = context.last_score_time-context.delay-self.decision_time
            n = context.remaining
            self.schedule = self.decision_time+(np.arange(n)+self.phase)*max(span, 0.)/n
        if self.cursor < len(self.schedule) and context.now+EPS >= self.schedule[self.cursor]:
            self.cursor += 1
            while self.cursor < len(self.schedule) and self.schedule[self.cursor] <= context.now+EPS:
                self.cursor += 1
            return True
        return False


def counterfactual_pair(record, window, context, issued_prefix, cap, delay, scale,
                        predictor_factory=None, continuation_factory=None, numerical=None):
    if context.remaining <= 0 or context.now+delay > context.last_score_time+EPS:
        raise ValueError('State has no useful legal request action')
    predictor_factory = predictor_factory or (lambda: Kinematic('damped', damping=.2))
    outcomes = []
    for action in (False, True):
        continuation = continuation_factory() if continuation_factory else None
        policy = BranchPolicy(issued_prefix, context.now, action, continuation)
        result = replay(record, window, predictor_factory(), policy, cap, delay, numerical=numerical)
        outcomes.append(result)
    wait, request = outcomes
    before = wait['times'] < context.now-EPS
    if not np.array_equal(wait['predictions'][before], request['predictions'][before]):
        raise ValueError('Counterfactuals disagree before the forced decision')
    return dict(error_gain=(wait['metrics']['mean_error']-request['metrics']['mean_error'])/scale,
                cost_change=request['metrics']['total_cost']-wait['metrics']['total_cost'],
                waiting_change=request['metrics']['waiting_union_s']-wait['metrics']['waiting_union_s'],
                wait_error=wait['metrics']['mean_error']/scale,
                request_error=request['metrics']['mean_error']/scale,
                wait_requests=wait['metrics']['requests'], request_requests=request['metrics']['requests'])


class DecisionRecorder:
    def __init__(self, policy):
        self.policy = policy
        self.states, self.issued = [], []

    def __call__(self, context):
        if context.remaining > 0 and context.now+context.delay <= context.last_score_time+EPS:
            self.states.append((context, tuple(self.issued)))
        action = bool(self.policy(context))
        if action:
            self.issued.append(context.now)
        return action
