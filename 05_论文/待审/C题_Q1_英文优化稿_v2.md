# Q1: matched-time state forecasting and observation reliability (v2 draft)

AI-assisted draft for review. Captions are external; numerical units remain native.

We represent the two hands by their 21-point geometric centers and velocities, yielding a twelve-dimensional state. Relative hand displacement is derived from this state and does not represent an inventory item. Forecasts are evaluated at the nearest observed target time, with a tolerance of 0.55 median sampling intervals and at least 0.4 s of available history. Targets beyond the recording endpoint are excluded. All methods share identical eligible hand-target pairs. Reported errors average hands within recordings, recordings within environments, and environments equally.

The development comparison includes last-value persistence, two-point extrapolation, a single fitted velocity damping coefficient, regularized multiscale velocities, a quality-aware constant-velocity filter, and a vector adaptation of the 1 euro filter followed by causal velocity extrapolation. The selected method is the quality-aware filter at 1/30 s and multiscale extrapolation at the remaining declared horizons. A strong scalar baseline explains much of the gain over unregularized extrapolation.

| split | dataset | linear | scalar_damped | multiscale | gain_vs_linear_pct | gain_vs_scalar_pct |
| --- | --- | --- | --- | --- | --- | --- |
| validation | EgoDemo | 0.01129 | 0.01082 | 0.01061 | 6.08988 | 2.00702 |
| validation | EgoPro | 0.01254 | 0.01201 | 0.01191 | 5.00556 | 0.86313 |
| validation | EgoStandard | 0.01631 | 0.01562 | 0.01547 | 5.19941 | 1.00123 |
| test | EgoDemo | 0.00874 | 0.00835 | 0.00813 | 6.98439 | 2.60698 |
| test | EgoPro | 0.01719 | 0.01638 | 0.01646 | 4.23609 | -0.50325 |
| test | EgoStandard | 0.01199 | 0.01216 | 0.01204 | -0.41020 | 1.02050 |

Prediction balls are calibrated from leave-environment residuals of the same point predictor. A nonnegative scale model uses observed speed, turning, forecast age and flags at the current and sampled historical endpoints. The balls describe conditional agreement with supplied unflagged numerical references, not independent physical accuracy or joint two-hand coverage. Empirical quantiles do not provide a distribution-free guarantee under environment shift.

| split | nominal | coverage | radius | pinball |
| --- | --- | --- | --- | --- |
| validation | 0.50000 | 0.48163 | 0.00936 | 0.00272 |
| validation | 0.80000 | 0.78612 | 0.01537 | 0.00244 |
| validation | 0.90000 | 0.88801 | 0.01973 | 0.00180 |
| validation | 0.95000 | 0.94258 | 0.02456 | 0.00122 |
| test | 0.50000 | 0.46925 | 0.00981 | 0.00255 |
| test | 0.80000 | 0.78770 | 0.01610 | 0.00224 |
| test | 0.90000 | 0.89385 | 0.02067 | 0.00161 |
| test | 0.95000 | 0.94990 | 0.02573 | 0.00108 |

Reliability is represented by four joint flag states, preserving coherent single-hand and any-hand probabilities. Models are calibrated with four environment-grouped folds. Probability targets concern the future endpoint; onset and recovery events are separately evaluated over the prediction window. Thresholds are selected from training out-of-fold predictions and transferred unchanged. Event recall, lead time and actual false-alarm time must be reported together. Detection of a current abnormal condition must not be presented as advance warning of a new one.

The frozen reliability choices are {"0.1": "numerical_transition", "0.5": "numerical_frequency"}. Three feature regimes compare numerical observations, head-camera video and all available views, including a fixed 5 versus 15 Hz ablation. Head-camera projection consistency supports the coordinate convention in a selected two-detection, unflagged subset; it does not supply independent ground truth. Missing wrist detections are not equivalent to bad original hand observations.

Task descriptions are weak multi-label annotations used only for retrospective explanation. Within-record centering reduces constant recording-level confounding, but unique full task descriptions and sparse environment support prevent independent causal attribution to task, operator and environment. Semantic endpoints never enter online features.

A delayed-response integration probe confirms that point forecasts use only returned request-time samples. Dense-history uncertainty transfers poorly to sparse updates; a separate empirical residual correction is reported together with its increased radius and limited observation-pattern scope. This probe does not replace the complete Q2 evaluation.

Models and thresholds were frozen at 2026-11-20T07:28:09.059360+00:00 before the official test evaluation began at 2026-11-20T07:28:09.606729+00:00. No model was reselected on test outcomes. Independent physical accuracy and the pending blinded human 2D audit remain outside the present claims.
