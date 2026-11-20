# Problem 1: Operational states and observation reliability — development draft

We represented each hand by the centroid of its 21 recorded keypoints and its velocity. The 12-dimensional state is shared with the subsequent missing-observation experiments; the inter-hand vector is derived from the two centroids. Record-local coordinates are retained, without treating hand states as product states.

The development split contains 55 training records (47,046 frames) and seven validation records (6,783 frames). All hand coordinates are finite and the numerical timestamps are strictly increasing. Flagged observations remain available for downweighted fitting and low-quality analysis but are excluded as positional error references. Unflagged measurements are conditional numerical references, not independent error-free physical ground truth.

A quality-aware constant-velocity state-space model was compared with holding the latest centroid, two-point extrapolation and a learned multiscale linear predictor. The latter combines velocities estimated over the previous 1, 3, 6 and 12 frames using shared scalar coefficients fitted on training records. Validation parameters were chosen using environment-balanced 0.1-second forecast error normalized by training hand scale.

| dataset | constant | linear | filter | multiscale_linear |
| --- | --- | --- | --- | --- |
| EgoDemo | 0.025904 | 0.013918 | 0.013470 | 0.012972 |
| EgoPro | 0.031680 | 0.018043 | 0.018132 | 0.016950 |
| EgoStandard | 0.040176 | 0.016405 | 0.015964 | 0.015523 |

All entries are mean Euclidean errors in native coordinate units, evaluated at the first existing sample at least 0.1 seconds after each forecast origin. This is a timestamp threshold, not an exactly identical horizon or a fixed three-frame shift. Actual mean horizons are recorded in `actual_prediction_horizons.csv`; all methods share the same target timestamps. The multiscale predictor improved over two-point extrapolation on each dataset. These are validation-selected development results, and final test trajectories have not been evaluated.

Reliability was defined as the probability that either hand would receive a bad-quality flag at least 0.1 seconds later. A dataset-conditioned flag-transition model achieved Brier score 0.015277, compared with 0.020381 for a training-frequency baseline. Numerical gradient boosting improved ranking discrimination but did not consistently outperform the transition model in probability error. Features extracted from 166 training/validation videos (24,750 sampled images) did not improve this reliability target in the current fusion model. The negative result is retained. Image features do not provide an independent 3D accuracy reference.

Environment bootstrap intervals, parameter sensitivity, leave-operator stress tests and retrospective stage summaries accompany the results. Validation includes only four environments; the EgoDemo validation subset contains no positive flags, so its AUC and average precision are undefined. Stage associations are retrospective and potentially confounded by task and environment.

Sources and experiment paths: `docs/analysis/2026初赛/C题_Q1结果_v1.md`, `artifacts/c_q1_v1/`. AI-assisted draft and analysis; independent human verification pending.
