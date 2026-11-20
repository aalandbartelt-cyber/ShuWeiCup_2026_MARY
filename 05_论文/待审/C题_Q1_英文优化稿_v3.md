# Q1: causal hand-state prediction, calibrated uncertainty, and video-assisted onset anticipation

Candidate section for review; AI-assisted drafting and computation by Codex. This section supersedes the v2 candidate for Q1. It is not a submission-ready complete contest paper. All coordinates and radii retain the supplied native units.

## Data characteristics and preprocessing

We first reconcile official record identifiers, file schemas, timestamps, and quality annotations across EgoDemo, EgoPro, and EgoStandard. The 62 training/validation recordings contain 53,829 synchronized two-hand timestamps, equivalent to 107,658 hand–time observations and 6,782,454 supplied coordinate values. These coordinate values are not independent statistical samples. The official test contributes seven further record identifiers to Figure 1A; its trajectories are excluded from the descriptive analyses in Figure 1B–F. The visual audit covers 166 development camera streams and 74,100 sampled frames at nominal 15 Hz. Of these, the 124 head-camera streams supply 54,038 frames; wrist streams were retained for exploratory comparisons.

Development recording durations range from 5.93 to 173.97 s. Supplied flags affect 750/107,658 hand observations (0.697%), but per-record fractions range from zero to 4.784%. Their concentration in short intervals motivates a separate onset target and explicit quality features. We found no nonfinite hand-coordinate samples, exact duplicate trajectories, or consecutive identical complete two-hand frames; no missing-value imputation or duplicate removal was therefore performed. Quality flags remain separate from numerical coordinates. We construct the geometric hand center from the 21 supplied landmarks without future interpolation, and video features use only the most recent already-available frame.

For 0.1 s forecasts, 1,486 hand observations fall within warmup and 406 have requested targets beyond their recording endpoints. The remaining 105,766 matched hand-target pairs divide into 105,057 unflagged numerical references and 709 flagged targets. Flagged observations remain available to the quality models; they are not globally deleted. Every eligible reference key agrees with the existing v3 training OOF/validation results. On identical admissible origins, nearest-sample matching reduces mean absolute target-time deviation from 15.493 to 0.709 ms in EgoDemo, from 9.543 to 0.085 ms in EgoPro, and from 0.429 to 0.268 ms in EgoStandard. These are preprocessing timing errors, not spatial prediction errors. Feature scales are subsequently estimated from the relevant training fold, and recording/environment balancing prevents long records from dominating evaluation.

![Q1 data characteristics and preprocessing](../../artifacts/c_q1_v3/data_characteristics/q1v3_data_characteristics.png)

**Figure 1.** Data characteristics and preprocessing for Q1. (A) Official training/validation/test record inventory. (B) Development-record duration and flagged-hand fraction. (C) Flag distribution over 40 relative-time bins per recording, grouped by dataset and ordered by duration; the labeled square-root color scale preserves the percentage values. (D) A deterministic training example showing the 21-landmark x-coordinate range and geometric hand center; the band denotes geometric spread, not uncertainty. (E) Target-time deviation before and after nearest admissible matching, using identical origins and pooled observation means. (F) Exact accounting from raw hand observations to matched targets. Flagged targets remain available for quality analysis and are excluded only from numerical reference scoring. A includes all 69 metadata entries; B–F use only the 62 development recordings. Source figure: `artifacts/c_q1_v3/data_characteristics/q1v3_data_characteristics.pdf`.

## State, information, and evaluation protocol

For hand \(a\in\{L,R\}\), let \(p_{a,k}(t)\in\mathbb R^3\) denote the supplied position of landmark \(k\), and define its geometric center as \(c_a(t)=21^{-1}\sum_{k=1}^{21}p_{a,k}(t)\). The shared state is \(s(t)=(c_L,v_L,c_R,v_R)\in\mathbb R^{12}\); relative hand displacement is derived from this state. It does not identify held objects. The supplied quality flags are distinct from independently measured physical error.

Every predictor uses only information available at its issuance time. After at least 0.4 s of history, the target is the observed timestamp nearest the requested horizon, within 0.55 median sampling intervals. Targets outside this tolerance or beyond the recording are excluded. Prediction error is evaluated only where the target's supplied hand flag is good. This eligibility rule is used for scoring, not for deciding whether an online forecast is issued. The principal horizon is 0.1 s; sensitivity experiments use 1/30, 0.2, 0.5, and 1 s. Comparisons use identical eligible targets.

The official training, validation, and test partitions contain 55, 7, and 7 recordings, respectively, spanning 16, 4, and 4 environment identifiers. These identifiers are not necessarily disjoint across dataset labels. Point errors average observations within each hand-record pair, hands within recordings, recordings within environments, and environments equally. Quality scores use recording/environment-balanced weights. Overlapping forecast windows are not independent samples. Training out-of-fold (OOF) predictions hold out complete environments; validation supports development choices. The official test had already been inspected during v2. Consequently, all v3 test numbers below are **fixed-version reevaluations of the previously observed official test**, not a fresh independent test.

## Adaptive displacement prediction

We construct ten causal vector experts: seven velocity extrapolations using lags of 1/30, 2/30, 0.1, 0.15, 0.2, 0.3, and 0.4 s, and three quadratic extrapolations fitted over the latest 5, 9, and 13 available samples. Each expert predicts displacement over the actual query interval. Let \(e_{a,k}(t,h)\) be an expert displacement divided by a training-derived hand scale \(d\). The prediction is

\[
\widehat c_a(t+h)=c_a(t)+d\sum_{k=1}^{10}
\left[b_{g,k}+\tfrac12\tanh f_{\theta,k}(x_a(t,h))\right]e_{a,k}(t,h),
\]

where \(g\) indexes the dataset and \(b_g\) is a regularized displacement baseline. These are bounded residual coefficients around the baseline, **not simplex mixture weights**. The input comprises expert norms, pairwise velocity cosines, current and sampled past flags, status duration, hand-shape singular values, head and interhand distances, hand side, dataset indicators, and already-matured linear prediction errors. Spatial norms, cosines, and singular values preserve the vector prediction's rotation equivariance under a consistent coordinate transformation.

The gate has two 48-unit SiLU hidden layers. It minimizes balanced Euclidean displacement error, with a 0.001 penalty on coefficient corrections. AdamW uses learning rate 0.001 and weight decay 0.01 for ten epochs. Three training seeds are averaged. Feature normalization and hand scales are fitted within the relevant training fold. The gate is evaluated against last-value persistence, two-point linear extrapolation, scalar velocity damping, and the earlier four-scale velocity combination. Other previously evaluated filters remain in the archived ablation results.

At 0.1 s, the adaptive model improves on the multiscale baseline by 4.98% in training environment OOF and 7.60% in development validation. Improvement occurs in 15/16 training and 4/4 validation environments. This does not imply improvement in every dataset at every horizon: EgoStandard has adverse comparisons at some nonprincipal horizons. An additional near-exact optimization of the linear initialization produces negligible change, indicating that the main gain comes from input-dependent corrections rather than merely extending the baseline solver's iteration limit.

| Fixed-version test reevaluation, 0.1 s | Scalar damping | Four-scale model | Adaptive model | Gain over four-scale |
| --- | ---: | ---: | ---: | ---: |
| All environments | 0.011781 | 0.011674 | 0.010826 | 7.27% |
| EgoDemo | 0.008348 | 0.008130 | 0.007446 | 8.42% |
| EgoPro | 0.016380 | 0.016463 | 0.015571 | 5.42% |
| EgoStandard | 0.012163 | 0.012039 | 0.011223 | 6.77% |

**Figure 2.** Center prediction at 0.1 s: (A) development validation; (B) fixed-version reevaluation of the previously observed test. Bars show environment-balanced mean error, multiplied by 1000, with direct numerical labels. Panels share the same zero-based scale. Source figure: `artifacts/c_q1_v3/fixed_figures/q1v3_fixed_state.pdf`.

## Conditional prediction balls and delayed feedback

The set for one hand is a Euclidean ball, \(B_{a,q}(t,h)=\{z:\|z-\widehat c_a(t+h)\|_2\le r_{a,q}(t,h)\}\), for nominal levels \(q\in\{0.5,0.8,0.9,0.95\}\). The present v3 conditional calibration is fitted for \(h=0.1\) s. It must not be transferred silently to other horizons or sparse acquisition patterns.

Quantile regressors predict normalized residual magnitude from the causal state features and summaries of already-arrived, qualified residuals. LightGBM quantile regressors use 15 leaves, 180 trees, learning rate 0.04, minimum leaf size 150, and L2 regularization 10. Empirical residual corrections partially pool dataset-specific shifts toward a global shift. All fitting uses training residuals from held-out environments. An additional nested assessment excludes each outer environment from both the inner point-prediction and residual-calibration fits.

Rare forecasts issued while the current hand is flagged receive a lower radius bound learned from the corresponding training OOF residuals. A cumulative maximum across nominal levels enforces nested balls. For a qualified reference that has actually arrived, the log-radius multiplier is updated using the radius issued with that forecast:

\[
\ell_{a,q}\leftarrow\operatorname{clip}\!\left(
\ell_{a,q}+\eta\min(30\Delta t,3)
\left[\mathbf1\{E_a>r^{\mathrm{issued}}_{a,q}\}-(1-q)\right],
-\log3,\log3\right),\qquad \eta=0.01.
\]

The first update uses the declared 1/30 s reference interval. Missing or unqualified references produce no update. The implementation stores issued radii and never recomputes past decisions using later information. This feedback is an empirical adaptation mechanism; it is not an implementation of a theorem establishing distribution-free coverage under the present temporal and environmental dependence.

| Nominal 90%, fixed-version test reevaluation | v2 coverage | v3 coverage | v2 radius | v3 radius | v2 / v3 pinball loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| All environments | 89.39% | 89.99% | 0.020666 | 0.019325 | 0.001612 / 0.001436 |
| EgoDemo | 90.71% | 90.06% | 0.014436 | 0.013921 | 0.000944 / 0.000971 |
| EgoPro | 82.04% | 90.21% | 0.025497 | 0.029013 | 0.002706 / 0.002226 |
| EgoStandard | 91.38% | 89.99% | 0.022428 | 0.019367 | 0.001562 / 0.001426 |

The previous EgoPro deficit is substantially reduced, accompanied by a 13.79% larger mean radius and lower pinball loss. Overall radius decreases by 6.49%. EgoDemo pinball loss increases slightly. The currently flagged test subset has only 66 qualified hand-target pairs, so its observed coverage is not strong evidence for rare-state reliability. The balls describe agreement with supplied unflagged coordinates; they do not establish physical accuracy or simultaneous two-hand coverage.

**Figure 3.** Prediction-ball calibration on the previously observed test. (A) Observed versus nominal coverage, with a linked local enlargement near nominal 90%. The earlier EgoPro point at 82.04% remains visible in the main plot and lies outside the enlargement. (B) Mean radius and observed coverage at nominal 90%, with direct percentage labels. Each of the six colored legend entries matches its actual dataset-method series, including line type, marker, and fill. The comparison concerns complete pipelines with different point predictors. Source figure: `artifacts/c_q1_v3/fixed_figures/q1v3_fixed_coverage.pdf`.

## Endpoint reliability and video-assisted onset anticipation

The endpoint quality state takes four values: neither hand flagged, only left flagged, only right flagged, and both flagged. A shared-hand 32-unit network adds bounded log-probability corrections to a numerical transition or frequency prior; softmax normalization preserves a coherent joint distribution. The implied probabilities are \(P_L=p_1+p_3\), \(P_R=p_2+p_3\), and \(P_{\mathrm{any}}=1-p_0\). At 0.1 s, the selected numerical model's aggregate endpoint Brier score is 0.004706 versus 0.004775 for its prior on fixed-version test reevaluation. This overall score is dominated by state persistence. The model detects no new onset events at its training 5% threshold, so its endpoint score must not be advertised as advance-warning performance.

A separate target addresses this limitation:

\[
Y^{\mathrm{onset}}_t=\mathbf1\{\exists u\in(t,t+0.5]:\text{a new supplied bad-flag episode begins at }u\},
\]

conditional on both hands being good at \(t\). This target retains events that begin and recover before the endpoint. It has 762 positive training origins and 215 positive validation origins, compared with 348 and 97 for endpoint abnormality in those same currently-good subsets. These are overlapping positive origins, not independent event counts.

Head-camera frames are sampled at 15 Hz. The detector is stateless, and each feature uses the latest available frame or a past frame. Numerical projection associates detections to the currently observed hands. Features summarize blur, proximity to image boundaries, detection/projection agreement, and motion relative to the background. Both views use the same fixed preprocessing; principal points are retained, and lens distortion is not corrected. The selected model does not require DINOv2 appearance vectors or wrist-camera identity assignment.

Matched numerical and video-assisted onset models each use 150 LightGBM trees with seven leaves, minimum leaf size 200, learning rate 0.03, and L2 regularization 20. Only the input features differ. Thresholds are fixed using negative examples in training environment OOF predictions.

| Onset anticipation over 0.5 s | Numerical Brier | Video Brier | Numerical AP | Video AP |
| --- | ---: | ---: | ---: | ---: |
| Training environment OOF | 0.014269 | 0.013990 | 0.034893 | 0.066384 |
| Development validation | 0.038553 | 0.037945 | 0.141873 | 0.202065 |
| Fixed-version test reevaluation | 0.013280 | 0.012908 | 0.063755 | 0.112596 |

Video reduces Brier loss by 1.95%, 1.58%, and 2.80%, respectively. On the reevaluated test, its AP increases by 76.61% relative to the matched numerical model. At the training nominal 5% false-alarm threshold, the numerical model detects 21/27 events with an actual environment-balanced false-alarm fraction of 12.09%; video detects 24/27 with 9.61%. Both have a median lead of 0.5 s. A nominal training threshold is not a guarantee of the realized test false-alarm rate. EgoDemo has no positive test onset origins; its AP is undefined rather than zero.

**Figure 4.** Video-assisted onset anticipation, conditional on both hands currently good. (A) Average precision for training OOF, development validation, and fixed-version test reevaluation, with bar values. (B) Test event recall versus actual false-alarm fraction. Point labels identify the 1%, 5%, and 10% nominal training threshold targets. Endpoint quality probabilities and window-onset probabilities are different quantities. Source figure: `artifacts/c_q1_v3/fixed_figures/q1v3_fixed_video_onset.pdf`.

We also test whether the model uses meaningful temporal video correspondence. Delaying video by 0.5, 1, or 2 s lowers validation AP from 0.2021 to 0.0565, 0.0812, and 0.0684. Permuting frames within recordings lowers it to 0.0719. The permutation is an intentionally noncausal falsification control, never a deployed input strategy. Environment-bootstrap intervals for Brier improvement are [0.57%, 3.78%] in training OOF and [−1.13%, 3.58%] in validation. The latter includes zero, and improvement occurs in only 2/4 validation environments. These development intervals are descriptive and conditional on model selection.

**Figure 5.** Validation controls that disrupt video timing. Bars display AP directly; the blue dashed line is the numerical onset baseline. Source figure: `artifacts/c_q1_v3/fixed_figures/q1v3_video_alignment_controls.pdf`.

Figure 6 connects the aggregate results to observed images. In this selected validation case, video risk crosses its threshold within the eligible window 0.467 s before a supplied right-hand bad-flag onset, while numerical onset risk does not cross its own threshold within that window. Missing detections and image motion are observable diagnostics, not independent evidence of physical failure. The screenshot sequence illustrates the computation; its selection after aggregate analysis precludes using it to estimate general performance.

**Figure 6.** Recorded keyframes and analysis around a validation onset. Columns 1–4 correspond to origins −0.467, −0.267, −0.067, and +0.100 s relative to the onset. A1–A4: latest available head-left frames, their ages, and a fixed crop rectangle. B1–B4: the enlarged crop, with detected 2D landmarks (green), projected supplied centers (white circles), and frozen 0.1 s center forecasts evaluated on the local GPU (blue arrows/crosses). L/R use supplied numerical identities. Projections use the camera pose available at the displayed frame; they do not predict the future camera image. C1–C4: background-compensated optical-flow speed over the same crop, with one shared scale in frame widths/s; values above the pooled 99th percentile saturate. D: numerical and video-assisted 0.5 s onset probabilities, fixed training thresholds, the eligible warning window (green), and supplied bad flags (orange). Numbered gray lines link to the frame columns; probabilities remain undefined during current bad states. The selected onset predictor uses both head views, while only the left view is displayed. Keypoints, flow, and onset probabilities were computed on CPU; the overlaid center forecasts were recomputed on GPU. This is an explanatory case, not independent physical validation. Source figure: `artifacts/c_q1_v3/video_case_figure/q1v3_video_keyframes.pdf`.

## Ablations, integration, and limits

The video conclusion is specific to onset anticipation. Global and hand-region DINOv2 appearance features, projected image geometry, articulated landmarks, stereo-ray anchors, neural probability corrections, and short GRU histories were also evaluated. They did not provide consistent improvements over the stronger numerical models for point prediction or the principal endpoint-quality target. A small initial 0.5 s video point-prediction gain disappeared in nested OOF residual training. These negative results remain archived; they are not omitted to imply general superiority of video or deep models.

The disclosed-prefix API reproduces the dense validation pipeline over 13,187 qualified hand-target pairs: maximum point-error difference is below \(10^{-7}\) native units, and published radii match exactly. Forecasts whose targets later become flagged are still issued; only already-arrived qualified references update calibration. This checks actual sequential execution, beyond equality at a few selected points.

A separate acquisition probe hides intermediate numerical observations and pixels during three fixed one-second windows per recording. Requests arrive every 0.2 or 0.5 s, with 0 or 0.1 s response delay. The adaptive point predictor reduces validation error relative to two-point extrapolation in all four patterns. A training-OOF conditional sparse calibration, however, attains only 76.17%–91.33% aggregate coverage at nominal 90%; restricting to genuinely unobserved moments reveals further deficits in some patterns. It is an integration diagnostic, not a completed Q2 acquisition policy or a coverage guarantee. Sparse calibration requires further work when Q2 is developed.

Task descriptions support retrospective interpretation only. Unique task descriptions, sparse environment support, and repeated observations prevent independent causal attribution to task, operator, or environment. Human 2D checks remain pending; no unfilled annotation is treated as validation. External visual estimates are not independent physical ground truth.

The numerical component was fixed at 2026-11-20 09:32:12 UTC and the quality/video component at 09:39:51 UTC, before their respective v3 test reevaluations. Their manifests bind source, checkpoints, parameters, and inherited v2 inputs. Video test extraction covers 14 head-camera streams and 12,246 sampled frames. Figures use embedded Times New Roman, external captions, exact colored legend samples, direct values, and linked enlargements. All results, including adverse subgroups, are retained. Local GPU resources were sufficient; no A100 was used.

## Method sources and artifact pointers

- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS. [Original proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). Cited for the boosting implementation, not as evidence of this task's empirical gain.
- Oquab, M., et al. *DINOv2*. [Official implementation](https://github.com/facebookresearch/dinov2). Used in archived appearance-feature ablations; not part of the selected onset predictor.
- Gibbs, I., and Candès, E. (2021). *Adaptive Conformal Inference Under Distribution Shift*. NeurIPS. [Original proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html). Related feedback-calibration context; the present log-radius update does not inherit that paper's guarantees.

Primary evidence: `artifacts/c_q1_v3/frozen_numeric/`, `frozen_quality/`, `numeric_oof/`, `calibration_nested/`, `flagged_calibration/`, `onset_h0.5/`, `interface/dense_validation/`, and `sparse_probe/`. Figure captions, source CSV hashes, and export checks are in `fixed_figures/`. Existing task-context analyses and legacy visual revisions remain separate from the new fixed-version results.
