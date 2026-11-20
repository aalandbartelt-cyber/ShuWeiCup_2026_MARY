#!/usr/bin/env python3
"""Render candidate-paper tables from existing frozen results; never fit models."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path("artifacts/c_q1_v3")
OUT = Path("05_论文/待审/C题_问题一稿_v1")
SOURCES = []


def read(path):
    path = Path(path)
    SOURCES.append(path)
    return pd.read_csv(path)


def table(name, caption, label, columns, header, rows):
    text = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{" + columns + "}", r"\toprule",
        " & ".join(header) + r"\\", r"\midrule",
        *[" & ".join(row) + r"\\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (OUT / "tables" / (name + ".tex")).write_text("\n".join(text) + "\n")


def main():
    point = read("artifacts/c_q1_review_repair/validation_point_metrics.csv").query("horizon == 0.1")
    rows = []
    for ds in ["ALL", "EgoDemo", "EgoPro", "EgoStandard"]:
        g = point[point.dataset.eq(ds)].set_index("method")
        values = [g.loc[m, "mean_error"] for m in ["constant", "linear", "scalar", "multiscale", "gate"]]
        rows.append(["All environments" if ds == "ALL" else ds,
                     *[f"{1000*v:.3f}" for v in values]])
    table("point", "Validation prediction error at 0.1 s",
          "tab:point", "lrrrrr", ["Dataset", "Hold last", "Two-point", "Scalar", "Four-scale", "Adaptive"], rows)

    old = read("artifacts/c_q1_v2/state_set_validation.csv")
    new = read(ROOT / "frozen_numeric/validation/radius_metrics.csv")
    old = old.query("horizon == 0.1 and nominal == 0.9 and scheme == 'adaptive' and context == 'all' and method == 'multiscale'").set_index("dataset")
    new = new.query("horizon == 0.1 and nominal == 0.9 and units == 'native' and condition == 'all'").set_index("dataset")
    rows = []
    for ds in ["ALL", "EgoDemo", "EgoPro", "EgoStandard"]:
        a,b = old.loc[ds],new.loc[ds]
        rows.append(["All" if ds=="ALL" else ds,
                     f"{100*a.coverage:.2f}",f"{100*b.coverage:.2f}",
                     f"{a.radius:.6f}",f"{b.radius:.6f}",
                     f"{a.pinball:.6f}",f"{b.pinball:.6f}"])
    table("coverage", "Validation coverage and prediction-ball size",
          "tab:coverage","lrrrrrr",["Dataset",r"$C_{\mathrm{base}}$",r"$C_{\mathrm{cond}}$",
          r"$r_{\mathrm{base}}$",r"$r_{\mathrm{cond}}$",r"$L_{\mathrm{base}}$",r"$L_{\mathrm{cond}}$"],rows)

    metrics=pd.concat([read(ROOT/"onset_h0.5/metrics.csv"),read(ROOT/"frozen_quality/test/onset_metrics.csv")])
    metrics=metrics.query("dataset == 'ALL'").set_index(["split","method"])
    rows=[]
    for split,label in [("train_oof","Training OOF"),("validation","Validation")]:
        a=metrics.loc[(split,"numeric_leaves7")];b=metrics.loc[(split,"diagnostics_leaves7")]
        rows.append([label,f"{a.brier:.6f}",f"{b.brier:.6f}",f"{a.ap:.6f}",f"{b.ap:.6f}"])
    table("onset","Window-onset prediction at 0.5 s",
          "tab:onset","lrrrr",["Split",r"Brier$_{\mathrm{num}}$",r"Brier$_{\mathrm{video}}$",r"AP$_{\mathrm{num}}$",r"AP$_{\mathrm{video}}$"],rows)

    params=read("artifacts/c_q1_v2/state_parameter_uncertainty.csv").query("horizon == 0.1")
    rows=[[f"{r.lag_s:.4f}",f"{r.coefficient:.4f}",f"[{r.low:.4f}, {r.high:.4f}]"] for r in params.itertuples()]
    table("baseline_uncertainty","Four-scale coefficients and 95\\% bootstrap intervals",
          "tab:coef","rrr",["Lag (s)","Coefficient","95\\% interval"],rows)
    purpose=read('artifacts/c_q1_review_repair/purpose_counts.csv')
    rows=[]
    for split,label in [('train','Training'),('validation','Validation'),('test','Test')]:
        for key,meaning in [('qualified_targets','Qualified targets'),('flagged_observations','Low-quality observations'),('excluded_prediction_origins','Excluded origins')]:
            r=purpose[purpose.split.eq(split)&purpose.purpose.eq(key)].iloc[0]
            rows.append([label,meaning,str(r.records),str(r.hand_intervals),f'{r.hand_samples:,}'])
    table('retention_counts','Records, intervals, and hand samples','tab:retention_counts','llrrr',
          ['Split','Purpose','Records','Intervals','Samples'],rows)
    sampling=read('artifacts/c_q1_review_repair/sampling_distribution.csv')
    rows=[[r.dataset,f'{r.intervals:,}',f'{r.p01_ms:.3f}',f'{r.median_ms:.3f}',f'{r.p99_ms:.3f}',str(r.gap_count)] for r in sampling.itertuples()]
    table('sampling','Sampling intervals and continuity','tab:sampling','lrrrrr',
          ['Dataset','Intervals','1st percentile','Median','99th percentile','Gaps'],rows)
    sem=read('artifacts/c_q1_review_repair/semantic_descriptives.csv')
    rows=[]
    for r in sem.itertuples():
        rows.append([r.action.replace('_','/').capitalize(),str(r.phase_intervals),str(r.environments),
                     f'{r.flagged_percent:.3f}' if pd.notna(r.flagged_percent) else '--',
                     f'{r.mean_log1p_speed:.3f}' if pd.notna(r.mean_log1p_speed) else '--'])
    table('semantic_descriptives','Descriptive statistics by semantic action','tab:semantic_desc','lrrrr',
          ['Action','Phase intervals','Environments','Flagged (\\%)','Mean log-speed'],rows)
    point_test=read(ROOT/'frozen_numeric/test/point_metrics.csv').query('horizon == .1')
    radius_test=read(ROOT/'frozen_numeric/test/radius_metrics.csv').query("horizon == .1 and nominal == .9 and units == 'native' and condition == 'all'")
    rows=[]
    for ds in ['ALL','EgoDemo','EgoPro','EgoStandard']:
        p=point_test[point_test.dataset.eq(ds)].set_index('method');r=radius_test[radius_test.dataset.eq(ds)].iloc[0]
        rows.append(['All' if ds=='ALL' else ds,f'{1000*p.loc["multiscale","mean_error"]:.3f}',
                     f'{1000*p.loc["gate","mean_error"]:.3f}',f'{100*r.coverage:.2f}',f'{r.radius:.6f}'])
    table('test_supplement','Previously observed test: supplementary numerical recheck','tab:test_supplement',
          'lrrrr',['Dataset','Four-scale error','Adaptive error','Coverage (\\%)','Radius'],rows)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(SOURCES)}
    (OUT/"table_sources.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print("Rendered eight tables from audited existing result files.")


if __name__=="__main__":
    main()
