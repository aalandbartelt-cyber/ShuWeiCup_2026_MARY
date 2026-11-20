#!/usr/bin/env python3
"""Read-only scientific checks and release index; never refits or reruns test."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path('artifacts/c_q1_v3')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []
    for path in [Path('artifacts/c_q1_v2/freeze.json'), ROOT/'frozen_numeric/freeze.json', ROOT/'frozen_quality/freeze.json']:
        spec = json.loads(path.read_text())
        bound = {**spec.get('inherited_v2_hashes', {}), **spec['source_hashes']}
        for name, expected in bound.items():
            assert digest(name) == expected, f'Frozen file changed: {name}'
        checks.append(dict(check='immutable_sources', manifest=str(path), files=len(bound)))
    for component in ['frozen_numeric', 'frozen_quality']:
        path = ROOT/component/'freeze.json'
        spec = json.loads(path.read_text())
        manifest = json.loads((ROOT/component/'test/manifest.json').read_text())
        assert manifest['freeze_sha256'] == digest(path)
        # Different evaluators name their start timestamp explicitly.
        start = manifest.get('started_utc', manifest.get('evaluation_started_utc'))
        if start is not None:
            assert spec['timestamp_utc'] < start
        assert spec['test_previously_observed'] is True
        checks.append(dict(check='test_provenance', component=component, start=start))
    for split in ['validation', 'test']:
        radius = np.load(ROOT/f'frozen_numeric/{split}/main_radii.npz')['radii']
        assert np.isfinite(radius).all() and (radius >= 0).all()
        assert (np.diff(radius, axis=1) >= -1e-12).all()
    for horizon in ['0.1', '0.5']:
        endpoint = pd.read_parquet(ROOT/f'frozen_quality/test/endpoint_{horizon}.parquet')
        p = endpoint[['p0', 'p1', 'p2', 'p3']].to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        # Four softmax entries and a three-seed average are stored as float32.
        assert np.max(np.abs(p.astype(np.float64).sum(axis=1)-1)) <= 4*np.finfo(p.dtype).eps
    onset = pd.read_parquet(ROOT/'frozen_quality/test/onset.parquet')
    assert (onset.current_state == 0).all()
    p = onset.filter(like='_onset_probability').to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    diagnostics = json.loads((ROOT/'frozen_quality/test_video/diagnostics/manifest.json').read_text())
    assert len({(r['episode'], r['view']) for r in diagnostics}) == 14
    assert sum(r['frames'] for r in diagnostics) == 12246
    checks.append(dict(check='probabilities_nested_balls_and_video_completeness', status='PASS'))
    for path in [ROOT/'interface/dense_validation/verification.json', ROOT/'head_diagnostic_adapter/verification.json']:
        result = json.loads(path.read_text())
        assert result['status'] == 'PASS' and result['test_used'] is False
    sparse = json.loads((ROOT/'sparse_probe/manifest.json').read_text())
    assert sparse['test_used'] is False
    assert sparse['source_hash'] == digest('scripts/replay_c_q1_v3_sparse.py')
    checks.append(dict(check='interface_and_adapter_validation', status='PASS', sparse_coverage_guarantee=False))
    figures = ROOT/'fixed_figures'
    contracts = json.loads((figures/'legend_contracts.json').read_text())
    review = json.loads((figures/'visual_review.json').read_text())
    assert len(contracts) == 4 and review['status'] == 'PASS'
    for name, item in contracts.items():
        assert item['pdf_sha256'] == digest(figures/f'{name}.pdf') == review['pdf_sha256'][name]
        for path, expected in item['source_csv_sha256'].items():
            assert digest(path) == expected
        assert item['fonts'] and all('TimesNewRoman' in row for row in item['fonts'])
    assert '45 passed' in (ROOT/'final_tests.log').read_text()
    checks.append(dict(check='four_reviewed_figures_and_project_tests', figures=4, tests=45))
    case=ROOT/'video_case_figure'
    provenance=json.loads((case/'provenance.json').read_text())
    case_review=json.loads((case/'visual_review.json').read_text())
    assert case_review['status']=='PASS' and provenance['split']=='validation'
    assert provenance['frame_panels']==12 and provenance['probability_panels']==1
    for path,expected in provenance['sources_sha256'].items():assert digest(path)==expected,path
    assert provenance['figure_sha256']['pdf']==digest(case/'q1v3_video_keyframes.pdf')==case_review['pdf_sha256']
    assert provenance['gpu']['cpu_max_coordinate_difference']<1e-6
    assert provenance['diagnostics_max_difference']<1e-5
    frames=pd.read_csv(case/'frame_analysis.csv')
    assert (frames.frame_s<=frames.origin_s+1e-12).all()
    assert len(frames)==8 and len(frames.column.unique())==4
    trace=pd.read_csv(case/'probability_trace.csv')
    assert trace.loc[trace.current_bad,'diagnostics_leaves7_onset_probability'].isna().all()
    assert provenance['lead_s']['numeric_leaves7'] is None
    assert abs(provenance['lead_s']['diagnostics_leaves7']-.466651)<1e-6
    checks.append(dict(check='reviewed_video_keyframes_gpu_parity_and_causality',frame_panels=12,probability_panels=1,status='PASS'))
    datafig=ROOT/'data_characteristics'
    dp=json.loads((datafig/'provenance.json').read_text());dr=json.loads((datafig/'visual_review.json').read_text())
    assert dr['status']=='PASS' and not dp['test_trajectories_read'] and not dp['models_fitted']
    assert dp['point_reference_keys_match_existing_results']
    assert dp['figure_sha256']['pdf']==digest(datafig/'q1v3_data_characteristics.pdf')==dr['pdf_sha256']
    for name,expected in dr['derived_table_sha256'].items():assert digest(datafig/name)==expected,name
    assert digest(datafig/'flow_layout_audit.json')==dr['flow_layout_sha256']
    layout=json.loads((datafig/'flow_layout_audit.json').read_text())
    assert layout['status']=='PASS' and layout['opaque_borders'] and not layout['clip_on']
    assert all(x>=.025 and y>=.025 and x+w<=.975 and y+h<=.975 for x,y,w,h in layout['boxes_axes_bounds'])
    for path,expected in dp['source_sha256'].items():assert digest(path)==expected,path
    for path,expected in dp['visual_timestamp_source_sha256'].items():assert digest(path)==expected,path
    counts=json.loads((datafig/'totals.json').read_text())
    assert counts['raw_hand_samples']==107658 and counts['timestamps']==53829
    assert counts['raw_hand_samples']==sum(counts[k] for k in ['warmup_excluded','boundary_excluded','timing_mismatch_excluded','matched_pairs'])
    assert counts['matched_pairs']==counts['qualified_targets']+counts['flagged_targets']==105766
    assert counts['qualified_targets']==105057 and counts['nonfinite_hand_samples']==0
    timing=pd.read_parquet(datafig/'time_alignment_pairs.parquet')
    assert (timing.groupby(['episode','origin']).size()==2).all()
    ages=pd.read_csv(datafig/'video_alignment_age.csv')
    assert (ages[['median_age_ms','p95_age_ms','max_age_ms']]>=0).all().all()
    inventory=json.loads((ROOT/'paper_figures/inventory.json').read_text())
    assert inventory['manuscript_sha256']==digest(inventory['manuscript'])
    assert [e['figure'] for e in inventory['figures']]==list(range(1,7))
    for item in inventory['figures']:assert digest(item['pdf'])==item['sha256']
    checks.append(dict(check='six_panel_data_audit_and_manuscript_numbering',figures_in_manuscript=6,status='PASS'))
    numerical = pd.read_csv(ROOT/'frozen_numeric/test/point_metrics.csv')
    numerical = numerical[numerical.horizon.eq(.1)].set_index(['method', 'dataset'])
    quality = pd.read_csv(ROOT/'frozen_quality/test/onset_metrics.csv').query('dataset == "ALL"').set_index('method')
    metrics = dict(
        point_gain_over_multiscale_pct=100*(1-numerical.loc[('gate','ALL'),'mean_error']/numerical.loc[('multiscale','ALL'),'mean_error']),
        point_gain_over_scalar_pct=100*(1-numerical.loc[('gate','ALL'),'mean_error']/numerical.loc[('scalar','ALL'),'mean_error']),
        onset_brier_gain_pct=100*(1-quality.loc['diagnostics_leaves7','brier']/quality.loc['numeric_leaves7','brier']),
        onset_ap_gain_pct=100*(quality.loc['diagnostics_leaves7','ap']/quality.loc['numeric_leaves7','ap']-1))
    document = Path('05_论文/待审/C题_Q1_英文优化稿_v3.md')
    text = document.read_text()
    for value in ['7.27%', '90.21%', '0.112596', '24/27', '9.61%', 'previously observed', '76.17%']:
        assert value in text, value
    results = dict(status='PASS', timestamp_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
                   metrics=metrics, interpretation='Previously observed test, fixed-version reevaluation',
                   limitations=['Sparse/delayed acquisition still undercovers in several declared patterns',
                                'No independent physical truth or completed human 2D annotation',
                                'Validation video-gain interval includes zero; not a universal environment guarantee'])
    (ROOT/'assessment/final_verification.json').write_text(json.dumps(results, indent=2)+'\n')
    paths = [ROOT/'frozen_numeric/freeze.json', ROOT/'frozen_quality/freeze.json',
             ROOT/'frozen_numeric/test/manifest.json', ROOT/'frozen_quality/test/manifest.json',
             ROOT/'interface/dense_validation/verification.json', ROOT/'sparse_probe/manifest.json',
             figures/'legend_contracts.json', figures/'visual_review.json', figures/'captions.md',
             case/'provenance.json',case/'visual_review.json',case/'caption.md',case/'legend_contracts.json',
             case/'frame_analysis.csv',case/'probability_trace.csv',case/'gpu_forecasts.npz',
             datafig/'provenance.json',datafig/'visual_review.json',datafig/'caption.md',datafig/'legend_contracts.json',
             datafig/'totals.json',datafig/'record_audit.csv',datafig/'eligibility_by_record.csv',
             datafig/'flow_layout_audit.json',datafig/'export_audit.json',
             Path('scripts/report_c_q1_v3_data_characteristics.py'),
             Path('docs/analysis/2026初赛/C题_Q1数据特征图设计与核验.md'),
             datafig/'quality_time_map.csv',datafig/'time_alignment_pairs.parquet',datafig/'time_alignment_summary.csv',
             datafig/'state_construction_example.csv',datafig/'video_alignment_age.csv',
             ROOT/'paper_figures/inventory.json',ROOT/'paper_figures/captions.md',
             Path('scripts/index_c_q1_v3_paper_figures.py'),
             ROOT/'assessment/final_verification.json', document,
             Path('docs/analysis/2026初赛/C题_Q1_v3交付与结论.md'),
             Path('02_口径与约定/图表规范.md'), Path('src/shuwei_c/q1v3_interface.py'),
             Path('src/shuwei_c/q1v3_prefix.py'), Path('src/shuwei_c/q1v3_online.py'),
             Path('scripts/replay_c_q1_v3_dense.py'), Path('scripts/replay_c_q1_v3_sparse.py'), Path(__file__)]
    release = dict(version='q1-v3-reviewed-candidate-20261120', status='candidate_for_paper_review',
                   scientific_freezes=['frozen_numeric/freeze.json','frozen_quality/freeze.json'],
                   scope='Completed Q1 optimization round; not complete Q2/Q3 or formal submission',
                   interface='Q1V3(frozen_numeric/freeze.json, dataset); dense 0.1 s calibration only',
                   files_sha256={str(p):digest(p) for p in paths})
    (ROOT/'release_manifest.json').write_text(json.dumps(release, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='PASS', checks=len(checks), metrics=metrics), indent=2))


if __name__ == '__main__':
    main()
