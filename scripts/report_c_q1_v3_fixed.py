#!/usr/bin/env python3
"""Figures from fixed-version results; exact encodings, direct values and zooms."""
from pathlib import Path
import hashlib,json,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from PIL import Image
from shuwei_c.figure_style import apply_style,check_figure,panel_label
from shuwei_c.figure_detail_v3 import label_bars,line_zoom

ROOT=Path('artifacts/c_q1_v3');OUT=ROOT/'fixed_figures';OUT.mkdir(exist_ok=True)
STYLE,_=apply_style();C=STYLE['palette'];contracts={}
plt.rcParams.update({'mathtext.cal':'Times New Roman','mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman'})


def save(fig,name,details):
    check_figure(fig,STYLE)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',dpi=600)
    fig.savefig(OUT/f'{name}_preview.png',bbox_inches='tight',dpi=140)
    legends=list(fig.legends)+[a.get_legend() for a in fig.axes if a.get_legend() is not None]
    fonts=subprocess.check_output(['pdffonts',str(OUT/f'{name}.pdf')],text=True).splitlines()[2:]
    assert fonts and all('TimesNewRoman' in line and ' yes ' in line for line in fonts)
    with Image.open(OUT/f'{name}.png') as im:assert min(im.info['dpi'])>599
    contracts[name]=dict(details=details,legend_labels=[t.get_text() for legend in legends for t in legend.get_texts()],fonts=fonts,pdf_sha256=hashlib.sha256((OUT/f'{name}.pdf').read_bytes()).hexdigest());plt.close(fig)


def main():
    assert (ROOT/'frozen_quality/test/manifest.json').exists();datasets=['EgoDemo','EgoPro','EgoStandard'];methods=[('scalar','Scalar velocity damping','orange','///'),('multiscale','Four-scale velocity combination','green','\\\\\\'),('gate','Adaptive displacement mixture','blue','')]
    fig,axes=plt.subplots(1,2,figsize=(7,3.7),layout='constrained',sharey=True);numbers=[]
    for ax,split,letter in zip(axes,['validation','test'],'AB'):
        frame=pd.read_csv(ROOT/f'frozen_numeric/{split}/point_metrics.csv');frame=frame[frame.horizon.eq(.1)].set_index(['method','dataset']);x=np.arange(3)
        for k,(method,label,color,hatch) in enumerate(methods):ax.bar(x+(k-1)*.24,1000*frame.loc[method].loc[datasets,'mean_error'],width=.23,color=C[color],hatch=hatch,edgecolor='white',linewidth=.4,label=label)
        ax.set_xticks(x,datasets);ax.set_ylim(0,21);ax.set_ylabel('Mean error (10⁻³ native units)');numbers+=label_bars(ax,'.2f',rotation=90);panel_label(ax,letter)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=1,frameon=False,fontsize=8,handlelength=3)
    save(fig,'q1v3_fixed_state',dict(bar_labels=numbers,panels={'A':'development validation','B':'previously observed test, fixed-version reevaluation'},units='10^-3 native coordinate units; bars start at zero'))
    old=pd.read_csv('artifacts/c_q1_v2/state_set_test.csv');old=old[(old.horizon==.1)&(old.context=='all')&(old.scheme=='adaptive')]
    new=pd.read_csv(ROOT/'frozen_numeric/test/radius_metrics.csv');new=new[(new.condition=='all')&(new.units=='native')]
    fig,axes=plt.subplots(1,2,figsize=(7,4.4),layout='constrained');handles={};labels={}
    for ds,color,marker in zip(datasets,['blue','green','orange'],['o','s','^']):
        for frame,style,name in [(old,'--','v2 adaptive'),(new,'-','v3 conditional + feedback')]:
            f=frame[frame.dataset==ds].sort_values('nominal');line,=axes[0].plot(f.nominal,f.coverage,color=C[color],marker=marker,linestyle=style,markerfacecolor='none' if style=='--' else C[color]);handles[(ds,style)]=line;labels[(ds,style)]=ds+': '+name
            r=f[f.nominal.eq(.9)].iloc[0];axes[1].plot(r.radius,r.coverage,color=C[color],marker=marker,linestyle='None',markerfacecolor='none' if style=='--' else C[color],markersize=7)
            axes[1].annotate(f'{100*r.coverage:.2f}%',(r.radius,r.coverage),xytext=(0,9 if style=='--' else -14),textcoords='offset points',ha='center',fontsize=8,color=C[color])
    equality,=axes[0].plot([.45,1],[.45,1],color=C['neutral'],linestyle='--');axes[0].legend([equality],['Observed = nominal'],loc='upper left',fontsize=8,frameon=False)
    reference=axes[1].axhline(.9,color=C['neutral'],linestyle='--');axes[1].legend([reference],['90% nominal coverage'],loc='lower left',fontsize=8,frameon=False)
    axes[0].set(xlabel='Nominal coverage',ylabel='Observed coverage');axes[1].set(xlabel='Mean 90% radius (native units)',ylabel='Observed coverage',ylim=(.79,.96))
    _,zoom=line_zoom(axes[0],[.52,.07,.45,.44],(.892,.908),(.89,.922),[.895,.900,.905],[.89,.90,.91,.92])
    order=[(ds,style) for style in ['--','-'] for ds in datasets];fig.legend([handles[k] for k in order],[labels[k] for k in order],loc='outside upper center',ncol=2,fontsize=8,frameon=False)
    for ax,letter in zip(axes,'AB'):panel_label(ax,letter)
    save(fig,'q1v3_fixed_coverage',dict(zoom=zoom,legend='Six actual dataset-method combinations; colors, markers, styles and fills copied from plotted curves',scope='Previously observed test; different point predictors, full-pipeline comparison'))
    dev=pd.read_csv(ROOT/'onset_h0.5/metrics.csv');test=pd.read_csv(ROOT/'frozen_quality/test/onset_metrics.csv');allmetrics=pd.concat([dev,test],ignore_index=True);allmetrics=allmetrics[allmetrics.dataset.eq('ALL')]
    event=pd.read_csv(ROOT/'frozen_quality/test/events.csv');event=event[event.target.eq('onset_within_window')];fig,axes=plt.subplots(1,2,figsize=(7,3.9),layout='constrained');x=np.arange(3);linehandles=[];legend=[];barhandles=[]
    for k,(method,label,color,marker,style,hatch) in enumerate([('numeric_leaves7','Numerical onset model','blue','s','--','/'),('diagnostics_leaves7','Video-assisted onset model','mauve','o','-','')]):
        f=allmetrics[allmetrics.method.eq(method)].set_index('split');values=f.loc[['train_oof','validation','fixed_test'],'ap'].to_numpy();bar=axes[0].bar(x+(k-.5)*.32,values,width=.3,color=C[color],hatch=hatch,edgecolor='white');barhandles.append(Patch(facecolor=C[color],edgecolor='white',hatch=hatch))
        g=event[event.method.eq(method)].sort_values('nominal_false_alarm');assert np.all(np.diff(g.false_alarm_fraction)>=-1e-10)
        line,=axes[1].plot(100*g.false_alarm_fraction,100*g.event_recall,color=C[color],marker=marker,linestyle=style);linehandles.append(line);legend.append(label)
        for r in g.itertuples():axes[1].annotate(f'{100*r.nominal_false_alarm:.0f}%',(100*r.false_alarm_fraction,100*r.event_recall),xytext=(3,8 if k else -13),textcoords='offset points',fontsize=8,color=C[color])
    axes[0].set_xticks(x,['Training OOF','Validation','Fixed test']);axes[0].set_ylabel('Average precision');axes[0].margins(y=.25);numbers=label_bars(axes[0],'.3f')
    axes[1].set_xlabel('Observed false-alarm fraction (%)');axes[1].set_ylabel('Detected onset events (%)');axes[1].set_ylim(-3,103);axes[1].margins(x=.15)
    fig.legend(list(zip(barhandles,linehandles)),legend,handler_map={tuple:HandlerTuple(ndivide=None)},handlelength=4,loc='outside upper center',ncol=2,frameon=False,fontsize=8)
    axes[1].text(.02,.98,'Point labels: training false-alarm target',transform=axes[1].transAxes,fontsize=8,va='top')
    for ax,letter in zip(axes,'AB'):panel_label(ax,letter)
    save(fig,'q1v3_fixed_video_onset',dict(bar_labels=numbers,target='Any new bad-flag onset within 0.5 s, conditional on currently good; not endpoint bad probability',event_scope='Previously observed test; thresholds fixed from training OOF negatives'))
    controls=pd.read_csv(ROOT/'onset_h0.5/audit/control_metrics.csv').query('dataset=="ALL"').set_index('method');names=['aligned','lag_0.5s','lag_1s','lag_2s','permuted_within_episode'];labels=['Correctly aligned video','Video delayed 0.5 s','Video delayed 1 s','Video delayed 2 s','Video permuted within record'];fig,ax=plt.subplots(figsize=(7,2.8),layout='constrained')
    bars=ax.barh(np.arange(5),controls.loc[names,'ap'],color=[C['mauve']]+[C['neutral']]*4);ax.set_yticks(np.arange(5),labels);ax.invert_yaxis();ax.set_xlabel('Validation average precision');ax.set_xlim(0,.24)
    for bar in bars:ax.annotate(f'{bar.get_width():.3f}',(bar.get_width(),bar.get_y()+bar.get_height()/2),xytext=(4,0),textcoords='offset points',va='center',fontsize=8)
    numeric=float(dev.query('dataset=="ALL" and split=="validation" and method=="numeric_leaves7"').ap.iloc[0]);line=ax.axvline(numeric,color=C['blue'],linestyle='--');ax.legend([line],['Numerical onset model'],loc='lower right',fontsize=8,frameon=False)
    save(fig,'q1v3_video_alignment_controls',dict(scope='Fixed development-selected 0.5 s onset model on validation',warning='Permutation intentionally breaks causality as a falsification control; never deployed'))
    captions='''Figure 1. Center prediction at 0.1 s: (A) development validation; (B) fixed-version reevaluation of the previously observed test. Each hand is averaged within a recording and each recording within an environment. Labels give native coordinate error multiplied by 1000; no centimetre claim is made. All bars start at zero and panels share the same scale.

Figure 2. Prediction-ball calibration on the previously observed test. (A) Observed versus nominal coverage, with a faithful local zoom around nominal 90%. The old EgoPro point at 82.04% lies outside the zoom and remains visible in the full-range plot. (B) Mean radius versus observed coverage at nominal 90%, with direct percentage labels. The six legend entries each reproduce an actual dataset-method combination. This compares complete pipelines with different point predictors. V3 includes training-OOF conditional quantiles, a current-flag radius floor, and delayed feedback using already-arrived qualified references. It does not establish physical or distribution-free coverage.

Figure 3. Video-assisted onset anticipation over a 0.5 s window, conditional on both hands currently good. (A) Average precision in training environment OOF, development validation, and fixed-version test reevaluation. (B) Test event recall versus actual false-alarm fraction. Point labels are nominal training false-alarm targets (1%, 5%, 10%); they are not assertions about test false-alarm rates. These onset probabilities are distinct from endpoint four-state probabilities. Both numeric and video models use the same tree capacity.

Figure 4. Validation falsification controls for the fixed video-assisted onset model. Shifting video into the past or permuting it within a record damages the correct temporal correspondence. The numerical model is shown as an actual blue dashed reference line. The permutation is an intentionally noncausal diagnostic control and is never used for deployment. Source-level numerical tables and all negative results remain available.

All captions are external to the figures. Exported fonts are genuine embedded Times New Roman; PDF/SVG/600 dpi PNG and previews are supplied. Codex visual review is recorded separately and does not imply user approval.
'''
    sources={
        'q1v3_fixed_state':[ROOT/'frozen_numeric'/s/'point_metrics.csv' for s in ['validation','test']],
        'q1v3_fixed_coverage':[Path('artifacts/c_q1_v2/state_set_test.csv'),ROOT/'frozen_numeric/test/radius_metrics.csv'],
        'q1v3_fixed_video_onset':[ROOT/'onset_h0.5/metrics.csv',ROOT/'frozen_quality/test/onset_metrics.csv',ROOT/'frozen_quality/test/events.csv'],
        'q1v3_video_alignment_controls':[ROOT/'onset_h0.5/audit/control_metrics.csv',ROOT/'onset_h0.5/metrics.csv']}
    for name,paths in sources.items():contracts[name]['source_csv_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'captions.md').write_text(captions);(OUT/'legend_contracts.json').write_text(json.dumps(contracts,indent=2)+'\n');(OUT/'export_audit.json').write_text(json.dumps(dict(status='PASS',figures=len(contracts),visual_review_pending=True,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
