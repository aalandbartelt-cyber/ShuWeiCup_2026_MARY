#!/usr/bin/env python3
"""Same-version development evidence; no test reread and no in-figure titles."""
from pathlib import Path
import json,sys,hashlib,subprocess
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
from shuwei_c.figure_detail_v3 import line_zoom,label_bars

ROOT=Path('artifacts/c_q1_v3');OUT=ROOT/'assessment';FIG=OUT/'figures';FIG.mkdir(exist_ok=True,parents=True)
STYLE,_=apply_style();C=STYLE['palette'];contracts={}
plt.rcParams.update({'mathtext.cal':'Times New Roman','mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman'})


def save(fig,name,encoding):
    check_figure(fig,STYLE);fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=600,bbox_inches='tight');fig.savefig(FIG/(name+'_preview.png'),dpi=140,bbox_inches='tight')
    legends=list(fig.legends)+[a.get_legend() for a in fig.axes if a.get_legend() is not None]
    contracts[name]=dict(encoding=encoding,legend_labels=[t.get_text() for l in legends for t in l.get_texts()],details=getattr(fig,'detail_contract',{}),pdf_sha256=hashlib.sha256((FIG/(name+'.pdf')).read_bytes()).hexdigest())
    plt.close(fig)


def main():
    f=pd.read_csv(ROOT/'numeric_oof/comparison.csv');h=pd.read_csv(ROOT/'horizon_transfer/comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(7,3.9),layout='constrained');datasets=['EgoDemo','EgoPro','EgoStandard'];x=np.arange(3)
    methods={'scalar':('Scalar velocity damping','orange','/'),'multiscale':('Four-scale velocity combination','green','\\'),'gate':('Adaptive displacement mixture','mauve','')}
    for i,(method,(label,color,hatch)) in enumerate(methods.items()):
        values=f[(f.split=='validation')&(f.method==method)].set_index('dataset').loc[datasets,'mean_error']
        axes[0].bar(x+(i-1)*.24,1000*values,width=.23,color=C[color],label=label,hatch=hatch,edgecolor='white',linewidth=.3)
        g=h[(h.dataset=='ALL')&(h.method==method)][['horizon','mean_error']].copy();m=f[(f.split=='validation')&(f.dataset=='ALL')&(f.method==method)].mean_error.iloc[0]
        g=pd.concat([g,pd.DataFrame([dict(horizon=.1,mean_error=m)])]).sort_values('horizon')
        axes[1].plot(g.horizon,g.mean_error,color=C[color],linestyle=['--','-.','-'][i],marker=['s','^','o'][i],label=label)
    axes[0].set_xticks(x,datasets);axes[0].set_ylabel('Mean error (10⁻³ native units)');axes[1].set_xlabel('Prediction horizon (s)');axes[1].set_ylabel('Mean error (native units)');axes[1].set_xscale('log');axes[1].set_yscale('log')
    numbers=label_bars(axes[0],'.2f',rotation=90);axes[0].margins(y=.25)
    _,zoom=line_zoom(axes[1],[.48,.08,.49,.44],(.097,.103),(.0104,.0122),[.098,.100,.102],[.0105,.011,.0115,.012])
    fig.detail_contract=dict(bar_labels=numbers,bar_multiplier=1000,zooms=[zoom])
    for ax,letter in zip(axes,'AB'):panel_label(ax,letter)
    lines,labels=axes[1].get_legend_handles_labels();handles=[(Patch(facecolor=C[color],edgecolor='white',hatch=hatch),line) for (_,color,hatch),line in zip(methods.values(),lines)]
    fig.legend(handles,labels,handler_map={tuple:HandlerTuple(ndivide=None)},handlelength=4,loc='outside upper center',ncol=1,frameon=False,fontsize=8)
    save(fig,'q1v3_state_development','Color and line/hatch identify the method; A is 0.1 s validation by dataset, B is environment-balanced validation across horizons.')
    old=pd.read_csv('artifacts/c_q1_v2/state_set_validation.csv');old=old[(old.horizon==.1)&(old.context=='all')&(old.scheme=='adaptive')]
    new=pd.read_csv(ROOT/'calibration_screen/validation_metrics.csv');new=new[(new.units=='native')&(new.method=='quantile_online_leaves15_dataset')]
    fig,axes=plt.subplots(1,2,figsize=(7,4.2),layout='constrained');combo_handles={};combo_labels={}
    for ds,color,marker in zip(datasets,['blue','green','orange'],['o','s','^']):
        for frame,style in [(old,'--'),(new,'-')]:
            g=frame[frame.dataset==ds].sort_values('nominal');line,=axes[0].plot(g.nominal,g.coverage,color=C[color],marker=marker,linestyle=style,markerfacecolor='none' if style=='--' else C[color])
            combo_handles[(ds,style)]=line;combo_labels[(ds,style)]=ds+(': v2 adaptive radius' if style=='--' else ': conditional radius')
            g=g[g.nominal.eq(.9)];axes[1].plot(g.radius,g.coverage,color=C[color],marker=marker,linestyle='None',markerfacecolor='none' if style=='--' else C[color],markersize=7)
            dx=-18 if ds=='EgoStandard' and style=='--' else 0
            axes[1].annotate(f'{100*g.coverage.iloc[0]:.2f}%',(g.radius.iloc[0],g.coverage.iloc[0]),xytext=(dx,9),textcoords='offset points',ha='center',color=C[color],fontsize=8)
    ref,=axes[0].plot([.45,1],[.45,1],'--',color=C['neutral']);axes[0].set_xlabel('Nominal coverage');axes[0].set_ylabel('Observed coverage')
    axes[0].legend([ref],['Observed = nominal'],loc='upper left',frameon=False,fontsize=8)
    ref2=axes[1].axhline(.9,color=C['neutral'],linestyle='--');axes[1].set_xlabel('Mean 90% radius (native units)');axes[1].set_ylabel('Observed coverage');axes[1].set_ylim(.86,.92)
    axes[1].legend([ref2],['90% nominal coverage'],loc='lower right',frameon=False,fontsize=8)
    order=[(ds,style) for style in ['--','-'] for ds in datasets]
    fig.legend([combo_handles[k] for k in order],[combo_labels[k] for k in order],loc='outside upper center',ncol=2,frameon=False,fontsize=8)
    _,zoom=line_zoom(axes[0],[.53,.08,.44,.44],(.892,.908),(.877,.906),[.895,.900,.905],[.88,.89,.90])
    fig.detail_contract=dict(legend_style_source='actual six plotted series, no abstract black/white proxies',zooms=[zoom])
    for ax,letter in zip(axes,'AB'):panel_label(ax,letter)
    save(fig,'q1v3_radius_development','Color and shape identify dataset; line/fill identify calibration version; gray references denote equality in A and 90% in B. Both use validation, each with its own point predictor.')
    q=pd.read_csv(ROOT/'quality_oof/metrics.csv');q=q[(q.dataset=='ALL')&(q.target=='any')&(q.condition=='all')]
    fig,axes=plt.subplots(1,2,figsize=(7,3.3),layout='constrained',sharex=True);names=['numeric','diagnostics','mixture'];labels=['Numerical residual','Video diagnostics','Equal mixture']
    for ax,split,letter in zip(axes,['train_oof','validation'],'AB'):
        g=q[q.split==split].set_index('method');baseline=g.loc['transition','brier'];gain=100*(1-g.loc[names,'brier']/baseline)
        bars=ax.barh(np.arange(3),gain,color=[C['blue'],C['mauve'],C['green']]);ax.set_yticks(np.arange(3),labels);ax.axvline(0,color=C['neutral'],linestyle='--');ax.set_xlabel('Brier reduction vs transition prior (%)');panel_label(ax,letter)
        for bar,value in zip(bars,gain):
            if abs(value)>5:ax.text(value/2,bar.get_y()+bar.get_height()/2,f'{value:+.2f}%',color='white',ha='center',va='center',fontsize=8)
            else:ax.annotate(f'{value:+.2f}%',(value,bar.get_y()+bar.get_height()/2),xytext=(3 if value>=0 else -3,0),textcoords='offset points',ha='left' if value>=0 else 'right',va='center',fontsize=8)
    axes[0].set_xlim(-16,6)
    fig.legend(handles=[Line2D([],[],color=C['neutral'],linestyle='--',label='Zero gain: transition-prior Brier')],loc='outside upper center',frameon=False,fontsize=8)
    save(fig,'q1v3_video_generalization','Rows explicitly identify methods. A is whole-environment OOF; B is official validation. Positive values improve Brier; negative values worsen it. No unexplained color encoding beyond row labels.')
    (FIG/'legend_contracts.json').write_text(json.dumps(contracts,indent=2)+'\n')
    captions='''Development figures; no new official-test evaluation.

Figure 1. Adaptive displacement prediction. (A) Mean Euclidean center error at 0.1 s, by dataset. (B) Error across five predeclared horizons. Each hand is averaged within recording and each recording within environment before equal environment averaging. Panel A labels use 10⁻³ native units. Panel B retains logarithmic full-range axes and adds a linear local zoom over 0.097–0.103 s and 0.0104–0.0122 native units, centered on the prespecified 0.1 s comparison. Gray selection boxes and connectors identify the source region; source values and styles are unchanged. Native units are those of the supplied coordinates; they are not asserted to be centimetres. Recipe transfer improves the aggregate across all five horizons, but EgoStandard worsens at 0.033, 0.5 and 1 s; see the complete comparison table.

Figure 2. Empirical prediction-ball calibration at 0.1 s. (A) Observed versus nominal validation coverage. (B) Coverage–radius comparison at the 90% nominal level. The six colored legend entries directly identify each dataset–method combination. Dashed curves/open symbols identify v2; solid curves/filled symbols identify v3. The inset shows nominal coverage 0.892–0.908 and observed coverage 0.877–0.906, with the selection box and connectors retained. At nominal 90%, the v2 EgoDemo/EgoPro points genuinely nearly coincide (88.15%/88.14%); their coordinates were not shifted. Panel B directly labels the observed percentages. The v3 radius uses conditional quantiles, already-matured residual feedback and dataset-shrunk correction. Point predictors differ between versions, so this is a full-pipeline comparison, not an isolated calibration ablation. Four validation environments and qualified supplied references do not establish independent physical or distribution-free guarantees.

Figure 3. The video improvement is not yet stable across environments. Relative Brier reduction against the same split's transition prior: (A) training environment OOF; (B) official validation. Positive bars improve and negative bars worsen the score; signed percentage values are written on the bars and both panels share the same horizontal scale. The equal numerical/diagnostic mixture improves validation but worsens aggregate OOF. Development-selected models and this failure are retained together. Labels describe supplied quality flags rather than physical tracking accuracy.

All figures were generated from recorded numerical experiments by Codex; visual inspection is recorded separately, and no user/human verification is implied.
'''
    (FIG/'captions.md').write_text(captions)
    audit=[]
    for file in FIG.glob('*.pdf'):
        fonts=subprocess.check_output(['pdffonts',str(file)],text=True).splitlines()[2:];assert fonts and all('TimesNewRoman' in r and ' yes ' in r for r in fonts)
        with Image.open(file.with_suffix('.png')) as im:assert min(im.info['dpi'])>599
        audit.append(dict(figure=file.stem,pdf_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),fonts=fonts))
    (FIG/'export_audit.json').write_text(json.dumps(dict(status='PASS',figures=audit,visual_review_pending=True),indent=2)+'\n')


if __name__=='__main__':main()
