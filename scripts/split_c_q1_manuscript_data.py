#!/usr/bin/env python3
"""Split the persisted six-panel data audit into a compact main figure and supplement."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np
import pandas as pd
from shuwei_c.figure_style import apply_style, check_figure, panel_label
from shuwei_c.q1 import contiguous_runs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/c_q1_v3/data_characteristics'
OUT = ROOT / 'artifacts/c_q1_manuscript/reflow/split_data'
PAPER = ROOT / '05_论文/待审/C题_问题一稿_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style, _ = apply_style()
    plt.rcParams.update({'font.size':8.5, 'axes.labelsize':9, 'xtick.labelsize':8.2,
                         'ytick.labelsize':8.2, 'legend.fontsize':8.2})
    c = style['palette']
    ds = ['EgoDemo', 'EgoPro', 'EgoStandard']
    inventory = pd.read_csv(DATA/'record_inventory.csv', index_col=0)
    totals = json.loads((DATA/'totals.json').read_text())
    fig, axes = plt.subplots(1,2,figsize=(5.35,2.8),layout='constrained',gridspec_kw={'width_ratios':[1,1.05]})
    ax = axes[0]
    left = np.zeros(3)
    for split,label,color,hatch in [('train','Training','blue',''),('validation','Validation','orange','//'),('test','Test metadata','neutral','..')]:
        values=inventory.loc[ds,split].to_numpy()
        ax.barh(np.arange(3),values,left=left,height=.58,color=c[color],hatch=hatch,edgecolor='white',linewidth=.4,label=label)
        for y,n,start in zip(range(3),values,left):
            ax.text(start+n/2,y,str(n),ha='center',va='center',fontsize=8.2,color='white' if split=='train' else c['text'])
        left += values
    for y,n in enumerate(left):
        ax.text(n+.6,y,f'n = {int(n)}',va='center',fontsize=8.2)
    ax.set(yticks=range(3),yticklabels=ds,xlabel='Recordings',xlim=(0,40),ylim=(2.55,-1.75))
    ax.legend(loc='upper right',frameon=False,fontsize=8.2)
    panel_label(ax,'A')
    ax=axes[1];ax.axis('off');panel_label(ax,'B')
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.006',
                     transform=ax.transAxes,facecolor=c[color]+'12',edgecolor=c[color],lw=.8,clip_on=False))
        ax.text(x,y,label,transform=ax.transAxes,fontsize=8.2,ha='center',va='center')
    for y,value,label in [(.88,totals['raw_hand_samples'],'Raw'),(.64,totals['raw_hand_samples']-totals['warmup_excluded'],'After warmup'),(.40,totals['matched_pairs'],'Matched')]:
        box(.5,y,.89,.13,f'{label}: {value:,}','blue')
    for y,label in [(.76,f'−{totals["warmup_excluded"]:,} warmup'),(.52,f'−{totals["boundary_excluded"]:,} endpoint')]:
        ax.annotate('',xy=(.5,y-.055),xytext=(.5,y+.055),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=c['neutral'],lw=.8))
        ax.text(.55,y,label,transform=ax.transAxes,fontsize=8.2,va='center')
    for x,value,label,color in [(.24,totals['qualified_targets'],'Qualified','green'),(.76,totals['flagged_targets'],'Flagged','orange')]:
        box(x,.12,.44,.20,f'{label}\n{value:,}',color)
        ax.annotate('',xy=(x,.235),xytext=(.5,.325),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=c['neutral'],lw=.8))

    def save(figure,name,local):
        check_figure(figure,style)
        for ext in ['pdf','svg','png']:
            figure.savefig(OUT/(name+'.'+ext),dpi=600,bbox_inches='tight')
        figure.savefig(OUT/(name+'_preview.png'),dpi=170,bbox_inches='tight')
        (PAPER/'figures'/local).write_bytes((OUT/(name+'.pdf')).read_bytes())
        plt.close(figure)
    save(fig,'q1_data_overview','main_01.pdf')

    audit=pd.read_csv(DATA/'record_audit.csv')
    heat=pd.read_csv(DATA/'quality_time_map.csv')
    example=pd.read_csv(DATA/'state_construction_example.csv')
    timing=pd.read_csv(DATA/'time_alignment_summary.csv')
    fig,axes=plt.subplots(2,2,figsize=(5.35,4.65),layout='constrained')
    ax=axes[0,0]
    for dataset,color,marker in zip(ds,['blue','green','orange'],['o','s','^']):
        g=audit[audit.dataset.eq(dataset)]
        ax.scatter(g.duration_s,g.flagged_hand_percent,s=17,c=c[color],marker=marker,edgecolors='white',linewidth=.3,label=f'{dataset} (n = {len(g)})')
    ax.set(xlabel='Recording duration (s)',ylabel='Flagged hand samples (%)',ylim=(-.15,audit.flagged_hand_percent.max()*1.3))
    ax.legend(loc='upper right',frameon=False,fontsize=8.2)
    ax=axes[0,1]
    cmap=LinearSegmentedColormap.from_list('flag_density',['#F0F2F3',c['orange']]);cmap.set_bad('#343A40')
    im=ax.imshow(heat.filter(regex='^bin_').to_numpy(),aspect='auto',extent=(0,100,len(heat),0),cmap=cmap,norm=PowerNorm(.5,vmin=0,vmax=100),interpolation='nearest')
    offset=0;yt=[]
    for dataset in ds:
        n=int(heat.dataset.eq(dataset).sum());yt.append(offset+n/2);offset+=n
        if offset<len(heat):ax.axhline(offset,color='white',lw=1.2)
    ax.set(xlabel='Elapsed recording (%)',yticks=yt,yticklabels=ds);ax.grid(False)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.07,pad=.14,ticks=[0,1,5,20,50,100]);cb.set_label('Flagged samples (%) [sqrt scale]',fontsize=8.2)
    ax=axes[1,0];tt=example.time_s.to_numpy()
    ax.fill_between(tt,example.joint_min_x,example.joint_max_x,color=c['neutral'],alpha=.3,label='21-landmark range')
    ax.plot(tt,example.centre_x,color=c['green'],lw=1.4,label='Hand-center x')
    for a,b in contiguous_runs(example.flag):ax.axvspan(tt[a],tt[min(b,len(tt)-1)],color=c['orange'],alpha=.22)
    handles,labels=ax.get_legend_handles_labels();handles.append(Patch(facecolor=c['orange'],alpha=.22));labels.append('Supplied bad flag')
    ax.legend(handles,labels,loc='upper left',frameon=False,fontsize=8.2)
    low,high=example.joint_min_x.min(),example.joint_max_x.max()
    ax.set(xlabel='Time in example (s)',ylabel='World x (native units)',ylim=(low-.04*(high-low),high+.65*(high-low)))
    ax=axes[1,1]
    for k,(rule,label,color,hatch) in enumerate([('First sample at/after target','First sample at/after target','neutral','//'),('Nearest admissible target','Nearest admissible target','green','')]):
        values=timing[timing.rule.eq(rule)].set_index('dataset').loc[ds,'mean_absolute_error_ms'].to_numpy()
        bars=ax.barh(np.arange(3)+(k-.5)*.3,values,height=.28,color=c[color],hatch=hatch,edgecolor='white',linewidth=.4,label=label)
        for bar,value in zip(bars,values):
            ax.annotate(f'{value:.3f}',(value,bar.get_y()+bar.get_height()/2),xytext=(3,0),textcoords='offset points',va='center',fontsize=8.2)
    ax.set(yticks=range(3),yticklabels=ds,xlabel='Mean timing error (ms)',xlim=(0,21),ylim=(2.6,-1.8))
    ax.legend(loc='upper right',frameon=False,fontsize=8.2)
    for ax,letter in zip(axes.flat,'ABCD'):panel_label(ax,letter)
    save(fig,'q1_data_diagnostics','appendix_B8.pdf')

    index=json.loads((PAPER/'figure_sources.json').read_text())
    base=index[0]
    base.update(layout_source=str((OUT/'q1_data_overview.pdf').relative_to(ROOT)),sha256=sha(PAPER/base['local']),layout_group='split_data',layout_note='Original panels A and F; relabeled A and B.')
    row=dict(number='B.8',source=base['source'],local='figures/appendix_B8.pdf',source_sha256=base['source_sha256'],sha256=sha(PAPER/'figures/appendix_B8.pdf'),layout_source=str((OUT/'q1_data_diagnostics.pdf').relative_to(ROOT)),layout_group='split_data',layout_note='Original panels B, C, D, E; relabeled A, B, C, D.')
    index=[r for r in index if r['number']!='B.8']+[row]
    (PAPER/'figure_sources.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    inputs=['record_inventory.csv','record_audit.csv','quality_time_map.csv','state_construction_example.csv','time_alignment_summary.csv','totals.json']
    (OUT/'provenance.json').write_text(json.dumps(dict(source_sha256={n:sha(DATA/n) for n in inputs},models_refitted=False,test_trajectories_read=False,script_sha256=sha(Path(__file__))),indent=2)+'\n')


if __name__=='__main__':
    main()
