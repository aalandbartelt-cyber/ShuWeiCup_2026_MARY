#!/usr/bin/env python3
"""Render validation-led Q1 figures from audited sources, without fitting."""
from pathlib import Path
import hashlib,json,sys,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shuwei_c.figure_style import apply_style,check_figure,panel_label

OUT=Path('artifacts/c_q1_review_repair'); PAPER=Path('05_论文/待审/C题_问题一稿_v1')
style,font=apply_style(); C=style['palette']
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
                    'mathtext.cal':'Times New Roman','mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman'})
DS=['EgoDemo','EgoPro','EgoStandard']; SOURCES={}; figures=[]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    p=Path(p);SOURCES[str(p)]=sha(p)
    return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)


def save(fig,name,local,number,meaning):
    check_figure(fig,style)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.05)
    fig.savefig(OUT/f'{name}_preview.png',dpi=160,bbox_inches='tight',pad_inches=.05)
    plt.close(fig);source=OUT/f'{name}.pdf';target=PAPER/'figures'/local
    shutil.copy2(source,target)
    figures.append(dict(number=number,source=str(source),local='figures/'+local,
        sha256=sha(target),source_sha256=sha(source),layout_note=meaning))


def main():
    points=read(OUT/'validation_point_metrics.csv'); p=points.query('horizon == 0.1')
    fig,axs=plt.subplots(1,2,figsize=(5.35,2.25),layout='constrained',gridspec_kw={'width_ratios':[1.2,1]})
    ax=axs[0];g=p[p.dataset.eq('ALL')].set_index('method')
    methods=['constant','linear','multiscale','gate']; labels=['Hold last','Two-point','Four-scale','Adaptive']
    val=np.array([g.loc[m,'mean_error']*1000 for m in methods]); colors=[C['neutral'],C['blue'],C['orange'],C['green']]
    ax.barh(np.arange(4),val,color=colors,height=.58)
    for y,v in enumerate(val):ax.text(v+.35,y,f'{v:.2f}',va='center',fontsize=9)
    ax.set(yticks=range(4),yticklabels=labels,xlabel='Mean error (native units × 1000)',xlim=(0,val.max()*1.24));ax.invert_yaxis()
    ax=axs[1];gains=[]
    for ds in DS:
        g=p[p.dataset.eq(ds)].set_index('method');gains.append(100*(1-g.loc['gate','mean_error']/g.loc['multiscale','mean_error']))
    ax.barh(range(3),gains,color=C['green'],height=.55)
    for y,v in enumerate(gains):ax.text(v+.2,y,f'{v:.2f}%',va='center',fontsize=9)
    ax.set(yticks=range(3),yticklabels=DS,xlabel='Reduction from four-scale (%)',xlim=(0,max(gains)*1.4));ax.invert_yaxis()
    for a,l in zip(axs,'AB'):panel_label(a,l)
    save(fig,'validation_motion','main_02.pdf','2','A: environment-balanced validation errors; B: dataset reductions on identical qualified targets.')

    old=read('artifacts/c_q1_v2/state_set_validation.csv').query("horizon == 0.1 and nominal == .9 and method == 'multiscale' and scheme == 'adaptive' and context == 'all'").set_index('dataset')
    new=read('artifacts/c_q1_v3/frozen_numeric/validation/radius_metrics.csv').query("horizon == .1 and nominal == .9 and units == 'native' and condition == 'all'").set_index('dataset')
    fig,axs=plt.subplots(1,2,figsize=(5.35,2.4),layout='constrained'); ds=['ALL',*DS]
    for ax,field,mult,label in zip(axs,['coverage','radius'],[100,1000],['Coverage (%)','Mean radius (native units × 1000)']):
        for k,(src,color,name) in enumerate([(old,'neutral','Motion-scaled'),(new,'green','Conditional + feedback')]):
            vals=src.loc[ds,field].to_numpy()*mult;y=np.arange(4)+(k-.5)*.3
            ax.barh(y,vals,height=.28,color=C[color],label=name)
            for yy,v in zip(y,vals):ax.text(v+.3,yy,f'{v:.2f}',fontsize=8.5,va='center')
        ax.set(yticks=range(4),yticklabels=['All',*DS],xlabel=label);ax.invert_yaxis()
    axs[0].set_xlim(0,110);axs[0].axvline(90,color=C['blue'],ls=':',lw=1)
    axs[1].set_xlim(0,34)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    for a,l in zip(axs,'AB'):panel_label(a,l)
    save(fig,'validation_coverage','main_03.pdf','3','Validation nominal-90% coverage and mean radius; bar labels eliminate overlapping traces. Different point predictors: pipeline comparison.')

    met=read('artifacts/c_q1_v3/onset_h0.5/metrics.csv'); met=met.query("split == 'validation' and dataset == 'ALL'").set_index('method')
    ev=read('artifacts/c_q1_v3/onset_h0.5/events.csv')
    fig,axs=plt.subplots(1,2,figsize=(5.35,2.45),layout='constrained')
    methods=['numeric_leaves7','diagnostics_leaves7'];names=['Numerical','Numerical + video']
    for i,(m,c) in enumerate(zip(methods,['blue','mauve'])):
        v=met.loc[m,'ap'];axs[0].bar(i,v,color=C[c],width=.5);axs[0].text(i,v+.008,f'{v:.3f}',ha='center')
        g=ev[ev.method.eq(m)].sort_values('nominal_false_alarm')
        axs[1].plot(100*g.false_alarm_fraction,100*g.event_recall,color=C[c],marker=['o','s'][i],label=names[i])
        for r in g.itertuples():
            axs[1].annotate(f'{100*r.nominal_false_alarm:.0f}%',(100*r.false_alarm_fraction,100*r.event_recall),
                            xytext=(4,-11 if i==0 else 7),textcoords='offset points',fontsize=8.5,color=C[c])
    axs[0].set(xticks=[0,1],xticklabels=['Numerical','+ video'],ylabel='Average precision',ylim=(0,.26))
    axs[1].set(xlabel='Realized false alarms (%)',ylabel='Event recall (%)',ylim=(15,102),xlim=(0,26))
    axs[1].legend(loc='lower right',fontsize=8.5,frameon=False)
    for a,l in zip(axs,'AB'):panel_label(a,l)
    save(fig,'validation_video','main_04.pdf','4','Validation AP and event recall versus false-alarm fraction; labels are thresholds fixed using training origins; 22 events, not independent origins.')

    rad=read(OUT/'horizon_baseline_radii.csv'); ra=rad[rad.dataset.eq('ALL')].sort_values('horizon')
    fig,axs=plt.subplots(2,2,figsize=(5.35,4.55),layout='constrained')
    for m,c,marker,label in [('constant','neutral','x','Hold last'),('linear','blue','s','Two-point'),('multiscale','orange','^','Four-scale'),('gate','green','o','Adaptive')]:
        g=points[points.dataset.eq('ALL')&points.method.eq(m)].sort_values('horizon')
        axs[0,0].plot(g.horizon,g.mean_error,color=C[c],marker=marker,label=label,lw=1.1)
    axs[0,0].set(xlabel='Forecast horizon (s)',ylabel='Mean error (native units)',xscale='log',yscale='log')
    axs[0,0].legend(fontsize=8.3,frameon=False,ncol=2,handlelength=1.1,columnspacing=.4,handletextpad=.3,loc='upper left')
    inset=axs[0,0].inset_axes([.53,.08,.44,.37]); inset.set_facecolor('white')
    for m,c,marker in [('linear','blue','s'),('multiscale','orange','^'),('gate','green','o')]:
        g=points[points.dataset.eq('ALL')&points.method.eq(m)].sort_values('horizon')
        inset.plot(g.horizon,g.mean_error,color=C[c],marker=marker,ms=3,lw=.9)
    inset.set(xlim=(.085,.215),ylim=(.008,.035),xticks=[.1,.2],yticks=[.01,.03])
    inset.tick_params(labelsize=8);inset.spines[['top','right']].set_visible(True)
    axs[0,0].indicate_inset_zoom(inset,edgecolor=C['text'],alpha=.65,lw=.7)
    axs[0,1].plot(ra.horizon,ra.radius,color=C['orange'],marker='^',label='Motion-scaled')
    axs[0,1].scatter([.1],[new.loc['ALL','radius']],color=C['green'],marker='o',label='Conditional, 0.1 s',zorder=4)
    axs[0,1].set(xlabel='Forecast horizon (s)',ylabel='Mean 90% radius (native units)',xscale='log',yscale='log');axs[0,1].legend(fontsize=8.5,frameon=False)
    axs[1,0].plot(ra.horizon,100*ra.coverage,color=C['orange'],marker='^')
    for r in ra.itertuples():axs[1,0].annotate(f'{100*r.coverage:.1f}',(r.horizon,100*r.coverage),xytext=(8,-12) if r.horizon==.1 else (0,6),textcoords='offset points',ha='center',fontsize=8.5)
    axs[1,0].scatter([.1],[100*new.loc['ALL','coverage']],color=C['green'],zorder=4);axs[1,0].axhline(90,color=C['neutral'],ls=':')
    axs[1,0].set(xlabel='Forecast horizon (s)',ylabel='Coverage (%)',xscale='log',ylim=(83,93))
    pivot=points.pivot(index=['dataset','horizon'],columns='method',values='mean_error');gain=100*(1-pivot.gate/pivot.multiscale)
    horizons=sorted(points.horizon.unique());a=np.array([[gain.loc[(ds,h)] for h in horizons] for ds in DS])
    from matplotlib.colors import LinearSegmentedColormap
    ax=axs[1,1];ax.grid(False);cmap=LinearSegmentedColormap.from_list('gain',[C['orange'],'#FFFFFF',C['green']])
    im=ax.imshow(a,cmap=cmap,vmin=-15,vmax=15,aspect='auto')
    for (i,j),v in np.ndenumerate(a):ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=8.5,color='white' if abs(v)>8 else C['text'])
    ax.set(yticks=range(3),yticklabels=DS,xticks=range(5),xticklabels=['1/30','.1','.2','.5','1'],xlabel='Forecast horizon (s)')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.1,pad=.25);cb.set_label('Error reduction from four-scale (%)',fontsize=8.5)
    for ax,l in zip(axs.flat,'ABCD'):panel_label(ax,l)
    save(fig,'horizon_diagnostics','appendix_B9.pdf','B.9','Validation point errors (5 horizons), existing baseline radii/coverage (4 horizons), and dataset gains. Conditional radii only at 0.1 s; no sparse-history claim.')

    env=read(OUT/'validation_environment_descriptives.csv')
    fig,axs=plt.subplots(1,2,figsize=(5.35,2.55),layout='constrained',gridspec_kw={'width_ratios':[1.8,1]})
    for i,r in env.iterrows():axs[0].plot([1000*r.gate_error,1000*r.multiscale_error],[i,i],color=C['neutral'],lw=1)
    for m,c,marker,label,dy in [('multiscale','orange','^','Four-scale',9),('gate','green','o','Adaptive',-12)]:
        vals=1000*env[m+'_error'];axs[0].plot(vals,range(len(env)),ls='',marker=marker,color=C[c],label=label)
        for i,v in enumerate(vals):axs[0].annotate(f'{v:.2f}',(v,i),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=8.5,color=C[c])
    axs[0].set(yticks=range(4),yticklabels=['E1','E2','E3','E4'],xlabel='Mean error (native units × 1000)',xlim=(6.5,15.7),ylim=(3.45,-.45))
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,fontsize=8.5,frameon=False)
    vals=100*env.flagged_fraction;axs[1].barh(range(4),vals,color=C['orange'],height=.55)
    for i,v in enumerate(vals):axs[1].text(v+.06,i,f'{v:.2f}',va='center',fontsize=8.5)
    axs[1].set(yticks=range(4),yticklabels=['E1','E2','E3','E4'],xlabel='Flagged hand samples (%)',xlim=(0,3.2));axs[1].invert_yaxis()
    for a,l in zip(axs,'AB'):panel_label(a,l)
    save(fig,'environment_diagnostics','appendix_B10.pdf','B.10','Validation environment means and record-balanced hand flag rates; E1–E4 map to source CSV and appendix text. Descriptive, no frame-independent confidence intervals.')

    failure_cases()
    inventory=json.loads((PAPER/'figure_sources.json').read_text()); replacements={f['number']:f for f in figures}
    inventory=[replacements.pop(f['number'],f) for f in inventory]+list(replacements.values())
    (PAPER/'figure_sources.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False)+'\n')
    (OUT/'figure_contracts.json').write_text(json.dumps(dict(figures=figures,source_sha256=SOURCES,font=font,rendering='Times New Roman; 600 dpi; final width 140.8 mm; no internal figure titles'),indent=2)+'\n')
    print('Rendered',len(figures),'review figures.')


def failure_cases():
    import torch
    from shuwei_c.data import Dataset
    from shuwei_c.q1v3_state import pack_records
    from shuwei_c.q1v3_learning import network,base_coefficients
    torch.set_num_threads(2)
    freeze=json.loads(Path('artifacts/c_q1_v3/frozen_numeric/freeze.json').read_text());config=json.loads(Path('configs/c_q1_v2.json').read_text());data=Dataset()
    baseline_path=Path('artifacts/c_q1_v2/state_models_development.json');SOURCES[str(baseline_path)]=sha(baseline_path)
    baseline=json.loads(baseline_path.read_text())['horizons']
    fig,axs=plt.subplots(2,1,figsize=(5.35,4.3),layout='constrained'); selections=[];rows=[]
    for ax,h,letter in zip(axs,[.5,1.],'AB'):
        f=read(OUT/f'points_h{h:.6f}.parquet');g=f[f.dataset.eq('EgoStandard')].copy()
        g['excess']=g.gate_error-g.multiscale_error;worst=g.loc[g.excess.idxmax()]
        rec=data.record(worst.episode);pack=pack_records([rec],h,config,freeze['noise']);meta=pack['meta']
        key=[k for k in freeze['point_checkpoints'] if abs(float(k)-h)<1e-10][0];pred=[]
        for path in freeze['point_checkpoints'][key]:
            SOURCES[path]=sha(path);model=torch.load(path,map_location='cpu',weights_only=False)
            x=torch.tensor(np.clip((pack['features']-model['mean'])/model['std'],-10,10),dtype=torch.float32)
            e=torch.tensor(pack['experts']);b=torch.tensor(base_coefficients(meta,model['base']))
            net=network(x.shape[1],e.shape[-1]);net.load_state_dict(model['state_dict']);net.eval()
            with torch.inference_mode():pred.append(torch.einsum('nck,nk->nc',e,b+.5*torch.tanh(net(x))).numpy())
        pred=np.mean(pred,axis=0); t,p,bad=rec.hands();c=p.mean(axis=2);hh=meta.hand.to_numpy(int);ii=meta.origin.to_numpy(int);jj=meta.target.to_numpy(int)
        forecast=c[ii,hh]+meta.scale.to_numpy()[:,None]*pred
        actual=c[jj,hh];err=np.linalg.norm(forecast-actual,axis=1)
        saved=f[f.episode.eq(rec.episode)].reset_index(drop=True)
        assert (saved[['origin','target','hand']].to_numpy()==meta[['origin','target','hand']].to_numpy()).all()
        maxdiff=float(np.max(abs(err-saved.gate_error.to_numpy())));print('CPU replay',h,maxdiff,flush=True)
        # Float32 normalized targets/checkpoints versus native float64 reference
        # subtraction differ slightly; record the discrepancy rather than round it away.
        assert maxdiff<1e-6
        take=(meta.hand==worst.hand)&(abs(meta.target_s-worst.target_s)<=1.0)
        actual=actual[take];forecast=forecast[take];tt=meta.loc[take,'target_s'].to_numpy()
        dimension=int(np.argmax(np.ptp(np.r_[actual,forecast],axis=0)));offset=float(actual[0,dimension]);anchor=float(worst.target_s)
        target_indices=jj[take];gap_positions=np.flatnonzero(np.diff(target_indices)>1)+1
        # Do not draw interpolated lines across omitted low-quality targets.
        def broken(values):return np.insert(values,gap_positions,np.nan)
        xx=broken(tt-anchor)
        ax.plot(xx,broken(actual[:,dimension]-offset),color=C['text'],lw=1.5,label='Qualified recorded center')
        ax.plot(xx,broken(forecast[:,dimension]-offset),color=C['green'],ls='--',lw=1.2,label='Adaptive forecast')
        coef=baseline[key]['family_models']['multiscale']['coefficients']
        base=c[ii,hh]+np.einsum('nck,k->nc',pack['experts'][...,[0,2,4,6]],coef)*meta.scale.to_numpy()[:,None]
        assert np.max(abs(np.linalg.norm(base-c[jj,hh],axis=1)-saved.multiscale_error.to_numpy()))<1e-6
        ax.plot(xx,broken(base[take,dimension]-offset),color=C['orange'],ls=':',lw=1.1,label='Four-scale forecast')
        ax.axvline(0,color=C['orange'],lw=.8);ax.set(xlabel='Time relative to selected failure target (s)',ylabel=f"Relative {'xyz'[dimension]} (native units)")
        lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+.18*(hi-lo))
        selected_meta=meta[(meta.origin==int(worst.origin))&(meta.hand==int(worst.hand))].iloc[0]
        origin_status='flagged origin' if bad[int(worst.origin),int(worst.hand)] else ('recently flagged history' if selected_meta.recent_bad else 'unflagged history')
        ax.text(.02,.96,f'h = {h:g} s; {origin_status}\nExcess error = {worst.excess:.4f}',transform=ax.transAxes,va='top',fontsize=8.5)
        if letter=='A':ax.legend(loc='lower left',fontsize=8.5,frameon=False)
        panel_label(ax,letter)
        selections.append(dict(episode=rec.episode,hand=int(worst.hand),horizon=h,target_s=anchor,origin=int(worst.origin),
            target=int(worst.target),gate_error=float(worst.gate_error),multiscale_error=float(worst.multiscale_error),
            excess_error=float(worst.excess),coordinate='xyz'[dimension],offset=offset,reproduced_error_max_difference=maxdiff,
            origin_flagged=bool(bad[int(worst.origin),int(worst.hand)]),recent_history_flagged=bool(selected_meta.recent_bad),
            omitted_target_gaps=int(len(gap_positions)),
            selection='Post hoc largest adaptive-minus-four-scale error among qualified EgoStandard validation targets; coordinate with largest plotted range'))
        for a,b,d,tm in zip(actual[:,dimension]-offset,forecast[:,dimension]-offset,base[take,dimension]-offset,tt-anchor):
            rows.append(dict(panel=letter,horizon=h,relative_target_s=tm,recorded_center=a,adaptive=b,four_scale=d))
    pd.DataFrame(rows).to_csv(OUT/'failure_trajectories.csv',index=False)
    (OUT/'failure_selection.json').write_text(json.dumps(selections,indent=2)+'\n')
    save(fig,'failure_trajectories','appendix_B11.pdf','B.11','Two post hoc validation failures at 0.5/1 s; native coordinate offset displayed; forecasts reproduced on CPU from unchanged checkpoints. Selection is descriptive, not additional independent evaluation.')


if __name__=='__main__':main()
