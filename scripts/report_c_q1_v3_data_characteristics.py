#!/usr/bin/env python3
"""Q1 data characteristics and actual preprocessing; no fitting or test-data EDA.

AI-assisted by Codex, 2026-11-20. Test contributes official metadata counts only.
"""
from pathlib import Path
import hashlib,json,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap,PowerNorm,to_rgba
from PIL import Image
from shuwei_c.data import Dataset
from shuwei_c.q1 import contiguous_runs,audit_records
from shuwei_c.q1v2_state import matched_targets
from shuwei_c.figure_style import apply_style,check_figure,panel_label

ROOT=Path('artifacts/c_q1_v3');OUT=ROOT/'data_characteristics'
DATASETS=['EgoDemo','EgoPro','EgoStandard']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True);style,font=apply_style();C=style['palette']
    plt.rcParams.update({'mathtext.cal':'Times New Roman','mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman'})
    frozen=json.loads((ROOT/'frozen_numeric/freeze.json').read_text())
    for path,expected in {**frozen['inherited_v2_hashes'],**frozen['source_hashes']}.items():assert sha(path)==expected,path
    data=Dataset();records=data.records('train')+data.records('validation');metadata=pd.DataFrame(data.split_rows)
    inventory=metadata.groupby(['dataset','split']).size().unstack(fill_value=0).reindex(DATASETS)[['train','validation','test']]
    inventory.to_csv(OUT/'record_inventory.csv')
    audit,intervals,duplicates=audit_records(records);old=pd.read_csv('artifacts/c_q1_v1/record_audit.csv').set_index('episode')
    pd.testing.assert_series_equal(audit.set_index('episode').frames.sort_index(),old.frames.sort_index())
    assert not duplicates and audit.nonfinite_hand_frames.sum()==0
    audit['flagged_hand_percent']=100*(audit.left_bad_frames+audit.right_bad_frames)/audit.total_hand_frames
    audit.to_csv(OUT/'record_audit.csv',index=False);intervals.to_csv(OUT/'quality_intervals.csv',index=False)
    timing=[];flow=[];heat=[];heat_ids=[];age_rows=[];example=None;canonical_hashes={};visual_hashes={}
    video=json.loads(Path('artifacts/c_video_v2_15hz/video_manifest.json').read_text());visuals={(v['episode'],v['view']):v for v in video}
    cache_meta={}
    for split,path in [('train',ROOT/'numeric_oof/train_oof.parquet'),('validation',ROOT/'frozen_numeric/validation/points_h0.100000.parquet')]:
        cache_meta[split]=pd.read_parquet(path,columns=['episode','origin','target','hand'])
    for rec in records:
        t,p,bad=rec.hands();canonical_hashes[rec.episode]=hashlib.sha256(t.tobytes()+p.tobytes()+bad.tobytes()).hexdigest();i,j=matched_targets(t,.1);warm=(t>=t[0]+.4-1e-9)&(np.arange(len(t))>=2)
        desired=t[warm]+.1;outside=desired>t[-1]+1e-9;inside=int(warm.sum()-outside.sum()-len(i));assert inside>=0
        count=dict(episode=rec.episode,dataset=rec.dataset,split=rec.split,raw_hand_samples=2*len(t),warmup_excluded=2*int((~warm).sum()),boundary_excluded=2*int(outside.sum()),timing_mismatch_excluded=2*inside,matched_pairs=2*len(i),flagged_targets=int(bad[j].sum()),qualified_targets=int((~bad[j]).sum()))
        assert count['raw_hand_samples']==sum(count[k] for k in ['warmup_excluded','boundary_excluded','timing_mismatch_excluded','matched_pairs'])
        assert count['matched_pairs']==count['flagged_targets']+count['qualified_targets'];flow.append(count)
        expected=pd.DataFrame([dict(episode=rec.episode,origin=int(a),target=int(b),hand=h) for a,b in zip(i,j) for h in range(2) if not bad[b,h]])
        got=cache_meta[rec.split];got=got[got.episode.eq(rec.episode)]
        pd.testing.assert_frame_equal(expected.sort_values(['origin','hand']).reset_index(drop=True),got.sort_values(['origin','hand']).reset_index(drop=True),check_dtype=False)
        ceiling=np.searchsorted(t,t[i]+.1-1e-9);common=ceiling<len(t)
        for label,targets in [('First sample at/after target',ceiling[common]),('Nearest admissible target',j[common])]:
            actual=t[targets]-t[i[common]];error=1000*abs(actual-.1)
            for a,b,h,e in zip(i[common],targets,actual,error):timing.append(dict(episode=rec.episode,dataset=rec.dataset,split=rec.split,origin=int(a),target=int(b),rule=label,actual_horizon_ms=1000*float(h),absolute_timing_error_ms=float(e)))
        bins=np.minimum(((t-t[0])/(t[-1]-t[0])*40).astype(int),39);values=np.full(40,np.nan)
        for k in range(40):
            mask=bins==k
            if mask.any():values[k]=100*bad[mask].mean()
        heat.append(values);heat_ids.append(rec.episode)
        for view in ['head_left','head_right']:
            entry=visuals[(rec.episode,view)];source=Path('artifacts/c_video_v2_15hz')/entry['features_file'];visual_hashes[str(source)]=sha(source);vt=np.array([json.loads(line)['time_s'] for line in source.open()]);vi=np.searchsorted(vt,t[i],side='right')-1;known=vi>=0
            age=t[i[known]]-vt[vi[known]];assert (age>=0).all()
            age_rows.append(dict(episode=rec.episode,view=view,known_origins=int(known.sum()),missing_origins=int((~known).sum()),median_age_ms=float(np.median(age)*1000),p95_age_ms=float(np.quantile(age,.95)*1000),max_age_ms=float(age.max()*1000)))
        if example is None and rec.split=='train' and rec.dataset=='EgoPro':
            runs=[(a,b) for a,b in contiguous_runs(bad.any(1)) if t[a]-t[0]>2 and t[-1]-t[a]>2]
            if runs:
                a,b=runs[0];hand=int(np.argmax(bad[a:b].sum(0)));take=(t>=t[a]-2)&(t<=t[a]+2);joint=p[take,hand,:,0]
                example=pd.DataFrame(dict(time_s=t[take],joint_min_x=joint.min(1),joint_max_x=joint.max(1),centre_x=joint.mean(1),flag=bad[take,hand]))
                example_id=dict(episode=rec.episode,hand=['left','right'][hand],event_s=float(t[a]),selection='First training EgoPro record in official order with an onset at least 2 seconds from both endpoints; first such onset; hand with most flags in that run')
    flow=pd.DataFrame(flow);timing=pd.DataFrame(timing);heat=np.array(heat)
    example.to_csv(OUT/'state_construction_example.csv',index=False);flow.to_csv(OUT/'eligibility_by_record.csv',index=False);timing.to_parquet(OUT/'time_alignment_pairs.parquet',index=False);pd.DataFrame(age_rows).to_csv(OUT/'video_alignment_age.csv',index=False)
    summary=timing.groupby(['dataset','rule']).agg(origins=('origin','size'),mean_absolute_error_ms=('absolute_timing_error_ms','mean'),median_actual_ms=('actual_horizon_ms','median'),p05_actual_ms=('actual_horizon_ms',lambda x:np.quantile(x,.05)),p95_actual_ms=('actual_horizon_ms',lambda x:np.quantile(x,.95))).reset_index();summary.to_csv(OUT/'time_alignment_summary.csv',index=False)
    quality=audit.sort_values(['dataset','duration_s','episode'],ascending=[True,False,True]);heat=heat[[heat_ids.index(u) for u in quality.episode]]
    hf=pd.DataFrame(heat,columns=[f'bin_{k:02d}' for k in range(40)]);hf.insert(0,'episode',quality.episode.to_list());hf.insert(1,'dataset',quality.dataset.to_list());hf.to_csv(OUT/'quality_time_map.csv',index=False)
    totals={k:int(flow[k].sum()) for k in flow if k not in ['episode','dataset','split']}
    totals.update(records_all=len(metadata),records_development=len(records),timestamps=int(audit.frames.sum()),coordinate_values=int(audit.frames.sum())*126,flagged_hand_samples=int((audit.left_bad_frames+audit.right_bad_frames).sum()),duplicate_trajectory_groups=len(duplicates),nonfinite_hand_samples=int(audit.nonfinite_hand_frames.sum()),videos_development=len(video),video_sampled_frames=sum(v['sampled_frames'] for v in video),head_streams=sum(v['view'].startswith('head_') for v in video))
    (OUT/'totals.json').write_text(json.dumps(totals,indent=2)+'\n')
    fig,axes=plt.subplots(3,2,figsize=(7,7.9),layout='constrained');ax=axes[0,0];left=np.zeros(3)
    for split,label,color,hatch in [('train','Training','blue',''),('validation','Validation','orange','//'),('test','Test metadata','neutral','..')]:
        values=inventory[split].to_numpy();ax.barh(np.arange(3),values,left=left,height=.62,color=C[color],hatch=hatch,label=label,edgecolor='white',linewidth=.4)
        for y,value,start in zip(range(3),values,left):ax.text(start+value/2,y,str(value),ha='center',va='center',fontsize=8,color='white' if split=='train' else C['text'])
        left+=values
    for y,total in enumerate(left):ax.text(total+.6,y,f'n = {int(total)}',va='center',fontsize=8)
    ax.set(yticks=range(3),yticklabels=DATASETS,xlabel='Recordings',xlim=(0,39),ylim=(2.6,-1.9));ax.legend(loc='upper right',ncol=1,fontsize=8,frameon=False,handlelength=2.)
    ax=axes[0,1]
    for ds,color,marker in zip(DATASETS,['blue','green','orange'],['o','s','^']):
        g=audit[audit.dataset.eq(ds)];ax.scatter(g.duration_s,g.flagged_hand_percent,s=19,marker=marker,color=C[color],edgecolors='white',linewidths=.3,label=f'{ds} (n = {len(g)})',alpha=.9)
    ax.set(xlabel='Recording duration (s)',ylabel='Flagged hand samples (%)',ylim=(-.15,max(audit.flagged_hand_percent)*1.22));ax.legend(fontsize=8,frameon=False,loc='upper right')
    ax=axes[1,0];cmap=LinearSegmentedColormap.from_list('flag_density',['#F0F2F3',C['orange']]);cmap.set_bad('#343A40')
    image=ax.imshow(heat,aspect='auto',interpolation='nearest',extent=(0,100,len(heat),0),cmap=cmap,norm=PowerNorm(.5,vmin=0,vmax=100));offset=0;yt=[]
    for ds in DATASETS:
        n=int(quality.dataset.eq(ds).sum());yt.append(offset+n/2);offset+=n
        if offset<len(heat):ax.axhline(offset,color='white',lw=1.2)
    ax.set(xlabel='Elapsed fraction of recording (%)',yticks=yt,yticklabels=DATASETS);ax.grid(False)
    cb=fig.colorbar(image,ax=ax,orientation='horizontal',fraction=.07,pad=.13,aspect=28,ticks=[0,1,5,20,50,100]);cb.set_label('Flagged hand samples (%) [sqrt color scale]',fontsize=8);cb.ax.tick_params(labelsize=8)
    ax=axes[1,1];tt=example.time_s.to_numpy();ax.fill_between(tt,example.joint_min_x,example.joint_max_x,color=C['neutral'],alpha=.3,label='21-landmark x range (min–max)');ax.plot(tt,example.centre_x,color=C['green'],lw=1.5,label='Hand-center x')
    for a,b in contiguous_runs(example.flag):ax.axvspan(tt[a],tt[min(b,len(tt)-1)],color=C['orange'],alpha=.22)
    handles,labels=ax.get_legend_handles_labels();handles.append(Patch(facecolor=C['orange'],alpha=.22));labels.append('Supplied bad flag')
    ax.legend(handles,labels,fontsize=8,frameon=False,loc='upper left');ax.set(xlabel='Time within example recording (s)',ylabel='World x (native units)')
    low=float(example.joint_min_x.min());high=float(example.joint_max_x.max());ax.set_ylim(low-.04*(high-low),high+.65*(high-low))
    ax=axes[2,0]
    for k,(rule,color,hatch) in enumerate([('First sample at/after target','neutral','//'),('Nearest admissible target','green','')]):
        values=summary[summary.rule.eq(rule)].set_index('dataset').loc[DATASETS,'mean_absolute_error_ms'].to_numpy()
        bars=ax.barh(np.arange(3)+(k-.5)*.3,values,height=.28,color=C[color],hatch=hatch,edgecolor='white',linewidth=.4,label=rule)
        for bar,value in zip(bars,values):ax.annotate(f'{value:.3f}',(value,bar.get_y()+bar.get_height()/2),xytext=(3,0),textcoords='offset points',va='center',fontsize=8)
    ax.set(yticks=range(3),yticklabels=DATASETS,xlabel='Mean absolute target-time error (ms)',xlim=(0,21),ylim=(2.55,-1.15));ax.legend(loc='upper right',fontsize=8,frameon=False,handlelength=2.5)
    ax=axes[2,1];ax.axis('off');flow_boxes=[]
    def box(x,y,w,h,label,color):
        patch=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.01,rounding_size=.015',facecolor=to_rgba(C[color],.10),edgecolor=C[color],linewidth=1.0,transform=ax.transAxes,clip_on=False,zorder=2)
        ax.add_patch(patch);flow_boxes.append(patch)
        ax.text(x,y,label,ha='center',va='center',fontsize=8,transform=ax.transAxes)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=C['neutral'],lw=1.))
    box(.5,.89,.80,.14,f'Raw hand–time observations\n{totals["raw_hand_samples"]:,}','blue')
    box(.5,.635,.80,.14,f'After 0.4 s warmup\n{totals["raw_hand_samples"]-totals["warmup_excluded"]:,}','blue')
    box(.5,.38,.80,.14,f'Time-matched 0.1 s pairs\n{totals["matched_pairs"]:,}','blue')
    arrow((.5,.81),(.5,.715));ax.text(.52,.763,f'−{totals["warmup_excluded"]:,} warmup',fontsize=8,ha='left',va='center',transform=ax.transAxes)
    assert totals['timing_mismatch_excluded']==0
    arrow((.5,.555),(.5,.46));ax.text(.52,.507,f'−{totals["boundary_excluded"]:,} record endpoint',fontsize=8,ha='left',va='center',transform=ax.transAxes)
    arrow((.35,.30),(.26,.22));arrow((.65,.30),(.74,.22));box(.26,.14,.40,.14,f'Qualified targets\n{totals["qualified_targets"]:,}','green');box(.74,.14,.40,.14,f'Flagged targets\n{totals["flagged_targets"]:,}','orange')
    box_bounds=[p.get_path().get_extents().bounds for p in flow_boxes]
    assert all(x>=.025 and y>=.025 and x+w<=.975 and y+h<=.975 for x,y,w,h in box_bounds),box_bounds
    (OUT/'flow_layout_audit.json').write_text(json.dumps(dict(status='PASS',boxes_axes_bounds=box_bounds,clip_on=False,opaque_borders=True,minimum_clearance_axes_fraction=.025),indent=2)+'\n')
    for ax,letter in zip(axes.flat,'ABCDEF'):panel_label(ax,letter)
    check_figure(fig,style);name='q1v3_data_characteristics'
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=600,bbox_inches='tight')
    fig.savefig(OUT/f'{name}_preview.png',dpi=160,bbox_inches='tight');plt.close(fig)
    caption=f'''Figure 1. Data characteristics and audited preprocessing for Q1. (A) Official recording counts: all 69 metadata entries, with training, validation, and test shown separately. (B) Duration and flagged-hand fraction for the 62 development recordings; each point is one recording, not an independent time sample. (C) Quality flags across 40 relative-time bins per development recording, grouped by dataset and sorted by decreasing duration. Each cell is the flagged fraction among both hands in that bin; empty bins remain missing. (D) A deterministic training example: the range of 21 supplied landmark x coordinates and their geometric center, with supplied bad-flag intervals. The band is geometric spread, not a confidence interval; center construction is not a smoothing claim. (E) Mean absolute deviation from the requested 0.1 s target on identical admissible origins: first sample at/after target versus the adopted nearest-sample rule (0.55 median-step tolerance). These descriptive means pool origins; they are not environment-balanced prediction-error scores. (F) Exact sample accounting: {totals['raw_hand_samples']:,} hand–time observations, {totals['warmup_excluded']:,} warmup exclusions, {totals['boundary_excluded']+totals['timing_mismatch_excluded']:,} endpoint/timing exclusions, and {totals['matched_pairs']:,} matched pairs. The {totals['flagged_targets']:,} flagged targets are excluded only from numerical reference scoring and remain available for quality/onset analysis; no entire recording is removed. All numerical distributions use training/validation only; test contributes metadata to A. No numerical hand values were missing and no exact duplicate trajectories were found. The 166 development videos supply {totals['video_sampled_frames']:,} sampled frames at nominal 15 Hz; image features use the last already-available frame. Training-only normalization and environment-grouped validation are applied downstream. Codex recomputed and plotted these diagnostics from recorded data; the figure does not establish independent physical accuracy.
'''
    caption=caption.replace('empty bins remain missing.','empty bins remain missing. A labeled square-root color normalization makes sparse flagged intervals visible without changing the percentage values.')
    (OUT/'caption.md').write_text(caption)
    inputs=[Path('configs/c_q1_v2.json'),Path('src/shuwei_c/data.py'),Path('src/shuwei_c/q1v2_state.py'),Path('src/shuwei_c/q1.py'),Path('artifacts/c_q1_v1/record_audit.csv'),Path('artifacts/c_video_v2_15hz/video_manifest.json'),ROOT/'frozen_numeric/freeze.json',ROOT/'numeric_oof/train_oof.parquet',ROOT/'frozen_numeric/validation/points_h0.100000.parquet',data.task_root/'record_split.csv',Path(__file__)]
    provenance=dict(status='generated_pending_visual_review',scope='Official metadata all splits; numerical/visual audit train+validation only',test_trajectories_read=False,models_fitted=False,point_reference_keys_match_existing_results=True,totals=totals,example=example_id,source_sha256={str(p):sha(p) for p in inputs},trajectory_hashes=dict(zip(audit.episode,audit.trajectory_sha256)),time_rule_comparison='Same admissible origins, first-after rule uses original 1e-9 s tolerance; no model predictions compared',figure_sha256={ext:sha(OUT/f'{name}.{ext}') for ext in ['pdf','svg','png']})
    provenance.update(canonical_time_position_flag_sha256=canonical_hashes,visual_timestamp_source_sha256=visual_hashes,heatmap_normalization='PowerNorm gamma=0.5, vmin=0, vmax=100; ticks remain percentages; missing distinct')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    contracts=dict(A=dict(source='record_inventory.csv',encoding='Blue training, orange hatched validation, gray dotted test metadata; exact counts'),B=dict(source='record_audit.csv',encoding='Blue circles EgoDemo, green squares EgoPro, orange triangles EgoStandard; one point per recording'),C=dict(source='quality_time_map.csv',encoding='Shared gray-to-orange linear 0–100% scale; missing dark gray; rows grouped by dataset'),D=dict(source='state_construction_example.csv',encoding='Gray landmark min–max, green arithmetic center, orange supplied-flag spans'),E=dict(source='time_alignment_summary.csv',encoding='Gray hatched first-after, green nearest admissible; labels are mean absolute timing error in ms'),F=dict(source='eligibility_by_record.csv',encoding='Blue source/eligibility stages, green qualified targets, orange flagged targets retained for quality tasks'))
    contracts['C']['encoding']='Shared gray-to-orange square-root color normalization over 0–100%; ticks are percentages; missing dark gray; rows grouped by dataset'
    (OUT/'legend_contracts.json').write_text(json.dumps(contracts,indent=2)+'\n')
    fonts=subprocess.check_output(['pdffonts',str(OUT/f'{name}.pdf')],text=True).splitlines()[2:];assert fonts and all('TimesNewRoman' in s and ' yes ' in s for s in fonts)
    with Image.open(OUT/f'{name}.png') as im:assert min(im.info['dpi'])>599
    (OUT/'export_audit.json').write_text(json.dumps(dict(status='PASS',fonts=fonts,font=font,panels=6,visual_review_pending=True),indent=2)+'\n')
    print(json.dumps(dict(totals=totals,time_alignment=summary.to_dict('records'),example=example_id),indent=2))


if __name__=='__main__':main()
