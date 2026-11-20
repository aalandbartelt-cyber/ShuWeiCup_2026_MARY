#!/usr/bin/env python3
"""Observed validation frames, cached diagnostics, and frozen-model GPU forecasts.

AI-assisted by Codex, 2026-11-20. Scientific figure; no generated imagery,
retuning, test selection, or changes to the fixed numerical/video components.
"""
from pathlib import Path
import hashlib,json,subprocess,sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parent)]
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,ConnectionPatch,Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from PIL import Image
from shuwei_c.data import Dataset
from shuwei_c.figure_style import apply_style,check_figure
from shuwei_c.q1v3_prefix import causal_pack
from shuwei_c.q1v3_learning import network
from shuwei_c.q1v3_state import DATASETS
from shuwei_c.q1v3_interface import Q1V3
from prepare_c_video import videos
from audit_c_q1_v2_geometry import intrinsics
from extract_c_q1_v3_visual import projected_hands,region_stats

ROOT=Path('artifacts/c_q1_v3');OUT=ROOT/'video_case_figure'
UID='3da72d6d-367e-438d-bb3b-b611675f1b7a';VIEW='head_left';EVENT=40.998633
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_project(rec,centres,stamp):
    camera=rec.streams[VIEW+'_camera'];i=np.searchsorted(camera['time'],stamp,side='right')-1;assert i>=0
    q=camera['quaternion'][i].reshape(4);rotation=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    xyz=(np.asarray(centres)-camera['position'][i].reshape(3))@rotation
    k,_,_=intrinsics(rec,VIEW,np.array([stamp]));fx,fy,cx,cy=k[0]
    assert (xyz[:,2]>0).all()
    return np.c_[fx*xyz[:,0]/xyz[:,2]+cx,fy*xyz[:,1]/xyz[:,2]+cy]


def gpu_points(rec,indices,spec):
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    t,p,bad=rec.hands();head=rec.streams['head'];scale=spec['noise'][rec.dataset]['hand_scale'];packs=[]
    for i in indices:
        hi=head['time']<=t[i]
        packs.append(causal_pack(t[:i+1],p[:i+1],bad[:i+1],head['time'][hi],head['position'][hi],rec.dataset,scale,t[i]+.1,.1))
    forecasts=[]
    for path in spec['point_checkpoints']['0.1']:
        m=torch.load(path,map_location='cpu',weights_only=False);net=network(len(m['feature_names']),10).cuda().eval();net.load_state_dict(m['state_dict'])
        x=np.concatenate([a['features'] for a in packs]);expert=np.concatenate([a['experts'] for a in packs]);assert all(a['feature_names']==m['feature_names'] for a in packs)
        x=torch.tensor(np.clip((x-m['mean'])/m['std'],-10,10),device='cuda',dtype=torch.float32)
        b=np.asarray(m['base']['coefficients'],np.float32)[DATASETS.index(rec.dataset)]
        with torch.inference_mode():
            coeff=torch.tensor(b,device='cuda')[None,:]+.5*torch.tanh(net(x));d=torch.einsum('nck,nk->nc',torch.tensor(expert,device='cuda'),coeff).cpu().numpy()
        forecasts.append(d.reshape(len(indices),2,3))
    forecast=p[indices].mean(axis=2)+scale*np.mean(forecasts,axis=0)
    # Independent CPU public inference path must agree at these exact queries.
    api=Q1V3(spec,rec.dataset);difference=0.
    for k,i in enumerate(indices):
        hi=head['time']<=t[i];reference,_=api.point(t[:i+1],p[:i+1],bad[:i+1],head['time'][hi],head['position'][hi],t[i]+.1,.1)
        difference=max(difference,float(np.max(np.abs(reference-forecast[k]))))
    assert difference<1e-6
    return forecast,dict(device=torch.cuda.get_device_name(),torch=torch.__version__,checkpoint_paths=spec['point_checkpoints']['0.1'],cpu_max_coordinate_difference=difference,forecast_horizon_s=.1)


def main():
    OUT.mkdir(exist_ok=True);style,font=apply_style();C=style['palette'];plt.rcParams.update({'mathtext.cal':'Times New Roman','mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman'})
    spec=json.loads((ROOT/'frozen_numeric/freeze.json').read_text());qs=json.loads((ROOT/'frozen_quality/freeze.json').read_text())
    for frozen in [spec,qs]:
        for path,expected in {**frozen['inherited_v2_hashes'],**frozen['source_hashes']}.items():assert sha(path)==expected,path
    rec=Dataset().record(UID);assert rec.split=='validation';t,p,bad=rec.hands();ei=int(np.argmin(abs(t-EVENT)));event=float(t[ei]);assert abs(event-EVENT)<1e-6 and bad[ei].any() and not bad[ei-1].any()
    # Fixed chronological offsets around one disclosed illustrative event.
    origins=ei+np.array([-14,-8,-2,3]);query=t[origins];forecast,gpu=gpu_points(rec,origins,spec)
    v=next(v for v in videos(rec) if v['view']==VIEW)
    detections_path=Path(f'artifacts/c_video_v2_15hz/{UID}__{VIEW}.jsonl');raw=[json.loads(line) for line in detections_path.open()];vt=np.array([r['time_s'] for r in raw]);ix=np.searchsorted(vt,query,side='right')-1
    assert (vt[ix]<=query).all() and (ix>0).all()
    source=ROOT/f'visual_dino15/{UID}__{VIEW}.npz';cached=np.load(source);np.testing.assert_array_equal(cached['time_s'],vt)
    frames={};cap=cv2.VideoCapture(str(v['path']))
    for j in sorted(set(ix.tolist()+[int(k-1) for k in ix])):
        cap.set(cv2.CAP_PROP_POS_FRAMES,raw[j]['frame_index']);ok,frame=cap.read();assert ok;frames[j]=frame
    cap.release();height,width=frames[int(ix[0])].shape[:2];projections,visible=projected_hands(rec,VIEW,vt[ix]);assert visible.all()
    centres=[];predicted=[];detected=[];heat=[];rows=[];maximum_stats_diff=0.
    for col,j in enumerate(ix):
        frame=frames[int(j)];stamp=float(vt[j]);centres.append(camera_project(rec,p[origins[col]].mean(axis=1),stamp));predicted.append(camera_project(rec,forecast[col],stamp))
        detected.append([np.asarray(d['landmarks'])[:,:2]*[width,height] for d in raw[j]['detections']])
        small=cv2.resize(frame,(320,round(height*320/width)),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        previous=cv2.cvtColor(cv2.resize(frames[int(j-1)],(320,gray.shape[0]),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        flow=cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0);background=np.median(flow.reshape(-1,2),axis=0);dt=vt[j]-vt[j-1]
        speed=np.linalg.norm(flow-background,axis=-1)/(dt*320);heat.append(speed)
        # Reconstruct exactly the motion statistics that fed the existing cache.
        reconstructed=region_stats(gray,flow,np.array([0,0,gray.shape[1],gray.shape[0]]),background,dt)
        maximum_stats_diff=max(maximum_stats_diff,float(np.max(abs(reconstructed-cached['diagnostics'][j,:12]))))
        for hand in range(2):rows.append(dict(column=col+1,hand=['left','right'][hand],origin_s=float(query[col]),frame_s=stamp,frame_index=int(raw[j]['frame_index']),frame_age_ms=float((query[col]-stamp)*1000),event_s=event,relative_origin_s=float(query[col]-event),current_flag=bool(bad[origins[col],hand]),detected_hands=len(raw[j]['detections']),gpu_forecast_x=float(predicted[-1][hand,0]),gpu_forecast_y=float(predicted[-1][hand,1]),current_center_x=float(centres[-1][hand,0]),current_center_y=float(centres[-1][hand,1])))
    assert maximum_stats_diff<1e-5,maximum_stats_diff
    # A fixed union crop across all columns preserves the same spatial scale.
    points=np.concatenate([projections.reshape(-1,2),np.array(predicted).reshape(-1,2),*([a for ds in detected for a in ds])]);lo=points.min(0)-25;hi=points.max(0)+25
    lo=np.maximum(lo,0);hi=np.minimum(hi,[width,height]);crop=np.r_[np.floor(lo),np.ceil(hi)].astype(int);x0,y0,x1,y1=crop
    assert x1>x0 and y1>y0
    limits=[(x0,y0,x1,y1)]*4
    roiheat=[a[max(0,round(y0*320/width)):min(a.shape[0],round(y1*320/width)),round(x0*320/width):round(x1*320/width)] for a in heat]
    vmax=float(np.quantile(np.concatenate([a.ravel() for a in roiheat]),.99));norm=Normalize(0,vmax,clip=True);cmap='magma'
    history=pd.read_parquet(ROOT/'onset_h0.5/validation.parquet');history=history[history.episode.eq(UID)]
    trace=pd.DataFrame({'time_s':t,'current_bad':bad.any(1)});trace=trace[(trace.time_s>=event-.75)&(trace.time_s<=event+.3)].merge(history[['time_s','numeric_leaves7_onset_probability','diagnostics_leaves7_onset_probability']],on='time_s',how='left');trace['relative_time_s']=trace.time_s-event
    thresholds={m:qs['onset']['models'][m]['thresholds']['0.05'] for m in ['numeric_leaves7','diagnostics_leaves7']}
    lead={}
    for name in thresholds:
        eligible=trace[(trace.relative_time_s>=-.5-1e-8)&(trace.relative_time_s<0)&(~trace.current_bad)];alerts=eligible[eligible[name+'_onset_probability']>thresholds[name]];lead[name]=None if alerts.empty else float(-alerts.relative_time_s.iloc[0])
    assert lead['numeric_leaves7'] is None and lead['diagnostics_leaves7'] is not None
    fig=plt.figure(figsize=(7,7.8));grid=fig.add_gridspec(5,4,left=.065,right=.975,bottom=.14,top=.91,wspace=.10,hspace=.28,height_ratios=[1.,1.,1.,.34,1.3]);axes=[]
    for col,j in enumerate(ix):
        rgb=cv2.cvtColor(frames[int(j)],cv2.COLOR_BGR2RGB);a=fig.add_subplot(grid[0,col]);b=fig.add_subplot(grid[1,col]);c=fig.add_subplot(grid[2,col]);axes.append((a,b,c))
        a.imshow(rgb);a.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=C['green'],linewidth=1.1));a.text(.56,1.03,f'Δt = {query[col]-event:+.2f} s',transform=a.transAxes,ha='center',fontsize=9)
        a.text(.5,-.10,f'Frame age: {(query[col]-vt[j])*1000:.0f} ms',transform=a.transAxes,ha='center',fontsize=8)
        b.imshow(rgb);b.set_xlim(x0,x1);b.set_ylim(y1,y0)
        for joints in detected[col]:
            for i,k in EDGES:b.plot(joints[[i,k],0],joints[[i,k],1],color=C['green'],lw=.85)
            b.scatter(joints[:,0],joints[:,1],s=4,c=C['green'],edgecolors='white',linewidths=.25,zorder=4)
        for hand in range(2):
            start=centres[col][hand];end=predicted[col][hand]
            b.plot(*start,'o',color='white',markeredgecolor=C['text'],markersize=4,zorder=5)
            b.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color=C['blue'],lw=1.3),zorder=5)
            b.plot(*end,'x',color=C['blue'],markersize=5,markeredgewidth=1.4,zorder=6)
            b.annotate(['L','R'][hand],start,xytext=(3,-11),textcoords='offset points',color='white',fontsize=8,bbox=dict(facecolor=C['text'],alpha=.75,edgecolor='none',pad=.6),zorder=7)
        b.text(.5,-.10,f'Detected hands: {len(detected[col])}',transform=b.transAxes,ha='center',fontsize=8)
        image=c.imshow(heat[col],extent=(0,width,height,0),norm=norm,cmap=cmap,interpolation='nearest');c.set_xlim(x0,x1);c.set_ylim(y1,y0)
        for hand in range(2):c.plot(*centres[col][hand],'o',color='white',markeredgecolor=C['text'],markersize=3)
        c.text(.5,-.10,f'ROI median: {np.median(roiheat[col]):.3f}',transform=c.transAxes,ha='center',fontsize=8)
        for row,ax in enumerate([a,b,c]):
            ax.set_xticks([]);ax.set_yticks([]);ax.grid(False);ax.text(0,1.02,f'{chr(65+row)}{col+1}',transform=ax.transAxes,fontweight='bold',fontsize=8)
            for spine in ax.spines.values():spine.set_visible(False)
        for xx in [x0,x1]:fig.add_artist(ConnectionPatch(xyA=(xx,y1),coordsA=a.transData,xyB=(xx,y0),coordsB=b.transData,color=C['neutral'],lw=.5,zorder=0))
    legend=[Line2D([],[],color=C['green'],marker='o',markersize=3,label='Detected 2D hand landmarks'),Line2D([],[],color=C['text'],marker='o',markerfacecolor='white',ls='None',label='Projected supplied center'),Line2D([],[],color=C['blue'],marker='x',label='GPU 0.1 s center forecast')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.52,.995),ncol=2,frameon=False,fontsize=8)
    ax=fig.add_subplot(grid[4,:]);ax.text(-.025,1.02,'D',transform=ax.transAxes,fontweight='bold',fontsize=10)
    ax.axvspan(-.5,0,color=C['green'],alpha=.09,zorder=0)
    for i in range(len(trace)):
        if trace.current_bad.iloc[i]:ax.axvspan(trace.relative_time_s.iloc[i],trace.relative_time_s.iloc[i]+np.median(np.diff(t)),color=C['orange'],alpha=.2,zorder=0)
    for name,label,color,linestyle,marker in [('numeric_leaves7','Numerical onset model','blue','--','s'),('diagnostics_leaves7','Video-assisted onset model','mauve','-','o')]:
        ax.plot(trace.relative_time_s,100*trace[name+'_onset_probability'],color=C[color],linestyle=linestyle,marker=marker,markersize=2.4,label=label)
        ax.axhline(100*thresholds[name],color=C[color],linestyle=':',lw=1,label=f'{label.split()[0]} threshold ({100*thresholds[name]:.2f}%)')
    ax.axvline(0,color=C['text'],lw=1.1,label='Selected onset (Δt = 0)')
    ax.set(xlim=(-.75,.3),ylim=(0,9),xlabel='Time relative to selected flag onset (s)',ylabel='0.5 s onset probability (%)')
    for i,qtime in enumerate(query-event):
        ax.axvline(qtime,color=C['neutral'],ls='--',lw=.6,zorder=0);ax.text(qtime,8.5,str(i+1),ha='center',fontsize=8,color=C['neutral'])
    handles,labels=ax.get_legend_handles_labels();handles += [Patch(facecolor=C['green'],alpha=.15),Patch(facecolor=C['orange'],alpha=.25)];labels += ['Eligible 0.5 s warning window','Supplied bad flag']
    ax.legend(handles,labels,loc='upper left',bbox_to_anchor=(0,-.33),ncol=3,fontsize=8,frameon=False,columnspacing=1.1,handlelength=2.)
    # Explicit continuous scale, shared by every heatmap; high values are capped.
    color_space=grid[3,1:3].get_position(fig)
    color_ax=fig.add_axes([color_space.x0,color_space.y0+.55*color_space.height,color_space.width,.011]);cb=fig.colorbar(image,cax=color_ax,orientation='horizontal',extend='max');cb.set_label('Background-compensated speed (frame widths/s)',fontsize=8);cb.ax.tick_params(labelsize=8,pad=1)
    ax.annotate(f'In-window video warning: {lead["diagnostics_leaves7"]:.2f} s lead',xy=(-lead['diagnostics_leaves7'],100*eligible.loc[eligible.relative_time_s.eq(-lead['diagnostics_leaves7']),'diagnostics_leaves7_onset_probability'].iloc[0]),xytext=(-.69,7.3),fontsize=8,arrowprops=dict(arrowstyle='->',color=C['mauve']),color=C['mauve'])
    check_figure(fig,style)
    name='q1v3_video_keyframes'
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=600,bbox_inches='tight')
    fig.savefig(OUT/f'{name}_preview.png',dpi=160,bbox_inches='tight');plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'frame_analysis.csv',index=False);trace.to_csv(OUT/'probability_trace.csv',index=False)
    np.savez_compressed(OUT/'gpu_forecasts.npz',origin_s=query,forecast_centres=forecast,current_centres=p[origins].mean(axis=2),projected_forecasts=np.array(predicted),projected_centres=np.array(centres))
    provenance=dict(status='generated_pending_visual_review',episode=UID,dataset=rec.dataset,split=rec.split,view=VIEW,event_s=event,frame_panels=12,probability_panels=1,
        selection='Illustrative video-only warning case in development validation, inspected after aggregate analysis; not a representative sample or an independent validation of accuracy',
        sources_sha256={str(path):sha(path) for path in [v['path'],detections_path,source,ROOT/'onset_h0.5/validation.parquet',ROOT/'frozen_numeric/freeze.json',ROOT/'frozen_quality/freeze.json',Path(__file__)]},
        crop_xyxy=crop.tolist(),shared_crop=True,heatmap_vmax_p99=vmax,heatmap_units='frame widths/s',diagnostics_max_difference=maximum_stats_diff,
        gpu=gpu,compute={'landmarks':'Existing MediaPipe IMAGE CPU extraction','motion':'Existing Farneback CPU extraction; exact reconstruction checked','center_forecast':'Frozen numerical gate, actual CUDA inference in this script','onset_probability':'Existing frozen LightGBM CPU prediction'},
        projection='Both current and future centers projected into the camera pose available at the displayed frame; forecast is not a future image localization claim',
        thresholds=thresholds,lead_s=lead,image_transform='Original RGB, shared spatial crop, overlays and scientific heatmaps; no brightness adjustment, synthetic fill or image generation',
        figure_sha256={ext:sha(OUT/f'{name}.{ext}') for ext in ['pdf','svg','png']})
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    contracts={
        'A_crop':dict(panels='A1–A4',encoding='Green rectangle and gray connectors',meaning='Same fixed image rectangle enlarged in row B; a display crop, not a changed model input',source='provenance.json:crop_xyxy'),
        'B_landmarks':dict(panels='B1–B4',encoding='Green connected points with white edges',meaning='Actual cached MediaPipe 2D detections; absent detections remain absent',source=str(detections_path)),
        'B_supplied':dict(panels='B1–B4 and C1–C4',encoding='White circles with dark edges',meaning='Supplied current numerical hand centers projected into the displayed-frame camera',source='frame_analysis.csv:current_center_x,current_center_y'),
        'B_forecast':dict(panels='B1–B4',encoding='Blue arrows and crosses',meaning='Frozen numerical 0.1 s forecasts evaluated with CUDA, projected in the same available camera',source='gpu_forecasts.npz; frame_analysis.csv:gpu_forecast_x,gpu_forecast_y'),
        'C_flow':dict(panels='C1–C4',encoding='Shared magma color map and labeled continuous colorbar',meaning='Previous-to-current flow minus median image flow; magnitude divided by elapsed seconds and image width',source=str(source),range=[0,vmax],overflow='saturate above pooled ROI p99; colorbar triangle'),
        'D_numeric':dict(panels='D',encoding='Blue dashed line and square markers',meaning='Numerical 0.5 s window-onset probability, only for currently good origins',source='probability_trace.csv:numeric_leaves7_onset_probability'),
        'D_video':dict(panels='D',encoding='Mauve solid line and circular markers',meaning='Video-assisted 0.5 s window-onset probability, only for currently good origins',source='probability_trace.csv:diagnostics_leaves7_onset_probability'),
        'D_thresholds':dict(panels='D',encoding='Blue and mauve dotted lines with exact matching colored legend samples',meaning='Separate fixed training-OOF thresholds at nominal 5% false-alarm target',source='provenance.json:thresholds'),
        'D_window':dict(panels='D',encoding='Pale green span',meaning='Eligible warning window from -0.5 s to the selected onset',source='event_s and fixed 0.5 s horizon'),
        'D_flags':dict(panels='D',encoding='Pale orange spans',meaning='Supplied current bad flag for either hand; not independent physical ground truth',source='probability_trace.csv:current_bad'),
        'D_linkage':dict(panels='D',encoding='Numbered gray dashed vertical lines and dark zero-time line',meaning='Frame-column origin times 1–4 and selected onset, respectively',source='frame_analysis.csv:relative_origin_s; selected event=0')}
    (OUT/'legend_contracts.json').write_text(json.dumps(contracts,indent=2)+'\n')
    caption=f'''Figure 5. Observed video and model analysis around a validation quality-flag onset. Columns 1–4 use origins {', '.join(f'{q-event:+.3f}' for q in query)} s relative to the selected onset. A1–A4 show the latest available head-left frames, with frame ages and the common crop rectangle. B1–B4 enlarge that same rectangle: green landmarks are existing MediaPipe detections; white circles are projected supplied centers; blue arrows and crosses are frozen 0.1 s numerical center forecasts recomputed on the local GPU. L/R follow the supplied numerical hand identities, not unverified detector handedness. All projections use the camera pose available at the displayed frame, so the crosses are not predictions of a future camera image. Missing detections remain missing. C1–C4 show previous-to-current, background-compensated Farneback speed over the same crop, with one shared scale in frame widths/s; the upper color limit is the pooled 99th percentile and larger values saturate. D shows the separately defined 0.5 s onset probabilities and their fixed training thresholds. Green shading is the eligible warning window, orange shading indicates supplied bad flags, and numbered gray lines link to the four frame columns. Probabilities are undefined during a currently bad state and are not connected through those gaps. Video first warns {lead['diagnostics_leaves7']:.3f} s before this onset; numerical risk does not cross its own threshold within this window. This is a selected explanatory case, not a performance estimate or proof of physical tracking failure. The onset predictor uses both head views; only the left view is shown for legibility. Keypoints, optical flow, and onset probabilities were computed on CPU; the overlaid numerical forecasts were evaluated on GPU. Images and numerical results are real recorded/computed data, not generated illustrations. Codex prepared this candidate; independent human 2D verification remains pending.
'''
    (OUT/'caption.md').write_text(caption)
    fonts=subprocess.check_output(['pdffonts',str(OUT/f'{name}.pdf')],text=True).splitlines()[2:];assert fonts and all('TimesNewRoman' in s and ' yes ' in s for s in fonts)
    with Image.open(OUT/f'{name}.png') as im:assert min(im.info['dpi'])>599
    (OUT/'export_audit.json').write_text(json.dumps(dict(status='PASS',fonts=fonts,frame_panels=12,probability_panels=1,figure_has_title=False,font=font,visual_review_pending=True),indent=2)+'\n')
    print(json.dumps(dict(gpu=gpu,diagnostics_max_difference=maximum_stats_diff,lead_s=lead,crop=crop.tolist()),indent=2))


if __name__=='__main__':main()
