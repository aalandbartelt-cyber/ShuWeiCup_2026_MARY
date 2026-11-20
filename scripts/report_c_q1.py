#!/usr/bin/env python3
"""Create Q1 tables, English figures, a Chinese results report and English draft.

AI-assisted: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
All plotted values are read from versioned experiment CSVs, never demonstration data.
"""
from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shuwei_c.data import balanced_weights
from shuwei_c.figure_style import apply_style,check_figure,panel_label

OUT=Path('artifacts/c_q1_v1');DOC=Path('docs/analysis/2026初赛');FIG=OUT/'figures'
STYLE,FONT_PATH=apply_style()
COLORS={model:STYLE['palette'][color] for model,color in STYLE['model_colors'].items()}
LABELS={'constant':'Hold last','linear':'Two-point linear','filter':'Quality-aware filter','multiscale_linear':'Multiscale linear',
        'dataset_rate':'Dataset frequency','flag_transition':'Flag transition','numeric_boosting_raw':'Numerical boosting',
        'multimodal_boosting_raw':'Numerical + video'}


def table(frame,columns=None,digits=5):
    f=frame[columns] if columns else frame
    def fmt(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
        return str(v)
    return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))


def save(fig,name):
    check_figure(fig,STYLE)
    fig.savefig(FIG/(name+'.png'),dpi=STYLE['raster_dpi'],bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def main():
    DOC.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    audit=pd.read_csv(OUT/'record_audit.csv');intervals=pd.read_csv(OUT/'interval_uses.csv')
    state=pd.read_csv(OUT/'state_validation_summary.csv');quality=pd.read_csv(OUT/'quality_validation_metrics.csv')
    prediction=pd.read_csv(OUT/'quality_validation_predictions.csv');stage=pd.read_csv(OUT/'retrospective_stage_statistics.csv')
    video=json.loads(Path('artifacts/c_video/video_manifest.json').read_text())
    state_model=json.loads((OUT/'state_model.json').read_text());multiscale=json.loads((OUT/'multiscale_model.json').read_text())
    retention=audit.groupby(['dataset','split']).agg(records=('episode','nunique'),environments=('environment','nunique'),
        frames=('frames','sum'),qualified_hand_frames=('qualified_hand_frames','sum'),total_hand_frames=('total_hand_frames','sum'),
        any_bad_frames=('any_bad_frames','sum')).reset_index()
    retention['retained_hand_percent']=100*retention.qualified_hand_frames/retention.total_hand_frames
    retention.to_csv(OUT/'retention_summary.csv',index=False)
    mainstate=state[(state.horizon=='0.1s')&(state.context=='all')]
    by_record=pd.read_csv(OUT/'state_validation_by_record.csv')
    actual=by_record[(by_record.horizon=='0.1s')&(by_record.context=='all')&(by_record.method=='linear')]
    actual=actual.groupby(['dataset','environment','episode']).actual_horizon_mean_s.mean().groupby(['dataset','environment']).mean().groupby('dataset').mean().reset_index()
    actual.to_csv(OUT/'actual_prediction_horizons.csv',index=False)
    methods=['constant','linear','filter','multiscale_linear']
    state_table=mainstate[mainstate.method.isin(methods)].pivot(index='dataset',columns='method',values='mean_euclidean_error').reset_index()
    state_table['gain_vs_two_point_percent']=100*(1-state_table.multiscale_linear/state_table.linear)
    state_table.to_csv(OUT/'state_main_table.csv',index=False)
    mainquality=quality[(quality.dataset=='ALL')&quality.method.isin(COLORS)].sort_values('brier')
    mainquality.to_csv(OUT/'quality_main_table.csv',index=False)
    # Figure 1: data coverage / flagged-rate shift.
    fig,axes=plt.subplots(1,2,figsize=(7.,3.35),layout='constrained')
    datasets=['EgoDemo','EgoPro','EgoStandard'];x=np.arange(3)
    for j,(split,color) in enumerate([('train','#466C9B'),('validation','#D88C32')]):
        f=retention[retention.split==split].set_index('dataset').loc[datasets]
        axes[0].bar(x+(j-.5)*.32,f.frames/1000,width=.3,color=color,label=split.title(),hatch=['','//'][j],edgecolor='white')
        axes[1].bar(x+(j-.5)*.32,100*f.any_bad_frames/f.frames,width=.3,color=color,label=split.title(),hatch=['','//'][j],edgecolor='white')
    for ax in axes:ax.set_xticks(x,datasets);ax.tick_params(axis='x',labelsize=9)
    axes[0].set(ylabel='Numerical frames (thousands)');panel_label(axes[0],'A')
    axes[1].set(ylabel='Frames with either hand flagged (%)');panel_label(axes[1],'B')
    axes[0].legend(frameon=False);save(fig,'q1_data_quality')
    # Figure 2: native-unit prediction error, same axes for the three datasets.
    fig,ax=plt.subplots(figsize=(7.,3.5),layout='constrained')
    for j,method in enumerate(methods):
        values=mainstate[mainstate.method==method].set_index('dataset').loc[datasets].mean_euclidean_error
        bars=ax.bar(x+(j-1.5)*.19,values,width=.18,label=LABELS[method],color=COLORS[method],hatch=['','//','..','xx'][j],edgecolor='white')
        ax.bar_label(bars,fmt='%.4f',fontsize=8,padding=3+8*(j%2))
    ax.set_xticks(x,datasets);ax.set(ylabel='Mean Euclidean error (native coordinate units)')
    ax.set_ylim(0,float(mainstate.mean_euclidean_error.max())*1.25)
    ax.legend(ncol=2,frameon=False,loc='upper left');save(fig,'q1_state_forecast')
    # Figure 3: prediction scores with proper baselines; zero-positive subset explicitly shown.
    qmethods=['dataset_rate','flag_transition','numeric_boosting_raw','multimodal_boosting_raw']
    fig,axes=plt.subplots(1,2,figsize=(7.,3.5),layout='constrained')
    for j,method in enumerate(qmethods):
        f=quality[(quality.dataset=='ALL')&(quality.method==method)].iloc[0]
        axes[0].barh(j,f.brier,color=COLORS[method]);axes[0].text(f.brier+.0003,j,f'{f.brier:.5f}',va='center',fontsize=9)
    axes[0].set_yticks(np.arange(4),[LABELS[m] for m in qmethods]);axes[0].invert_yaxis()
    axes[0].set(xlabel='Brier score (lower is better)',xlim=(0,.025));panel_label(axes[0],'A')
    for method,marker,linestyle in zip(['flag_transition','numeric_boosting_raw','multimodal_boosting_raw'],['o','s','^'],['-','--','-.']):
        w=balanced_weights(prediction.environment,prediction.episode);pp=prediction[method].to_numpy();yy=prediction.target_bad.to_numpy()
        xs=[];ys=[]
        for low,high in zip([0,.01,.05,.1,.2,.4,.6,.8],[.01,.05,.1,.2,.4,.6,.8,1.00001]):
            take=(pp>=low)&(pp<high)
            if take.sum()<5:continue
            xs.append(np.average(pp[take],weights=w[take]));ys.append(np.average(yy[take],weights=w[take]))
        axes[1].plot(xs,ys,marker=marker,linestyle=linestyle,label=LABELS[method],color=COLORS[method],markersize=4)
    axes[1].plot([0,1],[0,1],'--',color='#777777',linewidth=1)
    axes[1].set(xlabel='Predicted flag probability',ylabel='Observed flag fraction',xlim=(0,1),ylim=(0,1));panel_label(axes[1],'B')
    axes[1].legend(fontsize=8,frameon=False,loc='upper left');save(fig,'q1_reliability')
    # Figure 4: retrospective stage association, with explicit noncausal interpretation.
    stage_rows=[]
    for key,g in stage.groupby(['split','family','stage']):
        w=balanced_weights(g.environment,g.episode)
        clean=g[g.clean_speed.notna()]
        stage_rows.append(dict(zip(['split','family','stage'],key),
            bad_percent=float(np.average(g.any_bad_fraction,weights=w)*100),
            clean_speed=float(np.average(clean.clean_speed,weights=balanced_weights(clean.environment,clean.episode))) if len(clean) else None,
            environments=g.environment.nunique(),segments=len(g)))
    stage_table=pd.DataFrame(stage_rows);stage_table.to_csv(OUT/'stage_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.,3.3),layout='constrained')
    for ax,split,letter in zip(axes,['train','validation'],['A','B']):
        for family,color,marker,linestyle in zip(['P','A','L'],['#466C9B','#228B77','#D88C32'],['o','s','^'],['-','--','-.']):
            f=stage_table[(stage_table.split==split)&(stage_table.family==family)].set_index('stage').reindex(['early','middle','late'])
            ax.plot(np.arange(3),f.bad_percent,marker=marker,linestyle=linestyle,color=color,label=family)
        ax.set_xticks(np.arange(3),['Early third','Middle third','Late third'])
        ax.set(ylabel='Environment-balanced flagged frames (%)',
               ylim=(-.2,max(stage_table.bad_percent)*1.15))
        panel_label(ax,letter)
        ax.legend(title='Family',ncol=3,fontsize=9,frameon=False)
    save(fig,'q1_retrospective_stages')
    # Figure 5: actual record state and quality, deliberately a difficult validation case.
    chosen=audit[audit.split=='validation'].copy();chosen['rate']=chosen.any_bad_frames/chosen.frames
    chosen=chosen.sort_values('rate',ascending=False).iloc[0]
    z=np.load(OUT/'states'/f'{chosen.episode}.npz');t=z['time'];bad=z['raw_bad'].any(axis=1)
    centre=float(t[np.flatnonzero(bad)[0]]) if bad.any() else float(t[len(t)//2])
    low=max(float(t[0]),centre-1.);high=min(float(t[-1]),low+6.);take=(t>=low)&(t<=high)
    fig,axes=plt.subplots(3,1,figsize=(7.,6.2),sharex=True,layout='constrained')
    for j,(hand,color) in enumerate([('Left','#466C9B'),('Right','#228B77')]):
        clean=take&~z['raw_bad'][:,j]
        axes[0].plot(t[take],z['centroid'][take,j,0],color=color,linestyle=['-','--'][j],label=hand+' filtered')
        axes[0].scatter(t[clean],z['observed_centroid'][clean,j,0],s=7,color=color,alpha=.4)
        axes[1].plot(t[take],np.linalg.norm(z['velocity'][take,j],axis=-1),color=color,linestyle=['-','--'][j],label=hand)
    pred=prediction[(prediction.episode==chosen.episode)&(prediction.time_s>=low)&(prediction.time_s<=high)]
    axes[2].plot(pred.time_s,pred.flag_transition,color='#B85C7B',label='Future bad-flag probability')
    from shuwei_c.q1 import contiguous_runs
    for a,b in contiguous_runs(bad):
        start=max(low,float(t[a]));stop=min(high,float(t[b-1]+np.median(np.diff(t))))
        if stop<=start:continue
        for ax in axes:ax.axvspan(start,stop,color='#D88C32',alpha=.17)
    axes[0].set(ylabel='Centroid x\n(native units)')
    axes[1].set(ylabel='Speed\n(native units / s)')
    axes[2].set(ylabel='Probability',xlabel='Record-relative time (s)',ylim=(-.02,1.02),xlim=(low,high))
    for ax,letter in zip(axes,['A','B','C']):
        ax.legend(frameon=False,fontsize=8,loc='lower left' if letter=='A' else 'upper right');panel_label(ax,letter)
    save(fig,'q1_state_quality_example')
    (OUT/'example_selection.json').write_text(json.dumps(dict(episode=chosen.episode,dataset=chosen.dataset,
        rule='validation record with largest flagged-frame fraction; six seconds around first flag',
        start_s=low,end_s=high))+'\n')
    # Key-number provenance, figure contract and source hashes.
    keys=[]
    for row in state_table.itertuples(index=False):
        for metric in methods:keys.append(dict(key='q1_error_'+row.dataset+'_'+metric,value=getattr(row,metric),
            source='state_main_table.csv',condition='validation; 0.1s; qualified hand references; environment/episode balanced'))
    for row in mainquality.itertuples(index=False):keys.append(dict(key='q1_brier_'+row.method,value=row.brier,
            source='quality_main_table.csv',condition='validation; any hand flagged 0.1s ahead; environment/episode balanced'))
    pd.DataFrame(keys).to_csv(OUT/'key_numbers.csv',index=False)
    captions={
      'q1_data_quality':'Development-set numerical coverage and raw quality-flag frequency. A frame is flagged if either hand is flagged. Counts do not establish physical sensor accuracy. Test trajectories are unopened.',
      'q1_state_forecast':'Mean Euclidean forecast error at the first existing sample at least 0.1 seconds after each origin, against unflagged held-out numerical references. Actual horizons vary with sampling timestamps and are tabulated separately. Hands are averaged within episodes, episodes within environments, then environments. Parameters were selected on validation; results are developmental. Coordinate units are not independently verified.',
      'q1_reliability':'Prediction of either hand being flagged at least 0.1 seconds later. Scores balance environments and episodes. The transition baseline conditions on the current flag. Numerical/video boosting uses raw probabilities; the full comparison also includes training-only out-of-fold calibration. Bins with fewer than five frames are omitted from the calibration plot.',
      'q1_retrospective_stages':'Offline association between annotated-interval thirds and raw quality flags, grouped by sampling family. A: training. B: validation. Segment endpoints are used for this retrospective description only. Task/environment confounding prevents a causal interpretation.',
      'q1_state_quality_example':f'A deliberately difficult {chosen.dataset} validation interval, chosen from the record with the highest flagged-frame fraction. A: centroid x. B: speed. C: future quality-flag probability. Dots show unflagged numerical observations, curves show causal state estimates, and orange spans mark raw bad-quality intervals. No true positional error is computed within the flagged spans. The probability curve predicts a future raw quality flag, not physical localization failure.'}
    inventory=[]
    for name,caption in captions.items():
        (FIG/(name+'_caption.txt')).write_text(caption+'\nAI-assisted analysis and visualization; human verification pending.\n')
        inventory.append(dict(figure=name,question=1,png=str(FIG/(name+'.png')),pdf=str(FIG/(name+'.pdf')),caption=caption,
            figure_sha256=hashlib.sha256((FIG/(name+'.png')).read_bytes()).hexdigest(),visual_check='pending'))
    pd.DataFrame(inventory).to_csv(OUT/'figure_inventory.csv',index=False)
    n_train=int(audit[audit.split=='train'].frames.sum());n_val=int(audit[audit.split=='validation'].frames.sum())
    qbase=float(mainquality[mainquality.method=='dataset_rate'].brier.iloc[0]);qbest=float(mainquality[mainquality.method=='flag_transition'].brier.iloc[0])
    relative=100*(qbase-qbest)/qbase
    report=f'''# C 题第一问结果 v1（2026-11-20）

第一问已形成可复现的开发结果：三套数据完成共同状态建模、记录与连续区间用途划分、因果预测对照、可靠性模型比较、参数敏感性和不确定性分析，并实际使用了全部训练与验证视频。**当前是验证集开发结果，尚未冻结为最终版本，未使用测试轨迹。**

## 1. 可直接核查的结论

1. 55 条训练记录和 7 条验证记录共 {n_train+n_val:,} 个数值帧。全部可读，无非有限手坐标、时间倒退或超过 1.5 倍记录中位采样间隔的缺口；未发现完整轨迹重复。并不意味着每帧观测准确。
2. 双手状态统一为 21 个点的几何中心与速度，双手相对向量由中心作差得到。三套数据不强行假定数字索引 0 都是腕部，也不把记录世界坐标拼成一个超市。
3. 多时间尺度线性模型在三套验证数据的短时预测上均优于两点线性基线。目标为不早于 0.1 秒的首个样本，实际跨度见第 4 节。具体增益见下表；这是有限验证环境内的比较。
4. 未来质量标记预测中，质量转移模型的整体 Brier 为 **{qbest:.6f}**，训练数据集频率基线为 **{qbase:.6f}**，相对下降 **{relative:.2f}%**。数值提升树的排序指标较好，但概率误差尚未稳定优于质量转移模型。
5. 已处理训练/验证的 **{len(video)} 段视频、{sum(r['sampled_frames'] for r in video):,} 个抽样帧**。当前 MediaPipe 多视角摘要加入模型后没有提高整体可靠性预测，不将这一负结果隐藏，也不据此否定更强的视觉模型。

## 2. 数据保留和区间用途

{table(retention,['dataset','split','records','environments','frames','qualified_hand_frames','total_hand_frames','retained_hand_percent'],3)}

分母是手—帧：一帧有左右手两个观测。参考资格要求坐标有限、原始质量未标坏、时间有效；本版所有开发记录时间均连续。标坏帧保留作低质量现象分析和降权状态更新，不能充当真实位置误差的参照。区间按左右手分别切分，边界在 `interval_uses.csv`，半开索引 `[start_index, stop_index_exclusive)`；持续时间以最后时间减最初时间加记录中位采样间隔近似，不声称精确遮挡时长。

![数据覆盖与质量分布](../../../artifacts/c_q1_v1/figures/q1_data_quality.png)

完整原始样例、字段与划分的独立审计见《C题读题_Codex.md》及 `audit_c_inputs_20261120.py`。总包的 69 条记录包括 7 条测试记录；上述模型开发只读 62 条训练/验证轨迹。训练/测试操作者有重叠，未来结论应主要称跨环境泛化。

## 3. 状态模型、公式和假设

对手 h、帧 t，观测中心 `c[h,t] = sum(j=1..21) p[h,j,t]/21`。共同状态为 `[cL,vL,cR,vR]`，维度 12；相对向量 `r=cR-cL` 为派生量。原生位置单位未独立核验，速度单位为原生位置单位/秒。

局部匀速状态转移为 `F(dt)=[[1,dt],[0,1]]`，三个坐标共享同一时间结构、两手分别更新。过程协方差为 `q*[[dt^4/4,dt^3/2],[dt^3/2,dt^2]]`，相当于一步内随机常加速度。观测噪声对标坏手乘倍数。训练估计 q、R 是时序自一致性的稳健代理，不能唯一辨识真正的传感器物理噪声。更新使用 Joseph 协方差形式保持数值稳定。

验证网格为 q 倍率 {{0.1,1,10}}、R 倍率 {{1,10}}、质量降权倍率 {{1,10,100}}，共 18 组；选择结果为 `{state_model['selection']}`。q 倍率位于网格边界，应解释为本次候选范围内的选择，不能称连续参数全局最优。

增强的线性预测器用过去 1、3、6、12 帧形成四个平均速度，预测 `c(t+τ)=c(t)+τ*sum(a[k]*v[k](t))`。系数通过环境/记录等权的岭回归拟合，并向两点线性预测系数 `[1,0,0,0]` 收缩；三个坐标和左右手共享系数，因此保留平移、旋转等变性。0.1 秒系数为 `{multiscale['0.1s']['coefficients']}`。正则参数从 {{0.001,0.01,0.1,1,10,100}} 在验证集选择；拟合本身只用训练记录。

| 假设 | 依据与范围 | 失效时的影响/检查 |
| --- | --- | --- |
| 很短时间内运动可由局部速度描述 | 原始近 30 Hz 轨迹；预测跨度 0.1 秒 | 快速转向可能失配；保留误差尾部及标坏历史子集 |
| 三套数据共享状态定义和时间结构 | 都有两手 21×3 坐标 | 噪声和手尺度按数据集估计；不强行统一坐标原点 |
| 原始未标坏帧可作条件参考 | 官方允许合格记录作参照 | 未标坏不等于无误差；不能报告真实物理定位精度 |
| 标记历史可能预测后续标记 | 以训练记录估计转移 | 数据集/环境变化可能破坏校准；与运动、视频模型比较 |
| 阶段只用于事后解释 | 原始标注的片段三等分 | 不能在在线预测使用未来片段终点 |

## 4. 状态预测结果

{table(state_table,['dataset',*methods,'gain_vs_two_point_percent'],6)}

指标是先用截至 t 的信息预测，再在 **不早于 t+0.1 秒的首个真实样本**上求欧氏距离；不插值制造目标，也不会用当前更新后的状态和同一个输入比较来证明精度。受不规则时间步和浮点时间戳影响，“0.1s”是选择阈值，不是固定三帧或精确相同跨度。实际平均跨度如下，方法间使用完全相同的目标时刻；不能用跨数据集误差差异直接推断数据集难度。

{table(actual,digits=6)}

左右手先在记录内平均，再在环境内等权平均记录、环境间等权。`state_validation_by_record.csv` 保留实际预测跨度、每手样本量、90% 误差分位以及 recent_flag 子集。

![状态预测对照](../../../artifacts/c_q1_v1/figures/q1_state_forecast.png)

`noise_parameter_bootstrap.csv` 给出 q、R、手尺度的训练环境 bootstrap 区间；`multiscale_coefficient_uncertainty.csv` 给出固定已选正则下的系数环境 bootstrap 区间。两者均未计入再次模型选择的不确定性。验证集仅 4 个环境，不能将数千帧当成同等数量的独立试验。

## 5. 可靠性模型与多视角比较

可靠性目标明确为“至少 0.1 秒后的下一合适样本是否任一手被原始质量标记标坏”。这既不是商品状态/作业成功率，也不是三维定位误差概率。

- 简单基线：训练数据集频率；按数据集和当前是否标坏估计的质量转移概率。
- 数值模型：当前及历史速度、速度变化、手几何尺度、双手距离、已有标记和数据集类别；去掉质量历史另做消融。
- 视觉增强：各视角最近的不晚于决策时刻的抽样图像，最大陈旧时间 0.25 秒，输入手检测数、二维中心、空间跨度、框面积、左右手分类分数与图像年龄。左右手分类分数没有解释为定位准确率。
- 学习器：逻辑回归和直方图提升树；权重为环境等权、环境内记录等权。Platt 校准仅拟合训练侧留环境预测；验证集选择原始概率或校准版本。
- 图像按每秒 5 帧抽样、IMAGE 无状态模式运行，未将视频时序追踪的隐藏帧带入特征。Demo 按容器 FPS 与片段起始时间近似对齐；Pro/Standard 使用 frame CSV 时间戳。

{table(mainquality,['method','brier','log_loss','average_precision','auc','ece'],6)}

![可靠性比较与校准](../../../artifacts/c_q1_v1/figures/q1_reliability.png)

训练留操作者压力检验在 `quality_leave_operator_stress.csv`；它重新拟合训练子集上的模型，仅报告未校准版本，避免把包含被留操作者的全训练校准器带入检验。`quality_paired_vs_transition.csv` 按验证环境做固定模型配对 bootstrap；仅 4 个独立环境，区间为探索性证据。

重要失败/边界：EgoDemo 验证记录没有正质量标记，AUC/AP 不可定义，文件保留为空；此处低 Brier 不能证明检出能力。提升树/视频的更多参数未自动转化为更好的概率预测。当前未证明手中心估计的三维物理准确度；已准备 34 帧独立二维人工核验包，但尚未收到人工标签。

## 6. 任务、环境和阶段差异

![困难片段中的状态与质量](../../../artifacts/c_q1_v1/figures/q1_state_quality_example.png)

上述时间线刻意选取验证集中标坏帧比例最高记录的一段，展示状态、速度与质量如何变化；橙色区间只作低质量诊断，不在其中计算真实位置误差。样例选取规则保存在 `example_selection.json`。

![阶段质量差异](../../../artifacts/c_q1_v1/figures/q1_retrospective_stages.png)

这里用原始语义片段的前、中、后三分之一作事后阶段。P/A/L 是官方片段抽样类别，并不等同具体机器人技能。`retrospective_stage_statistics.csv` 与 `stage_summary.csv` 同时报阶段质量、清洁参考速度、双手距离和环境数。阶段/类别与环境的分布有混杂，当前结果只能说明关联；不可声称某类动作导致遮挡。未把阶段标签输入在线模型。

## 7. 第一问交付与后续接口

| 内容 | 位置 | 状态 |
| --- | --- | --- |
| 记录/区间用途与保留量 | `artifacts/c_q1_v1/record_audit.csv`、`interval_uses.csv`、`retention_summary.csv` | 已运行 |
| 12 维状态与派生相对向量 | `artifacts/c_q1_v1/states/*.npz`，62 条记录 | 已导出；Q2 后续统一接入该版本 |
| 模型与参数敏感性/不确定性 | `state_model.json`、`multiscale_model.json`、各参数 CSV | 已运行 |
| 质量模型、逐帧预测、对照/消融 | `quality_models.joblib`、`quality_validation_predictions.csv`、`quality_validation_metrics.csv` | 已运行 |
| 原始结果→关键数字→图 | `key_numbers.csv`、`figure_inventory.csv`、`figures/` | 同版生成 |
| 三维外部真值/人工二维核验 | `artifacts/c_video/human_2d/` | 独立三维真值缺失；二维人工回填待完成 |
| 测试集评价 | 官方 7 条测试记录 | 未启动；模型最终冻结后进行 |

复现顺序（工作目录为仓库根目录）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/run_c_q1.py --stage state
python scripts/prepare_c_video.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/run_c_q1.py --stage quality
OPENBLAS_NUM_THREADS=1 python scripts/extend_c_q1_dynamics.py
python scripts/report_c_q1.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_c_q1.py tests/test_c_protocol.py
```

数值依赖见 `requirements-c.txt`；视频环境另需 `requirements-c-vision.txt`。本地数值入口是现有 mathorcup Python，视觉入口是 `artifacts/c_vision_env/bin/python`。报告与模型均为 Codex 生成的可审阅工作产物；尚无队员独立复核结论。当前无 commit/push，也没有改写正式论文或正式提交包。
'''
    # Captions are external Markdown paragraphs, never part of the image.
    names=['q1_data_quality','q1_state_forecast','q1_reliability','q1_state_quality_example','q1_retrospective_stages']
    for number,name in enumerate(names,1):
        marker=f'../../../artifacts/c_q1_v1/figures/{name}.png)'
        report=report.replace(marker,marker+f'\n\n*Figure Q1-{number}. {captions[name]}*')
    report+='\n出图规范：[项目图表规范](../../../02_口径与约定/图表规范.md)。全部候选图使用 Times New Roman、外置图题/图注，保留 PDF/SVG/600 DPI PNG。\n'
    (DOC/'C题_Q1结果_v1.md').write_text(report)
    # Standalone English subsection draft; not a fabricated finished competition paper.
    english=f'''# Problem 1: Operational states and observation reliability — development draft

We represented each hand by the centroid of its 21 recorded keypoints and its velocity. The 12-dimensional state is shared with the subsequent missing-observation experiments; the inter-hand vector is derived from the two centroids. Record-local coordinates are retained, without treating hand states as product states.

The development split contains 55 training records ({n_train:,} frames) and seven validation records ({n_val:,} frames). All hand coordinates are finite and the numerical timestamps are strictly increasing. Flagged observations remain available for downweighted fitting and low-quality analysis but are excluded as positional error references. Unflagged measurements are conditional numerical references, not independent error-free physical ground truth.

A quality-aware constant-velocity state-space model was compared with holding the latest centroid, two-point extrapolation and a learned multiscale linear predictor. The latter combines velocities estimated over the previous 1, 3, 6 and 12 frames using shared scalar coefficients fitted on training records. Validation parameters were chosen using environment-balanced 0.1-second forecast error normalized by training hand scale.

{table(state_table,['dataset',*methods],6)}

All entries are mean Euclidean errors in native coordinate units, evaluated at the first existing sample at least 0.1 seconds after each forecast origin. This is a timestamp threshold, not an exactly identical horizon or a fixed three-frame shift. Actual mean horizons are recorded in `actual_prediction_horizons.csv`; all methods share the same target timestamps. The multiscale predictor improved over two-point extrapolation on each dataset. These are validation-selected development results, and final test trajectories have not been evaluated.

Reliability was defined as the probability that either hand would receive a bad-quality flag at least 0.1 seconds later. A dataset-conditioned flag-transition model achieved Brier score {qbest:.6f}, compared with {qbase:.6f} for a training-frequency baseline. Numerical gradient boosting improved ranking discrimination but did not consistently outperform the transition model in probability error. Features extracted from {len(video)} training/validation videos ({sum(r['sampled_frames'] for r in video):,} sampled images) did not improve this reliability target in the current fusion model. The negative result is retained. Image features do not provide an independent 3D accuracy reference.

Environment bootstrap intervals, parameter sensitivity, leave-operator stress tests and retrospective stage summaries accompany the results. Validation includes only four environments; the EgoDemo validation subset contains no positive flags, so its AUC and average precision are undefined. Stage associations are retrospective and potentially confounded by task and environment.

Sources and experiment paths: `docs/analysis/2026初赛/C题_Q1结果_v1.md`, `artifacts/c_q1_v1/`. AI-assisted draft and analysis; independent human verification pending.
'''
    draft=Path('05_论文/待审');draft.mkdir(parents=True,exist_ok=True)
    (draft/'C题_Q1_英文结果初稿.md').write_text(english)
    print(json.dumps(dict(report=str(DOC/'C题_Q1结果_v1.md'),figures=len(captions),state_rows=len(state),quality_rows=len(quality)),ensure_ascii=False))


if __name__=='__main__':main()
