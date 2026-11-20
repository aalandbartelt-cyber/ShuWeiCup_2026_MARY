#!/usr/bin/env python3
"""Publish reproducible Q1 v2 development/test tables and caption-free figures.

AI-assisted by Codex, 2026-11-20. Every number comes from persisted experiments.
"""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shuwei_c.figure_style import apply_style,check_figure,panel_label
from shuwei_c.data import Dataset
from prepare_c_video import videos

OUT=Path('artifacts/c_q1_v2');FIG=OUT/'figures';STYLE,FONT=apply_style();C=STYLE['palette']
EXPORT=OUT;FIGURES_ONLY=False;LEGEND_REVISION=False;LEGEND_CONTRACTS={}
plt.rcParams.update({'mathtext.cal':'Times New Roman','mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman'})
LABEL={'linear':'Two-point linear','scalar_damped':'Scalar damping','multiscale':'Multiscale','quality_filter':'Quality-aware KF','one_euro':'1 euro + velocity','constant':'Hold last'}
COLOR={'linear':C['blue'],'scalar_damped':C['mauve'],'multiscale':C['green'],'quality_filter':C['orange'],'one_euro':C['neutral'],'constant':C['neutral']}


def table(frame,digits=5):
    def fmt(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
        return str(v)
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in r)+' |' for r in frame.itertuples(index=False,name=None))


def save(fig,name):
    if LEGEND_REVISION:
        from shuwei_c.figure_legends_v3 import revise
        LEGEND_CONTRACTS[name]=revise(fig,name)
    check_figure(fig,STYLE)
    for suffix in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{suffix}',dpi=600,bbox_inches='tight')
    fig.savefig(FIG/f'{name}_preview.png',dpi=130,bbox_inches='tight');plt.close(fig)


def quality_rows():
    result=[]
    for tag in ['numerical','video5','video15']:
        folder=OUT/('quality_'+tag);models=json.loads((folder/'models.json').read_text());metrics=pd.read_csv(folder/'metrics.csv')
        for h,choices in models.items():
            for name,spec in choices.items():
                f=metrics[(metrics.horizon==float(h))&(metrics.method==name+'_'+spec['calibration'])].copy()
                f['model']=tag+'_'+name;f['feature_group']=name;f['tag']=tag;result.append(f)
    return pd.concat(result,ignore_index=True)


def main():
    FIG.mkdir(exist_ok=True);freeze=json.loads((OUT/'freeze.json').read_text());test_manifest=json.loads((OUT/'test_run_manifest.json').read_text())
    state=pd.read_csv(OUT/'state_summary.csv');test=pd.read_csv(OUT/'state_test_summary.csv');quality=quality_rows();quality.to_csv(EXPORT/'quality_development_comparison.csv',index=False)
    qtest=pd.read_csv(OUT/'quality_test_metrics.csv');sets=pd.read_csv(OUT/'state_set_validation.csv');testsets=pd.read_csv(OUT/'state_set_test.csv')
    tables=[]
    for split,frame in [('validation',state),('test',test)]:
        f=frame[(frame.horizon==.1)&(frame.context=='all')].pivot(index='dataset',columns='method',values='mean_error').reset_index()
        f['gain_vs_linear_pct']=100*(1-f.multiscale/f.linear);f['gain_vs_scalar_pct']=100*(1-f.multiscale/f.scalar_damped);f.insert(0,'split',split);tables.append(f)
    main_table=pd.concat(tables,ignore_index=True);main_table.to_csv(EXPORT/'state_main_table.csv',index=False)
    # Main point predictions and horizon sensitivity.
    fig,axes=plt.subplots(2,1,figsize=(7,6.3),layout='constrained');ds=['EgoDemo','EgoPro','EgoStandard'];x=np.arange(3)
    for j,method in enumerate(['linear','scalar_damped','multiscale','quality_filter']):
        values=main_table[main_table.split=='validation'].set_index('dataset').loc[ds,method]
        axes[0].bar(x+(j-1.5)*.2,values,width=.19,label=LABEL[method],color=COLOR[method],hatch=['//','..','','xx'][j],edgecolor='white')
    axes[0].set_xticks(x,ds);axes[0].set_ylabel('Mean error (native units)');axes[0].legend(ncol=2,frameon=False);panel_label(axes[0],'A')
    curves=[]
    for h in freeze['state']['horizons']:
        f=pd.read_parquet(OUT/f'state_validation_{float(h):.6f}.parquet');f['normalized']=f.error/f.scale
        s=f.groupby(['method','environment','episode','hand']).normalized.mean().groupby(['method','environment','episode']).mean().groupby(['method','environment']).mean().groupby('method').mean()
        for method,value in s.items():curves.append(dict(horizon=float(h),method=method,error=value))
    curves=pd.DataFrame(curves);curves.to_csv(EXPORT/'state_horizon_curves.csv',index=False)
    for method,marker,ls in zip(['linear','scalar_damped','multiscale','quality_filter','one_euro'],['o','s','^','D','x'],['--',':','-','-.',':']):
        f=curves[curves.method==method].sort_values('horizon');axes[1].plot(f.horizon,f.error,color=COLOR[method],label=LABEL[method],marker=marker,linestyle=ls,markersize=4)
    axes[1].set(xscale='log',yscale='log',xlabel='Forecast horizon (s)',ylabel='Error / training hand scale');axes[1].set_xticks([1/30,.1,.2,.5,1],['0.033','0.1','0.2','0.5','1.0']);axes[1].legend(ncol=2,frameon=False,fontsize=8);panel_label(axes[1],'B');save(fig,'q1v2_state')
    # Calibration belongs to the same selected point predictor.
    fig,axes=plt.subplots(1,2,figsize=(7,3.3),layout='constrained')
    for ax,frame,letter in zip(axes,[sets,testsets],['A','B']):
        for dataset,color,marker in zip(ds,[C['blue'],C['green'],C['orange']],['o','s','^']):
            f=frame[(frame.horizon==.1)&(frame.context=='all')&(frame.scheme=='adaptive')&(frame.dataset==dataset)]
            ax.plot(f.nominal,f.coverage,label=dataset,color=color,marker=marker,markersize=4)
        ax.plot([0,1],[0,1],color=C['neutral'],linestyle='--');ax.set(xlabel='Nominal coverage',ylabel='Observed coverage',xlim=(.45,1),ylim=(0,1));panel_label(ax,letter)
    axes[0].legend(frameon=False,fontsize=8);save(fig,'q1v2_coverage')
    names=['numerical_frequency','numerical_transition','numerical_duration','numerical_geometry','video5_head_video','video15_head_video','video15_all_video']
    labels=['Frequency','Flag transition','Flag + duration','Motion + geometry','Head video, 5 Hz','Head video, 15 Hz','All video, 15 Hz'];colors=[C['neutral'],C['green'],C['orange'],C['blue'],C['mauve'],C['mauve'],C['mauve']]
    fig,axes=plt.subplots(1,2,figsize=(7,3.7),layout='constrained')
    for ax,condition,value,letter in [(axes[0],'all','brier','A'),(axes[1],'currently_good','auc','B')]:
        f=quality[(quality.horizon==.1)&(quality.dataset=='ALL')&(quality.target=='any')&(quality.condition==condition)].set_index('model').loc[names]
        ax.barh(np.arange(len(names)),f[value],color=colors,hatch=['','//','..','xx','//','..','xx']);ax.set_yticks(np.arange(len(names)),labels if ax==axes[0] else []);ax.invert_yaxis();panel_label(ax,letter)
        ax.set_xlabel('Brier score' if value=='brier' else 'AUC while currently normal')
        if value=='auc':ax.set_xlim(0,1)
    save(fig,'q1v2_quality')
    event_frames=[]
    for tag in ['numerical','video5','video15']:
        f=pd.read_csv(OUT/('quality_'+tag)/'events.csv');f['model']=tag+'_'+f.method;event_frames.append(f)
    events=pd.concat(event_frames);events.to_csv(EXPORT/'quality_event_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,3.4),layout='constrained')
    for name,color,marker,ls in zip(['numerical_duration','numerical_motion','numerical_geometry','video15_head_video'],[C['orange'],C['green'],C['blue'],C['mauve']],['o','s','^','D'],['-','--','-.',':']):
        f=events[(events.horizon==.1)&(events['mode']=='onset')&(events.model==name)].sort_values('nominal_false_alarm')
        assert np.all(np.diff(f.false_alarm_fraction)>=-1e-12)
        label={'numerical_duration':'Flag + duration','numerical_motion':'Motion + history','numerical_geometry':'Motion + geometry','video15_head_video':'Head video, 15 Hz'}[name]
        ax.plot(100*f.false_alarm_fraction,100*f.event_recall,marker=marker,color=color,linestyle=ls,label=label)
    ax.set(xlabel='False alarm time among eligible negatives (%)',ylabel='Detected onset events (%)',ylim=(-2,102));ax.legend(frameon=False,ncol=2);save(fig,'q1v2_events')
    # Within-record associations do not identify causal task effects.
    effects=pd.read_csv(OUT/'semantic_within_record_effects.csv');terms=['open_close','pick','place','transport','arrange','phase_middle','phase_late']
    term_labels=['Open / close','Pick / take','Place / put','Transport','Arrange','Middle third','Late third']
    fig,axes=plt.subplots(1,2,figsize=(7,3.8),layout='constrained')
    for ax,target,factor,letter in [(axes[0],'bad_fraction',100,'A'),(axes[1],'mean_log_speed',1,'B')]:
        f=effects[effects.target==target].set_index('term').loc[terms]
        ax.hlines(np.arange(len(terms)),f.low*factor,f.high*factor,color=C['blue'],linewidth=2);ax.scatter(f.coefficient*factor,np.arange(len(terms)),color=C['blue'],s=18)
        ax.axvline(0,color=C['neutral'],linestyle='--');ax.set_yticks(np.arange(len(terms)),term_labels if ax==axes[0] else []);ax.invert_yaxis();panel_label(ax,letter)
        ax.set_xlabel('Flag rate contrast (percentage points)' if factor==100 else 'Log-speed contrast')
    save(fig,'q1v2_context')
    sparse=pd.read_csv(OUT/'q2_sparse_coverage.csv');fig,axes=plt.subplots(1,2,figsize=(7,3.3),layout='constrained');x=np.arange(4)
    for j,(scheme,color,hatch) in enumerate([('dense_Q1',C['orange'],'//'),('sparse_recalibrated',C['green'],'')]):
        f=sparse[sparse.scheme==scheme].sort_values(['interval','delay'])
        axes[0].bar(x+(j-.5)*.32,100*f.covered,width=.31,color=color,hatch=hatch,label=['Q1 radius','Sparse correction'][j]);axes[1].bar(x+(j-.5)*.32,f.evaluated_radius,width=.31,color=color,hatch=hatch)
    ticks=['0.2 / 0','0.2 / 0.1','0.5 / 0','0.5 / 0.1']
    for ax,letter in zip(axes,['A','B']):ax.set_xticks(x,ticks,rotation=25);ax.set_xlabel('Request interval / delay (s)');panel_label(ax,letter)
    axes[0].axhline(90,color=C['neutral'],linestyle='--');axes[0].set_ylabel('Observed coverage (%)');axes[1].legend(*axes[0].get_legend_handles_labels(),loc='upper left',frameon=False,fontsize=8);axes[1].set_ylabel('Mean radius (native units)');save(fig,'q1v2_sparse_interface')
    # Three median validation examples and the largest automatic disagreement.
    geometry=pd.read_csv(OUT/'camera_projection_diagnostics.csv');g=geometry[geometry.split=='validation'];chosen=[]
    for dataset in ds:
        sub=g[g.dataset==dataset];chosen.append(sub.iloc[np.argmin(abs(sub.normalized_error-sub.normalized_error.median()))])
    chosen.append(g.loc[g.normalized_error.idxmax()]);chosen=pd.DataFrame(chosen);chosen.to_csv(EXPORT/'geometry_figure_examples.csv',index=False)
    data=Dataset();fig,axes=plt.subplots(2,2,figsize=(7,6.2),layout='constrained')
    for ax,(_,row),letter in zip(axes.flat,chosen.iterrows(),list('ABCD')):
        rec=data.record(row.episode);video=next(v for v in videos(rec) if v['view']==row['view']);cap=cv2.VideoCapture(str(video['path']));cap.set(cv2.CAP_PROP_POS_FRAMES,int(row.source_frame));ok,img=cap.read();cap.release();assert ok
        ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));ax.scatter([row.hand0_x,row.hand1_x],[row.hand0_y,row.hand1_y],color=C['green'],marker='x',s=45,linewidth=1.4,label='Projected center')
        raw=next(json.loads(line) for line in (Path('artifacts/c_video')/f'{row.episode}__{row["view"]}.jsonl').open() if json.loads(line)['frame_index']==row.source_frame)
        detected=np.array([np.array(d['landmarks'])[:,:2].mean(axis=0) for d in raw['detections']])*[row.image_width,row.image_height]
        ax.scatter(detected[:,0],detected[:,1],facecolors='none',edgecolors=C['orange'],s=55,linewidth=1.4,label='Detected center');ax.set_axis_off();panel_label(ax,letter)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False);save(fig,'q1v2_geometry')
    if FIGURES_ONLY:
        (FIG/'legend_contracts.json').write_text(json.dumps(LEGEND_CONTRACTS,ensure_ascii=False,indent=2)+'\n')
        original=(OUT/'figures/captions.md').read_text()
        (FIG/'captions.md').write_text(original+'\n\n## 图形细节修订说明\n\n本目录修订同版数据的表达，不重做实验。状态图 A 的数字以 10⁻³ 原始单位显示；B 的局部放大范围为 0.09—0.11 秒，放大窗为线性坐标。覆盖率图 A 放大名义 0.89—0.91、观测 0.868—0.912 范围。灰色选区框和连接线标明放大位置；原始值和系列样式不变。各柱/条的关键数值直接标注，事件图彩色点旁的 1%/5%/10% 为训练名义误报目标，方法图例与真实点形和线型相同。记录见 legend_contracts.json；统计含义保持不变。\n')
        print('Generated seven figures with explicit legend contracts');return
    quality_main=quality[(quality.horizon==.1)&(quality.dataset=='ALL')&(quality.target=='any')&(quality.condition=='all')][['model','brier','ap','auc']]
    qcurrent=quality[(quality.horizon==.1)&(quality.dataset=='ALL')&(quality.target=='any')&(quality.condition=='currently_good')][['model','brier','ap','auc','n','positive']]
    testmain=qtest[(qtest.horizon==.1)&(qtest.dataset=='ALL')&(qtest.target=='any')&(qtest.condition=='all')][['method','selected','brier','ap','auc']]
    calibration=pd.concat([sets.assign(split='validation'),testsets.assign(split='test')]);calibration=calibration[(calibration.horizon==.1)&(calibration.dataset=='ALL')&(calibration.context=='all')&(calibration.scheme=='adaptive')][['split','nominal','coverage','radius','pinball']]
    failures=[]
    for split in ['validation','test']:
        f=pd.read_parquet(OUT/f'state_{split}_0.100000.parquet');f=f[f.method=='multiscale']
        for criterion,index in [('largest_error',f.error.idxmax()),('median_error',(f.error-f.error.median()).abs().idxmin())]:
            row=f.loc[index].to_dict();row.update(split=split,selection=criterion);failures.append(row)
    pd.DataFrame(failures).to_csv(EXPORT/'state_failure_examples.csv',index=False)
    context=json.loads((OUT/'semantic_protocol.json').read_text());coverage=pd.read_csv(OUT/'semantic_coverage.csv');gains=pd.read_csv(OUT/'state_paired_gains.csv');ablation=pd.read_csv(OUT/'history_ablation.csv')
    selected_quality={h:r['tag']+'_'+r['method'] for h,r in freeze['quality'].items()}
    captions='''# Q1 v2 图题与图注（外置）\n\n所有论文图均使用 Times New Roman；无图内标题。PDF/SVG 为矢量输出，PNG 为 600 DPI。原始视频画面中的物品文字不属于绘图标注。\n\n1. **q1v2_state — Matched-time hand-state forecasting.** A：验证集 0.1 s 同样本比较，位置为原生坐标单位；B：环境等权的归一化误差随预测跨度变化。KF 和 1 euro 的参数分别由验证集选择；不能将验证选择后的曲线当独立确认。\n2. **q1v2_coverage — Calibration of the selected predictor.** A：验证；B：冻结后测试。每手三维球，0.1 s，运动自适应残差尺度；非双手联合覆盖，非独立物理真值。\n3. **q1v2_quality — Probability scores and early-warning ranking.** A：任一手未来原始坏标记的总体 Brier；B：当前双手均正常时的 AUC。视频/几何模块是输入增量消融，不是独立物理定位精度比较。\n4. **q1v2_events — Onset detection under transferred thresholds.** 23 个符合窗口资格的新发事件；点对应训练留环境预测的 1%、5%、10% 名义误报分位。横轴为验证实际误报时间占比，分母是当前正常且未来窗口无新发事件的采样时刻。纵轴是事件微平均召回；不把名义阈值当验证保证。\n5. **q1v2_context — Within-record semantic associations.** 多标签关键词弱语义、区间三等分回顾性阶段；控制记录常量后的线性描述性关联，横线为 2000 次环境重采样区间。非因果效应；稀少/不可识别项须结合支持量表解释。\n6. **q1v2_sparse_interface — Uncertainty under sparse delayed responses.** A：90% 名义球的验证覆盖；B：平均半径。训练留环境残差仅针对声明的周期/延迟校准。不能推广为任意 Q2 策略的保证。\n7. **q1v2_geometry — Automatic projection consistency examples.** A/B/C：三套数据验证条件子集的中位一致性例；D：该子集最大分歧例。绿叉为原始三维手中心的相机投影，橙圈为自动二维 21 点均值。所有案例已满足原始双方未标坏、自动检测双手；不是无条件精度，更不是独立三维真值。案例记录见 geometry_figure_examples.csv。\n'''
    (FIG/'captions.md').write_text(captions)
    report=f'''# C 题第一问优化结果 v2

2026-11-20。AI 辅助代码与报告，独立人工二维核验尚未回填。正式论文仍在待审；本报告基于已执行实验，不代表无条件物理精度。

## 1. 结论与相对 v1 的改进

本轮真正增加的是公平时间协议、强简单基线、与点预测一致的预测球、环境留出校准、事件级预警评估、相机几何核对和冻结后测试。多尺度预测在验证主跨度有增益，但单参数速度收缩已解释大部分收益；不能把复杂模型的全部收益归因于多时间尺度。视觉模块是否获胜按完整结果判定，不以使用 GPU 或模型大小作为创新依据。

冻结后的测试进一步限制了结论：相对两点线性，EgoDemo 改善 6.98%、EgoPro 改善 4.24%，EgoStandard 退化 0.41%；在 EgoPro 上多尺度也比单参数收缩差 0.50%。因此不能声称多尺度全面优越。90% 球总体测试覆盖 89.39%，但 EgoPro 只有 82.04%，必须保留分层失校准这一失败结果。

v1 的“0.1 秒”按首个不早于目标的样本取值，不同数据集平均跨度偏长。v2 使用最近真实目标、0.55 个中位采样间隔容差，越界剔除；各方法的样本键完全相同。因此 v1/v2 原始误差不可直接当同协议改进率。

训练 55 条 / 16 环境，验证 7 条 / 4 环境，冻结后测试 {test_manifest['test_records']} 条 / {test_manifest['environments']} 环境。点预测评估按手→记录→环境等权；拟合矩在记录内合并合格手坐标，残差校准按环境→记录→合格行等权。明确区分拟合与评价权重。位置仍为原生单位，未宣称厘米精度。

## 2. 点预测与参数解释

共同状态为双手 21 点几何中心及速度，共 12 维；手间相对位移派生，不当作商品状态。0.1 秒采用四尺度线性速度组合，最长历史 0.4 秒。系数见 state_models_development.json，按环境重采样的系数区间见 state_parameter_uncertainty.csv；这些区间固定了已选超参数，并未包含选模不确定性。

{table(main_table[['split','dataset','linear','scalar_damped','multiscale','quality_filter','gain_vs_linear_pct','gain_vs_scalar_pct']])}

验证配对环境增益（百分比需乘 100）：

{table(gains[['dataset','baseline','gain','low','high','environments','records_improved','records']])}

只有 4 个验证环境；单数据集只有 1 个环境时区间退化，不能据此作总体显著性结论。验证选择与验证区间共享数据，冻结后测试用于确认。均值最优不等于尾部最优；逐记录 P90 和异常历史结果完整保留在 state_summary.csv / state_test_summary.csv。

固定已选正则强度后的历史长度消融：

{table(ablation[['lags','longest_lag_s','normalized_error']])}

1/30 秒选质量滤波，其余跨度选多尺度；更强平滑的 1 euro + 速度候选并未赢得主跨度。不能因该结果否定 1 euro 在其他噪声/交互场景的作用。

## 3. 同一预测器的不确定性

用 16 次留环境点预测残差拟合非负尺度：常数项、速度×跨度、转向×跨度、当前及四个历史采样端点是否曾标坏。最后一项不是整段历史所有帧的最大值。再用留环境尺度残差的经验分位形成每手三维球；没有分布无关或物理精度保证。

{table(calibration)}

固定半径与自适应半径的覆盖—半径—径向分位损失均公开。1 秒、EgoStandard 和异常历史有明显失配，不用总体覆盖掩盖分层问题。模型仅条件于“目标原始未标坏且数值有限”的参照；未标坏不等于独立真值。

## 4. 质量发生、持续与恢复

四状态联合分布表示双手均正常、仅左坏、仅右坏、双手坏，保证单手与任一手概率自洽。概率目标是未来时点的原始坏标记；事件预警评估另以整个未来窗口的新发/恢复转变为标签。连续坏帧合并为事件，左/右截尾和无足够历史窗口不纳入相应事件数。

训练侧 4 折按整个环境划分，用留出预测拟合温度及 1/5/10% 名义误报阈值；所有阈值以严格大于比较处理离散并列分数。验证选择分类器与是否温度校准；模型和阈值冻结后才打开测试。全局基准先验向总体四状态频率作小幅收缩，尚不能解释为已验证的完整层次贝叶斯模型。

验证总体：

{table(quality_main)}

当前双手正常条件：

{table(qcurrent)}

冻结质量选择为 `{json.dumps(selected_quality,ensure_ascii=False)}`。按方案“差异微小优先简单”具体化：距最低验证 Brier 不超过 1% 时优先更少输入；此规则在 15 Hz 评估前写入冻结脚本。动作持续时间/几何可能改善排序或事件召回，仍不能取代总体概率校准证据。完整事件表现见 quality_event_comparison.csv；名义训练误报率不等于迁移后的实际误报率。

冻结后测试主跨度：

{table(testmain)}

## 5. 视频、几何和更强视觉候选

5 Hz 与 15 Hz 均覆盖开发集 166 个视频；使用相同 MediaPipe IMAGE 模式和权重，头部/全部视角分开消融。15 Hz 有 {sum(m['sampled_frames'] for m in json.loads(Path('artifacts/c_video_v2_15hz/video_manifest.json').read_text()))} 个实际采样画面。头部逐手特征由可用三维投影与二维检测作一对一分配；腕部仍是视角级汇总，因为缺少本轮核实的腕相机世界外参。左右手分类分数不当作准确性置信度，无检测也不当作原始手观测坏的真值。

训练侧比较 32 种约定，三套数据均选择 wxyz、标准相机到世界旋转、正 z 前向。验证双方未标坏且检出双手的子集，归一化中心距离中位数约 0.8%；最大图像—数值最近时标差约 1.53 微秒。该差值只是提供时标的内部一致性，不能证明传感器绝对同步。EgoDemo 头部相机主点与图像中心/分辨率已逐视频核对；腕部主点不居中，本模型没有对腕相机套用该假设。

WildHands 等透视感知模型可提供相机与裁剪归一化设计灵感，但当前自动二维一致性已较好，主要瓶颈是可靠性标记迁移、少量异常事件及独立真值缺失。5/15 Hz 全视角增量已做实测；没有证据支持继续堆叠或微调更大三维模型能改善本题的质量标记目标。因此本轮不开展云端训练，也不宣称运行过该候选。独立人工 34 帧核验包仍待回填，本轮图像人工智能审阅只做几何诊断。

## 6. 任务、环境与阶段

原始语义区间覆盖 {int(coverage.covered.sum())}/{int(coverage.frames.sum())} 帧（{100*coverage.covered.sum()/coverage.frames.sum():.2f}%）。采用公开固定关键词的多标签弱语义，并将给定区间回顾性分三段。区间终点、文字描述和任务哈希不进入在线预测。62 份逐记录完整任务描述均不同，没有同一完整描述跨环境的重复；粗动作语义可以跨环境。不能将环境、操作者和具体任务效应都解释为已经独立辨识。

按记录中心化后估计语义/阶段关联，环境聚类重采样给区间，支持量和秩保存在 semantic_within_record_effects.csv / semantic_summary.csv / context_confounding.csv。描述可以包含多个动作，弱标签不是逐帧抓取真值；三等分也不是精确操作阶段。关联用于定位失效条件，不能作因果归因。

设计矩阵 8 列、秩为 7：关键词 reach 没有命中样本，该项不可估计，已从系数图排除。原始广义逆输出中的该项零值仅是代数占位，不表示“伸手动作没有影响”。

## 7. Q2 接口与稀疏回传

PrefixPredictor 只接收已经获得的时标、双手中心和质量；140 个真实记录前缀与批量预测一致。另对 7 条记录的未来手/头/相机位置和质量实际扰动，截断前数值/几何特征不变。视频采用 as-of 图像，图像时标晚于决策时不可用。

代表性回放固定每条验证记录 20/50/80% 位置的一秒窗口、0.2/0.5 秒请求间隔、0/0.1 秒延迟。回包只包含请求时刻的样本；隐藏区间不补送，三方法同请求预算。Q1 的完整历史区间在该模式下明显偏窄；训练侧 55 条、16 环境的留出点预测残差给出 90% 球乘数 {json.loads((OUT/'q2_sparse_calibration.json').read_text())['multiplier']:.4f}，覆盖改善同时付出半径代价：

{table(sparse)}

此乘数仅适用所检验回传模式。完整 Q2 策略、任意遮蔽强度或超过 1 秒外推仍须独立评价；不将此集成检查写成第二问已经完成。

## 8. 对优秀论文的实质借鉴与剩余边界

- 个人 MathorCup 论文：采用收缩强基线、留组残差和配对环境区间；不沿用未经独立检验的“高覆盖即准确”。
- 个人国赛论文：增加真实未来扰动、接口账本、模型版本对齐，避免 Q1 最优点预测与 Q2 另一模型区间拼接。
- 2025 C2561635：保留状态—预测—风险的可解释链；用真实回放证据代替只画框架。
- C2556711 / C2556821：借鉴参数与灵敏度对照，明确概率和阈值定义；未把其自述结果当本题有效性证明。
- A2553180 / B2549191：补齐失效、混杂、类别不平衡与消融；参考文本中不一致数字没有继承。

深度逐篇证据与页码见 C题_Q1深度评审与优化路线_20261120.md。尚缺独立物理真值与人工二维回填；当前结果足以支持限定范围的 Q1 模型，但不足以声称任意环境、任意遮挡或厘米级精度。测试结果完整保留，不按测试重选模型。

## 9. 冻结、复现与产物

冻结 UTC：{freeze['timestamp_utc']}；测试开始：{test_manifest['started_utc']}。freeze.json 含模型、参数、阈值及代码哈希；test_run_manifest.json 含冻结文件哈希。主脚本按 run_c_q1_v2_state → audit_c_q1_v2_geometry → 视频提取 → run_c_q1_v2_quality（三组）→ context/interface/sparse/additional → freeze → test → report 顺序运行。numeric 环境为 mathorcup，视觉环境为 artifacts/c_vision_env。

图：artifacts/c_q1_v2/figures/，图题外置 captions.md，Times New Roman，PDF/SVG/600 DPI PNG；预览文件仅用于审阅。数值/字体/测试验收另见 verification.json。原始数据与大产物不入 Git；没有修改正式论文或交付目录。
'''
    Path('docs/analysis/2026初赛/C题_Q1优化结果_v2.md').write_text(report)
    english=f'''# Q1: matched-time state forecasting and observation reliability (v2 draft)

AI-assisted draft for review. Captions are external; numerical units remain native.

We represent the two hands by their 21-point geometric centers and velocities, yielding a twelve-dimensional state. Relative hand displacement is derived from this state and does not represent an inventory item. Forecasts are evaluated at the nearest observed target time, with a tolerance of 0.55 median sampling intervals and at least 0.4 s of available history. Targets beyond the recording endpoint are excluded. All methods share identical eligible hand-target pairs. Reported errors average hands within recordings, recordings within environments, and environments equally.

The development comparison includes last-value persistence, two-point extrapolation, a single fitted velocity damping coefficient, regularized multiscale velocities, a quality-aware constant-velocity filter, and a vector adaptation of the 1 euro filter followed by causal velocity extrapolation. The selected method is the quality-aware filter at 1/30 s and multiscale extrapolation at the remaining declared horizons. A strong scalar baseline explains much of the gain over unregularized extrapolation.

{table(main_table[['split','dataset','linear','scalar_damped','multiscale','gain_vs_linear_pct','gain_vs_scalar_pct']])}

Prediction balls are calibrated from leave-environment residuals of the same point predictor. A nonnegative scale model uses observed speed, turning, forecast age and flags at the current and sampled historical endpoints. The balls describe conditional agreement with supplied unflagged numerical references, not independent physical accuracy or joint two-hand coverage. Empirical quantiles do not provide a distribution-free guarantee under environment shift.

{table(calibration)}

Reliability is represented by four joint flag states, preserving coherent single-hand and any-hand probabilities. Models are calibrated with four environment-grouped folds. Probability targets concern the future endpoint; onset and recovery events are separately evaluated over the prediction window. Thresholds are selected from training out-of-fold predictions and transferred unchanged. Event recall, lead time and actual false-alarm time must be reported together. Detection of a current abnormal condition must not be presented as advance warning of a new one.

The frozen reliability choices are {json.dumps(selected_quality)}. Three feature regimes compare numerical observations, head-camera video and all available views, including a fixed 5 versus 15 Hz ablation. Head-camera projection consistency supports the coordinate convention in a selected two-detection, unflagged subset; it does not supply independent ground truth. Missing wrist detections are not equivalent to bad original hand observations.

Task descriptions are weak multi-label annotations used only for retrospective explanation. Within-record centering reduces constant recording-level confounding, but unique full task descriptions and sparse environment support prevent independent causal attribution to task, operator and environment. Semantic endpoints never enter online features.

A delayed-response integration probe confirms that point forecasts use only returned request-time samples. Dense-history uncertainty transfers poorly to sparse updates; a separate empirical residual correction is reported together with its increased radius and limited observation-pattern scope. This probe does not replace the complete Q2 evaluation.

Models and thresholds were frozen at {freeze['timestamp_utc']} before the official test evaluation began at {test_manifest['started_utc']}. No model was reselected on test outcomes. Independent physical accuracy and the pending blinded human 2D audit remain outside the present claims.
'''
    Path('05_论文/待审/C题_Q1_英文优化稿_v2.md').write_text(english)
    (FIG/'style_manifest.json').write_text(json.dumps(dict(font=FONT,style=STYLE,captions='captions.md',titles_inside=False,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')}),indent=2)+'\n')
    print('Created seven caption-free figure sets, Chinese report and English draft')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--legend-revision',action='store_true');args=parser.parse_args()
    if args.legend_revision:
        FIG=Path('artifacts/c_q1_v3/figures_legend_revision');FIG.mkdir(parents=True,exist_ok=True)
        EXPORT=FIG;FIGURES_ONLY=True;LEGEND_REVISION=True
    main()
