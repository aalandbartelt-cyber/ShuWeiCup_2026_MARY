"""Explicit legend contracts for the user's Q1 figure revision request."""
from matplotlib.lines import Line2D
from .figure_style import apply_style,panel_label
from .figure_detail_v3 import line_zoom,label_bars

STYLE,_=apply_style();COLORS=STYLE['palette']
CONTRACTS={
 'q1v2_state':{'encoding':'color + line/hatch = forecasting method','labels':{
    'Two-point linear':'Two-point velocity extrapolation','Scalar damping':'Single-coefficient velocity damping',
    'Multiscale':'Four-scale velocity combination','Quality-aware KF':'Quality-aware Kalman filter',
    '1 euro + velocity':'Speed-adaptive filter + velocity'},'panels':'A: 0.1 s error; B: horizon sensitivity'},
 'q1v2_coverage':{'encoding':'color + marker = dataset; dashed gray = equality reference',
    'required':['EgoDemo','EgoPro','EgoStandard','Ideal coverage: observed = nominal'],'panels':'A: validation; B: previously frozen test'},
 'q1v2_quality':{'encoding':'row label identifies input/model; bars in both panels refer to the same row',
    'panels':'A: any-hand Brier; B: AUC conditional on currently normal hands'},
 'q1v2_events':{'encoding':'line color and marker = feature set; direct percentage labels = training OOF false-alarm target',
    'required':['Flag + duration','Motion + history','Motion + geometry','Head video, 15 Hz'],
    'panels':'single panel; x is actual validation false-alarm fraction, not training target'},
 'q1v2_context':{'encoding':'point = coefficient; interval = environment-bootstrap uncertainty; dashed gray = zero',
    'required':['Within-record coefficient','95% confidence interval (environment bootstrap)','Zero coefficient'],
    'panels':'A: bad-flag fraction; B: log-speed'},
 'q1v2_sparse_interface':{'encoding':'bar color/hatch = calibration regime; dashed gray = nominal coverage',
    'required':['Dense-history calibration','Sparse-response calibration','Nominal coverage: 90%'],
    'panels':'A: empirical coverage; B: radius'},
 'q1v2_geometry':{'encoding':'green cross = supplied 3D projection; orange circle = automatic 2D detection',
    'required':['Projection of supplied 3D hand center','Mean of detected 2D landmarks'],
    'panels':'A/B/C: per-dataset median agreement; D: largest disagreement in eligible validation subset'},
}


def all_legends(fig):
    return list(fig.legends)+[ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]


def revise(fig,name):
    contract=CONTRACTS[name];details={}
    if name=='q1v2_state':
        colors={'Two-point linear':COLORS['blue'],'Scalar damping':COLORS['orange'],'Multiscale':COLORS['green'],'Quality-aware KF':COLORS['mauve'],'1 euro + velocity':COLORS['neutral']}
        for bars in fig.axes[0].containers:
            for patch in bars:patch.set_height(1000*patch.get_height());patch.set_facecolor(colors[bars.get_label()])
        details['bar_labels']=label_bars(fig.axes[0],'.2f',factor=1000)
        maximum=max(p.get_height() for p in fig.axes[0].patches);fig.axes[0].set_ylim(0,maximum*1.27);fig.axes[0].set_ylabel('Mean error (10⁻³ native units)')
        for line in fig.axes[1].lines:line.set_color(colors[line.get_label()])
        for ax in fig.axes:
            handles,labels=ax.get_legend_handles_labels()
            ax.legend(handles,[contract['labels'].get(s,s) for s in labels],ncol=2,frameon=False,fontsize=8,loc='upper left')
        _,z=line_zoom(fig.axes[1],[.60,.08,.37,.43],(.09,.11),(.26,.38),[.09,.1,.11],[.28,.32,.36]);details['zooms']=[z]
    elif name=='q1v2_coverage':
        for ax in fig.axes:ax.lines[-1].set_label('Ideal coverage: observed = nominal')
        fig.axes[0].legend(frameon=False,fontsize=8,loc='upper left')
        _,z=line_zoom(fig.axes[0],[.53,.10,.44,.43],(.89,.91),(.868,.912),[.89,.90,.91],[.88,.90]);details['zooms']=[z]
    elif name=='q1v2_quality':
        fig.axes[0].set_yticks(range(7),['Dataset flag frequency','Current-to-future flag transition','Flags + status duration',
            'Numerical history + camera geometry','Numerical + head video (5 Hz)','Numerical + head video (15 Hz)','Numerical + all views (15 Hz)'],fontsize=8)
        details['bar_labels']=[]
        for i,ax in enumerate(fig.axes):
            details['bar_labels']+=label_bars(ax,'.5f' if i==0 else '.3f')
            ax.set_xlim(0,max(b.get_width() for b in ax.patches)*1.28 if i==0 else 1.)
    elif name=='q1v2_events':
        ax=fig.axes[0];lines=list(ax.lines);ax.get_legend().remove()
        offsets=[[(0,-14),(0,8),(0,8)],[(10,-9),(9,-6),(0,8)],[(-8,9),(0,9),(0,8)],[(-4,-13),(-5,9),(0,8)]]
        direct=[]
        for i,line in enumerate(lines):
            for j,value in enumerate([1,5,10]):
                ax.annotate(f'{value}%',(line.get_xdata()[j],line.get_ydata()[j]),xytext=offsets[i][j],textcoords='offset points',ha='center',fontsize=8,color=line.get_color())
                direct.append(dict(method=line.get_label(),training_false_alarm=value/100,x=float(line.get_xdata()[j]),y=float(line.get_ydata()[j])))
        fig.set_size_inches(7,4.2);fig.legend(lines,[line.get_label() for line in lines],loc='outside upper center',ncol=2,frameon=False,fontsize=8)
        ax.text(.98,.05,'Point labels: training false-alarm targets',transform=ax.transAxes,ha='right',fontsize=8)
        details['direct_threshold_labels']=direct
    elif name=='q1v2_context':
        original=list(fig.axes);row_names=[t.get_text() for t in original[0].get_yticklabels()];source=[]
        for ax in original:
            segments=ax.collections[0].get_segments();points=ax.collections[1].get_offsets()
            source.append(dict(intervals=segments,points=points,xlim=ax.get_xlim(),ylim=ax.get_ylim(),xlabel=ax.get_xlabel()))
        fig.clear();axes=fig.subplots(2,2,gridspec_kw={'width_ratios':[3.5,1.6]});details['interval_labels']=[]
        for k,(plot_ax,text_ax,data) in enumerate(zip(axes[:,0],axes[:,1],source)):
            for i,(segment,point) in enumerate(zip(data['intervals'],data['points'])):
                low,high=float(segment[0,0]),float(segment[1,0]);value=float(point[0])
                plot_ax.hlines(i,low,high,color=COLORS['blue'],linewidth=2);plot_ax.scatter([value],[i],color=COLORS['blue'],s=18)
                precision=2 if k==0 else 3;label=f'{value:.{precision}f} [{low:.{precision}f}, {high:.{precision}f}]'
                text_ax.text(0,i,label,va='center',fontsize=8)
                details['interval_labels'].append(dict(panel='AB'[k],term=row_names[i],estimate=value,low=low,high=high,label=label))
            plot_ax.axvline(0,color=COLORS['neutral'],linestyle='--');plot_ax.set(xlim=data['xlim'],ylim=data['ylim'],xlabel=data['xlabel']);plot_ax.set_yticks(range(len(row_names)),row_names);panel_label(plot_ax,'AB'[k])
            text_ax.set(xlim=(0,1),ylim=data['ylim']);text_ax.set_axis_off();text_ax.text(0,1.02,'Estimate [95% CI]',transform=text_ax.transAxes,fontsize=8)
        handles=[Line2D([],[],marker='o',color=COLORS['blue'],linestyle='None',label='Within-record coefficient'),
            Line2D([],[],color=COLORS['blue'],linewidth=2,label='95% confidence interval (environment bootstrap)'),
            Line2D([],[],color=COLORS['neutral'],linestyle='--',label='Zero coefficient')]
        fig.set_size_inches(7,6.2);fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False,fontsize=8)
    elif name=='q1v2_sparse_interface':
        handles,_=fig.axes[0].get_legend_handles_labels();fig.axes[1].get_legend().remove()
        reference=Line2D([],[],color=COLORS['neutral'],linestyle='--')
        fig.set_size_inches(7,3.8);fig.legend([*handles,reference],contract['required'],loc='outside upper center',ncol=2,frameon=False,fontsize=8)
        details['bar_labels']=label_bars(fig.axes[0],'.1f')+label_bars(fig.axes[1],'.3f')
        fig.axes[0].set_ylim(0,104);fig.axes[1].margins(y=.20)
    elif name=='q1v2_geometry':
        handles,_=fig.axes[0].get_legend_handles_labels()
        for legend in list(fig.legends):legend.remove()
        fig.legend(handles,contract['required'],loc='outside lower center',ncol=1,frameon=False,fontsize=9)
    text=[text.get_text() for legend in all_legends(fig) for text in legend.get_texts()]
    missing=set(contract.get('required',[]))-set(text)
    if missing:raise ValueError(f'Unexplained legend entries for {name}: {missing}')
    return dict(contract,actual_legend_labels=text,details=details)
