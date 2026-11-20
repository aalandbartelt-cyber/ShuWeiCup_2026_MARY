"""Faithful local zooms and direct numerical labels for scientific plots."""
import numpy as np
from matplotlib.colors import to_rgba


def line_zoom(ax,bounds,xlim,ylim,xticks,yticks,lines=None):
    lines=list(ax.lines) if lines is None else lines
    inset=ax.inset_axes(bounds);inset.set_zorder(6)
    for line in lines:
        copied,=inset.plot(line.get_xdata(),line.get_ydata(),color=line.get_color(),linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),marker=line.get_marker(),markersize=line.get_markersize(),
            markerfacecolor=line.get_markerfacecolor(),markeredgecolor=line.get_markeredgecolor(),
            markeredgewidth=line.get_markeredgewidth(),alpha=line.get_alpha())
        np.testing.assert_array_equal(copied.get_xdata(),line.get_xdata());np.testing.assert_array_equal(copied.get_ydata(),line.get_ydata())
        assert to_rgba(copied.get_color())==to_rgba(line.get_color())
        assert copied.get_linestyle()==line.get_linestyle() and copied.get_marker()==line.get_marker()
        assert to_rgba(copied.get_markerfacecolor())==to_rgba(line.get_markerfacecolor())
        assert to_rgba(copied.get_markeredgecolor())==to_rgba(line.get_markeredgecolor())
    inset.set(xlim=xlim,ylim=ylim,xticks=xticks,yticks=yticks)
    inset.tick_params(axis='both',labelsize=8,pad=1,length=2)
    for spine in inset.spines.values():spine.set_visible(True);spine.set_color('#9A9A9A');spine.set_linewidth(.8)
    indicator=ax.indicate_inset_zoom(inset,edgecolor='#9A9A9A',linewidth=.8,alpha=.8)
    # A local zoom is explicitly linear, even when the full context uses logs.
    assert inset.get_xscale()=='linear' and inset.get_yscale()=='linear'
    return inset,dict(xlim=list(xlim),ylim=list(ylim),scale='linear',source_series=len(lines),source_values_identical=True,source_styles_identical=True)


def label_bars(ax,fmt='.2f',factor=1.,rotation=0):
    from matplotlib.container import BarContainer
    labels=[]
    for bars in ax.containers:
        if not isinstance(bars,BarContainer):continue
        values=np.asarray(bars.datavalues)*factor
        text=[format(v,fmt) for v in values]
        ax.bar_label(bars,labels=text,padding=3,fontsize=8,rotation=rotation,bbox=dict(facecolor='white',edgecolor='none',pad=.2))
        labels.extend(dict(value=float(v),label=t) for v,t in zip(values,text))
    return labels
