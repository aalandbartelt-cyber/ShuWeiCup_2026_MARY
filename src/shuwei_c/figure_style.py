"""Project figure contract: genuine Times New Roman, external captions, semantic colors.

AI-assisted: OpenAI Codex, GPT-6 family, 2026-11-20.
User requirements override the optional generic skill theme starter.
"""
from pathlib import Path
import json
from matplotlib import font_manager,rcParams


def apply_style():
    config=json.loads((Path(__file__).resolve().parents[2]/'configs/figure_style.json').read_text())
    # Register local licensed fonts if present. Do not redistribute font files.
    for directory in [Path.home()/'.local/share/fonts',Path('/mnt/c/Windows/Fonts')]:
        for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
            path=directory/name
            if path.exists():font_manager.fontManager.addfont(str(path))
    font=font_manager.findfont(font_manager.FontProperties(family=config['font_family']),fallback_to_default=False)
    actual=font_manager.FontProperties(fname=font).get_name()
    if actual!='Times New Roman':raise RuntimeError(f'Incorrect figure font: {actual}')
    rcParams.update({'font.family':config['font_family'],'font.size':config['base_font_pt'],
        'axes.labelsize':config['base_font_pt'],'xtick.labelsize':config['tick_font_pt'],
        'ytick.labelsize':config['tick_font_pt'],'legend.fontsize':config['legend_font_pt'],
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.7,
        'grid.color':config['palette']['grid'],'axes.axisbelow':True,'text.color':config['palette']['text'],
        'axes.labelcolor':config['palette']['text'],'xtick.color':config['palette']['text'],
        'ytick.color':config['palette']['text'],'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
        'mathtext.bf':'Times New Roman:bold'})
    return config,font


def check_figure(fig,config):
    if getattr(fig,'_suptitle',None) is not None and fig._suptitle.get_text():
        raise ValueError('Figure titles belong in the paper caption, not the image')
    for ax in fig.axes:
        if any(ax.get_title(loc=loc) for loc in ['left','center','right']):
            raise ValueError('Axis/subplot titles are disabled; use panel letters and external captions')
    from matplotlib.text import Text
    for artist in fig.findobj(match=Text):
        if artist.get_visible() and artist.get_text().strip():
            if artist.get_fontsize()<config['minimum_text_pt']-1e-9:
                raise ValueError(f'Text smaller than minimum: {artist.get_text()}')
            if artist.get_fontfamily()!=[config['font_family']]:
                raise ValueError(f'Unexpected font family: {artist.get_fontfamily()}')


def panel_label(ax,letter):
    ax.text(-.03,1.025,letter,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom',ha='left')
