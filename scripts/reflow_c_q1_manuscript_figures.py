#!/usr/bin/env python3
"""Reflow existing Q1 plots for the manuscript; isolate all rendering outputs.

Scientific models, frozen results, original figure scripts and original exports
are unchanged. The video renderer checks four existing validation forecasts.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
PAPER = ROOT / '05_论文/待审/C题_问题一稿_v1'
OUT = ROOT / 'artifacts/c_q1_manuscript/reflow'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('groups', nargs='*', default=['fixed', 'progress', 'v2', 'data', 'video'])
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    import matplotlib.pyplot as plt
    from shuwei_c.figure_style import apply_style
    original_save = Figure.savefig
    records = {}

    # Recompose at the final physical width, preserving a readable font size.
    def save(self, filename, *a, **kw):
        if not getattr(self, '_paper_reflowed', False):
            name = Path(filename).stem
            w, h = self.get_size_inches()
            self.set_size_inches(5.35, h * .83)
            if name == 'q1v3_video_alignment_controls':
                self.set_size_inches(5.0, 2.35)
            for text in self.findobj(match=Text):
                text.set_fontsize(max(8.2, min(9.5, text.get_fontsize())))
                replacements = {
                    'Four-scale velocity combination': 'Four-scale velocity model',
                    'Adaptive displacement mixture': 'Adaptive displacement model',
                    'v2 adaptive radius': 'Motion-scaled radius',
                    'v2 adaptive': 'Motion-scaled radius',
                    'v3 conditional + feedback': 'Conditional radius + feedback',
                    'Point labels: training false-alarm target': 'Labels: training targets',
                    'First sample at/after target': 'First sample at/after target',
                    '21-landmark x range (min–max)': '21-landmark range',
                    'Mean absolute target-time error (ms)': 'Mean target-time error (ms)',
                    'Time within example recording (s)': 'Time in example (s)',
                    'Elapsed fraction of recording (%)': 'Elapsed recording (%)',
                    'Flagged hand samples (%) [sqrt color scale]': 'Flagged samples (%) [sqrt scale]',
                    'Brier reduction vs transition prior (%)': 'Brier reduction (%)',
                    'Numerical history + camera geometry': 'Numerical + geometry',
                    'Current-to-future flag transition': 'Flag transition',
                    'Numerical + head video (15 Hz)': 'Head video (15 Hz)',
                    'Numerical + head video (5 Hz)': 'Head video (5 Hz)',
                    'Numerical + all views (15 Hz)': 'All views (15 Hz)',
                    '95% confidence interval (environment bootstrap)': '95% interval (environment bootstrap)',
                }
                value = text.get_text()
                for old, new in replacements.items():
                    value = value.replace(old, new)
                if name in {'q1v3_fixed_coverage', 'q1v3_radius_development'}:
                    value = value.replace(': ', ':\n')
                text.set_text(value)
            if name == 'q1v3_data_characteristics':
                from matplotlib.patches import FancyBboxPatch
                colors = apply_style()[0]['palette']
                # Rebuild the accounting tree for its smaller printed panel.
                ax = self.axes[5]
                ax.clear()
                ax.axis('off')
                ax.text(-.03, 1.025, 'F', transform=ax.transAxes,
                        fontsize=9.5, fontweight='bold', va='bottom')
                def box(x, y, w, h, label, color):
                    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                        boxstyle='round,pad=0.008', transform=ax.transAxes,
                        facecolor=color+'12', edgecolor=color, linewidth=.8,
                        clip_on=False))
                    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.2,
                            va='center', ha='center')
                for y, label in [(.9, 'Raw: 107,658'), (.64, 'After warmup: 106,172'),
                                 (.38, 'Matched: 105,766')]:
                    box(.5, y, .88, .13, label, colors['blue'])
                box(.24, .12, .46, .20, 'Qualified\n105,057', colors['green'])
                box(.76, .12, .44, .20, 'Flagged\n709', colors['orange'])
                for y, label in [(.77, '−1,486 warmup'), (.51, '−406 endpoint')]:
                    ax.annotate('', xy=(.5,y-.055), xytext=(.5,y+.055),
                                xycoords='axes fraction', arrowprops={'arrowstyle':'->',
                                'color':colors['neutral'], 'lw':.8})
                    ax.text(.55,y,label,transform=ax.transAxes,fontsize=8.2,va='center')
                for x in [.24,.76]:
                    ax.annotate('',xy=(x,.235),xytext=(.5,.31),xycoords='axes fraction',
                                arrowprops={'arrowstyle':'->','color':colors['neutral'],'lw':.8})
                self.axes[4].set_ylim(2.6, -1.8)
            if name == 'q1v3_fixed_coverage':
                for text in self.axes[1].texts:
                    if text.get_text() in {'90.71%', '90.06%'}:
                        old = text.get_position()
                        text.set_position((5, old[1]))
                        text.set_ha('left')
            if name == 'q1v2_state':
                # Reserve an upper band for both legend rows above all bar labels.
                self.axes[0].set_ylim(0, 24)
            if name == 'q1v3_video_keyframes':
                self.set_size_inches(5.35, 6.9)
                ax = self.axes[-2] if len(self.axes) > 13 else self.axes[-1]
                # Locate the probability axis independently of colorbar order.
                ax = next(x for x in self.axes if x.get_ylabel() == '0.5 s onset probability (%)')
                handles, labels = ax.get_legend_handles_labels()
                old = ax.get_legend()
                # Keep shaded-span handles from the existing seven-entry legend.
                handles = old.legend_handles
                labels = [x.get_text() for x in old.get_texts()]
                ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, -.37),
                          ncol=2, fontsize=8.2, frameon=False, columnspacing=.8, handlelength=1.8)
                for legend in self.legends:
                    for t in legend.get_texts():
                        t.set_fontsize(8.2)
                # More room for the bottom legend; preserve all twelve frames.
                self.subplots_adjust(bottom=.17)
            self._paper_reflowed = True
            self.canvas.draw()
            records[name] = {'canvas_inches': self.get_size_inches().tolist(),
                             'minimum_base_font_pt': 8.2}
        return original_save(self, filename, *a, **kw)

    Figure.savefig = save
    mapping = {
        'fixed': 'report_c_q1_v3_fixed',
        'progress': 'report_c_q1_v3_progress',
        'v2': 'report_c_q1_v2',
        'data': 'report_c_q1_v3_data_characteristics',
        'video': 'report_c_q1_v3_video_case',
    }
    original_sources = json.loads((PAPER / 'figure_sources.json').read_text())
    before = {r['source']: sha(ROOT / r['source']) for r in original_sources}
    for group in args.groups:
        directory = OUT / group
        directory.mkdir(parents=True, exist_ok=True)
        name = mapping[group]
        spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if group == 'v2':
            module.FIG = module.EXPORT = directory
            module.FIGURES_ONLY = module.LEGEND_REVISION = True
        elif group == 'progress':
            module.FIG = directory
        else:
            module.OUT = directory
        print('Reflowing', group, flush=True)
        module.main()
        plt.close('all')
    for r in original_sources:
        assert sha(ROOT / r['source']) == before[r['source']], r['source']
        if r.get('layout_group') == 'split_data':
            continue
        candidates = list(OUT.glob('*/' + Path(r['source']).name))
        if candidates:
            source = candidates[0]
            (PAPER / r['local']).write_bytes(source.read_bytes())
            r['source_sha256'] = before[r['source']]
            r['sha256'] = sha(source)
            r['layout_source'] = str(source.relative_to(ROOT))
            r['layout_note'] = 'Reflowed at manuscript width; scientific source and values retained.'
    (PAPER / 'figure_sources.json').write_text(json.dumps(original_sources, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'layout_record.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
