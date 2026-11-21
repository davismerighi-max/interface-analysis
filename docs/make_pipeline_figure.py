"""Draw the pipeline as a step-by-step flow diagram.

    python make_pipeline_figure.py

Writes fig_pipeline_flow.png/.pdf. Everything is laid out in INCH coordinates
(ax.set_xlim(0, FW), ax.set_ylim(0, FH)) so that FancyBboxPatch corner rounding stays
circular; in axes-fraction coordinates the rounding warps with the aspect ratio.
Text is wrapped against MEASURED glyph widths (TextPath), never a guessed character
width, and every box height is computed from its own wrapped content.
"""
import sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

FW = 7.2
INK, MUT = fs.INK, '#6E6E6E'
C_IN, C_STEP, C_OUT = '#EAF0F7', '#FFFFFF', '#FDF0E4'
E_IN, E_STEP, E_OUT = '#9DB9D6', '#B9B9B9', '#E7B98A'
C_OPT, C_CHIP, E_CHIP = '#F4F4F2', '#F7F9FB', '#CFDCE9'
TOOL = '#2C6FAD'                      # software is always written in this blue
# There is no second pathway through the pipeline: the same steps always run, and a few
# of them ADAPT to what the model contains. These badges mark exactly which ones.
BADGE = {'nuc':   ('#E7EEF7', '#3C6FA8', 'nucleic-aware'),
         'shape': ('#FBF1E6', '#A8762F', '--shape'),
         'prot':  ('#F1F1EF', '#6F6F6F', 'protein only'),
         'lig':   ('#F7F1E9', '#8A6A4B', 'ligands / cofactors')}
# One continuous chain, grouped by colour rather than boxed: each script owns a hue, and
# every chip, icon and connector in that stretch carries it.
GROUP = [dict(chip='#EDF3FA', edge='#B4CCE3', dark='#2C6FAD', icon='#4A7FB5'),
         dict(chip='#EDF5F1', edge='#B2D5C4', dark='#2E7D5B', icon='#3E8C6A'),
         dict(chip='#F3F0F8', edge='#C8BCDD', dark='#6B4E9E', icon='#7A5FA6')]

MARG, GAPC = 0.10, 0.32               # outer margin, gap between the three columns
CW = (FW - 2 * MARG - 2 * GAPC) / 3.0  # column width
CHIPW = CW - 0.26                      # sub-step chip
TXTW = CHIPW - 0.20                    # text width inside a chip

_FP = {'sans': FontProperties(family='DejaVu Sans'),
       'sansb': FontProperties(family='DejaVu Sans', weight='bold'),
       'sansi': FontProperties(family='DejaVu Sans', style='italic'),
       'mono': FontProperties(family='DejaVu Sans Mono')}


def tw(s, fontsize, face='sans'):
    """Width of a string in INCHES, measured from the glyph outlines."""
    return TextPath((0, 0), s, size=fontsize,
                    prop=_FP[face]).get_extents().width / 72.0 if s else 0.0


def lh_for(fontsize):
    return 1.42 * fontsize / 72.0


def split(text, width, fontsize, face='sans'):
    lines, cur = [], ''
    for word in text.split(' '):
        trial = (cur + ' ' + word).strip()
        if tw(trial, fontsize, face) <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = word
    if cur:
        lines.append(cur)
    return lines


def box(ax, x, y, w, h, fc, ec, lw=0.9, r=0.055, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def arrow(ax, p0, p1, color='#8A8A8A', lw=1.5, ms=9, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms,
                                 color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=4))


def wrap(ax, x, y, text, width, fontsize=5.3, color=INK, weight='normal',
         ha='left', style='normal'):
    face = 'sansb' if weight == 'bold' else ('sansi' if style == 'italic' else 'sans')
    lines = split(text, width, fontsize, face)
    lh = lh_for(fontsize)
    for i, ln in enumerate(lines):
        ax.text(x, y - i * lh, ln, fontsize=fontsize, color=color, ha=ha,
                va='top', fontweight=weight, style=style, zorder=5)
    return y - len(lines) * lh


def files(ax, x, y, items, width, fontsize=5.0):
    lh = lh_for(fontsize) * 1.18
    for nm, note in items:
        ax.text(x, y, nm, fontsize=fontsize, color=INK, family='DejaVu Sans Mono',
                ha='left', va='top', zorder=5)
        if note and tw(nm, fontsize, 'mono') + tw(note, fontsize - 0.4) + 0.07 <= width:
            ax.text(x + width, y, note, fontsize=fontsize - 0.4, color=MUT,
                    ha='right', va='top', zorder=5)
        y -= lh
    return y


# ---------------------------------------------------------------- content
# Each sub-step carries the files IT writes, so an output is attached to the step that
# produces it rather than pooled in a list at the bottom.
STEPS = [
    dict(tag='01', cmd='01_interface_analysis.py  MODEL.pdb',
         flow=[dict(act='read the coordinate file', icon='file',
                    tool='Biopython\nPDBParser / MMCIFParser'),
               dict(act='drop waters; keep bound ligands, cofactors and ions; set '
                        'hydrogens aside; make residue numbers unique', icon='filter',
                    tool='Biopython', flag='lig',
                    out=['residue_renumbering.csv', 'ligands.csv']),
               dict(act='group chains into entities by sequence identity', icon='group',
                    tool='symmetry.py', flag='nuc'),
               dict(act='superpose the copies, keep the operators that map the whole '
                        'chain set onto itself, read off the point group', icon='sym',
                    tool='symmetry.py\nKabsch superposition', flag='nuc',
                    out=['run_config.json', 'chain_geometry.csv',
                         'chain_environments.csv']),
               dict(act='list contacting chain pairs and sort them into symmetry orbits',
                    icon='contact', tool='SciPy cKDTree', out=['orbits.csv']),
               dict(act='buried surface per atom and per residue, for every pair',
                    icon='sasa', tool='Biopython\nShrake-Rupley',
                    out=['interfaces_summary.csv', 'interface_residues.csv']),
               dict(act='assign salt bridges, hydrogen bonds, hydrophobic contacts, '
                        'disulfides and metal bridges', icon='bond',
                    tool='geometry, from the model', flag='nuc',
                    out=['interface_bonds.csv', 'interface_hydrophobic.csv',
                         'interface_links.csv']),
               dict(act='buried area, hydrogen bonds and salt bridges of every bound '
                        'ligand against each neighbouring chain', icon='lig',
                    tool='geometry, from the model', flag='lig',
                    out=['ligand_contacts.csv']),
               dict(act='shape complementarity and gap volume', icon='fit',
                    tool='interface_shape.py\nSciPy + scikit-image', flag='shape',
                    out=['interface_shape.csv'])],
         opts='--resolution  --names  --contact-cut  --sasa-points  --intra  '
              '--shape  --shape-spacing  --shape-workers  --no-ligands  '
              '--min-ligand-atoms'),
    dict(tag='02', cmd='02_derive_tables.py  MODEL.pdb',
         flow=[dict(act='read run_config.json and the step-01 tables', icon='table',
                    tool='pandas'),
               dict(act='secondary structure per residue', icon='helix', tool='pydssp',
                    flag='prot', out=['secondary_structure.csv']),
               dict(act='burial inside the WHOLE assembly, not only pairwise',
                    icon='burial', tool='Biopython\nShrake-Rupley',
                    out=['residue_burial_assembly.csv']),
               dict(act='footprints per symmetry-independent environment; one row per '
                        'symmetry-unique interface', icon='group', tool='pandas',
                    out=['footprints_by_environment.csv', 'subunit_budget.csv',
                         'table_interfaces_unique.csv', 'interface_bonds_unique.csv']),
               dict(act='chemistry, energetics, stacking geometry and packing per '
                        'category', icon='chart', tool='pandas', flag='nuc',
                    out=['interface_orbit_chemistry.csv', 'interface_energetics.csv',
                         'interface_composition_by_category.csv',
                         'interaction_density_by_category.csv', 'stacking_geometry.csv',
                         'aromatic_interactions.csv']),
               dict(act='tests with the symmetry copies as replicates, then write the '
                        'report', icon='stats', tool='SciPy stats',
                    out=['stats_summary.txt'])],
         opts='(no options — the settings come from run_config.json)'),
    dict(tag='03', cmd='03_make_figures.py  MODEL.pdb',
         flow=[dict(act='one footprint figure per chain type: burial, bonds, density, '
                        'ΔG, conservation', icon='chart',
                    tool='matplotlib + figstyle.py', flag='nuc',
                    out=['figN_<protein>_footprints']),
               dict(act='contact map per interface, and the unrolled contact topology',
                    icon='map', tool='matplotlib',
                    out=['fig_contact_maps', 'figS1_contact_maps_all',
                         'fig_architecture']),
               dict(act='area partition, chemistry, inventory and packing figures',
                    icon='stats', tool='matplotlib', flag='shape',
                    out=['fig_area_partition', 'fig_chemistry', 'fig_inventory',
                         'fig_packing']),
               dict(act='residue-level interaction networks, bond length to scale',
                    icon='net', tool='matplotlib',
                    out=['fig_interaction_networks_N']),
               dict(act='model with buried area in the B-factor column, plus two '
                        'ChimeraX command files', icon='cube', tool='plain text',
                    out=['model_interfaces.pdb', 'chimerax_colour_only.cxc',
                         'chimerax_colour_by_interaction.cxc'])],
         opts='--per-line  --colours  --bsa-min  --solvation-track  --no-interaction-track  '
              '--network-min  --phobic-min  --network-per-page  --sc-offset  '
              '--conservation  --msa'),
]


def icon(ax, kind, cx, cy, s=0.075, col='#5E7E9E'):
    """Small drawn pictogram for a step. Everything is a primitive, so it stays sharp
    at any size and carries no third-party trademark."""
    import numpy as _np
    L = dict(lw=0.8, color=col, solid_capstyle='round', zorder=5)
    P = dict(lw=0.8, edgecolor=col, facecolor='none', zorder=5)
    if kind == 'file':
        ax.add_patch(mpl.patches.Polygon([(cx - s * .6, cy - s), (cx - s * .6, cy + s),
                                          (cx + s * .25, cy + s), (cx + s * .6, cy + s * .55),
                                          (cx + s * .6, cy - s)], closed=True, **P))
        ax.plot([cx + s * .25, cx + s * .25, cx + s * .6], [cy + s, cy + s * .55,
                                                            cy + s * .55], **L)
    elif kind == 'filter':
        ax.add_patch(mpl.patches.Polygon([(cx - s, cy + s * .8), (cx + s, cy + s * .8),
                                          (cx + s * .18, cy - s * .1)], closed=True, **P))
        ax.plot([cx + s * .18, cx + s * .18], [cy - s * .1, cy - s], **L)
    elif kind == 'group':
        for k, w in enumerate((0.9, 0.55, 0.9)):
            ax.plot([cx - s * w, cx + s * w], [cy + s * (0.7 - 0.7 * k)] * 2, **L)
    elif kind == 'sym':
        ax.add_patch(mpl.patches.Circle((cx, cy), s * .9, **P))
        for a in (90, 210, 330):
            ax.add_patch(mpl.patches.Circle((cx + s * .5 * _np.cos(_np.radians(a)),
                                             cy + s * .5 * _np.sin(_np.radians(a))),
                                            s * .18, facecolor=col, edgecolor='none',
                                            zorder=5))
    elif kind == 'contact':
        ax.add_patch(mpl.patches.Circle((cx - s * .45, cy), s * .55, **P))
        ax.add_patch(mpl.patches.Circle((cx + s * .45, cy), s * .55, **P))
    elif kind == 'sasa':
        ax.add_patch(mpl.patches.Circle((cx - s * .15, cy), s * .8, **P))
        ax.add_patch(mpl.patches.Circle((cx + s * .75, cy + s * .45), s * .3,
                                        facecolor=col, edgecolor='none', zorder=5))
    elif kind == 'bond':
        ax.add_patch(mpl.patches.Circle((cx - s * .7, cy), s * .25, facecolor=col,
                                        edgecolor='none', zorder=5))
        ax.add_patch(mpl.patches.Circle((cx + s * .7, cy), s * .25, facecolor=col,
                                        edgecolor='none', zorder=5))
        ax.plot([cx - s * .4, cx + s * .4], [cy, cy], ls=(0, (1.2, 1.2)), **L)
    elif kind == 'fit':
        t = _np.linspace(-1, 1, 40)
        ax.plot(cx + s * t, cy + s * .22 + s * .35 * _np.cos(_np.pi * t), **L)
        ax.plot(cx + s * t, cy - s * .22 + s * .35 * _np.cos(_np.pi * t), **L)
    elif kind == 'helix':
        t = _np.linspace(0, 1, 50)
        ax.plot(cx - s + 2 * s * t, cy + s * .6 * _np.sin(2.6 * _np.pi * t), **L)
    elif kind == 'lig':
        import numpy as _np
        a = _np.deg2rad(_np.arange(6) * 60 + 30)
        ax.add_patch(mpl.patches.Polygon(
            _np.column_stack([cx + s * .62 * _np.cos(a) - s * .25,
                              cy + s * .62 * _np.sin(a)]), closed=True, **P))
        ax.plot([cx + s * .37, cx + s * .95], [cy + s * .32, cy + s * .62], **L)
        ax.add_patch(mpl.patches.Circle((cx + s * .95, cy + s * .62), s * .17,
                                        facecolor=col, edgecolor='none', zorder=5))
    elif kind == 'burial':
        ax.add_patch(mpl.patches.Circle((cx, cy), s * .9, **P))
        ax.add_patch(mpl.patches.Circle((cx, cy), s * .4, facecolor=col,
                                        edgecolor='none', zorder=5))
    elif kind == 'table':
        ax.add_patch(mpl.patches.Rectangle((cx - s * .9, cy - s * .75), s * 1.8,
                                           s * 1.5, **P))
        ax.plot([cx - s * .9, cx + s * .9], [cy + s * .2] * 2, **L)
        ax.plot([cx, cx], [cy - s * .75, cy + s * .75], **L)
    elif kind == 'stats':
        for k, h in enumerate((0.5, 1.0, 0.72)):
            ax.add_patch(mpl.patches.Rectangle((cx - s * .85 + k * s * .62, cy - s * .8),
                                               s * .42, s * 1.6 * h * .9,
                                               facecolor=col, edgecolor='none', zorder=5))
    elif kind == 'chart':
        ax.plot([cx - s * .85, cx - s * .85, cx + s * .9], [cy + s * .8, cy - s * .8,
                                                            cy - s * .8], **L)
        t = _np.linspace(0, 1, 40)
        ax.plot(cx - s * .7 + s * 1.5 * t, cy - s * .5 + s * 1.1 * t ** 0.5 *
                (0.6 + 0.4 * _np.sin(3 * t)), **L)
    elif kind == 'map':
        ax.add_patch(mpl.patches.Rectangle((cx - s * .9, cy - s * .9), s * 1.8, s * 1.8,
                                           **P))
        for a, b in ((0, 0), (1, 1), (2, 2), (1, 2)):
            ax.add_patch(mpl.patches.Rectangle((cx - s * .9 + a * s * .6,
                                                cy - s * .9 + b * s * .6), s * .6,
                                               s * .6, facecolor=col, edgecolor='none',
                                               alpha=0.75, zorder=5))
    elif kind == 'net':
        pts = [(cx - s * .8, cy + s * .7), (cx + s * .8, cy + s * .2),
               (cx - s * .3, cy - s * .8)]
        for p, q in ((0, 1), (0, 2), (1, 2)):
            ax.plot([pts[p][0], pts[q][0]], [pts[p][1], pts[q][1]], **L)
        for p in pts:
            ax.add_patch(mpl.patches.Circle(p, s * .25, facecolor=col,
                                            edgecolor='none', zorder=6))
    elif kind == 'cube':
        ax.add_patch(mpl.patches.Rectangle((cx - s * .9, cy - s * .9), s * 1.4,
                                           s * 1.4, **P))
        ax.add_patch(mpl.patches.Polygon([(cx - s * .9, cy + s * .5),
                                          (cx - s * .45, cy + s * .9),
                                          (cx + s * .9, cy + s * .9),
                                          (cx + s * .5, cy + s * .5)], closed=True, **P))
        ax.plot([cx + s * .5, cx + s * .5, cx + s * .9], [cy - s * .9, cy + s * .5,
                                                          cy + s * .9], **L)


# ---------------------------------------------------------------- measure
GAP_CHIP, LH_A, LH_T, LH_O = 0.062, lh_for(5.0), lh_for(4.5), lh_for(4.4) * 1.13
GUT = 0.24
CHIPX = 0.10 + GUT
CHIPW2 = CW - CHIPX - 0.10
TXTW2 = CHIPW2 - 0.16
for st in STEPS:
    st['chip'] = []
    for f in st['flow']:
        la = split(f['act'], TXTW2, 5.0)
        lt = [l for part in f['tool'].split('\n') for l in split(part, TXTW2, 4.5)]
        lo = f.get('out', [])
        h = (0.065 + len(la) * LH_A + 0.015 + len(lt) * LH_T
             + (0.030 + len(lo) * LH_O if lo else 0.0)
             + (0.105 if f.get('flag') else 0.0))
        st['chip'].append((f, la, lt, lo, h))
    st['opt'] = split('options:  ' + st['opts'], CHIPW2 - 0.16, 4.6)
    st['opt_h'] = 0.075 + len(st['opt']) * lh_for(4.6)
    st['run'] = (0.34 + sum(c[4] for c in st['chip'])
                 + GAP_CHIP * (len(st['flow']) - 1) + 0.10 + st['opt_h'])

BODY = max(s['run'] for s in STEPS) + 0.30          # room for the rail under the chain
TITLE, INROW, FOOT = 0.97, 0.60, 1.26
FH = TITLE + INROW + BODY + FOOT

fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')

y = FH - 0.22
ax.text(MARG, y, 'Interface analysis pipeline — one chain of steps, from model to figures',
        fontsize=9.2, color=INK, va='top', fontweight='bold')
wrap(ax, MARG, y - 0.20, 'The whole analysis is a single sequence; the three scripts are '
     'colour-coded stretches of it, not separate programs, and they hand over through one '
     'working directory. Each step names the library that does the work and the files it '
     'writes. There is no alternative route — the same steps always run, and the badged '
     'ones adapt to what the model contains.', FW - 2 * MARG, 5.6, MUT)
EXPL = {'nuc': 'behaves differently when DNA or RNA chains are present',
        'prot': 'skipped for nucleic-acid chains',
        'shape': 'runs only if you ask for it',
        'lig': 'only does something when the model has them'}
bx, by = MARG, y - 0.50
for _k in ('nuc', 'prot', 'shape', 'lig'):
    fc, tc, bt = BADGE[_k]
    bw = tw(bt, 4.2) + 0.10
    expl = EXPL[_k]
    need = bw + 0.06 + tw(expl, 4.6)
    if bx > MARG and bx + need > FW - MARG:      # wrap to a second row
        bx, by = MARG, by - 0.125
    box(ax, bx, by, bw, 0.072, fc, tc, lw=0.5, r=0.030, z=4)
    ax.text(bx + bw / 2, by + 0.036, bt, fontsize=4.2, color=tc, ha='center',
            va='center', zorder=5)
    ax.text(bx + bw + 0.06, by + 0.036, expl, fontsize=4.6, color=MUT, va='center')
    bx += need + 0.22

ytop = FH - TITLE - INROW
hin = 0.40
box(ax, MARG, ytop + 0.12, CW * 0.80, hin, C_IN, E_IN)
wrap(ax, MARG + 0.07, ytop + 0.12 + hin - 0.045, 'MODEL.pdb  /  MODEL.cif',
     CW * 0.80 - 0.14, 5.4, INK, 'bold')
wrap(ax, MARG + 0.07, ytop + 0.12 + hin - 0.165,
     'the refined model, in the working directory', CW * 0.80 - 0.14, 4.7, MUT)
arrow(ax, (MARG + CW * 0.40, ytop + 0.11), (MARG + CW * 0.40, ytop + 0.005),
      lw=1.8, ms=11, color=GROUP[0]['dark'])

rail = ytop - BODY + 0.14
ends = []
for k, st in enumerate(STEPS):
    g = GROUP[k]
    x0 = MARG + k * (CW + GAPC)
    # header pill: the script name, in the colour that stretch of the chain carries
    hh = 0.21
    box(ax, x0, ytop - hh, CW, hh, g['dark'], g['dark'], lw=0, r=0.045, z=3)
    ax.text(x0 + 0.10, ytop - hh / 2, st['tag'], fontsize=6.1, color='white',
            ha='left', va='center', fontweight='bold', zorder=5)
    ax.text(x0 + 0.30, ytop - hh / 2, st['cmd'], fontsize=5.6, color='white',
            ha='left', va='center', family='DejaVu Sans Mono', zorder=5)
    arrow(ax, (x0 + CHIPX + CHIPW2 / 2, ytop - hh - 0.005),
          (x0 + CHIPX + CHIPW2 / 2, ytop - hh - 0.085), lw=1.2, ms=7.5, color=g['edge'])

    yy = ytop - hh - 0.095
    for i, (f, la, lt, lo, ch) in enumerate(st['chip']):
        box(ax, x0 + CHIPX, yy - ch, CHIPW2, ch, g['chip'], g['edge'], lw=0.7,
            r=0.042, z=3)
        yt = yy - 0.040
        for ln in la:
            ax.text(x0 + CHIPX + 0.08, yt, ln, fontsize=5.0, color='#2F2F2F',
                    va='top', zorder=5)
            yt -= LH_A
        yt -= 0.015
        for jj, ln in enumerate(lt):
            ax.text(x0 + CHIPX + 0.08, yt, ln, fontsize=4.5,
                    color=g['dark'] if jj == 0 else MUT, va='top', zorder=5,
                    fontweight='bold' if jj == 0 else 'normal')
            yt -= LH_T
        if lo:
            yt -= 0.022
            ax.plot([x0 + CHIPX + 0.08, x0 + CHIPX + CHIPW2 - 0.08], [yt + 0.012] * 2,
                    lw=0.5, color='#E4D3BE', zorder=4)
            yt -= 0.004
            for nm in lo:
                ax.text(x0 + CHIPX + 0.10, yt, nm, fontsize=4.4, color='#B07A3A',
                        va='top', family='DejaVu Sans Mono', zorder=5)
                yt -= LH_O
        if f.get('flag'):
            fc, tc, bt = BADGE[f['flag']]
            bw = tw(bt, 4.2) + 0.10
            bx2 = x0 + CHIPX + CHIPW2 - 0.07 - bw
            box(ax, bx2, yy - ch + 0.032, bw, 0.072, fc, tc, lw=0.5, r=0.030, z=4)
            ax.text(bx2 + bw / 2, yy - ch + 0.068, bt, fontsize=4.2, color=tc,
                    ha='center', va='center', zorder=5)
        ax.text(x0 + 0.115, yy - 0.062, str(i + 1), fontsize=4.6, color=g['edge'],
                ha='center', va='center', fontweight='bold', zorder=5)
        icon(ax, f['icon'], x0 + 0.115, yy - 0.190, col=g['icon'])
        if i < len(st['chip']) - 1:
            xa = x0 + CHIPX + CHIPW2 / 2
            arrow(ax, (xa, yy - ch - 0.008), (xa, yy - ch - GAP_CHIP + 0.008),
                  lw=1.1, ms=6.4, color=g['edge'])
        yy -= ch + GAP_CHIP
    yy += GAP_CHIP - 0.055
    box(ax, x0 + CHIPX, yy - st['opt_h'], CHIPW2, st['opt_h'], '#F6F6F4', '#E2E2E0',
        lw=0.6, r=0.038)
    ytx = yy - 0.040
    for ln in st['opt']:
        ax.text(x0 + CHIPX + 0.07, ytx, ln, fontsize=4.6, color='#4A4A4A', va='top',
                zorder=5)
        ytx -= lh_for(4.6)
    ends.append((x0, yy - st['opt_h']))

# ---- the chain continues from one stretch to the next, around the outside
for k in range(len(STEPS) - 1):
    g = GROUP[k + 1]
    x0, yend = ends[k]
    xm = x0 + CHIPX + CHIPW2 / 2
    xn = MARG + (k + 1) * (CW + GAPC)
    xg = xn - GAPC / 2
    ax.plot([xm, xm], [yend - 0.025, rail], lw=1.8, color=g['edge'],
            solid_capstyle='round', zorder=3)
    ax.plot([xm, xg], [rail, rail], lw=1.8, color=g['edge'],
            solid_capstyle='round', zorder=3)
    ax.plot([xg, xg], [rail, ytop - 0.105], lw=1.8, color=g['edge'],
            solid_capstyle='round', zorder=3)
    arrow(ax, (xg, ytop - 0.105), (xn - 0.015, ytop - 0.105), lw=1.8, ms=11,
          color=g['edge'])
    ax.text((xm + xg) / 2, rail + 0.045, 'all of it is read next', fontsize=4.7,
            color=MUT, ha='center', va='bottom')

# ---------------------------------------------------------------- footer row
fy = ytop - BODY - 0.10
box(ax, MARG, fy - 0.40, CW * 2 + GAPC, 0.40, C_OPT, '#D8D8D4', lw=0.7)
wrap(ax, MARG + 0.09, fy - 0.045, 'optional input to step 03:  a ConSurf .grades file or '
     'an aligned FASTA. You supply it; nothing is ever fetched from the network.',
     CW * 2 + GAPC - 0.18, 5.0, '#4A4A4A')
x3 = MARG + 2 * (CW + GAPC)
box(ax, x3, fy - 0.40, CW, 0.40, '#FAFAF8', '#C9C9C5', lw=0.8)
wrap(ax, x3 + 0.09, fy - 0.045, 'optional:  04_pisa_concordance.py PDBID — cross-check '
     'against PISA. The only step that touches the network.', CW - 0.18, 5.0, MUT)

wrap(ax, MARG, fy - 0.52, 'Every file named above is written into the one working '
     'directory, which is how the stretches hand over.  What changes with the input: '
     'point group, chain count and entity count are measured, never declared, so a C1 '
     'dimer and an icosahedral capsid take the same path. A protein–DNA complex or a '
     'ribosome runs the badged steps differently — C1′ or P instead of CA as the '
     'superposition anchor, nucleobases added to the stacking search, phosphate oxygens '
     'counted as anionic, no secondary structure for those chains — but no step is added '
     'or removed. Waters are the only thing discarded: metal ions are kept and reported '
     'when they bridge two chains, and bound ligands and cofactors are kept and treated '
     'like any other residue, so a cofactor sitting in a shared active site is found '
     'without being asked for.',
     FW - 2 * MARG, 5.2, MUT)

for ext in ('png', 'pdf'):
    fig.savefig(f'fig_pipeline_flow.{ext}', dpi=400 if ext == 'png' else None,
                facecolor='white')
print(f'wrote fig_pipeline_flow.png / .pdf   ({FW} x {FH:.2f} in)')
