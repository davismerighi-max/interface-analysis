"""Step 04 (optional) — cross-check our interfaces against PDBePISA.

    python 04_pisa_concordance.py 9O7M [--assembly 1]

PISA is the reference implementation for interface area, and it is open: PDBe serve PISA
2.0 results for every deposited entry over a public REST API, so for a DEPOSITED structure
the comparison can be made automatically. An unpublished model cannot be checked this way
— PISA has no upload API, only the interactive server — so run this on the deposited entry
that most resembles your model, or after deposition.

Why bother: PISA's own output is hard to read for a large assembly, because it lists every
interface in the whole thing with no notion of which ones are symmetry copies of each
other. This script does that matching for you and reports the agreement as one table and
one figure, so what you get is "our areas agree with PISA to within X %" rather than
several hundred rows to compare by hand.

Writes pisa_concordance.csv and fig_pisa_concordance.png/.pdf.
"""
import sys, os, json, argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ap = argparse.ArgumentParser()
ap.add_argument('pdb_id', help='4-character PDB code of the deposited entry')
ap.add_argument('--assembly', default='1')
ap.add_argument('--pisa-json', default=None, metavar='FILE',
                help='read a previously saved PISA response instead of fetching it. Use this '
                     'on a cluster compute node or any machine without outbound network, and '
                     'to make the comparison reproducible once the paper is written.')
ap.add_argument('--save-json', default=None, metavar='FILE',
                help='write the fetched PISA response here, for later offline reuse')
args = ap.parse_args()
PID = args.pdb_id.lower()

mpl.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 600, 'font.size': 8,
                     'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'sans-serif', 'pdf.fonttype': 42})

if args.pisa_json:
    print(f'reading PISA results from {args.pisa_json} (no network used)', flush=True)
    payload = json.load(open(args.pisa_json))
else:
    url = f'https://www.ebi.ac.uk/pdbe/api/pisa/interfaces/{PID}/{args.assembly}'
    print(f'fetching {url}', flush=True)
    r = requests.get(url, timeout=300)
    if r.status_code != 200:
        sys.exit(f'PDBePISA returned {r.status_code} for {PID} assembly {args.assembly}. '
                 'PISA results exist only for deposited entries; an unpublished model has to '
                 'go through the interactive server at '
                 'https://www.ebi.ac.uk/msd-srv/prot_int/pistart.html')
    payload = r.json()
    if args.save_json:
        json.dump(payload, open(args.save_json, 'w'))
        print(f'saved the raw PISA response to {args.save_json}', flush=True)
top = payload[PID]
asm = top['assembly']
print(f"PISA {top['pisa_version']}: {asm['interface_count']} interfaces, "
      f"buried {float(asm['buried_surface_area']):,.0f} A2, "
      f"solvation energy gain {float(asm['solvation_energy_gain']):,.1f} kcal/mol", flush=True)

rows = []
for it in asm['interfaces']:
    ch = [m['chain_id'] for m in it['molecules']]
    if len(ch) != 2:
        continue
    rows.append(dict(pisa_id=it['interface_id'], pc1=ch[0], pc2=ch[1],
                     pisa_area=float(it['interface_area']),
                     pisa_dg=float(it['solvation_energy']),
                     pisa_pvalue=float(it['p_value']),
                     pisa_nres=int(it['number_interface_residues']),
                     pisa_hb=int(it['number_hydrogen_bonds']),
                     pisa_sb=int(it['number_salt_bridges'])))
PI = pd.DataFrame(rows)
PI['key'] = [tuple(sorted(x)) for x in zip(PI.pc1, PI.pc2)]
PI = PI.sort_values('pisa_area', ascending=False).drop_duplicates('key')

S = pd.read_csv('interfaces_summary.csv')
S['key'] = [tuple(sorted(x)) for x in zip(S.c1, S.c2)]
M = S.merge(PI, on='key', how='outer', indicator=True)
both = M[M._merge == 'both'].copy()
both['area_diff_pct'] = 100 * (both.area - both.pisa_area) / both.pisa_area

print(f'\nours: {len(S)} contacting pairs   PISA: {len(PI)} chain pairs   matched: {len(both)}')
only_us = M[M._merge == 'left_only']
only_pi = M[M._merge == 'right_only']
if len(only_us):
    print(f'  {len(only_us)} pair(s) only in ours (PISA drops interfaces below its own '
          'area cut-off): ' + ', '.join(f'{a}-{b}' for a, b in only_us.key.head(8)))
if len(only_pi):
    print(f'  {len(only_pi)} pair(s) only in PISA: '
          + ', '.join(f'{a}-{b}' for a, b in only_pi.key.head(8)))

rp = np.nan
if len(both) > 1:
    rp = np.corrcoef(both.area, both.pisa_area)[0, 1]
    med = both.area_diff_pct.median()
    mad = (both.area_diff_pct - med).abs().median()
    print(f'\narea concordance: r = {rp:.4f}, median difference {med:+.1f} % '
          f'(median absolute deviation {mad:.1f} %), n = {len(both)}')
    print('largest disagreements:')
    idx = both.area_diff_pct.abs().sort_values(ascending=False).index
    for _, x in both.reindex(idx).head(5).iterrows():
        print(f'   {x.key[0]}-{x.key[1]}: ours {x.area:8.1f}  PISA {x.pisa_area:8.1f} A2  '
              f'({x.area_diff_pct:+.1f} %)')
    if 'dg_solv' in both.columns and both.dg_solv.notna().all():
        rg = np.corrcoef(both.dg_solv, both.pisa_dg)[0, 1]
        print(f'\nsolvation energy: r = {rg:.3f} against PISA. Ours uses Eisenberg-McLachlan '
              'parameters and PISA its own, so agreement in rank is the meaningful part, '
              'not agreement in value.')
    print('\nHydrogen bonds and salt bridges are NOT expected to agree closely: PISA applies '
          'its own geometric criteria and, for a model without hydrogens, both tools are '
          'guessing at donor geometry.')

both.drop(columns=['_merge']).to_csv('pisa_concordance.csv', index=False)

fig = plt.figure(figsize=(7.2, 3.1))
gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.88, bottom=0.17, wspace=0.30)
ax = fig.add_subplot(gs[0, 0])
lim = [0, max(both.area.max(), both.pisa_area.max()) * 1.08]
ax.plot(lim, lim, lw=0.8, color='#BBBBBB', zorder=1)
cats = list(both.ctg.dropna().unique()) if 'ctg' in both.columns else []
if len(cats) > 1:
    cols = {c: fs.FAMILIES[i % len(fs.FAMILIES)]['pal'] for i, c in enumerate(cats)}
    for c in cats:
        sub = both[both.ctg == c]
        ax.scatter(sub.pisa_area, sub.area, s=16, color=cols[c], edgecolor='white',
                   linewidth=0.4, zorder=3, label=c)
    ax.legend(frameon=False, fontsize=6, loc='upper left', handletextpad=0.3)
else:
    ax.scatter(both.pisa_area, both.area, s=16, color='#6FA3D2', edgecolor='white',
               linewidth=0.4, zorder=3)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('PISA interface area (Å$^2$)'); ax.set_ylabel('this pipeline (Å$^2$)')
ax.set_title(f'{PID.upper()} — buried area, r = {rp:.4f}', loc='left', fontsize=8, pad=3)
ax.spines[['left', 'bottom']].set_color('#AAAAAA'); ax.tick_params(colors='#666666')

ax2 = fig.add_subplot(gs[0, 1])
ax2.axhline(0, lw=0.8, color='#BBBBBB', zorder=1)
ax2.scatter(both.pisa_area, both.area_diff_pct, s=16, color='#6FA3D2',
            edgecolor='white', linewidth=0.4, zorder=3)
ax2.set_xlabel('PISA interface area (Å$^2$)')
ax2.set_ylabel('difference from PISA (%)')
ax2.set_title(f'median {both.area_diff_pct.median():+.1f} %', loc='left', fontsize=8, pad=3)
ax2.spines[['left', 'bottom']].set_color('#AAAAAA'); ax2.tick_params(colors='#666666')
fig.savefig('fig_pisa_concordance.png', dpi=600); fig.savefig('fig_pisa_concordance.pdf')
print('\nwrote pisa_concordance.csv and fig_pisa_concordance.png/.pdf')
