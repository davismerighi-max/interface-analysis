#!/usr/bin/env python
"""
05_assemblies.py — which parts of this model could be stable assemblies on their own?

Run after 01 and 02, in the same directory:

    python 05_assemblies.py MODEL.cif [--render] [--min-kept 0.50] [--strong-kept 0.67]

WHAT IT DOES
The chains and their interfaces form a graph. Remove the weakest interfaces first and
the graph falls apart in a definite order; every piece that appears on the way down is
a candidate assembly. This is the idea PISA uses to enumerate assemblies (Krissinel &
Henrick, J Mol Biol 2007; Krissinel, Acta Cryst D 2011). Symmetry-equivalent interfaces
are the same interface, so a whole orbit is always cut at once.

Each candidate is classified from the share of its buried area that stays INSIDE it
("kept"), using a stated heuristic — PISA itself ranks by ΔG_diss, not by a contact
share, and there is no published cut-off for this:
    kept ≥ 67 %   strong candidate   breaks less than half of what it keeps
    kept ≥ 50 %   candidate          keeps more than it breaks
    kept < 50 %   not supported by contacts alone
    any kept      covalent unit      every chain is joined by disulfides/isopeptides,
                                     so it cannot fall apart under non-reducing conditions

Disulfides are NOT weighed as area — no amount of surface is equivalent to a covalent
bond. They are tracked separately: interface orbits carrying a bonded one are never cut
in the first (oxidising) pass; a second (reducing) pass cuts everything, and a piece
whose boundary crosses a bonded disulfide is marked as separable only under reducing
conditions.

WHAT IT DOES NOT DO
It does not compute PISA's ΔG_diss (that needs an entropy term this pipeline does not
model), and for a cryo-EM particle there is no crystal lattice: the question is not "is
this interface real" but "could this piece exist on its own". The table is evidence to
weigh and to test (SEC-MALS, mass photometry, non-reducing SDS-PAGE, 2D classes).

OUTPUTS
assembly_candidates.csv   one row per kind of candidate, with the verdict and the reason
assemblies_summary.txt    the same, as a readable report
assembly_interfaces.csv   the interfaces, strongest first
assembly_NN_*.pdb / .cif  coordinates of every candidate (--no-pdb to skip)
fig_assemblies.png/.pdf   licorice renders of the main candidates (--render)
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('model')
ap.add_argument('--min-area', type=float, default=0.0,
                help='ignore interfaces smaller than this (Å²) when building the graph')
ap.add_argument('--min-kept', type=float, default=0.50,
                help='kept share for "candidate" (default 0.50)')
ap.add_argument('--strong-kept', type=float, default=0.67,
                help='kept share for "strong candidate" (default 0.67)')
ap.add_argument('--no-pdb', action='store_true', help='do not write per-assembly models')
ap.add_argument('--render', action='store_true', help='draw fig_assemblies.png')
ap.add_argument('--colours', default=None, metavar='SPEC',
                help='protein colours, e.g. "tube=#7FA8D4,belt=#E8884A" (default: the '
                     'colours from the --proteins FASTA of step 01, else the palette)')
args = ap.parse_args()
RPT = runlog.Report('05', model=args.model)

with RPT.guard('read step 01-02 outputs', ok_detail=None, fatal=True):
    for f in ('run_config.json', 'interfaces_summary.csv', 'chain_geometry.csv'):
        if not os.path.exists(f):
            raise FileNotFoundError(f'{f} is missing: run steps 01 and 02 first, here')
    CFG = json.load(open('run_config.json'))
    S = pd.read_csv('interfaces_summary.csv')
    GEO = pd.read_csv('chain_geometry.csv').set_index('chain')
    if 'orbit' not in S.columns:
        S['orbit'] = range(len(S))
    if 'ctg' not in S.columns:
        S['ctg'] = S.get('e1', 'chain') + '–' + S.get('e2', 'chain')
if args.min_area > 0:
    S = S[S.area >= args.min_area]
ENT = GEO.entity_name.to_dict()
chains = list(GEO.index)
RPT.ok('read step 01-02 outputs', f'{len(chains)} chains, {len(S)} interfaces')
if len(chains) < 2 or not len(S):
    RPT.skip('assemblies', 'fewer than two contacting chains: nothing to decompose')
    RPT.done(); sys.exit(0)

# ------------------------------------------------------------------ core residues
core = {}
with RPT.guard('core residues', ok_detail=None):
    if os.path.exists('residue_burial_assembly.csv') and os.path.exists('interface_residues.csv'):
        BUR = pd.read_csv('residue_burial_assembly.csv')
        IR = pd.read_csv('interface_residues.csv')
        iso = BUR.set_index(['chain', 'resi']).sasa_iso.to_dict()
        IR['iso'] = [iso.get((c, r), np.nan) for c, r in zip(IR.chain, IR.resi)]
        IR['frac'] = IR.dsasa / IR.iso.replace(0, np.nan)
        core = IR[IR.frac > 0.95].groupby(['c1', 'c2']).size().to_dict()
        RPT.ok('core residues', 'counted (buried > 95 % of isolated surface)')
    else:
        RPT.skip('core residues', 'step 02 tables not found; core-residue counts left out')

# ------------------------------------------------------------------ covalent links
COV, COVRES = {}, {}
with RPT.guard('disulfides and cross-links', ok_detail=None):
    if os.path.exists('interface_links.csv'):
        LK = pd.read_csv('interface_links.csv')
        LK = LK[LK.kind.isin(['disulfide', 'isopeptide']) & LK.bonded.astype(bool)]
        for _, l in LK.iterrows():
            k = tuple(sorted((l.c1, l.c2)))
            COV[k] = COV.get(k, 0) + 1
            COVRES.setdefault(k, []).append(
                f'{l.c1}:{l.resn1}{int(l.resi1)}–{l.c2}:{l.resn2}{int(l.resi2)}')
    RPT.ok('disulfides and cross-links',
         f'{sum(COV.values())} bonded inter-chain link(s)' if COV else 'none between chains')

# ------------------------------------------------------------------ the graph
EDGE = {}
for _, r in S.iterrows():
    k = tuple(sorted((r.c1, r.c2)))
    EDGE[(r.c1, r.c2)] = dict(area=float(r.area), dg=float(r.get('dg_solv', 0) or 0),
                              nhb=int(r.get('n_hbond', 0) or 0),
                              nsb=int(r.get('n_saltbridge', 0) or 0), ctg=r.ctg,
                              orbit=int(r.orbit), core=core.get((r.c1, r.c2), 0),
                              cov=COV.get(k, 0), covres=COVRES.get(k, []))
ADJ = {c: [] for c in chains}
for (a, b), e in EDGE.items():
    if a in ADJ and b in ADJ:
        ADJ[a].append((b, e)); ADJ[b].append((a, e))


def components(keep):
    seen, out = set(), []
    for c in chains:
        if c in seen:
            continue
        stack, comp = [c], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            for y, e in ADJ[x]:
                if id(e) in keep and y not in comp:
                    stack.append(y)
        seen |= comp
        out.append(frozenset(comp))
    return out


def covalently_whole(comp):
    """True if every chain of the piece is reachable through covalent links alone."""
    comp = set(comp)
    start = next(iter(comp)); seen, stack = set(), [start]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        for y, e in ADJ[x]:
            if y in comp and e['cov'] > 0 and y not in seen:
                stack.append(y)
    return seen == comp


def summarise(comp):
    inn = dict(area=0.0, dg=0.0, nhb=0, nsb=0, n=0, cov=0, res=[])
    out = dict(area=0.0, dg=0.0, nhb=0, nsb=0, n=0, cov=0, res=[], core=[])
    for (a, b), e in EDGE.items():
        ia, ib = a in comp, b in comp
        if ia and ib:
            d = inn
        elif ia or ib:
            d = out; d['core'].append(e['core'])
        else:
            continue
        d['area'] += e['area']; d['dg'] += e['dg']; d['nhb'] += e['nhb']
        d['nsb'] += e['nsb']; d['n'] += 1; d['cov'] += e['cov']; d['res'] += e['covres']
    return inn, out


def record(comp, found, source):
    if len(comp) < 2 or comp in found:
        return
    inn, out = summarise(comp)
    ce = pd.Series([ENT[c] for c in comp]).value_counts().to_dict()
    found[comp] = dict(
        chains=''.join(sorted(comp)), n_chains=len(comp),
        composition=' + '.join(f'{v}×{k}' for k, v in sorted(ce.items())),
        source=source, n_internal=inn['n'], internal_area=inn['area'],
        internal_dg=inn['dg'], area_per_chain=inn['area'] / len(comp),
        n_boundary=out['n'], boundary_area=out['area'], boundary_dg=out['dg'],
        boundary_hbond=out['nhb'], boundary_salt=out['nsb'],
        internal_covalent=inn['cov'], boundary_covalent=out['cov'],
        internal_links='; '.join(inn['res']), boundary_links='; '.join(out['res']),
        covalent_unit=bool(inn['cov']) and covalently_whole(comp),
        kept=inn['area'] / max(inn['area'] + out['area'], 1e-9))


ORB = {}
for e in EDGE.values():
    ORB.setdefault(e['orbit'], []).append(e['area'])
ORB = {k: float(np.median(v)) for k, v in ORB.items()}
COVORB = {e['orbit'] for e in EDGE.values() if e['cov'] > 0}
found = {}
with RPT.guard('dissociation scan', ok_detail=None, fatal=True):
    for reducing in (False, True):
        cuttable = [k for k in ORB if reducing or k not in COVORB]
        order = sorted(cuttable, key=lambda k: ORB[k])
        for i in range(len(order) + 1):
            cut = set(order[:i])
            keep = {id(e) for e in EDGE.values() if e['orbit'] not in cut}
            for comp in components(keep):
                record(comp, found, 'reducing' if reducing else 'oxidising')
    # Covalent units: chains joined only through bonded disulfides. When a disulfide is
    # present in some symmetry copies and not others, these are smaller than any orbit
    # cut can produce, and they are exactly what survives on a non-reducing gel.
    keep = {id(e) for e in EDGE.values() if e['cov'] > 0}
    for comp in components(keep):
        record(comp, found, 'covalent')
    RPT.ok('dissociation scan', f'{len(ORB)} interface orbits, {len(found)} pieces found'
         + (f'; {len(COVORB)} orbit(s) carry a disulfide' if COVORB else ''))

A = pd.DataFrame(found.values())
A['_key'] = (A.n_chains.astype(str) + '|' + A.composition + '|'
             + A.n_internal.astype(str) + '|' + A.n_boundary.astype(str) + '|'
             + A.internal_covalent.astype(str))
A['copies'] = A.groupby('_key').chains.transform('size')
A = (A.sort_values(['n_chains', 'internal_area'], ascending=False)
       .drop_duplicates('_key').drop(columns='_key').reset_index(drop=True))


# ------------------------------------------------------------------ verdicts
def verdict(r):
    k = 100 * r.kept
    if r.n_boundary == 0:
        cls, why = 'whole model', 'the complete model: nothing has to break'
    elif k >= 100 * args.strong_kept:
        cls = 'strong candidate'
        why = (f'strong candidate: keeps {k:.0f} % (≥ {100*args.strong_kept:.0f} %), '
               'breaks less than half of what it keeps')
    elif k >= 100 * args.min_kept:
        cls = 'candidate'
        why = (f'candidate: keeps {k:.0f} % (≥ {100*args.min_kept:.0f} %), keeps more '
               'than it breaks')
    elif r.covalent_unit:
        cls = 'covalent unit'
        why = (f'covalent unit: keeps only {k:.0f} % by area (< {100*args.min_kept:.0f} %), '
               f'but all {r.n_chains} chains are joined by {r.internal_covalent:.0f} '
               f'disulfide(s) ({r.internal_links}); under non-reducing conditions it '
               'cannot fall apart')
    else:
        cls = 'not supported by contacts'
        why = (f'not supported by contacts: keeps {k:.0f} % (< {100*args.min_kept:.0f} %), '
               'would break more than it keeps')
    if r.internal_covalent and cls not in ('covalent unit', 'whole model'):
        why += (f'; held internally by {r.internal_covalent:.0f} disulfide(s) '
                f'({r.internal_links})')
    if r.boundary_covalent:
        why += (f'; BUT {r.boundary_covalent:.0f} disulfide(s) tie it to the rest '
                f'({r.boundary_links}): separable only under reducing conditions')
    if r.n_boundary and r.boundary_dg > 0:
        why += (f'; its boundary costs solvation energy (+{r.boundary_dg:.1f} kcal/mol): '
                'a polar seam, typically sensitive to pH and salt')
    return pd.Series(dict(**{'class': cls}, reason=why))


A = pd.concat([A, A.apply(verdict, axis=1)], axis=1)
RANK = {'whole model': 0, 'strong candidate': 1, 'candidate': 2, 'covalent unit': 3,
        'not supported by contacts': 4}
A = (A.assign(_r=A['class'].map(RANK))
       .sort_values(['_r', 'kept', 'n_chains'], ascending=[True, False, False])
       .drop(columns='_r').reset_index(drop=True))
A.insert(0, 'rank', range(len(A)))

# ------------------------------------------------------------------ coordinates
A['file'] = ''
if args.no_pdb:
    RPT.skip('assembly coordinates', '--no-pdb')
else:
    with RPT.guard('assembly coordinates', ok_detail=None):
        from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO, Select
        par = MMCIFParser(QUIET=True) if args.model.lower().endswith(('.cif', '.mmcif')) \
            else PDBParser(QUIET=True)
        st = par.get_structure('m', args.model)
        # PDB format holds 1-character chain IDs and < 100,000 atoms; beyond that, mmCIF
        big = (any(len(c.id) > 1 for c in st[0])
               or sum(1 for _ in st[0].get_atoms()) > 99999)
        io = MMCIFIO() if big else PDBIO()
        io.set_structure(st)
        ext = 'cif' if big else 'pdb'

        class Keep(Select):
            def __init__(self, ch): self.ch = set(ch)
            def accept_chain(self, c): return c.id in self.ch

        for i, r in A.iterrows():
            name = ('assembly_00_whole.' + ext if r['class'] == 'whole model' else
                    f'assembly_{int(r["rank"]):02d}_{r.n_chains}chains.{ext}')
            ids = [c.id for c in st[0] if c.id in set(r.chains)] if big else list(r.chains)
            try:
                io.save(name, Keep(ids)); A.at[i, 'file'] = name
            except Exception as exc:
                RPT.fail(f'coordinates {name}', exc, sys.exc_info()[2])
        RPT.ok('assembly coordinates', f'{(A.file != "").sum()} {ext.upper()} file(s)')

# ------------------------------------------------------------------ tables
COLS = ['rank', 'class', 'kept', 'n_chains', 'composition', 'copies', 'chains',
        'internal_area', 'boundary_area', 'internal_covalent', 'boundary_covalent',
        'internal_links', 'boundary_links', 'internal_dg', 'boundary_dg',
        'boundary_hbond', 'boundary_salt', 'source', 'file', 'reason']
OUT = A[COLS].copy()
OUT['kept'] = (100 * OUT.kept).round(1)
OUT = OUT.rename(columns={'kept': 'kept_pct', 'internal_area': 'internal_area_A2',
                          'boundary_area': 'boundary_area_A2',
                          'internal_covalent': 'disulfides_inside',
                          'boundary_covalent': 'disulfides_across_boundary',
                          'internal_dg': 'internal_dG_solv', 'boundary_dg': 'boundary_dG_solv'})
OUT.to_csv('assembly_candidates.csv', index=False)
A.to_csv('assemblies.csv', index=False)

IF = pd.DataFrame([dict(c1=a, c2=b, **{k: v for k, v in e.items() if k != 'covres'})
                   for (a, b), e in EDGE.items()])
IF = (IF.groupby('ctg').agg(n=('area', 'size'), area=('area', 'median'),
                            dg=('dg', 'median'), hbond=('nhb', 'median'),
                            salt=('nsb', 'median'), core=('core', 'median'),
                            core_max=('core', 'max'), disulfides=('cov', 'sum'))
        .reset_index().sort_values('area', ascending=False))
IF.to_csv('assembly_interfaces.csv', index=False)

# ------------------------------------------------------------------ report
import textwrap
L = [f'model {CFG.get("model")} | point group {CFG.get("point_group")} | '
     f'{len(chains)} chains | {len(EDGE)} contacting pairs', '',
     'CANDIDATE ASSEMBLIES', '']
hdr = (f'{"#":>3} {"class":<26} {"kept":>5} {"chains":>6}  {"composition":<30} '
       f'{"copies":>6} {"internal Å²":>11} {"boundary Å²":>11} {"S–S in":>6} {"S–S out":>7}')
L += [hdr, '-' * len(hdr)]
for _, r in A.iterrows():
    L.append(f'{r["rank"]:>3} {r["class"]:<26} {100*r.kept:>4.0f}% {r.n_chains:>6}  '
             f'{r.composition[:30]:<30} {r.copies:>6} {r.internal_area:>11,.0f} '
             f'{r.boundary_area:>11,.0f} {r.internal_covalent:>6.0f} {r.boundary_covalent:>7.0f}')
    for w in textwrap.wrap(r.reason, 96):
        L.append(f'{"":>8}{w}')
    if r.file:
        L.append(f'{"":>8}coordinates: {r.file}')
L += ['', 'RULE USED',
      f'  kept ≥ {100*args.strong_kept:.0f} %   strong candidate   breaks less than half of what it keeps',
      f'  kept ≥ {100*args.min_kept:.0f} %   candidate          keeps more than it breaks',
      f'  kept < {100*args.min_kept:.0f} %   not supported by contacts alone',
      '  covalent unit     every chain joined by disulfides: survives non-reducing conditions',
      'kept = buried area inside the piece ÷ (inside + across its boundary). The cut-offs',
      'are a stated heuristic, not a published standard (PISA ranks by ΔG_diss). Change',
      'them with --min-kept and --strong-kept.', '',
      'INTERFACES, strongest first',
      f'{"category":<28} {"n":>4} {"area Å²":>9} {"ΔGsolv":>8} {"H-b":>5} {"salt":>5} '
      f'{"core res":>12} {"S–S":>4}']
for _, r in IF.iterrows():
    L.append(f'{r.ctg:<28} {r.n:>4} {r.area:>9,.0f} {r.dg:>8.2f} {r.hbond:>5.0f} '
             f'{r.salt:>5.0f} {r.core:>4.0f} (max {r.core_max:.0f}) {r.disulfides:>4.0f}')
L += ['', 'core residues = buried > 95 % of their isolated accessible surface; Duarte et al.',
      '(BMC Bioinformatics 2012) find biological interfaces typically have ≥ 6.',
      'A positive ΔGsolv means burying that surface COSTS solvation free energy: the interface',
      'is held by hydrogen bonds and salt bridges rather than the hydrophobic effect.', '',
      'CAVEAT. None of this is PISA\'s ΔG_diss. Treat the table as evidence to weigh and to test',
      '(SEC-MALS, mass photometry, non-reducing SDS-PAGE, 2D classes), not as a verdict.']
open('assemblies_summary.txt', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
n_c = {c: int((A['class'] == c).sum()) for c in RANK}
RPT.ok('candidate table', 'assembly_candidates.csv: ' + ', '.join(f'{v} {k}' for k, v in n_c.items() if v))

# ------------------------------------------------------------------ figure
if not args.render:
    RPT.skip('fig_assemblies', 'not requested (--render)')
else:
    with RPT.guard('fig_assemblies', ok_detail='written'):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from render_assembly import load_ca, licorice
        CA = load_ca(args.model)
        PAL = ['#7FA8D4', '#E8884A', '#95C892', '#B48BC4', '#D4B36A', '#7FC4C4']
        ents = [e['name'] for e in CFG.get('entities', [])] or sorted(set(ENT.values()))
        cmap = {e: PAL[i % len(PAL)] for i, e in enumerate(ents)}
        cmap.update(CFG.get('colours') or {})
        if args.colours:
            for tok in args.colours.split(','):
                if '=' in tok:
                    k, v = tok.split('=', 1); cmap[k.strip()] = v.strip()
        colmap = {c: cmap.get(ENT[c], '#9A9A9A') for c in chains}
        show = A[A['class'] != 'not supported by contacts'].head(4)
        n = max(len(show), 1)
        fig, axs = plt.subplots(1, n, figsize=(1.85 * n, 2.3), squeeze=False)
        for ax, (_, r) in zip(axs[0], show.iterrows()):
            sub = {c: CA[c] for c in r.chains if c in CA}
            licorice(ax, sub, colmap, elev=62, azim=20, dpi=600)
            ax.set_title(f'{r.n_chains} chains · {r.composition}\n'
                         f'kept {100*r.kept:.0f} %  ·  {r["class"]}',
                         fontsize=5.6, color='#333333', pad=3)
        fig.subplots_adjust(0.005, 0.01, 0.995, 0.86, wspace=0.03)
        fig.savefig('fig_assemblies.png', dpi=600); fig.savefig('fig_assemblies.pdf')
        plt.close(fig)
RPT.done()
