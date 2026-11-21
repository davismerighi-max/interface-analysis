"""Step 03 — figures, for any assembly and any point group.

    python 03_make_figures.py MODEL.{pdb,cif}

Generalised from the tail1-specific original. Entity names, colours, the number of
footprint figures and the number of budget rows all follow the model.

Two adaptations matter scientifically:

* In a HOMO-OLIGOMER every partner is the same protein, so "buried by entity X" says
  nothing. The footprint bars then colour by which distinct INTERFACE the surface
  belongs to instead, which is the informative question for a single-protein shell.
* When the resolution implies low bond confidence, the contact maps still draw the
  geometry but the bond overlays are labelled tentative in the legend, so a reader
  cannot mistake them for assignments.

Figures: fig1..figN footprints (one per entity), fig_architecture, fig_inventory,
fig_chemistry, fig_contact_maps, fig_area_partition, figS1_contact_maps_all.
"""
import sys, os, json, argparse, re, collections
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ap = argparse.ArgumentParser(description='Step 03 — figures and a ChimeraX colouring script.')
ap.add_argument('model', nargs='?', default='model.pdb')
ap.add_argument('--per-line', type=int, default=350, metavar='N',
                help='residues per line in the footprint figures; long chains are wrapped '
                     'onto several rows so the bars stay readable (default 350)')
ap.add_argument('--colours', default=None, metavar='SPEC',
                help='pin protein colours, e.g. "tube=#8CB4DC,belt=#F79E72,fibre=#95C892". '
                     'Unnamed proteins keep the default palette.')
ap.add_argument('--conservation', default=None, metavar='FILE',
                help='optional per-residue conservation. Accepts a ConSurf .grades file or a '
                     'CSV/TSV with columns resi,score and optionally protein and chain. '
                     'Adds a conservation track under each footprint panel.')
ap.add_argument('--msa', default=None, metavar='SPEC',
                help='compute conservation from an ALIGNED FASTA, e.g. "aln.fasta" or '
                     '"tube=tube.fasta,belt=belt.fasta". The first sequence is taken as the '
                     'reference. This is NOT ConSurf: it is a Henikoff-weighted Shannon '
                     'entropy binned into 9 grades, which needs no network and no external '
                     'program. Prefer --conservation with a real ConSurf .grades file when '
                     'you have one.')
ap.add_argument('--solvation-track', action='store_true',
                help='add a per-residue solvation strip under each panel: how much each '
                     'residue gains or pays by being buried (Eisenberg-McLachlan). This is '
                     'what fills the gaps in the bond track - apolar burial that pays for '
                     'itself, or polar burial that costs.')
ap.add_argument('--no-interaction-track', action='store_true',
                help='omit the per-residue interaction-type track from the footprint figures')
ap.add_argument('--network-min', type=int, default=3, metavar='N',
                help='an interface is drawn in the interaction-network figure only if it '
                     'makes at least N specific interactions (default 3). Interfaces '
                     'carrying a covalent link are always drawn, however few.')
ap.add_argument('--phobic-min', type=int, default=3, metavar='N',
                help='draw a hydrophobic edge in the network figure when two residues make '
                     'at least N apolar atom contacts (default 5). This is what holds a '
                     'non-specific interface together; set higher to thin it out.')
ap.add_argument('--network-per-page', type=int, default=6, metavar='N',
                help='panels per network figure; the rest spill onto further figures')
ap.add_argument('--sc-offset', type=float, default=0.0, metavar='X',
                help='shift the Sc reference bands down by X. The bands are drawn on the '
                     'scale of the original sc program; this implementation reads about '
                     '0.05 low against it, so --sc-offset 0.05 puts the band edges on the '
                     'measured scale instead.')
ap.add_argument('--bsa-min', type=float, default=10.0, metavar='A2',
                help='minimum buried area for a residue to be coloured in the ChimeraX '
                     'script; everything below stays grey (default 10 A2)')
args = ap.parse_args()
MODEL = args.model
PERLINE = max(20, args.per_line)
CFG = json.load(open('run_config.json'))
mpl.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 600, 'font.size': 8,
                     'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'sans-serif', 'pdf.fonttype': 42,
                     'svg.fonttype': 'none'})
ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'MSE': 'M'}
# nucleotides keep their own one-letter form so a DNA/RNA chain reads as a sequence
ONE.update({n: n[-1].lower() for n in
            ('A', 'C', 'G', 'U', 'T', 'DA', 'DC', 'DG', 'DT', 'DU', 'I')})
CLSORD = ['charged', 'polar', 'aromatic', 'aliphatic', 'Gly / Pro', 'nucleotide',
          'ligand', 'other']
BLAB = {'salt': 'salt bridge', 'hbond': 'hydrogen bond',
        'stack': 'aromatic stacking', 'catpi': 'cation–π'}
LOWCONF = CFG.get('bond_confidence') in ('low', 'unknown')
# When no --resolution was given the confidence is 'unknown', and there is no number
# to quote; say so rather than printing 'None Å'.
LOWTXT = (f'tentative at {CFG["resolution"]} Å' if CFG.get('resolution')
          else 'tentative — resolution not given')

GEO = pd.read_csv('chain_geometry.csv').set_index('chain')
S = pd.read_csv('interfaces_summary.csv')
ENV = pd.read_csv('footprints_by_environment.csv')
SS = pd.read_csv('secondary_structure.csv')
ORBC = pd.read_csv('interface_orbit_chemistry.csv')
AR = pd.read_csv('aromatic_interactions.csv')
B = pd.read_csv('interface_bonds.csv')
BUDG = pd.read_csv('subunit_budget.csv')
COMP = pd.read_csv('interface_composition_by_category.csv', index_col=0)
DEN = pd.read_csv('interaction_density_by_category.csv', index_col=0)

ENTS = [e['name'] for e in CFG['entities']]
PALS = fs.assign_palettes(ENTS)
if args.colours:                                   # user-pinned colours, by protein name
    for item in args.colours.split(','):
        if '=' not in item:
            continue
        k, v = item.split('=', 1)
        k = k.strip()
        if k in PALS:
            PALS[k] = fs.family_from_hex(v, k)
        else:
            print(f'  --colours: no protein called {k!r}; known: {", ".join(ENTS)}', flush=True)
PAL = {e: PALS[e]['pal'] for e in ENTS}
DARK = {e: PALS[e]['dark'] for e in ENTS}
HOMO = len(ENTS) == 1
PARTKEY = 'pctg' if HOMO else 'pfam'
CATS = list(S.groupby('ctg').area.sum().sort_values(ascending=False).index)
if HOMO:                                   # colour by interface, not by partner protein
    base = PALS[ENTS[0]]
    PARTCOL = {c: fs.shade_of(base, i, len(CATS)) for i, c in enumerate(CATS)}
else:
    PARTCOL = dict(PAL)
CATCOL = {}
for c in CATS:
    first = next((e for e in ENTS if c.startswith(e)), ENTS[0])
    sibs = [x for x in CATS if x.startswith(first)]
    CATCOL[c] = fs.shade_of(PALS[first], sibs.index(c), len(sibs))
SSc = SS.set_index(['chain', 'resi']).ss if len(SS) else pd.Series(dtype=object)
Bk = B.set_index(['c1', 'c2']).sort_index()
ARk = AR.set_index(['c1', 'c2']).sort_index() if len(AR) else None
prot = dict(zip(GEO.index, GEO.entity_name))
env_of = dict(zip(GEO.index, GEO.environment))
REP = S.sort_values('area', ascending=False).groupby('orbit').first().reset_index()

p = MMCIFParser(QUIET=True) if MODEL.lower().endswith(('.cif', '.mmcif')) else PDBParser(QUIET=True)
model = p.get_structure('m', MODEL)[0]
for ch in model:
    for r in list(ch):
        if r.id[0] != ' ':
            ch.detach_child(r.id); continue
        for a in [a for a in r if a.element == 'H']:
            r.detach_child(a.get_id())
for ch in list(model):
    if len(list(ch)) == 0:
        model.detach_child(ch.id)
coords = {c.id: {r.id[1]: np.array([a.coord for a in r]) for r in c} for c in model}


# ------------------------------------------------------------------ conservation (optional)
def load_conservation(path):
    """Read per-residue conservation, either a ConSurf .grades file or a simple table.

    Returns {protein_name: {resi: grade 1-9}} plus the raw scores. A ConSurf run is done
    on one protein, so an unlabelled file is applied to every protein whose numbering it
    covers; a 'protein' column overrides that.
    """
    if path is None:
        return {}, {}
    grades, scores = {}, {}
    txt = open(path, encoding='utf-8', errors='replace').read()
    rows = []
    try:                                           # a table with a header we recognise
        sep = '\t' if txt.count('\t') > txt.count(',') else ','
        T = pd.read_csv(path, sep=sep, comment='#')
        cols = {c.lower().strip(): c for c in T.columns}
        rc = next((cols[k] for k in ('resi', 'pos', 'position', 'residue') if k in cols), None)
        sc = next((cols[k] for k in ('grade', 'score', 'conservation', 'cons') if k in cols), None)
        pc = next((cols[k] for k in ('protein', 'entity', 'chain') if k in cols), None)
        if rc and sc:
            for _, r in T.iterrows():
                rows.append((str(r[pc]) if pc else None, int(r[rc]), float(r[sc])))
    except Exception:
        pass
    if not rows:                                   # ConSurf .grades: POS SEQ 3LATOM SCORE COLOR
        for line in txt.split('\n'):
            m = re.match(r'\s*(\d+)\s+(\w)\s+\S+\s+(-?[\d.]+)\s+(\d)', line)
            if m:
                rows.append((None, int(m.group(1)), float(m.group(4))))
    if not rows:
        print(f'  --conservation: could not read {path}; skipping', flush=True)
        return {}, {}
    hi = max(v for _, _, v in rows)
    lo = min(v for _, _, v in rows)
    for pname, resi, v in rows:
        g = int(round(v)) if 1 <= hi <= 9 and lo >= 1 else \
            int(round(1 + 8 * (v - lo) / (hi - lo))) if hi > lo else 5
        targets = [pname] if pname in ENTS else ENTS
        for t in targets:
            grades.setdefault(t, {})[resi] = min(9, max(1, g))
            scores.setdefault(t, {})[resi] = v
    src = 'ConSurf grades' if not any(p for p, _, _ in rows) else 'table'
    print(f'  conservation loaded from {path} ({src}, {len(rows)} residues)', flush=True)
    return grades, scores



# --------------------------------------------------- conservation from an alignment
def read_fasta(path):
    seqs, name, buf = [], None, []
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.rstrip('\n\r')
        if line.startswith('>'):
            if name is not None:
                seqs.append((name, ''.join(buf)))
            name, buf = line[1:].strip(), []
        elif line.strip():
            buf.append(line.strip())
    if name is not None:
        seqs.append((name, ''.join(buf)))
    return seqs


def grades_from_msa(path):
    """Per-site conservation grades 1-9 from an aligned FASTA, reference = first sequence.

    Sequences are weighted by the Henikoff & Henikoff (1994) position-based scheme, so a
    pile of near-identical homologs does not masquerade as conservation. Each column is
    then scored by 1 - H/Hmax, where H is the weighted Shannon entropy over the 20 amino
    acids, and columns are binned into 9 grades by rank, as ConSurf does with its rates.

    This is not ConSurf. ConSurf estimates an evolutionary RATE with Rate4Site under an
    explicit substitution model and a phylogeny; entropy is a cruder, model-free proxy.
    The two correlate well but are not interchangeable, and the grades here must not be
    reported as ConSurf grades.
    """
    seqs = read_fasta(path)
    if len(seqs) < 2:
        print(f'  --msa: {path} holds fewer than 2 sequences; skipping', flush=True)
        return None
    L = len(seqs[0][1])
    if any(len(s) != L for _, s in seqs):
        print(f'  --msa: {path} is not aligned (unequal lengths); skipping', flush=True)
        return None
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    M = np.array([[c.upper() for c in s] for _, s in seqs])
    # Henikoff weights: a residue in a column with few distinct types counts for less
    w = np.zeros(len(seqs))
    for j in range(L):
        col = M[:, j]
        keep = np.isin(col, list(AA))
        if keep.sum() < 2:
            continue
        types, counts = np.unique(col[keep], return_counts=True)
        share = {t: 1.0 / (len(types) * c) for t, c in zip(types, counts)}
        for i in np.where(keep)[0]:
            w[i] += share[col[i]]
    w = w / w.sum() if w.sum() > 0 else np.full(len(seqs), 1.0 / len(seqs))
    hmax = np.log(len(AA))
    score, gapfrac = np.zeros(L), np.zeros(L)
    for j in range(L):
        col = M[:, j]
        keep = np.isin(col, list(AA))
        gapfrac[j] = 1.0 - keep.mean()
        if keep.sum() == 0:
            score[j] = np.nan
            continue
        ww = w[keep] / w[keep].sum()
        p = np.array([ww[col[keep] == a].sum() for a in AA])
        p = p[p > 0]
        score[j] = 1.0 - (-(p * np.log(p)).sum()) / hmax
    # a column that is mostly gaps carries little information: damp it
    score = score * (1.0 - gapfrac) ** 0.5
    ref = M[0]
    pos = [j for j in range(L) if ref[j] in AA]
    vals = np.array([score[j] for j in pos], float)
    ok = np.isfinite(vals)
    gr = np.full(len(vals), 5, int)
    if ok.sum() >= 9:
        ranks = np.argsort(np.argsort(vals[ok]))
        gr[ok] = np.clip((ranks * 9 // ok.sum()) + 1, 1, 9)
    print(f'  conservation computed from {path}: {len(seqs)} sequences, '
          f'{len(pos)} reference positions (entropy-based grades, NOT ConSurf)', flush=True)
    return list(gr)


def apply_msa(spec):
    """Map --msa grades onto residue numbers, per protein."""
    out = {}
    if not spec:
        return out
    items = []
    for part in spec.split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            items.append((k.strip(), v.strip()))
        else:
            items.append((None, part.strip()))
    for pname, path in items:
        gr = grades_from_msa(path)
        if gr is None:
            continue
        resnums = {}
        for e in ENTS:
            ch = next((c for c in GEO.index if prot[c] == e), None)
            if ch is not None:
                resnums[e] = sorted(r.id[1] for r in model[ch])
        target = pname if pname in ENTS else min(
            resnums, key=lambda e: abs(len(resnums[e]) - len(gr)))
        rn = resnums[target]
        if len(rn) != len(gr):
            n = min(len(rn), len(gr))
            print(f'  --msa: reference has {len(gr)} residues but {target} has {len(rn)} in '
                  f'the model; aligning the first {n} by order', flush=True)
            rn, gr = rn[:n], gr[:n]
        out[target] = {int(r): int(g) for r, g in zip(rn, gr)}
    return out


CONS, CONSRAW = load_conservation(args.conservation)
RES_ALL = pd.read_csv('interface_residues.csv')
SOLV = {}
if 'dg_solv' in RES_ALL.columns:
    _per = RES_ALL.groupby(['chain', 'resi']).dg_solv.sum()
    for _e in ENTS:
        _chs = [c for c in GEO.index if prot[c] == _e]
        _d = collections.defaultdict(float)
        for _c in _chs:
            if _c in _per.index.get_level_values(0):
                for _r, _v in _per.loc[_c].items():
                    _d[int(_r)] += _v / len(_chs)
        SOLV[_e] = dict(_d)

for _ent, _g in apply_msa(args.msa).items():
    CONS.setdefault(_ent, _g)        # an explicit .grades file always wins



MUTED = '#8A8A8A'


def sc_size(area):
    """Marker AREA in points^2 for an interface of that buried area, used by both the
    scatter and its size key so the legend circles are literally to scale."""
    return np.clip(np.asarray(area, float) / 16.0, 8.0, 112.0)


def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.tick_params(colors='#666666')


def save(fig, name):
    fig.savefig(name + '.png', dpi=600); fig.savefig(name + '.pdf'); plt.close(fig)
    print('wrote', name, flush=True)


# ------------------------------------------- per-residue interaction-type profiles
# The footprint says how much surface a residue buries; it says nothing about HOW that
# surface is held. These counts answer the second question on the same residue axis:
# how many salt bridges, hydrogen bonds, stacking and cation-pi contacts each residue
# makes across the interfaces of its environment, averaged over the symmetry copies.
INTORD = ['salt', 'hbond', 'stack', 'catpi']
KINDMAP = {'saltbridge': 'salt', 'hbond': 'hbond', 'pi_stack_T': 'stack',
           'pi_stack_parallel': 'stack', 'cation_pi': 'catpi'}
INTRES = collections.defaultdict(collections.Counter)
for _df in (B, AR):
    if _df is None or not len(_df):
        continue
    for _c1, _r1, _c2, _r2, _k in zip(_df.c1, _df.ri1, _df.c2, _df.ri2, _df.kind):
        _kk = KINDMAP.get(str(_k))
        if _kk is None:
            continue
        INTRES[(_c1, int(_r1))][_kk] += 1
        INTRES[(_c2, int(_r2))][_kk] += 1
HAS_INT = bool(INTRES) and not args.no_interaction_track


def interaction_profile(env, lo, hi):
    """Mean count of each interaction type per residue, per monomer of this environment."""
    chs = [c for c in env_of if env_of[c] == env]
    n = max(1, len(chs))
    out = {k: np.zeros(hi - lo + 1) for k in INTORD}
    for c in chs:
        for ri in range(lo, hi + 1):
            cnt = INTRES.get((c, ri))
            if cnt:
                for k, v in cnt.items():
                    out[k][ri - lo] += v / n
    return out



# ============================================================ footprints
def footprint_fig(ent, fname, ph=1.15, dh=0.42, chh=0.13, ih=0.52, sh=0.13):
    """One figure per protein: buried area per residue, per symmetry-independent environment.

    Long chains are wrapped onto several lines, because a 1,000-residue protein squeezed
    onto one axis hides exactly the per-residue detail the figure exists to show. The
    number of lines is chosen so they come out near-equal rather than leaving a nearly
    empty last line, and every line spans the same number of residues so bar widths stay
    comparable down the whole figure.

    Axes are placed in inches rather than by gridspec, so the spacing is the same whether
    the protein needs one line or six.
    """
    sub_all = ENV[ENV.entity_name == ent]
    envs = sorted(sub_all.env.unique())
    parts = [k for k in sub_all[PARTKEY].unique()]
    parts = [c for c in CATS if c in parts] if HOMO else [e for e in ENTS if e in parts]
    lo, hi = int(sub_all.resi.min()), int(sub_all.resi.max())
    span = hi - lo + 1
    nseg = max(1, int(round(span / PERLINE)))
    seglen = int(np.ceil(span / nseg))
    segs = [(lo + i * seglen, lo + (i + 1) * seglen - 1) for i in range(nseg)]
    cons = CONS.get(ent)
    solv = SOLV.get(ent) if args.solvation_track else None
    pivs = {}
    for e in envs:
        sdf = sub_all[sub_all.env == e]
        P = sdf.pivot_table(index='resi', columns=PARTKEY, values='dsasa',
                            aggfunc='sum').reindex(range(lo, hi + 1)).fillna(0)
        for k in parts:
            if k not in P:
                P[k] = 0.0
        pivs[e] = P[parts]
    ymax = max(25.0, np.ceil(max(p.sum(axis=1).max() for p in pivs.values()) / 25) * 25)
    dmax = max(gaussian_filter1d(p.sum(axis=1).to_numpy(), 3).max() for p in pivs.values())
    iprof = {e: interaction_profile(e, lo, hi) for e in envs} if HAS_INT else {}
    slim = max(1e-9, float(np.percentile(np.abs(list(solv.values())), 97))) if solv else 1.0
    imax = float(np.ceil(max([np.sum(list(p.values()), axis=0).max()
                              for p in iprof.values()] + [1.0])))

    # All three y-labels are pinned to one x in axes coordinates, to the LEFT of every
    # tick label. Letting matplotlib place them puts each one just outside its own tick
    # labels, which on the short conservation strip lands it on top of the density
    # axis's "0" tick.
    YLAB_X = -0.058
    # The legend is grouped into columns, one group per column, so its height is set by
    # the LONGEST group. Reserve the top margin from that count rather than a constant,
    # or the bottom row of the legend lands on the first panel's title.
    n_int_kinds = sum(1 for k in INTORD
                      if HAS_INT and any(iprof[e][k].sum() > 0 for e in envs)) if HAS_INT else 0
    leg_rows = max(1 + len(parts), (1 + n_int_kinds) if HAS_INT else 0, 4,
                   3 if solv else 0, 3 if cons else 0)
    TOP = 0.34 + 0.108 * leg_rows               # inches: figure title + grouped legend
    BOT = 0.40
    GAP_AX, GAP_SEG, GAP_ENV, TITLE_H = 0.05, 0.30, 0.34, 0.20
    # a wider gap before the conservation strip: its label and the density label are
    # both rotated and sit in the same column, so too little space makes the two read
    # as one phrase
    GAP_CONS = 0.17
    ints = HAS_INT
    seg_h = (ph + GAP_AX + dh + ((GAP_AX + ih) if ints else 0)
             + ((GAP_CONS + sh) if solv else 0) + ((GAP_CONS + chh) if cons else 0))
    env_h = TITLE_H + nseg * seg_h + (nseg - 1) * GAP_SEG
    H = TOP + len(envs) * env_h + (len(envs) - 1) * GAP_ENV + BOT
    W = 7.2
    L, Wd = 0.085, 0.900
    fig = plt.figure(figsize=(W, H))

    def axes_at(y_top_in, h_in):
        return fig.add_axes([L, (H - y_top_in - h_in) / H, Wd, h_in / H])

    y = TOP
    for k, e in enumerate(envs):
        P = pivs[e]; tot = P.sum(axis=1)
        nco = sum(1 for c in env_of if env_of[c] == e)
        fig.text(L, (H - y - 0.06) / H,
                 f'environment {"/".join(e)}  ({nco} copies)   ·   '
                 f'{tot.sum():.0f} Å$^2$ buried per monomer',
                 fontsize=7.5, ha='left', va='top', color=fs.INK)
        y += TITLE_H
        for si, (s0, s1) in enumerate(segs):
            ax = axes_at(y, ph)
            x0, x1 = s0 - 4, s0 + seglen + 3
            ax.set_xlim(x0, x1); ax.set_ylim(-0.19 * ymax, ymax * 1.32); fig.canvas.draw()
            bot = np.zeros(len(P))
            for kk in parts:
                v = P[kk].to_numpy()
                if v.sum() > 0:
                    fs.sbar(ax, P.index.to_numpy(), v, bottoms=bot, width=1.0,
                            color=PARTCOL.get(kk, '#999999'), round_pt=1.0, gap_pt=0.35)
                bot = bot + v
            ch = e[0]
            ri = np.array([r for r in P.index if s0 <= r <= s1 and (ch, r) in SSc.index]) \
                if len(SSc) else np.array([])
            if len(ri) > 3:
                fs.ss_ribbon(ax, ri, [SSc[(ch, r)] for r in ri], -0.155 * ymax, 0.095 * ymax)
            minsep = max(1.0, (x1 - x0) / 26.0)
            seg_tot = tot[(tot.index >= s0) & (tot.index <= s1)]
            sel = []
            for r in seg_tot.sort_values(ascending=False).index:
                if seg_tot[r] <= 0:
                    break
                if all(abs(r - t) > minsep for t in sel):
                    sel.append(r)
                if len(sel) == 6:
                    break
            for r in sel:
                ax.plot([r, r], [tot[r] + ymax * 0.02, ymax * 1.0], lw=0.45, color='0.68', zorder=1)
                rn = sub_all[sub_all.resi == r].resn.iloc[0]
                ax.text(r, ymax * 1.02, f'{ONE.get(rn, rn)}{r}', rotation=90, ha='center',
                        va='bottom', fontsize=6, color=fs.INK)
            ax.set_yticks(np.arange(0, ymax + 1, 50 if ymax > 120 else 25))
            tidy(ax); ax.spines['bottom'].set_visible(False)
            ax.tick_params(labelbottom=False, bottom=False)
            ax.set_ylabel('buried Å$^2$', fontsize=7)
            ax.yaxis.set_label_coords(YLAB_X, 0.5)
            y += ph + GAP_AX

            if ints:
                axi = axes_at(y, ih)
                axi.set_xlim(x0, x1); axi.set_ylim(0, imax * 1.15); fig.canvas.draw()
                xs = np.arange(lo, hi + 1)
                m = (xs >= s0) & (xs <= s1)
                bot = np.zeros(m.sum())
                for k in INTORD:
                    v = iprof[e][k][m]
                    if v.sum() > 0:
                        fs.sbar(axi, xs[m], v, bottoms=bot, width=1.0,
                                color=fs.INTCOL[k], round_pt=1.0, gap_pt=0.35)
                    bot = bot + v
                step = 1 if imax <= 4 else 2
                axi.set_yticks(np.arange(0, imax + 0.1, step))
                tidy(axi); axi.spines['bottom'].set_visible(False)
                axi.tick_params(labelbottom=False, bottom=False, labelsize=5.5)
                axi.set_ylabel('bonds', fontsize=6.5)
                axi.yaxis.set_label_coords(YLAB_X, 0.5)
                y += ih + GAP_AX

            axd = axes_at(y, dh)
            pad = 15                             # smooth with a margin, then crop
            w = P.index[(P.index >= s0 - pad) & (P.index <= s1 + pad)]
            axd.set_xlim(x0, x1); fig.canvas.draw()
            fs.density_plot(axd, w.to_numpy(), tot.reindex(w).fillna(0).to_numpy(),
                            DARK[ent], vmax=dmax)
            axd.set_xlim(x0, x1)
            axd.set_yticks([0, np.floor(dmax / 10) * 10]); tidy(axd)
            axd.tick_params(labelsize=5.5); axd.set_ylabel('density', fontsize=6.5)
            axd.yaxis.set_label_coords(YLAB_X, 0.5)
            last = axd
            y += dh
            if solv:
                y += GAP_CONS
                axs_ = axes_at(y, sh)
                xs2 = np.array([r for r in range(s0, s1 + 1) if r in solv])
                if len(xs2):
                    vals = np.array([solv[r] for r in xs2])
                    lim = max(1e-9, np.percentile(np.abs(list(solv.values())), 97))
                    axs_.bar(xs2, np.ones(len(xs2)), width=1.0, align='center',
                             color=fs.solv_cmap()((np.clip(vals, -lim, lim) / lim + 1) / 2),
                             edgecolor='none', zorder=2)
                axs_.set_xlim(x0, x1); axs_.set_ylim(0, 1); axs_.set_yticks([])
                axs_.set_ylabel('ΔG', fontsize=6.5)
                axs_.yaxis.set_label_coords(YLAB_X, 0.45)
                for sp in ('top', 'right', 'left'):
                    axs_.spines[sp].set_visible(False)
                axs_.spines['bottom'].set_color('#AAAAAA')
                axs_.tick_params(colors='#666666', labelsize=5.5)
                axd.tick_params(labelbottom=False, bottom=False)
                axd.spines['bottom'].set_visible(False)
                last = axs_
                y += sh
            if cons:
                y += GAP_CONS
                axc = axes_at(y, chh)
                seg_r = [r for r in range(s0, s1 + 1) if r in cons]
                if seg_r:
                    axc.bar(seg_r, [1] * len(seg_r), width=1.0, align='center',
                            color=[fs.CONSURF[cons[r] - 1] for r in seg_r],
                            edgecolor='none', zorder=2)
                axc.set_xlim(x0, x1); axc.set_ylim(0, 1); axc.set_yticks([])
                axc.set_ylabel('cons.', fontsize=6.5)
                axc.yaxis.set_label_coords(YLAB_X, 0.45)
                for sp in ('top', 'right', 'left'):
                    axc.spines[sp].set_visible(False)
                axc.spines['bottom'].set_color('#AAAAAA')
                axc.tick_params(colors='#666666', labelsize=5.5)
                last.tick_params(labelbottom=False, bottom=False)
                last.spines['bottom'].set_visible(False)
                last = axc
                y += chh
            if si == nseg - 1 and k == len(envs) - 1:
                last.set_xlabel('residue number')
            y += GAP_SEG if si < nseg - 1 else 0
        y += GAP_ENV
    lab = 'buried at' if HOMO else 'buried by'
    BLANK = mpl.patches.Patch(fc='none', ec='none', label=' ')

    def title(t):
        return mpl.patches.Patch(fc='none', ec='none',
                                 label=r'$\bf{' + t.replace(' ', r'\ ') + '}$')

    groups = [[title(lab)] + [mpl.patches.Patch(fc=PARTCOL.get(k, '#999999'), ec='none',
                                                label=str(k)) for k in parts]]
    if HAS_INT:
        g = [title('interaction')]
        g += [mpl.patches.Patch(fc=fs.INTCOL[k], ec='none', label=t)
              for k, t in [('salt', 'salt bridge'), ('hbond', 'hydrogen bond'),
                           ('stack', 'aromatic stacking'), ('catpi', 'cation–π')]
              if any(iprof[e][k].sum() > 0 for e in envs)]
        groups.append(g)
    groups.append([title('structure'),
                   mpl.patches.Patch(fc='#6E7B8B', ec='none', label='α-helix'),
                   mpl.patches.Patch(fc='#AFB7C0', ec='none', label='β-strand'),
                   mpl.patches.Patch(fc=DARK[ent], ec='none', label='interaction density')])
    if cons:
        groups.append([title('conservation'),
                       mpl.patches.Patch(fc=fs.CONSURF[0], ec='none', label='variable'),
                       mpl.patches.Patch(fc=fs.CONSURF[-1], ec='none', label='conserved')])
    rows = max(len(g) for g in groups)
    h = [x for g in groups for x in (g + [BLANK] * (rows - len(g)))]
    fig.text(L, 1 - 0.13 / H, f'{ent} — interface footprint per symmetry-independent '
             f'environment', fontsize=8.5, ha='left', va='top', color=fs.INK)
    fig.legend(handles=h, frameon=False, fontsize=6.2, loc='upper left',
               ncol=len(groups), bbox_to_anchor=(L, 1 - 0.26 / H),
               handlelength=1.0, columnspacing=1.6, handletextpad=0.4,
               labelspacing=0.30)
    if solv:
        # a continuous quantity needs a scale, not two swatches
        cb_w, cb_h, cb_top = 1.35, 0.085, 0.34
        x0f = 0.985 - cb_w / 7.2
        axcb = fig.add_axes([x0f, (H - cb_top - cb_h) / H, cb_w / 7.2, cb_h / H])
        mp = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-slim, slim), cmap=fs.solv_cmap())
        cb = fig.colorbar(mp, cax=axcb, orientation='horizontal')
        cb.set_ticks([-slim, 0, slim])
        cb.set_ticklabels([f'{-slim:.0f}', '0', f'+{slim:.0f}'])
        cb.ax.tick_params(labelsize=5.5, colors='#666666', length=2, pad=1.5)
        cb.outline.set_visible(False)
        fig.text(x0f, (H - cb_top + 0.05) / H, r'$\bf{burial\ \Delta G}$  kcal/mol',
                 fontsize=6.2, ha='left', va='bottom', color=fs.INK)
        fig.text(x0f, (H - cb_top - cb_h - 0.17) / H, 'pays', fontsize=5.6,
                 ha='left', va='top', color=fs.SOLV_GOOD)
        fig.text(0.985, (H - cb_top - cb_h - 0.17) / H, 'costs', fontsize=5.6,
                 ha='right', va='top', color=fs.SOLV_BAD)
    save(fig, fname)


for i, e in enumerate(ENTS, start=1):
    if len(ENV[ENV.entity_name == e]):
        footprint_fig(e, f'fig{i}_{e}_footprints')

# ============================================================ architecture
fig = plt.figure(figsize=(7.2, 4.0))
gs = fig.add_gridspec(1, 1, left=0.085, right=0.76, top=0.92, bottom=0.13)
ax = fig.add_subplot(gs[0, 0])
envs_all = sorted(set(env_of.values()))
ENVCOL = {}
for e in ENTS:
    mine = [x for x in envs_all if prot[x[0]] == e]
    for i, x in enumerate(mine):
        ENVCOL[x] = fs.shade_of(PALS[e], i, len(mine))
for _, r in S.iterrows():
    t1, z1 = GEO.loc[r.c1, 'theta'], GEO.loc[r.c1, 'z']
    t2, z2 = GEO.loc[r.c2, 'theta'], GEO.loc[r.c2, 'z']
    lw = 0.22 + 3.0 * r.area / max(1.0, S.area.max())
    if abs(t1 - t2) > 180:
        s = 360 if t2 < t1 else -360
        ax.plot([t1, t2 + s], [z1, z2], lw=lw, color='#C9CDD2', zorder=1, solid_capstyle='round')
        ax.plot([t1 - s, t2], [z1, z2], lw=lw, color='#C9CDD2', zorder=1, solid_capstyle='round')
    else:
        ax.plot([t1, t2], [z1, z2], lw=lw, color='#C9CDD2', zorder=1, solid_capstyle='round')
for c in GEO.index:
    col = ENVCOL[env_of[c]]
    ax.scatter(GEO.loc[c, 'theta'], GEO.loc[c, 'z'], s=120, color=col,
               edgecolor='white', lw=0.9, zorder=3)
    dark = sum(int(col[i:i + 2], 16) for i in (1, 3, 5)) / 3 < 150
    ax.text(GEO.loc[c, 'theta'], GEO.loc[c, 'z'], c, ha='center', va='center',
            fontsize=5.4, color='white' if dark else '#333333', zorder=4, fontweight='bold')
ax.set_xticks(np.arange(-180, 181, 60)); ax.set_xlim(-195, 195)
ax.set_xlabel('angular position around the symmetry axis (°)')
ax.set_ylabel('position along axis (Å)')
ax.set_title(f'Contact topology, cylinder unrolled — point group {CFG["point_group"]}; '
             'line width ∝ buried area', fontsize=8, loc='left', pad=4, color=fs.INK)
tidy(ax)
yy = 0.92
for e in ENTS:
    mine = [x for x in envs_all if prot[x[0]] == e]
    hh = [mpl.patches.Patch(fc=ENVCOL[x], ec='white', lw=0.5, label='/'.join(x)) for x in mine]
    lg = fig.legend(handles=hh, frameon=False, fontsize=6.0, loc='upper left',
                    bbox_to_anchor=(0.775, yy), handlelength=1.1, labelspacing=0.4,
                    title=e, title_fontsize=6.6)
    lg._legend_box.align = 'left'
    yy -= 0.10 + 0.055 * len(mine)
save(fig, 'fig_architecture')

# ============================================================ inventory
fig = plt.figure(figsize=(7.2, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.42,
                      left=0.085, right=0.795, top=0.87, bottom=0.30)
ax = fig.add_subplot(gs[0, 0])
for i, c in enumerate(CATS):
    v = ORBC[ORBC.ctg == c].area.values
    ax.scatter(np.full(len(v), i) + np.linspace(-0.14, 0.14, max(len(v), 1))[:len(v)], v,
               s=20, color=CATCOL[c], edgecolor='white', lw=0.5, zorder=3)
    ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color='#555555', lw=1.1, zorder=4)
    ax.text(i, ORBC.area.max() * 1.10, f'{len(v)}', ha='center', fontsize=5.8, color='#888888')
ax.text(-0.85, ORBC.area.max() * 1.10, 'n =', ha='left', fontsize=5.8, color='#888888')
ax.set_xticks(range(len(CATS)))
ax.set_xticklabels(CATS, fontsize=6.2, rotation=36, ha='right', rotation_mode='anchor')
ax.set_ylabel('buried area per interface (Å$^2$)')
ax.set_ylim(0, ORBC.area.max() * 1.20); ax.set_xlim(-0.9, len(CATS) - 0.1)
ax.set_title('a   One dot = one symmetry-unique interface', fontsize=8, loc='left', pad=4, color=fs.INK)
tidy(ax)
ax2 = fig.add_subplot(gs[0, 1])
rows = sorted(set(BUDG.env), key=lambda e: (ENTS.index(prot[e[0]]), e))
y = np.arange(len(rows))[::-1]
xmax = BUDG.groupby('env').total.mean().max() * 1.25
ax2.set_xlim(0, xmax); ax2.set_ylim(-0.8, len(rows) - 0.2); fig.canvas.draw()
for yy_, e in zip(y, rows):
    left = 0.0
    for part in ENTS:
        w = BUDG[BUDG.env == e][part].mean()
        if w > 1:
            fs.sbarh(ax2, yy_, w, left=left, height=0.62, color=PAL[part], round_pt=2.0, gap_pt=1.1)
        left += w
    ax2.text(left + xmax * 0.015, yy_, f'{left:.0f}', va='center', fontsize=6, color='#555555')
ax2.set_yticks(y); ax2.set_yticklabels(['/'.join(e) for e in rows], fontsize=6.2)
for lbl, e in zip(ax2.get_yticklabels(), rows):
    lbl.set_color(DARK[prot[e[0]]])
ax2.set_xlabel('buried area per subunit (Å$^2$)')
ax2.set_title('b   Interface budget of each symmetry-independent subunit',
              fontsize=8, loc='left', pad=4, color=fs.INK)
tidy(ax2)
hh = [mpl.patches.Patch(fc=PAL[p_], ec='none', label=f'buried by {p_}') for p_ in ENTS]
lg = fig.legend(handles=hh, frameon=False, fontsize=6.4, loc='upper left',
                bbox_to_anchor=(0.808, 0.80), handlelength=1.1, labelspacing=0.5,
                title='partner', title_fontsize=6.8)
lg._legend_box.align = 'left'
save(fig, 'fig_inventory')

# ============================================================ chemistry
present = [c for c in CLSORD if c in COMP.columns and COMP[c].sum() > 0]
fig = plt.figure(figsize=(7.2, 4.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.52,
                      left=0.165, right=0.80, top=0.88, bottom=0.12)
y = np.arange(len(CATS))[::-1]
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1.0); ax.set_ylim(-0.8, len(CATS) - 0.2)
fig.canvas.draw()
for yy_, c in zip(y, CATS):
    left = 0.0
    for cl in present:
        w = float(COMP.loc[c, cl]) if c in COMP.index else 0.0
        if w > 0.002:
            fs.sbarh(ax, yy_, w, left=left, height=0.62,
                     color=fs.CLSCOL.get(cl, '#AAAAAA'), round_pt=1.6, gap_pt=1.0)
        left += w
ax.set_yticks(y); ax.set_yticklabels(CATS, fontsize=6.4)
ax.set_xlabel('fraction of buried interface area'); ax.set_xticks([0, .25, .5, .75, 1])
ax.set_title('a   Chemistry of the buried surface', fontsize=8, loc='left', pad=4, color=fs.INK)
tidy(ax)
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, max(1e-9, float(DEN.sum(axis=1).max())) * 1.20)
ax2.set_ylim(-0.8, len(CATS) - 0.2); fig.canvas.draw()
for yy_, c in zip(y, CATS):
    left = 0.0
    for k in ['salt', 'hbond', 'stack', 'catpi']:
        w = float(DEN.loc[c, k]) if c in DEN.index else 0.0
        if w > 0.02:
            fs.sbarh(ax2, yy_, w, left=left, height=0.62, color=fs.INTCOL[k],
                     round_pt=1.6, gap_pt=1.0)
        left += w
    ax2.text(left + 0.4, yy_, f'{left:.1f}', va='center', fontsize=6, color='#555555')
ax2.set_yticks(y); ax2.set_yticklabels([])
ax2.set_xlabel('specific interactions per 1000 Å$^2$ buried')
ttl = 'b   Directional interactions, normalised for interface size'
if LOWCONF:
    ttl += f'\n     ({LOWTXT})'
ax2.set_title(ttl, fontsize=8, loc='left', pad=4, color=fs.INK)
tidy(ax2)
h1 = [mpl.patches.Patch(fc=fs.CLSCOL.get(c, '#AAAAAA'), ec='none', label=c) for c in present]
l1 = fig.legend(handles=h1, frameon=False, fontsize=6.3, loc='upper left',
                bbox_to_anchor=(0.815, 0.86), handlelength=1.1, labelspacing=0.45,
                title='residue class', title_fontsize=6.8); l1._legend_box.align = 'left'
h2 = [mpl.patches.Patch(fc=fs.INTCOL[k], ec='none', label=BLAB[k])
      for k in ['salt', 'hbond', 'stack', 'catpi']]
l2 = fig.legend(handles=h2, frameon=False, fontsize=6.3, loc='upper left',
                bbox_to_anchor=(0.815, 0.50), handlelength=1.1, labelspacing=0.45,
                title='interaction type', title_fontsize=6.8); l2._legend_box.align = 'left'
save(fig, 'fig_chemistry')

# ============================================================ contact maps
def draw_map(ax, c1, c2, tick_every=8):
    r1, r2 = sorted(coords[c1]), sorted(coords[c2])
    P1 = np.vstack([coords[c1][i] for i in r1])
    i1 = np.concatenate([[j] * len(coords[c1][i]) for j, i in enumerate(r1)])
    P2 = np.vstack([coords[c2][i] for i in r2])
    i2 = np.concatenate([[j] * len(coords[c2][i]) for j, i in enumerate(r2)])
    D = np.full((len(r1), len(r2)), np.nan)
    for i, js in enumerate(cKDTree(P1).query_ball_tree(cKDTree(P2), 6.0)):
        if not js:
            continue
        d = np.linalg.norm(P1[i] - P2[js], axis=1); a = i1[i]
        for jj, dd in zip(np.array(js), d):
            b = i2[jj]
            if np.isnan(D[a, b]) or dd < D[a, b]:
                D[a, b] = dd
    m1 = np.nanmin(np.where(np.isnan(D), 99, D), axis=1) < 5.0
    m2 = np.nanmin(np.where(np.isnan(D), 99, D), axis=0) < 5.0
    if m1.sum() == 0 or m2.sum() == 0:
        return None, None, None
    sub = D[np.ix_(m1, m2)]; l1 = np.array(r1)[m1]; l2 = np.array(r2)[m2]
    f1, f2 = prot[c1], prot[c2]
    pm = ax.pcolormesh(np.arange(len(l2) + 1), np.arange(len(l1) + 1),
                       np.ma.masked_invalid(sub), cmap=fs.ramp(DARK[f1]),
                       vmin=2.5, vmax=6.0, linewidth=0, rasterized=True)
    bb = ax.get_window_extent()
    cell = min(bb.width / len(l2), bb.height / len(l1)) / ax.figure.dpi * 72
    md = float(np.clip(0.34 * cell, 0.7, 2.6)); mt = float(np.clip(0.62 * cell, 1.3, 3.6))
    lw = float(np.clip(0.14 * cell, 0.35, 0.95))
    for src in ([Bk] if ARk is None else [Bk, ARk]):
        for key, flip in (((c1, c2), False), ((c2, c1), True)):
            if key not in src.index:
                continue
            for _, row in src.loc[[key]].iterrows():
                a, b = (row.ri2, row.ri1) if flip else (row.ri1, row.ri2)
                if a not in l1 or b not in l2:
                    continue
                x = np.where(l2 == b)[0][0]; y = np.where(l1 == a)[0][0]
                if src is Bk and row.kind == 'saltbridge':
                    ax.add_patch(mpl.patches.Rectangle((x, y), 1, 1, fc='none', ec=fs.BOND,
                                                       lw=lw, zorder=5, joinstyle='round'))
                elif src is Bk:
                    ax.plot(x + .5, y + .5, marker='o', ms=md, color=fs.BOND, zorder=4)
                else:
                    ax.plot(x + .5, y + .5, marker='^', ms=mt, mfc='none', mec=fs.BOND,
                            mew=0.6, zorder=5)
    s1, s2 = max(1, len(l1) // tick_every), max(1, len(l2) // tick_every)
    ax.set_yticks(np.arange(len(l1))[::s1] + .5)
    ax.set_yticklabels(l1[::s1], fontsize=5.5, color=DARK[f1])
    ax.set_xticks(np.arange(len(l2))[::s2] + .5)
    ax.set_xticklabels(l2[::s2], fontsize=5.5, rotation=90, color=DARK[f2])
    ax.tick_params(length=2, colors='#888888')
    for sp in ax.spines.values():
        sp.set_color('#CCCCCC')
    return pm, l1, l2


tent = ' (tentative at this resolution)' if LOWCONF else ''
MAPLEG = [mpl.patches.Patch(fc='none', ec=fs.BOND, lw=0.8, label='salt bridge (outlined cell)' + tent),
          plt.Line2D([], [], marker='o', ls='', color=fs.BOND, ms=2.2, label='hydrogen bond' + tent),
          plt.Line2D([], [], marker='^', ls='', mfc='none', mec=fs.BOND, mew=0.6, ms=3.4,
                     label='aromatic stacking / cation–π')]
top3 = REP.sort_values('area', ascending=False).head(3)
fig, axs = plt.subplots(1, len(top3), figsize=(7.2, 2.95), squeeze=False)
axs = axs[0]
fig.subplots_adjust(left=0.065, right=0.975, top=0.90, bottom=0.30, wspace=0.52)
fig.canvas.draw()
pm = None
for ax, (_, r) in zip(axs, top3.iterrows()):
    pm, _, _ = draw_map(ax, r.c1, r.c2)
    ax.set_xlabel(f'residue, chain {r.c2}  ({prot[r.c2]})', color=DARK[prot[r.c2]])
    ax.set_ylabel(f'residue, chain {r.c1}  ({prot[r.c1]})', color=DARK[prot[r.c1]])
    ax.set_title(f'{r.ctg}  ·  {r.area:.0f} Å$^2$', fontsize=8, loc='left', pad=3, color=fs.INK)
    cb = fig.colorbar(pm, ax=ax, fraction=0.045, pad=0.03)
    cb.set_ticks([3, 4, 5, 6]); cb.ax.tick_params(labelsize=5.2, length=2, colors='#888888')
    for sp in cb.ax.spines.values():
        sp.set_color('#CCCCCC')
if pm is not None:
    cb.set_label('min. heavy-atom distance (Å)', fontsize=6)
fig.legend(handles=MAPLEG, frameon=False, fontsize=6.0, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.015), handletextpad=0.35, columnspacing=1.4)
save(fig, 'fig_contact_maps')

# ============================================================ area partition
TOT = S.area.sum()
catsum = S.groupby('ctg').area.sum().sort_values(ascending=False)
fig = plt.figure(figsize=(7.2, 3.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.52,
                      left=0.20, right=0.785, top=0.86, bottom=0.155)
ax = fig.add_subplot(gs[0, 0]); y = np.arange(len(catsum))[::-1]
ax.set_xlim(0, max(46.0, 100 * catsum.max() / TOT * 1.55)); ax.set_ylim(-0.8, len(catsum) - 0.2)
fig.canvas.draw()
for yy_, (c, v) in zip(y, catsum.items()):
    fs.sbarh(ax, yy_, 100 * v / TOT, left=0, height=0.60, color=CATCOL[c], round_pt=2.0, gap_pt=0)
    ax.text(100 * v / TOT + 1.0, yy_, f'{100*v/TOT:.1f}%   ({v:,.0f} Å$^2$)',
            va='center', fontsize=6, color='#555555')
ax.set_yticks(y); ax.set_yticklabels(catsum.index, fontsize=6.4)
ax.set_xlabel('share of all buried surface in the model (%)')
ax.set_title(f'a   Total buried surface = {TOT:,.0f} Å$^2$', fontsize=8, loc='left', pad=4, color=fs.INK)
tidy(ax)
ax2 = fig.add_subplot(gs[0, 1]); y2 = np.arange(len(ENTS))[::-1]
ax2.set_xlim(0, 100); ax2.set_ylim(-0.8, len(ENTS) - 0.2); fig.canvas.draw()
for yy_, e in zip(y2, ENTS):
    tot = BUDG[BUDG.entity_name == e][ENTS].mean(); pct = 100 * tot / max(tot.sum(), 1e-9)
    left = 0.0
    for part in ENTS:
        w = pct[part]
        if w > 0.4:
            fs.sbarh(ax2, yy_, w, left=left, height=0.60, color=PAL[part], round_pt=2.0, gap_pt=1.1)
            if w > 8:
                ax2.text(left + w / 2, yy_, f'{w:.0f}%', ha='center', va='center',
                         fontsize=6, color='#FFFFFF', fontweight='bold')
        left += w
    ax2.text(50, yy_ + 0.46, f'{tot.sum():,.0f} Å$^2$ total', ha='center', va='bottom',
             fontsize=5.8, color='#8A8A8A')
ax2.set_yticks(y2); ax2.set_yticklabels([f'{e}\nsubunit' for e in ENTS], fontsize=7)
for lbl, e in zip(ax2.get_yticklabels(), ENTS):
    lbl.set_color(DARK[e])
ax2.set_xlabel('share of that subunit’s buried surface (%)'); ax2.set_xticks([0, 25, 50, 75, 100])
ax2.set_title('b   How one subunit spends its interface budget', fontsize=8, loc='left', pad=4, color=fs.INK)
tidy(ax2)
hh = [mpl.patches.Patch(fc=PAL[p_], ec='none', label=f'with {p_}') for p_ in ENTS]
lg = fig.legend(handles=hh, frameon=False, fontsize=6.4, loc='upper left',
                bbox_to_anchor=(0.80, 0.60), handlelength=1.1, labelspacing=0.5,
                title='partner', title_fontsize=6.8); lg._legend_box.align = 'left'
save(fig, 'fig_area_partition')


# ============================================================ packing quality
# Buried area says how MUCH surface is hidden; Sc and the gap index say how WELL the two
# surfaces fit. A large interface with a low Sc and a wide gap is a loose contact, which
# is a different structural claim from a small, tightly interdigitated one.
#
# Reference bands. Lawrence & Colman (1993) report Sc ~0.70-0.76 for the interfaces of
# oligomeric proteins and ~0.70-0.74 for protease-inhibitor complexes, ~0.64-0.68 for
# antibody-antigen, and clearly lower values for contacts that are not biological. The
# bands below are those classes, drawn on the ORIGINAL sc scale. This grid
# implementation reads about 0.05 low against that scale (0.65 here for trypsin-BPTI,
# 2PTC), so a point sitting just under a band edge is at that edge after calibration;
# --sc-offset shifts the bands if you would rather read them on the measured scale.
SC_BANDS = [(0.70, 1.00, '#E8F0EA', 'tight'),
            (0.60, 0.70, '#F2F4EF', 'typical\ncomplex'),
            (0.45, 0.60, '#FAF3EA', 'loose'),
            (0.00, 0.45, '#F9EEEB', 'very\nloose')]
SC_REF = 0.70
SC_MAX = 0.75

if 'sc' in S.columns and S.sc.notna().any():
    SHP = S.dropna(subset=['sc'])
    off = args.sc_offset
    order = list(SHP.groupby('ctg').sc.median().sort_values(ascending=False).index)
    fig = plt.figure(figsize=(7.2, 4.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.16, 1.0], wspace=0.52,
                          left=0.135, right=0.665, top=0.905, bottom=0.295)

    def band_edges():
        for lo, hi, col, lab in SC_BANDS:
            yield max(0.0, lo - off), min(SC_MAX, hi - off), col, lab

    ax = fig.add_subplot(gs[0, 0])
    for lo, hi, col, lab in band_edges():
        if hi > lo:
            ax.axhspan(lo, hi, facecolor=col, edgecolor='none', zorder=0)
    for lo, hi, col, lab in band_edges():            # white rule between the classes
        if 0 < lo < SC_MAX:
            ax.axhline(lo, color='white', lw=1.1, zorder=1)
    if SC_REF - off < SC_MAX:
        ax.axhline(SC_REF - off, color='#6B6B6B', lw=0.9, ls=(0, (4, 2.5)), zorder=2)
    for i, c in enumerate(order):
        v = SHP[SHP.ctg == c].sc.values
        jit = np.linspace(-0.19, 0.19, len(v)) if len(v) > 1 else np.array([0.0])
        ax.scatter(i + jit, v, s=11, color=CATCOL[c], edgecolor='white',
                   linewidth=0.35, zorder=4)
        ax.plot([i - 0.30, i + 0.30], [np.median(v)] * 2, color=fs.INK, lw=1.3, zorder=5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=38, ha='right', fontsize=5.9)
    ax.set_xlim(-0.62, len(order) - 0.4 + 1.30)
    ax.set_ylim(0, SC_MAX)
    ax.set_yticks(np.arange(0, SC_MAX + 0.01, 0.15))
    ax.set_ylabel('shape complementarity  $S_\\mathrm{c}$')
    ax.set_title('a   How well the two surfaces fit', fontsize=8, loc='left',
                 pad=4, color=fs.INK)
    tidy(ax)
    for lo, hi, col, lab in band_edges():          # hung from the top of each zone
        if hi - lo > 0.045:
            ax.text(len(order) - 0.4 + 1.30 - 0.08, hi - 0.012, lab, fontsize=5.4,
                    color='#8A8A8A', ha='right', va='top', linespacing=1.25,
                    multialignment='right', zorder=2)

    ax2 = fig.add_subplot(gs[0, 1])
    for c in order:
        d = SHP[SHP.ctg == c]
        ax2.scatter(d.sc, d.gap_index, s=sc_size(d.area), color=CATCOL[c],
                    edgecolor='white', linewidth=0.4, zorder=3, label=c)
    ax2.set_xlim(0, SC_MAX)
    y0, y1 = ax2.get_ylim()
    ax2.set_ylim(y0 - (y1 - y0) * 0.11, y1)
    # The classes are defined on Sc ALONE, so they are drawn as a strip under the x axis
    # rather than as regions of the plane: the gap index is independent information and
    # a point is not reclassified by moving up or down.
    sy0, sy1 = ax2.get_ylim()[0], ax2.get_ylim()[0] + (y1 - y0) * 0.070
    for lo, hi, col, lab in band_edges():
        if hi > lo:
            ax2.add_patch(mpl.patches.Rectangle((lo, sy0), hi - lo, sy1 - sy0,
                                                facecolor=col, edgecolor='white',
                                                linewidth=1.1, zorder=2))
    if SC_REF - off < SC_MAX:
        ax2.plot([SC_REF - off] * 2, [sy0, sy1], color='#6B6B6B', lw=0.9,
                 ls=(0, (4, 2.5)), zorder=3)
    ax2.text(0.015, (sy0 + sy1) / 2, 'class on $S_\\mathrm{c}$ only', fontsize=4.8,
             color='#8A8A8A', ha='left', va='center', zorder=4)
    ax2.set_xlabel('shape complementarity  $S_\\mathrm{c}$')
    ax2.set_ylabel('gap index  (Å$^3$ per Å$^2$)')
    ax2.set_title('b   Tight fit, or wide gaps', fontsize=8, loc='left',
                  pad=4, color=fs.INK)
    ax2.annotate('tighter', xy=(0.985, 0.12), xycoords='axes fraction', ha='right',
                 va='bottom', fontsize=5.4, color='#9A9A9A')
    ax2.annotate('looser', xy=(0.02, 0.97), xycoords='axes fraction', ha='left',
                 va='top', fontsize=5.4, color='#9A9A9A')
    tidy(ax2)

    hh = [mpl.lines.Line2D([], [], marker='o', ls='', ms=4.2, mfc=CATCOL[c],
                           mec='white', mew=0.4, label=c) for c in order]
    lg = fig.legend(handles=hh, frameon=False, fontsize=6.2, loc='upper left',
                    bbox_to_anchor=(0.685, 0.935), labelspacing=0.52,
                    handletextpad=0.5, title='interface', title_fontsize=6.6)
    lg._legend_box.align = 'left'

    axk = fig.add_axes([0.690, 0.085, 0.30, 0.295]); axk.axis('off')
    axk.set_xlim(0, 1); axk.set_ylim(0, 1)
    axk.text(0.02, 0.985, 'buried area (Å$^2$)', fontsize=6.6, color=fs.INK,
             va='top', ha='left')
    for i, (rep, lab) in enumerate([(125, '< 250'), (500, '250 – 750'),
                                    (1000, '750 – 1250'), (1400, '> 1250')]):
        yy = 0.78 - i * 0.235
        axk.scatter([0.085], [yy], s=sc_size(rep), color='#BDBDBD',
                    edgecolor='white', linewidth=0.4)
        axk.text(0.20, yy, lab, fontsize=6.0, color=fs.INK, va='center', ha='left')

    fig.text(0.085, 0.030,
             'Classes are those of Lawrence & Colman (1993) on the original sc scale. '
             'This grid implementation reads about 0.10 below that scale, so the class '
             'names are pessimistic;\nuse the figure to rank interfaces within one run '
             'at one grid spacing.',
             fontsize=5.3, color='#8A8A8A', va='bottom', linespacing=1.5)
    save(fig, 'fig_packing')

# ============================================================ all interfaces
REPs = REP.sort_values('area', ascending=False).reset_index(drop=True)
ncol = 5; nrow = int(np.ceil(len(REPs) / ncol))
# Margins in inches, not in figure fractions: with one or two rows a fractional margin
# is only a few points and the panel titles and the footnote collide with the axes.
_FH = 1.42 * nrow + 0.86
fig, axs = plt.subplots(nrow, ncol, figsize=(7.2, _FH), squeeze=False)
fig.subplots_adjust(left=0.035, right=0.995, top=1 - 0.34 / _FH, bottom=0.52 / _FH,
                    wspace=0.42, hspace=0.80)
fig.canvas.draw()
for k in range(nrow * ncol):
    ax = axs[k // ncol][k % ncol]
    if k >= len(REPs):
        ax.axis('off'); continue
    r = REPs.iloc[k]
    pm, l1, l2 = draw_map(ax, r.c1, r.c2, tick_every=4)
    if pm is None:
        ax.axis('off'); continue
    ax.set_title(f'{r.c1}–{r.c2}  ·  {r.ctg}\n{r.area:.0f} Å$^2$', fontsize=5.6,
                 loc='left', pad=2, color=fs.INK)
    for lbl in ax.get_yticklabels():
        lbl.set_color(DARK[prot[r.c1]]); lbl.set_fontsize(4.2)
    for lbl in ax.get_xticklabels():
        lbl.set_color(DARK[prot[r.c2]]); lbl.set_fontsize(4.2)
# The caveat goes on its own line: added to the end, the footnote runs off both edges.
_foot = ('interface residues only; y = first chain, x = second · outlined cell = salt bridge'
         '  ● hydrogen bond  △ aromatic stacking / cation–π · hue = y-axis protein')
if LOWCONF:
    _foot += f'\nbond assignments {LOWTXT}'
fig.text(0.5, 0.045 / _FH, _foot, ha='center', va='bottom', fontsize=5.4,
         color='#666666', linespacing=1.4)
save(fig, 'figS1_contact_maps_all')
print('ALL FIGURES DONE')

# ============================================================ interaction networks
# One panel per SYMMETRY-UNIQUE interface, drawn from its representative chain pair, so
# a 30-chain assembly with 105 contacts produces a few readable diagrams instead of a
# hundred near-identical ones. Residues are nodes; every specific interaction is an
# edge, coloured as in the footprint track. Covalent links are drawn whatever their
# number, because one disulfide matters more than ten van der Waals contacts.
LINKS = pd.read_csv('interface_links.csv') if os.path.exists('interface_links.csv') \
    else pd.DataFrame(columns=['kind', 'bonded', 'c1', 'resi1', 'resn1', 'c2', 'resi2', 'resn2'])
COVAL = fs.COVALENT
PHOBCOL = '#C8C8C8'
PHOB = pd.read_csv('interface_hydrophobic.csv') if os.path.exists('interface_hydrophobic.csv') \
    else pd.DataFrame(columns=['c1', 'c2', 'ri1', 'rn1', 'ri2', 'rn2', 'n_contacts'])


def edges_for(c1, c2):
    """Every interaction between this chain pair: specific, covalent and hydrophobic."""
    out, cov, phob = [], [], []
    for _, r in B[((B.c1 == c1) & (B.c2 == c2)) | ((B.c1 == c2) & (B.c2 == c1))].iterrows():
        a = (r.ri1, r.rn1, r.ri2, r.rn2) if r.c1 == c1 else (r.ri2, r.rn2, r.ri1, r.rn1)
        out.append((int(a[0]), a[1], int(a[2]), a[3], KINDMAP.get(r.kind, 'hbond'),
                    float(r.dist)))
    if len(AR):
        for _, r in AR[((AR.c1 == c1) & (AR.c2 == c2)) |
                       ((AR.c1 == c2) & (AR.c2 == c1))].iterrows():
            a = (r.ri1, r.rn1, r.ri2, r.rn2) if r.c1 == c1 else (r.ri2, r.rn2, r.ri1, r.rn1)
            out.append((int(a[0]), a[1], int(a[2]), a[3], KINDMAP.get(r.kind, 'stack'),
                        float(r.dist)))
    if len(LINKS):
        L = LINKS[(LINKS.kind.isin(['disulfide', 'isopeptide'])) & (LINKS.bonded == True)]
        for _, r in L[((L.c1 == c1) & (L.c2 == c2)) | ((L.c1 == c2) & (L.c2 == c1))].iterrows():
            a = (r.resi1, r.resn1, r.resi2, r.resn2) if r.c1 == c1 else \
                (r.resi2, r.resn2, r.resi1, r.resn1)
            cov.append((int(a[0]), a[1], int(a[2]), a[3], r.kind, float(r.distance)))
    if len(PHOB):
        P = PHOB[((PHOB.c1 == c1) & (PHOB.c2 == c2)) | ((PHOB.c1 == c2) & (PHOB.c2 == c1))]
        for _, r in P[P.n_contacts >= args.phobic_min].iterrows():
            a = (r.ri1, r.rn1, r.ri2, r.rn2) if r.c1 == c1 else (r.ri2, r.rn2, r.ri1, r.rn1)
            phob.append((int(a[0]), a[1], int(a[2]), a[3], 'phobic', int(r.n_contacts)))
    return out, cov, phob


DMIN, DMAX, LWMAX, LWMIN = 2.4, 4.2, 1.9, 0.55


def lw_for_distance(d):
    """Shorter bond, thicker line. Bond strength falls off steeply with distance, so a
    2.6 A hydrogen bond should not look the same as a 3.4 A one. The mapping is linear
    between DMIN and DMAX and is drawn as a tapered wedge in the figure legend."""
    return float(np.interp(d, [DMIN, DMAX], [LWMAX, LWMIN]))


def draw_network(ax, c1, c2, ctg, area, eds, cov, phob):
    allm = eds + cov + phob
    left = sorted({(e[0], e[1]) for e in allm})
    right = sorted({(e[2], e[3]) for e in allm})
    n = max(len(left), len(right), 1)
    fsz = 6.0 if n <= 12 else (5.2 if n <= 20 else 4.4)
    ypos = lambda lst: {k: (1.0 - (i + 0.5) / max(len(lst), 1)) for i, k in enumerate(lst)}
    yl, yr = ypos(left), ypos(right)
    xl, xr = 0.20, 0.80
    for a_i, a_n, b_i, b_n, _k, nc in phob:
        ax.plot([xl, xr], [yl[(a_i, a_n)], yr[(b_i, b_n)]], lw=float(np.interp(nc, [3, 12], [0.5, 1.5])),
                color=PHOBCOL, zorder=0, solid_capstyle='round')
    for a_i, a_n, b_i, b_n, kind, d in eds:
        ax.plot([xl, xr], [yl[(a_i, a_n)], yr[(b_i, b_n)]], lw=lw_for_distance(d),
                color=fs.INTCOL.get(kind, '#999999'), zorder=1, solid_capstyle='round',
                alpha=0.92)
    for a_i, a_n, b_i, b_n, kind, d in cov:
        ax.plot([xl, xr], [yl[(a_i, a_n)], yr[(b_i, b_n)]], lw=2.6, color=COVAL,
                zorder=2, solid_capstyle='round')
    for lst, x, yy, ch, ha in [(left, xl, yl, c1, 'right'), (right, xr, yr, c2, 'left')]:
        col = ENVCOL.get(env_of.get(ch), '#999999')
        for k in lst:
            ax.scatter([x], [yy[k]], s=64, color=col, edgecolor='white', linewidth=0.7,
                       zorder=3)
            ax.text(x + (-0.045 if ha == 'right' else 0.045), yy[k],
                    f'{ONE.get(k[1], k[1])}{k[0]}', fontsize=fsz, ha=ha, va='center',
                    color=fs.INK, zorder=4)
    ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.06, 1.06); ax.axis('off')
    # chain letter AND its symmetry environment: with several conformers of one protein,
    # the chain alone does not say which fold is being shown
    for x, ch in ((xl, c1), (xr, c2)):
        ax.text(x, 1.012, f'{ch}  ({"/".join(env_of.get(ch, ch))})', fontsize=6.0,
                ha='center', va='bottom', color=fs.INK, fontweight='bold')
    ax.set_title(f'{ctg}\n{area:.0f} Å$^2$  ·  {len(eds) + len(cov)} bonds'
                 + (f'  ·  {len(phob)} apolar' if phob else ''),
                 fontsize=6.5, loc='center', pad=12, color=fs.INK, linespacing=1.45)


panels = []
for _, r in REP.iterrows():
    eds, cov, phob = edges_for(r.c1, r.c2)
    if len(eds) + len(cov) + len(phob) >= args.network_min or cov:
        panels.append((r.c1, r.c2, r.ctg, r.area, eds, cov, phob,
                       len(eds) + len(cov) + 0.01 * len(phob)))
panels.sort(key=lambda p: -p[7])
if panels:
    per = max(1, args.network_per_page)
    pages = [panels[i:i + per] for i in range(0, len(panels), per)]
    for pi, page in enumerate(pages):
        ncol = min(3, len(page))
        nrow = int(np.ceil(len(page) / ncol))
        tallest = max(max(len({(e[0], e[1]) for e in p[4] + p[5] + p[6]}),
                          len({(e[2], e[3]) for e in p[4] + p[5] + p[6]})) for p in page)
        rh = min(3.4, max(1.5, 0.17 * tallest + 0.7))
        FH = rh * nrow + 1.80
        fig = plt.figure(figsize=(7.2, FH))
        gs = fig.add_gridspec(nrow, ncol, left=0.03, right=0.97,
                              top=1 - 1.55 / FH, bottom=0.26 / FH,
                              hspace=0.34, wspace=0.26)
        for k, p in enumerate(page):
            draw_network(fig.add_subplot(gs[k // ncol, k % ncol]), *p[:7])
        hh = [mpl.patches.Patch(fc=fs.INTCOL[k], ec='none', label=t)
              for k, t in [('salt', 'salt bridge'), ('hbond', 'hydrogen bond'),
                           ('stack', 'aromatic stacking'), ('catpi', 'cation–π')]
              if any(e[4] == k for p in page for e in p[4])]
        if any(p[5] for p in page):
            hh.append(mpl.patches.Patch(fc=COVAL, ec='none', label='disulfide / covalent'))
        if any(p[6] for p in page):
            hh.append(mpl.patches.Patch(fc=PHOBCOL, ec='none',
                                        label=f'hydrophobic (≥{args.phobic_min} contacts)'))
        fig.legend(handles=hh, frameon=False, fontsize=6.3, loc='upper center',
                   ncol=len(hh), bbox_to_anchor=(0.5, 1 - 0.40 / FH),
                   handlelength=1.1, columnspacing=1.4, handletextpad=0.4)
        # key for the line width: a tapered wedge drawn with the SAME mapping the edges
        # use, so the reader can read a bond length off the thickness of any line.
        KW, KH = 1.75, 0.30                       # inches
        axk = fig.add_axes([0.5 - KW / 2 / 7.2, 1 - 0.80 / FH, KW / 7.2, KH / FH])
        axk.set_xlim(DMIN, DMAX); axk.set_ylim(-1, 1); axk.axis('off')
        dd = np.linspace(DMIN, DMAX, 200)
        half = np.array([lw_for_distance(x) for x in dd]) / 2.0      # points
        pt2y = 2.0 / (KH * 72.0)                  # ylim span per point of height
        axk.fill_between(dd, -half * pt2y, half * pt2y, facecolor='#9A9A9A',
                         edgecolor='none', zorder=2)
        for x in (2.5, 3.0, 3.5, 4.0):
            axk.text(x, -0.95, f'{x:g}', fontsize=5.4, ha='center', va='bottom',
                     color=fs.INK, alpha=0.75)
        axk.text(DMIN - 0.06, 0, 'bond length (Å)', fontsize=5.8, ha='right',
                 va='center', color=fs.INK)
        ttl = 'Residue-level interaction networks of the symmetry-unique interfaces'
        if len(pages) > 1:
            ttl += f'  ({pi + 1} of {len(pages)})'
        fig.text(0.03, 1 - 0.13 / FH, ttl, fontsize=8.5, ha='left', va='top', color=fs.INK)
        save(fig, 'fig_interaction_networks' if len(pages) == 1
             else f'fig_interaction_networks_{pi + 1}')
    print(f'  {len(panels)} of {len(REP)} symmetry-unique interfaces drawn '
          f'(>= {args.network_min} interactions, or any covalent link)', flush=True)



# ============================================================ ChimeraX colouring
def write_chimerax(bsa_min):
    """Write a structure that carries the colouring, plus two small ChimeraX scripts.

      model_interfaces.pdb        OPEN THIS. B-factor = conservation grade (1-9) when a
                                  conservation file was given, otherwise buried area in
                                  A^2. Occupancy = index of the symmetry environment, so
                                  the figure palette can be reproduced from this file
                                  alone. REMARK lines carry the key and the commands.
      chimerax_colour_only.cxc    palette by symmetry environment, matching the
                                  footprint figures.
      chimerax_colour_by_interaction.cxc
                                  palette by the kind of interaction each residue makes,
                                  matching the interaction track in those figures.

    Neither script opens a structure. A .cxc that opens a file only works when that file
    sits beside it; a colour-only script works from any folder once a model is loaded,
    which is the one behaviour that never surprises anyone.
    """
    BUR = pd.read_csv('residue_burial_assembly.csv')
    sel = BUR[BUR.buried >= bsa_min]
    bur_of = {(r.chain, int(r.resi)): float(r.buried) for _, r in BUR.iterrows()}
    envs_order = sorted(set(env_of.values()))
    env_idx = {e: i + 1 for i, e in enumerate(envs_order)}
    have_cons = bool(CONS)

    def ranges(nums):
        out, nums = [], sorted(int(n) for n in nums)
        for n in nums:
            if out and n == out[-1][1] + 1:
                out[-1][1] = n
            else:
                out.append([n, n])
        return ','.join(f'{a}-{b}' if b > a else f'{a}' for a, b in out)

    for c in model:
        gr = CONS.get(prot.get(c.id))
        for r in c:
            buried = bur_of.get((c.id, r.id[1]), 0.0)
            bval = float(gr.get(r.id[1], 0)) if gr else buried
            occ = float(env_idx.get(env_of.get(c.id), 0)) if buried >= bsa_min else 0.0
            for a in r:
                a.set_bfactor(round(bval, 2))
                a.set_occupancy(occ)

    bcol = 'ConSurf-style conservation grade 1-9 (0 = not scored)' if have_cons \
        else 'buried surface area in A^2'
    remarks = [
        'REMARK   1 Written by 03_make_figures.py (interface analysis pipeline).',
        f'REMARK   1 B-FACTOR  = {bcol}',
        'REMARK   1 OCCUPANCY = index of the symmetry environment, 0 if the residue',
        f'REMARK   1             buries less than {bsa_min:g} A^2 against another chain.',
        'REMARK   1 This file is self-contained: open it in ChimeraX and paste',
        'REMARK   1 either command below. No other file is needed.',
        'REMARK   1',
        'REMARK   1   continuous:  color bfactor palette "#D9D9D9:#F5C26B:#B33A3A"',
        'REMARK   1   by environment:',
        'REMARK   1   ' + 'color byattribute occupancy palette ' +
        '/'.join(['0,#D9D9D9'] + [f'{env_idx[e]},{mpl.colors.to_hex(ENVCOL.get(e, "#999999"))}'
                                  for e in envs_order]),
        'REMARK   1',
        'REMARK   1 environment key:']
    for ent in ENTS:
        for e in sorted({x for ch, x in env_of.items() if prot[ch] == ent}):
            chs = ','.join(sorted(ch for ch in env_of if env_of[ch] == e))
            remarks.append(f'REMARK   1   {env_idx[e]:>2}  {ent:<12} {e:<8} '
                           f'{mpl.colors.to_hex(ENVCOL.get(e, "#999999"))}  chains {chs}')

    struct_name = 'model_interfaces.pdb'
    try:
        io = PDBIO(); io.set_structure(model); io.save(struct_name)
    except Exception:
        struct_name = 'model_interfaces.cif'
        io = MMCIFIO(); io.set_structure(model); io.save(struct_name)
    if struct_name.endswith('.pdb'):
        body = open(struct_name).read()
        open(struct_name, 'w').write('\n'.join(remarks) + '\n' + body)
    print(f'wrote {struct_name}  (B-factor = {bcol}; occupancy = environment index)',
          flush=True)

    colour_cmds = ['hide atoms', 'show cartoons', 'color all #D9D9D9',
                   'set bgColor white', 'lighting soft', '']
    for c in sorted(set(BUR.chain)):
        rs = sel[sel.chain == c].resi
        if not len(rs):
            continue
        col = mpl.colors.to_hex(ENVCOL.get(env_of.get(c), '#999999'))
        # no trailing comments: ChimeraX reads a '#' mid-line as a model specifier
        colour_cmds.append(f'color /{c}:{ranges(rs)} {col}')
    key = ['', '# environment key, matching the footprint figures:']
    for ent in ENTS:
        for e in sorted({x for ch, x in env_of.items() if prot[ch] == ent}):
            key.append(f'#   {ent:<12} {e:<8} '
                       f'{mpl.colors.to_hex(ENVCOL.get(e, "#999999"))}')

    head2 = [f'# Interface colouring for {os.path.basename(MODEL)}',
             '#',
             '# Colour commands only: this script opens nothing, so it works from any',
             f'# folder. Open your structure first (or {struct_name}), then run this.',
             '#',
             f'# Residues burying at least {bsa_min:g} A^2 are coloured by symmetry',
             '# environment, exactly as in the footprint figures; the rest stay grey.',
             '']
    open('chimerax_colour_only.cxc', 'w').write(
        '\n'.join(head2 + colour_cmds + key) + '\n')

    # ---- second script: colour by the kind of interaction a residue makes ----
    # A residue can make several kinds at once, so one colour per residue has to be a
    # choice. The most specific and most structurally constraining wins: a salt bridge
    # over cation-pi, over stacking, over a hydrogen bond. Residues that bury surface
    # but make none of these are left in a neutral tone - that is the "non-specific
    # contact" class, and seeing how much of an interface it covers is the point.
    PRIORITY = [('covalent', 'disulfide / covalent'), ('salt', 'salt bridge'),
                ('catpi', 'cation-pi'), ('stack', 'aromatic stacking'),
                ('hbond', 'hydrogen bond')]
    NONSPEC = '#C8C8C8'
    covres = set()
    if len(LINKS):
        _L = LINKS[(LINKS.kind.isin(['disulfide', 'isopeptide'])) & (LINKS.bonded == True)]
        for _, r in _L.iterrows():
            covres.add((r.c1, int(r.resi1))); covres.add((r.c2, int(r.resi2)))
    by_class = collections.defaultdict(lambda: collections.defaultdict(list))
    for _, r in sel.iterrows():
        key = (r.chain, int(r.resi))
        cnt = INTRES.get(key)
        cls = 'covalent' if key in covres else \
            next((k for k, _ in PRIORITY[1:] if cnt and cnt.get(k)), 'none')
        by_class[cls][r.chain].append(int(r.resi))
    # a covalently linked residue may fall below the area cut-off and still matter
    for ch, ri in covres:
        if ri not in by_class['covalent'][ch]:
            by_class['covalent'][ch].append(ri)
    icmds = ['hide atoms', 'show cartoons', 'color all #EDEDED',
             'set bgColor white', 'lighting soft', '']
    order = [k for k, _ in PRIORITY][::-1] + ['none']      # specific drawn last, on top
    for cls in order:
        if cls not in by_class:
            continue
        col = NONSPEC if cls == 'none' else (fs.COVALENT if cls == 'covalent'
                                             else mpl.colors.to_hex(fs.INTCOL[cls]))
        lab = 'buried, no specific interaction' if cls == 'none' else \
            dict(PRIORITY)[cls]
        icmds.append(f'# {lab}')
        for c, rs in sorted(by_class[cls].items()):
            icmds.append(f'color /{c}:{ranges(rs)} {col}')
        icmds.append('')
    # Salt bridges are the one class where the chemistry is worth seeing atom by atom:
    # show the side chains and recolour the heteroatoms by element, so the charged
    # groups read directly as blue nitrogen against red oxygen. "byhetero" leaves carbon
    # at the class colour, which keeps the residue recognisable as a salt bridge.
    if 'covalent' in by_class:
        icmds.append('# disulfide / covalent side chains, by element (S yellow)')
        for c, rs in sorted(by_class['covalent'].items()):
            icmds.append(f'show /{c}:{ranges(rs)} & sidechain atoms')
            icmds.append(f'color /{c}:{ranges(rs)} & sidechain byhetero')
        icmds.append('')
    if 'salt' in by_class:
        icmds.append('# salt-bridge side chains, heteroatoms by element (N blue, O red)')
        for c, rs in sorted(by_class['salt'].items()):
            # "sidechain" includes CA; "sideonly" does not, which leaves the stick
            # floating with no bond back to the cartoon.
            icmds.append(f'show /{c}:{ranges(rs)} & sidechain atoms')
            icmds.append(f'color /{c}:{ranges(rs)} & sidechain byhetero')
        icmds.append('')
    ikey = ['# key (most specific interaction per residue wins):']
    for cls, lab in PRIORITY:
        if cls in by_class:
            n = sum(len(v) for v in by_class[cls].values())
            cc = fs.COVALENT if cls == 'covalent' else mpl.colors.to_hex(fs.INTCOL[cls])
            ikey.append(f'#   {lab:<22} {cc}  {n} residues')
    if 'none' in by_class:
        n = sum(len(v) for v in by_class['none'].values())
        ikey.append(f'#   {"buried, no specific":<22} {NONSPEC}  {n} residues')
    ikey.append('#   salt-bridge side chains are shown as sticks with N blue and O red')
    ihead = [f'# Interface colouring BY INTERACTION TYPE for {os.path.basename(MODEL)}',
             '#',
             '# Colour commands only: open your structure first, then run this.',
             '# Companion to the interaction track in the footprint figures, so the same',
             '# residue carries the same colour in 2D and in 3D.',
             '']
    open('chimerax_colour_by_interaction.cxc', 'w').write(
        '\n'.join(ihead + icmds + ikey) + '\n')
    print(f'wrote chimerax_colour_only.cxc and chimerax_colour_by_interaction.cxc '
          f'({len(sel)} residues >= {bsa_min:g} A^2 coloured)', flush=True)


write_chimerax(args.bsa_min)
