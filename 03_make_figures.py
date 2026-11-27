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
ap.add_argument('--solvation-track', dest='solvation_track', action='store_true',
                default=True,
                help='per-residue Eisenberg solvation ΔG strip under each footprint. On '
                     'by default; use --no-solvation-track to leave it out.')
ap.add_argument('--no-solvation-track', dest='solvation_track', action='store_false',
                help='drop the ΔG strip from the footprint figures.')
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
ap.add_argument('--partner-colours', default='auto', choices=['auto', 'molecule', 'subunit'],
                help='colour the partner segments of the budget and footprint figures by '
                     'partner MOLECULE, or by partner SUBUNIT (symmetry environment) in shades '
                     'of the molecule colour, so e.g. the two strands of a DNA are told apart. '
                     'auto (default): budgets by subunit; footprints by subunit when the model '
                     'has at most 6 environments, otherwise by molecule.')
ap.add_argument('--bsa-min', type=float, default=10.0, metavar='A2',
                help='minimum buried area for a residue to be coloured in the ChimeraX '
                     'script; everything below stays grey (default 10 A2)')
ap.add_argument('--no-render', dest='render', action='store_false',
                help='skip the two licorice renders of the whole model (coloured by burying '
                     'partner and by interaction type)')
ap.add_argument('--render-view', default='62,20', metavar='ELEV,AZIM',
                help='camera for those renders, in degrees: elevation 0 = side view with the '
                     'symmetry axis up, 90 = looking down the axis (default 62,20, the view '
                     'used by 05_assemblies.py)')
args = ap.parse_args()
MODEL = args.model
PERLINE = max(20, args.per_line)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog
RPT = runlog.Report('03', model=MODEL)
with RPT.guard('read step 01-02 outputs', ok_detail=None, fatal=True):
    if not os.path.exists('run_config.json'):
        raise FileNotFoundError('run_config.json is missing: run 01_interface_analysis.py '
                                'first, in this same folder')
    CFG = json.load(open('run_config.json'))
RPT.ok('read step 01-02 outputs', f'{CFG.get("n_chains")} chains, {CFG.get("point_group")}')
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
# bound ligands named with --ligand-names in step 01 are labelled by that name
LIGNAME = dict(CFG.get('ligand_names') or {})
LIGCOL = dict(CFG.get('ligand_colours') or {})
ONE.update({k: v + ' ' for k, v in LIGNAME.items()})
CLSORD = ['charged', 'polar', 'aromatic', 'aliphatic', 'Gly / Pro', 'nucleotide',
          'ligand', 'other']
BLAB = {'salt': 'salt bridge', 'hbond': 'hydrogen bond',
        'stack': 'aromatic stacking', 'catpi': 'cation–π'}
LOWCONF = CFG.get('bond_confidence') in ('low', 'unknown')
# When no --resolution was given the confidence is 'unknown', and there is no number
# to quote; say so rather than printing 'None Å'.
LOWTXT = (f'tentative at {CFG["resolution"]} Å' if CFG.get('resolution')
          else 'tentative — resolution not given')


def env_str(env, per_line=12, sep='/'):
    """Chain list of a symmetry environment, wrapped onto as many lines as it needs.

    A homo-oligomer puts every copy in ONE environment, so this list can be 36 chains
    long. Written on a single line it runs off the figure and over its neighbours;
    wrapping keeps the information instead of truncating it."""
    ch = list(env)
    return '\n'.join(sep.join(ch[i:i + per_line]) for i in range(0, len(ch), per_line))


def env_tag(env, lim=8):
    """One-line form, for a running headline that cannot grow downwards."""
    ch = list(env)
    return '/'.join(ch) if len(ch) <= lim else f'{ch[0]}…{ch[-1]}'

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

# two entities may share a name (parts of one molecule matched to the same FASTA record)
ENTS = list(dict.fromkeys(e['name'] for e in CFG['entities']))
PALS = fs.assign_palettes(ENTS)
_cfgcol = CFG.get('colours') or {}
for _k, _v in _cfgcol.items():                     # colours from the --proteins FASTA
    try:
        if _k in PALS:
            PALS[_k] = fs.family_from_hex(_v, _k)
    except Exception as _exc:
        RPT.default('protein colours', f'{_k}: colour {_v} could not be used ({_exc}); default')
if _cfgcol and not RPT._has('protein colours'):
    RPT.ok('protein colours', 'from the FASTA: ' + ', '.join(f'{k} {v}' for k, v in _cfgcol.items()))
if args.colours:                                   # user-pinned colours, by protein name
    for item in args.colours.split(','):
        if '=' not in item:
            continue
        k, v = item.split('=', 1)
        k = k.strip()
        if k in PALS:
            PALS[k] = fs.family_from_hex(v, k)
        else:
            RPT.default('protein colours (--colours)',
                      f'no protein called {k!r} (known: {", ".join(ENTS)}); default kept')
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
# Subunit colours: each symmetry environment (a chain, or a set of symmetry copies) gets a
# shade of its molecule's colour -- the scheme of the contact-topology plot -- so partner
# segments can say WHICH subunit buries a surface, e.g. which strand of a DNA.
ENVORD, ENVCOL, ENVLAB = [], {}, {}
for _e in ENTS:
    _mine = sorted({env_of[c] for c in GEO.index if GEO.loc[c, 'entity_name'] == _e})
    for _i, _x in enumerate(_mine):
        ENVORD.append(_x)
        ENVCOL[_x] = fs.shade_of(PALS[_e], _i, len(_mine))
        ENVLAB[_x] = f'{_e} {env_tag(_x)}' if len(_mine) > 1 else _e
ENV_ENT = {x: GEO.loc[x[0], 'entity_name'] for x in ENVORD}
_split = any(sum(1 for x in ENVORD if ENV_ENT[x] == e) > 1 for e in ENTS)
SUB_BUDGET = args.partner_colours != 'molecule' and _split
SUB_FOOT = (not HOMO and _split and 'penv' in ENV.columns
            and (args.partner_colours == 'subunit'
                 or (args.partner_colours == 'auto' and len(ENVORD) <= 6)))
if SUB_FOOT:
    PARTKEY = 'penv'
    PARTCOL = dict(ENVCOL)
PARTLAB = dict(ENVLAB) if SUB_FOOT else {}
# buried area of every chain, split by partner subunit (both directions of each pair)
_bp = [(r.c1, env_of[r.c2], r.area) for r in S.itertuples()] + \
      [(r.c2, env_of[r.c1], r.area) for r in S.itertuples()]
BPE = (pd.DataFrame(_bp, columns=['chain', 'penv', 'area'])
       .pivot_table(index='chain', columns='penv', values='area', aggfunc='sum')
       .reindex(index=list(GEO.index), columns=ENVORD).fillna(0.0))
_out = BPE.copy(); _out.columns = [ENVLAB[x] for x in BPE.columns]
_out.insert(0, 'environment', [env_of[c] for c in _out.index])
_out.insert(1, 'molecule', [GEO.loc[c, 'entity_name'] for c in _out.index])
_out.to_csv('subunit_budget_by_partner_subunit.csv', index_label='chain')
if SUB_BUDGET:
    BKEYS, BCOL, BLAB_ = ENVORD, ENVCOL, ENVLAB
    BUD = BPE
else:
    BKEYS, BCOL, BLAB_ = ENTS, PAL, {e: e for e in ENTS}
    BUD = BPE.T.groupby(lambda x: ENV_ENT[x]).sum().T.reindex(columns=ENTS).fillna(0.0)
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
    seqs = path if isinstance(path, list) else read_fasta(path)
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


def read_alignment(path):
    """[(name, aligned sequence)] from aligned FASTA, A3M, Clustal or Stockholm.

    A3M (ColabFold, HHblits) marks insertions relative to the query in lower case; those
    are dropped so every row has the query's columns. Clustal and Stockholm are read with
    Bio.AlignIO when the file does not look like FASTA."""
    head = open(path, encoding='utf-8', errors='replace').read(2000).lstrip()
    if head.startswith('>'):
        seqs = [(n, s) for n, s in read_fasta(path) if not n.startswith('#')]
        if seqs and len({len(s) for _, s in seqs}) > 1 and any(
                c.islower() for _, s in seqs[1:] for c in s):
            seqs = [(n, ''.join(c for c in s if not c.islower() and c != '.'))
                    for n, s in seqs]
        return seqs
    from Bio import AlignIO
    for fmt in ('stockholm', 'clustal', 'phylip-relaxed'):
        try:
            aln = AlignIO.read(path, fmt)
            return [(r.id, str(r.seq)) for r in aln]
        except Exception:
            continue
    return []


def _model_seq(entity):
    """(one-letter sequence, residue numbers) of `entity`: the union, by residue number,
    of all its chains, so a molecule modelled in pieces (e.g. two halves of one DNA, or a
    protein split over two chains) maps onto its alignment as a whole."""
    from Bio.SeqUtils import seq1
    seen = {}
    for c in GEO.index:
        if prot[c] == entity:
            for r in model[c]:
                if r.id[0] == ' ' and r.id[1] not in seen:
                    seen[r.id[1]] = r.get_resname()
    nums = sorted(seen)
    return ''.join(seq1(seen[n]) for n in nums), nums


def _map_reference(ref, mseq):
    """{index in ref: index in mseq} from a global alignment with free end gaps, plus the
    identity over aligned pairs. A full-length reference against a model with missing
    termini or loops maps correctly instead of being shifted."""
    from Bio import Align
    from Bio.Align import substitution_matrices
    al = Align.PairwiseAligner()
    al.mode = 'global'
    al.substitution_matrix = substitution_matrices.load('BLOSUM62')
    al.open_gap_score, al.extend_gap_score = -10, -0.5
    try:                                   # Biopython >= 1.85 names
        al.end_insertion_score = 0.0
        al.end_deletion_score = 0.0
    except AttributeError:                 # older Biopython
        al.target_end_gap_score = al.query_end_gap_score = 0.0
    clean = lambda s: ''.join(c if c in 'ACDEFGHIKLMNPQRSTVWY' else 'X' for c in s.upper())
    a = al.align(clean(ref), clean(mseq))[0]
    mp, same = {}, 0
    for (r0, r1), (m0, m1) in zip(*a.aligned):
        for k in range(r1 - r0):
            mp[r0 + k] = m0 + k
            same += ref[r0 + k].upper() == mseq[m0 + k]
    return mp, (same / len(mp) if mp else 0.0)


def apply_msa(spec):
    """Map --msa grades onto residue numbers, per protein, through a sequence alignment."""
    out = {}
    if not spec:
        if not args.conservation:
            RPT.skip('conservation', 'no alignment given (--msa or --conservation): '
                                     'footprints drawn without a conservation track')
        else:
            RPT.ok('conservation', f'from {os.path.basename(args.conservation)}')
        return out
    items = []
    for part in spec.split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            items.append((k.strip(), v.strip()))
        else:
            items.append((None, part.strip()))
    notes, bad = [], []
    for pname, path in items:
        try:
            if not os.path.exists(path):
                bad.append(f'{path}: file not found'); continue
            seqs = read_alignment(path)
            if len(seqs) < 2 or len({len(s) for _, s in seqs}) > 1:
                bad.append(f'{os.path.basename(path)}: not an alignment of >= 2 sequences '
                           'of equal length'); continue
            gr = grades_from_msa(seqs)            # no temporary file in the folder
            if gr is None:
                bad.append(f'{os.path.basename(path)}: could not be scored'); continue
            ref = ''.join(c for c in seqs[0][1] if c.upper() in 'ACDEFGHIKLMNPQRSTVWY')
            cands = [pname] if pname in ENTS else ENTS
            best = None
            for e in cands:
                mseq, rn = _model_seq(e)
                if not mseq:
                    continue
                mp, ident = _map_reference(ref, mseq)
                cover = len(mp) / max(len(mseq), 1)
                score = ident * cover
                if best is None or score > best[0]:
                    best = (score, e, mp, rn, ident, cover)
            if best is None or best[4] < 0.30 or best[5] < 0.50:
                why = (f'best match {best[1]} at {100*best[4]:.0f} % identity over '
                       f'{100*best[5]:.0f} % of the model' if best else 'no protein to match')
                bad.append(f'{os.path.basename(path)}: first sequence does not match the '
                           f'model ({why})'); continue
            _, e, mp, rn, ident, cover = best
            if pname and pname not in ENTS:
                notes.append(f'"{pname}" is not a protein name in this run; matched by '
                             f'sequence to {e}')
            out[e] = {int(rn[mp[k]]): int(gr[k]) for k in mp if k < len(gr)}
            notes.append(f'{e}: {len(seqs)} sequences, {len(out[e])} residues graded '
                         f'({100*ident:.0f} % identity to the model)')
        except Exception as ex:
            bad.append(f'{os.path.basename(path)}: {type(ex).__name__}: {ex}')
    if out and not bad:
        RPT.ok('conservation', '; '.join(notes) + ' — entropy grades, not ConSurf')
    elif out:
        RPT.warn('conservation', '; '.join(notes) + '. NOT used: ' + '; '.join(bad))
    else:
        RPT.fail('conservation', 'no alignment could be used, plotted without conservation: '
                 + '; '.join(bad))
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


def _wrap(t, width=18):
    """Break a long label after '–', '/', '_' or a space so user-chosen protein names of any
    length stay inside their axes. Labels that already have line breaks are left alone."""
    if not isinstance(t, str) or len(t) <= width or '$' in t:
        return t
    if '\n' in t:
        return '\n'.join(_wrap(x, width) for x in t.split('\n'))
    toks = re.findall(r'[^–/_ ]+[–/_ ]?|[–/_ ]', t)
    lines, cur = [], ''
    for tok in toks:
        if cur and len(cur) + len(tok.rstrip()) > width:
            lines.append(cur.rstrip()); cur = ''
        cur += tok
    if cur.strip():
        lines.append(cur.rstrip())
    return '\n'.join(lines)


def fit_labels(fig):
    """Wrap over-long tick labels and legend entries in every axes of `fig`."""
    fig.canvas.draw()
    for ax in fig.axes:
        for an in ('x', 'y'):
            labs = getattr(ax, f'get_{an}ticklabels')()
            texts = [t.get_text() for t in labs]
            if not any(len(t) > 18 and '$' not in t for t in texts):
                continue
            props = [(t.get_color(), t.get_fontsize(), t.get_rotation(), t.get_ha(),
                      t.get_rotation_mode()) for t in labs]
            getattr(ax, f'set_{an}ticks')(getattr(ax, f'get_{an}ticks')())
            new = getattr(ax, f'set_{an}ticklabels')([_wrap(t) for t in texts])
            for t, (c, fsz, rot, ha, rm) in zip(new, props):
                t.set_color(c); t.set_fontsize(fsz); t.set_rotation(rot)
                t.set_ha(ha); t.set_rotation_mode(rm); t.set_linespacing(1.05)
        lg = ax.get_legend()
        for leg in ([lg] if lg else []):
            for t in leg.get_texts():
                t.set_text(_wrap(t.get_text(), 26))
    for leg in getattr(fig, 'legends', []):
        for t in leg.get_texts():
            t.set_text(_wrap(t.get_text(), 26))
    # A legend in the right-hand column can still run off the canvas with a long protein
    # name: measure it and wrap narrower until it fits.
    W = fig.get_size_inches()[0] * fig.dpi
    for leg in getattr(fig, 'legends', []):
        for width in (20, 16, 13, 10):
            fig.canvas.draw()
            if leg.get_window_extent().x1 <= W - 2:
                break
            for t in leg.get_texts():
                t.set_text(_wrap(t.get_text().replace('\n', ''), width))


def _report_text_clashes(fig, name):
    """Development check (INTERACCT_CHECK_TEXT=1): list pairs of text items whose boxes
    overlap on the page. Text of hidden axes is skipped, and tick labels are compared only
    with text that is not a tick label of the same axes (rotated tick labels have
    overlapping boxes without overlapping letters)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ticks = {}
    for ax_ in fig.axes:
        for t_ in ax_.xaxis.get_ticklabels(which='both') + ax_.yaxis.get_ticklabels(which='both'):
            ticks[id(t_)] = ax_
    items = []
    for t_ in fig.findobj(mpl.text.Text):
        s_ = t_.get_text().strip()
        if not s_ or not t_.get_visible() or t_.get_alpha() == 0:
            continue
        ax_ = ticks.get(id(t_), t_.axes)
        if id(t_) in ticks:
            if not ax_.get_visible() or not ax_.axison:
                continue
            axis_ = ax_.xaxis if any(t_ is q for q in ax_.xaxis.get_ticklabels(which='both')) \
                else ax_.yaxis
            if not axis_.get_visible():
                continue
        elif ax_ is not None and not ax_.get_visible():
            continue
        try:
            bb = t_.get_window_extent(r)
        except Exception:
            continue
        if bb.width < 1 or bb.height < 1:
            continue
        items.append((t_, s_, bb.expanded(0.98, 0.94), ticks.get(id(t_))))
    hits = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a[3] is not None and a[3] is b[3]:
                continue
            if a[2].overlaps(b[2]):
                hits.append((a[1][:40], b[1][:40]))
    for a, b in hits:
        print(f'  TEXT CLASH in {name}: {a!r} <> {b!r}', flush=True)


def save(fig, name):
    try:
        fit_labels(fig)
    except Exception:
        pass                                   # cosmetic only: never block a figure
    if os.environ.get('INTERACCT_CHECK_TEXT'):
        try:
            _report_text_clashes(fig, name)
        except Exception as _e:
            print(f'  text check failed on {name}: {_e}', flush=True)
    fig.savefig(name + '.png', dpi=600); fig.savefig(name + '.pdf'); plt.close(fig)
    print('wrote', name, flush=True)


# ------------------------------------------- per-residue interaction-type profiles
# The footprint says how much surface a residue buries; it says nothing about HOW that
# surface is held. These counts answer the second question on the same residue axis:
# how many salt bridges, hydrogen bonds, stacking and cation-pi contacts each residue
# makes across the interfaces of its environment, averaged over the symmetry copies -
# and how many partner residues it packs against through apolar (C/S-C/S) contacts.
# A hydrophobic contact is counted once per partner residue with at least --phobic-min
# apolar atom contacts, the same rule as the hydrophobic edges of the network figure, so
# the track counts residue-residue contacts on one scale with the bonds.
INTORD = ['salt', 'hbond', 'stack', 'catpi', 'phob']
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
try:
    _PH = pd.read_csv('interface_hydrophobic.csv')
    _PH = _PH[_PH.n_contacts >= args.phobic_min]
    for _c1, _r1, _c2, _r2 in zip(_PH.c1, _PH.ri1, _PH.c2, _PH.ri2):
        INTRES[(_c1, int(_r1))]['phob'] += 1
        INTRES[(_c2, int(_r2))]['phob'] += 1
except Exception:                      # no table, or an empty one: no hydrophobic counts
    pass
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
with RPT.guard('footprint figures', ok_detail='fig1..N_<protein>_footprints'):
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
        parts = ([c for c in CATS if c in parts] if HOMO else
                 [x for x in ENVORD if x in parts] if SUB_FOOT else
                 [e for e in ENTS if e in parts])
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
            # residues this environment's chains actually contain: a molecule modelled in
            # pieces (two halves of one DNA) shares the axis, but each piece draws its
            # density, ΔG and conservation only where it has residues
            _have = {r.id[1] for c in env_of if env_of[c] == e and prot.get(c) == ent
                     for r in model[c] if r.id[0] == ' '}
            _hlo, _hhi = (min(_have), max(_have)) if _have else (lo, hi)
            fig.text(L, (H - y - 0.06) / H,
                     f'environment {env_tag(e)}  ({nco} copies)   ·   '
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
                    if imax <= 3:
                        axi.set_yticks(np.arange(0, imax + 0.1, 1))
                    else:                        # a short track: 0, middle, top at most
                        _top = int(np.floor(imax))
                        axi.set_yticks([0, _top] if _top < 6 else [0, _top // 2, _top])
                    tidy(axi); axi.spines['bottom'].set_visible(False)
                    axi.tick_params(labelbottom=False, bottom=False, labelsize=5.5)
                    axi.set_ylabel('contacts', fontsize=6.5)
                    axi.yaxis.set_label_coords(YLAB_X, 0.5)
                    y += ih + GAP_AX

                axd = axes_at(y, dh)
                pad = 15                             # smooth with a margin, then crop
                w = P.index[(P.index >= max(s0 - pad, _hlo)) & (P.index <= min(s1 + pad, _hhi))]
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
                    xs2 = np.array([r for r in range(s0, s1 + 1) if r in solv and r in _have])
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
                    seg_r = [r for r in range(s0, s1 + 1) if r in cons and r in _have]
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
                                                    label=PARTLAB.get(k, str(k))) for k in parts]]
        if HAS_INT:
            g = [title('interaction')]
            g += [mpl.patches.Patch(fc=fs.INTCOL[k], ec='none', label=t)
                  for k, t in [('salt', 'salt bridge'), ('hbond', 'hydrogen bond'),
                               ('stack', 'aromatic stacking'), ('catpi', 'cation–π'),
                               ('phob', 'hydrophobic contact')]
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
        _ttl = fig.text(L, 1 - 0.13 / H, f'{ent} — interface footprint per symmetry-independent '
                        f'environment', fontsize=8.5, ha='left', va='top', color=fs.INK)
        if solv:
            # shrink (down to 6.5 pt) so the title stops short of the ΔG colour bar
            _lim = (0.985 - 1.35 / 7.2 - 0.015) * fig.get_size_inches()[0] * fig.dpi
            fig.canvas.draw()
            _x1 = _ttl.get_window_extent().x1
            _x0 = _ttl.get_window_extent().x0
            if _x1 > _lim:
                _ttl.set_fontsize(max(6.5, 8.5 * (_lim - _x0) / (_x1 - _x0)))
                fig.canvas.draw()
                if _ttl.get_window_extent().x1 > _lim:
                    _ttl.set_text(f'{ent} —\ninterface footprint per symmetry-independent environment')
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
with RPT.guard('fig_architecture', ok_detail='written'):
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(1, 1, left=0.085, right=0.76, top=0.92, bottom=0.13)
    ax = fig.add_subplot(gs[0, 0])
    envs_all = sorted(set(env_of.values()))
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
    _zr = float(GEO.z.max() - GEO.z.min())
    if _zr < 10.0:
        # one layer: the axial spread is rounding noise, so show ±20 Å around it and say so
        _zc = float(GEO.z.mean()); ax.set_ylim(_zc - 20, _zc + 20)
        ax.text(0.99, 0.03, f'all chains within {max(_zr, 0.1):.1f} Å along the axis (one layer)',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=5.6, color='#8A8A8A')
    ax.set_xlabel('angular position around the symmetry axis (°)')
    ax.set_ylabel('position along axis (Å)')
    ax.set_title(f'Contact topology, cylinder unrolled — point group {CFG["point_group"]}; '
                 'line width ∝ buried area', fontsize=8, loc='left', pad=4, color=fs.INK)
    tidy(ax)
    yy = 0.92
    for e in ENTS:
        mine = [x for x in envs_all if prot[x[0]] == e]
        labs = [env_str(x, 9) for x in mine]
        hh = [mpl.patches.Patch(fc=ENVCOL[x], ec='white', lw=0.5, label=t)
              for x, t in zip(mine, labs)]
        lg = fig.legend(handles=hh, frameon=False, fontsize=6.0, loc='upper left',
                        bbox_to_anchor=(0.775, yy), handlelength=1.1, labelspacing=0.4,
                        title=e, title_fontsize=6.6)
        lg._legend_box.align = 'left'
        yy -= 0.10 + 0.055 * sum(t.count('\n') + 1 for t in labs)
    # Key for the line width, drawn with the SAME mapping as the edges above, so a thickness
    # can be read back as an area instead of only being compared with its neighbours.
    _amax = max(1.0, S.area.max())
    _KW, _KH = 1.15, 0.26                                   # inches
    _axk = fig.add_axes([0.782, max(yy - 0.17, 0.08), _KW / 7.2, _KH / 4.0])
    _axk.set_xlim(0, _amax); _axk.set_ylim(-1, 1); _axk.axis('off')
    _aa = np.linspace(0, _amax, 200)
    _half = (0.22 + 3.0 * _aa / _amax) / 2.0                # points
    _pt2y = 2.0 / (_KH * 72.0)
    _axk.fill_between(_aa, -_half * _pt2y, _half * _pt2y, facecolor='#C9CDD2',
                      edgecolor='none')
    _step = 10 ** np.floor(np.log10(_amax))
    if _amax / _step < 2:
        _step /= 2
    for _v in np.arange(_step, _amax * 1.001, _step):
        _axk.text(_v, -0.85, f'{_v:,.0f}', fontsize=5.0, ha='center', va='top',
                  color=fs.INK, alpha=0.75)
    _axk.text(0, 1.0, 'buried area (Å$^2$)', fontsize=5.8, ha='left', va='bottom',
              color=fs.INK)
    save(fig, 'fig_architecture')

# ---------------------------------------------------------------- bar geometry
# A horizontal bar must come out the same thickness whatever the model contains. With a
# fixed canvas, a run with one interface class stretched that single bar over the whole
# panel. So the plotting area is sized in INCHES from the number of rows and the figure
# grows or shrinks with it; the legend column sets a floor on the height, and the axes
# stay anchored at the top, so a short plot simply leaves space below it.
ROW_IN = 0.42                      # inches per row slot (bar plus its share of the gap)


def bar_geom(nrows, head=0.50, foot=0.52, legend_in=0.0):
    """(figure height, top, bottom) in inches / figure fractions for nrows bars."""
    axh = ROW_IN * (nrows + 0.6)
    fh = head + max(axh, legend_in) + foot
    top = 1.0 - head / fh
    return fh, top, top - axh / fh


def bar_ylim(ax, n, nmax):
    """n rows inside a box sized for nmax, hung from the top so the pitch is identical."""
    ax.set_ylim(n - 0.2 - (nmax + 0.6), n - 0.2)


def refit_bars(fig, gs, legs, nrows, head=0.50, foot=0.52, lg_x=0.815, lg_gap=0.18):
    """Resize the canvas so the bars keep ROW_IN per row AND the legend column fits.

    The legend height depends on the font, not on the figure, so it is measured once
    and is still valid after the resize. With many rows the bars set the height; with
    one or two rows the legend does, and the axes simply sit at the top of it."""
    fig.canvas.draw()
    lh = [lg.get_window_extent().height / fig.dpi for lg in legs]
    need = sum(lh) + lg_gap * max(len(lh) - 1, 0)
    axh = ROW_IN * (nrows + 0.6)
    fh = head + max(axh, need) + foot
    fig.set_size_inches(7.2, fh)
    top = 1.0 - head / fh
    gs.update(top=top, bottom=top - axh / fh)
    yy = top - 0.01
    for lg, h in zip(legs, lh):
        lg.set_bbox_to_anchor((lg_x, yy), transform=fig.transFigure)
        yy -= (h + lg_gap) / fh
    return fh


# ============================================================ inventory
with RPT.guard('fig_inventory', ok_detail='written'):
    _rows = sorted(set(BUDG.env), key=lambda e: (ENTS.index(prot[e[0]]), e))
    _FH, _TOP, _BOT = bar_geom(len(_rows), head=0.50, foot=1.00)
    fig = plt.figure(figsize=(7.2, _FH))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.42,
                          left=0.085, right=0.795, top=_TOP, bottom=_BOT)
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
    rows = _rows
    y = np.arange(len(rows))[::-1]
    xmax = BUDG.groupby('env').total.mean().max() * 1.25
    ax2.set_xlim(0, xmax); bar_ylim(ax2, len(rows), len(rows)); fig.canvas.draw()
    for yy_, e in zip(y, rows):
        left = 0.0
        _row = BUD.loc[[c for c in BUD.index if env_of[c] == e]].mean()
        for part in BKEYS:
            w = float(_row.get(part, 0.0))
            if w > 1:
                fs.sbarh(ax2, yy_, w, left=left, height=0.62, color=BCOL[part], round_pt=2.0, gap_pt=1.1)
            left += w
        ax2.text(left + xmax * 0.015, yy_, f'{left:.0f}', va='center', fontsize=6, color='#555555')
    ax2.set_yticks(y); ax2.set_yticklabels([env_str(e, 9) for e in rows], fontsize=6.2)
    for lbl, e in zip(ax2.get_yticklabels(), rows):
        lbl.set_color(DARK[prot[e[0]]])
    ax2.set_xlabel('buried area per subunit (Å$^2$)')
    ax2.set_title('b   Interface budget of each symmetry-independent subunit',
                  fontsize=8, loc='left', pad=4, color=fs.INK)
    tidy(ax2)
    _used = [p_ for p_ in BKEYS if BUD[p_].sum() > 0]
    hh = [mpl.patches.Patch(fc=BCOL[p_], ec='none', label=f'buried by {BLAB_[p_]}') for p_ in _used]
    lg = fig.legend(handles=hh, frameon=False, fontsize=6.4, loc='upper left',
                    bbox_to_anchor=(0.808, _TOP - 0.02), handlelength=1.1, labelspacing=0.5,
                    title='partner', title_fontsize=6.8)
    lg._legend_box.align = 'left'
    refit_bars(fig, gs, [lg], len(_rows), head=0.50, foot=1.00, lg_x=0.808)
    _W, _H0 = fig.get_size_inches()
    _MINA = 1.6                                   # inches: minimum height of the dot plot
    _pa, _pb = ax.get_position(), ax2.get_position()
    if _pa.height * _H0 < _MINA:
        _top_in = (1 - _pa.y1) * _H0              # head, kept
        _hb_in = _pb.height * _H0                 # bar panel keeps its row pitch
        _H1 = _H0 + (_MINA - _pa.height * _H0)
        fig.set_size_inches(_W, _H1)
        _y1 = 1 - _top_in / _H1
        ax.set_position([_pa.x0, _y1 - _MINA / _H1, _pa.width, _MINA / _H1])
        ax2.set_position([_pb.x0, _y1 - _hb_in / _H1, _pb.width, _hb_in / _H1])
        lg.set_bbox_to_anchor((0.808, _y1 - 0.01), transform=fig.transFigure)
    save(fig, 'fig_inventory')

# ============================================================ chemistry
with RPT.guard('fig_chemistry', ok_detail='written'):
    present = [c for c in CLSORD if c in COMP.columns and COMP[c].sum() > 0]
    _HEAD = 0.62 if LOWCONF else 0.50
    _FOOT = 0.84 if 'phob' in DEN.columns else 0.52   # the footnote needs its own band
    _FH, _TOP, _BOT = bar_geom(len(CATS), head=_HEAD, foot=_FOOT)
    fig = plt.figure(figsize=(7.2, _FH))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.52,
                          left=0.165, right=0.80, top=_TOP, bottom=_BOT)
    y = np.arange(len(CATS))[::-1]
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1.0); bar_ylim(ax, len(CATS), len(CATS))
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
    _IK = ['salt', 'hbond', 'stack', 'catpi'] + (['phob'] if 'phob' in DEN.columns else [])
    ax2.set_xlim(0, max(1e-9, float(DEN[_IK].sum(axis=1).max())) * 1.32)
    bar_ylim(ax2, len(CATS), len(CATS)); fig.canvas.draw()
    for yy_, c in zip(y, CATS):
        left = 0.0
        for k in _IK:
            w = float(DEN.loc[c, k]) if c in DEN.index else 0.0
            if w > 0.02:
                fs.sbarh(ax2, yy_, w, left=left, height=0.62, color=fs.INTCOL[k],
                         round_pt=1.6, gap_pt=1.0)
            left += w
        _ph = float(DEN.loc[c, 'phob']) if ('phob' in _IK and c in DEN.index) else 0.0
        # the label keeps bonds and apolar contacts apart: they are different kinds of count
        _lab = f'{left - _ph:.1f}' + (f' + {_ph:.1f}' if _ph > 0.05 else '')
        ax2.text(left + ax2.get_xlim()[1] * 0.012, yy_, _lab, va='center', fontsize=6,
                 color='#555555')
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlabel('interactions per 1000 Å$^2$ buried')
    ttl = 'b   Interactions' if 'phob' in _IK else 'b   Directional interactions'
    if LOWCONF:
        ttl += f'\n     ({LOWTXT})'
    ax2.set_title(ttl, fontsize=8, loc='left', pad=4, color=fs.INK)
    tidy(ax2)
    h1 = [mpl.patches.Patch(fc=fs.CLSCOL.get(c, '#AAAAAA'), ec='none', label=c) for c in present]
    l1 = fig.legend(handles=h1, frameon=False, fontsize=6.3, loc='upper left',
                    bbox_to_anchor=(0.815, _TOP - 0.02), handlelength=1.1, labelspacing=0.45,
                    title='residue class', title_fontsize=6.8); l1._legend_box.align = 'left'
    h2 = [mpl.patches.Patch(fc=fs.INTCOL[k], ec='none',
                            label=BLAB[k] if k != 'phob' else 'hydrophobic contact')
          for k in _IK]
    l2 = fig.legend(handles=h2, frameon=False, fontsize=6.3, loc='upper left',
                    bbox_to_anchor=(0.815, _TOP - 0.40), handlelength=1.1, labelspacing=0.45,
                    title='interaction type', title_fontsize=6.8); l2._legend_box.align = 'left'
    _fh = refit_bars(fig, gs, [l1, l2], len(CATS), head=_HEAD, foot=_FOOT, lg_x=0.828)
    if 'phob' in _IK:
        # hung just under the x labels, wherever the legend column puts the axes
        fig.text(0.165, ax.get_position().y0 - 0.50 / _fh, 'b: salt bridges, hydrogen bonds, stacking and cation–π are '
                 f'counted per bond; hydrophobic contacts per residue pair with ≥ {CFG.get("phobic_min", 3)} apolar\n'
                 'atom contacts. The label gives bonds + hydrophobic pairs.', fontsize=5.2,
                 color='#888888', ha='left', va='top', linespacing=1.4)
    save(fig, 'fig_chemistry')

# ============================================================ contact maps
with RPT.guard('fig_contact_maps', ok_detail='written'):
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
with RPT.guard('fig_area_partition', ok_detail='written'):
    TOT = S.area.sum()
    catsum = S.groupby('ctg').area.sum().sort_values(ascending=False)
    _STEP = 1.5 if SUB_BUDGET else 1.2          # row pitch of panel b, in panel-a rows
    _RB = _STEP * (len(ENTS) - 1) + 1.55        # panel b height, in rows
    _N = max(len(catsum), _RB - 0.6)
    _FH, _TOP, _BOT = bar_geom(_N, head=0.50, foot=0.56)
    fig = plt.figure(figsize=(7.2, _FH))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.52,
                          left=0.20, right=0.785, top=_TOP, bottom=_BOT)
    ax = fig.add_subplot(gs[0, 0]); y = np.arange(len(catsum))[::-1]
    _pmax = 100 * catsum.max() / TOT
    _xmax = 100.0 if _pmax > 60 else max(46.0, _pmax * 1.55)
    ax.set_xlim(0, _xmax); bar_ylim(ax, len(catsum), _N)
    fig.canvas.draw()
    for yy_, (c, v) in zip(y, catsum.items()):
        p_ = 100 * v / TOT
        fs.sbarh(ax, yy_, p_, left=0, height=0.60, color=CATCOL[c], round_pt=2.0, gap_pt=0)
        if p_ > 0.55 * _xmax:            # no room to the right: write it inside the bar
            _cc = mpl.colors.to_rgb(CATCOL[c])
            _light = (0.299 * _cc[0] + 0.587 * _cc[1] + 0.114 * _cc[2]) > 0.62
            ax.text(p_ - 0.02 * _xmax, yy_, f'{p_:.1f}%   ({v:,.0f} Å$^2$)', ha='right',
                    va='center', fontsize=6, color='#333333' if _light else '#FFFFFF',
                    fontweight='bold')
        else:
            ax.text(p_ + 0.01 * _xmax, yy_, f'{p_:.1f}%   ({v:,.0f} Å$^2$)',
                    va='center', fontsize=6, color='#555555')
    ax.set_yticks(y); ax.set_yticklabels(catsum.index, fontsize=6.4)
    ax.set_xlabel('share of all buried surface in the model (%)')
    if _xmax == 100.0:
        ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title(f'a   Total buried surface = {TOT:,.0f} Å$^2$', fontsize=8, loc='left', pad=4, color=fs.INK)
    tidy(ax)
    ax2 = fig.add_subplot(gs[0, 1])
    # Rows sit further apart than in panel a: above each bar there is a thin bar per partner
    # molecule (its subunits combined) and then the total, and both need their own room.
    y2 = _STEP * np.arange(len(ENTS))[::-1]
    ax2.set_xlim(0, 100); ax2.set_ylim(-0.6, _STEP * (len(ENTS) - 1) + 0.95); fig.canvas.draw()

    def _ink(col):
        _cc = mpl.colors.to_rgb(col)
        return '#333333' if (0.299 * _cc[0] + 0.587 * _cc[1] + 0.114 * _cc[2]) > 0.62 else '#FFFFFF'

    for yy_, e in zip(y2, ENTS):
        tot = BUD.loc[[c for c in BUD.index if prot[c] == e]].mean()
        pct = 100 * tot / max(tot.sum(), 1e-9)
        left, span = 0.0, {}
        for part in BKEYS:
            w = float(pct.get(part, 0.0))
            if w > 0.4:
                fs.sbarh(ax2, yy_, w, left=left, height=0.60, color=BCOL[part], round_pt=2.0, gap_pt=1.1)
                if w > 5:                           # every segment above 5 %, number only
                    ax2.text(left + w / 2, yy_, f'{w:.0f}', ha='center', va='center',
                             fontsize=5.2, color=_ink(BCOL[part]), fontweight='bold')
                g = ENV_ENT.get(part, part)
                a0, a1 = span.get(g, (left, left))
                span[g] = (min(a0, left), left + w)
            left += w
        if SUB_BUDGET:
            # one thin bar per partner molecule, spanning its subunits: their combined share
            for g, (a0, a1) in span.items():
                w = a1 - a0
                fs.sbarh(ax2, yy_ + 0.44, w, left=a0, height=0.17, color=PAL[g],
                         round_pt=1.2, gap_pt=1.1)
                if w > 5:
                    ax2.text(a0 + w / 2, yy_ + 0.44, f'{w:.0f}', ha='center', va='center',
                             fontsize=4.6, color=_ink(PAL[g]), fontweight='bold')
        ax2.text(50, yy_ + (0.60 if SUB_BUDGET else 0.40), f'{tot.sum():,.0f} Å$^2$ total',
                 ha='center', va='bottom', fontsize=5.8, color='#8A8A8A')
    ax2.set_yticks(y2); ax2.set_yticklabels([f'{e}\nsubunit' for e in ENTS], fontsize=7)
    for lbl, e in zip(ax2.get_yticklabels(), ENTS):
        lbl.set_color(DARK[e])
    ax2.set_xlabel('share of that subunit’s buried surface (%)'); ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_title('b   Interface budget', fontsize=8, loc='left', pad=4, color=fs.INK)
    tidy(ax2)
    _used = [p_ for p_ in BKEYS if BUD[p_].sum() > 0]
    hh = [mpl.patches.Patch(fc=BCOL[p_], ec='none', label=f'with {BLAB_[p_]}') for p_ in _used]
    if SUB_BUDGET:
        hh.append(mpl.patches.Patch(fc='#BDBDBD', ec='none', label='thin bar: subunits\nof one molecule together'))
    lg = fig.legend(handles=hh, frameon=False, fontsize=6.4, loc='upper left',
                    bbox_to_anchor=(0.80, _TOP - 0.02), handlelength=1.1, labelspacing=0.5,
                    title='partner', title_fontsize=6.8); lg._legend_box.align = 'left'
    refit_bars(fig, gs, [lg], _N, head=0.50, foot=0.56, lg_x=0.80)
    # The two panels count different things - interface categories on the left, entities on
    # the right - so they get their OWN axes heights at the shared row pitch, hung from the
    # same top line. Stretching the short one to match the tall one leaves a block of empty
    # axis under three bars.
    _pos = ax2.get_position()
    _h2 = ROW_IN * _RB / fig.get_size_inches()[1]
    ax2.set_position([_pos.x0, _pos.y1 - _h2, _pos.width, _h2])
    save(fig, 'fig_area_partition')


# ============================================================ packing quality
with RPT.guard('fig_packing', ok_detail='written'):
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

    if not ('sc' in S.columns and S.sc.notna().any()):
        RPT.skip('fig_packing', 'no shape complementarity in step 01 (it was run with --no-shape)')
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
        ax.set_title('a   Surface fit', fontsize=8, loc='left',
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
        ax2.set_title('b   Fit vs gap', fontsize=8, loc='left',
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
with RPT.guard('figS1_contact_maps_all', ok_detail='written'):
    REPs = REP.sort_values('area', ascending=False).reset_index(drop=True)
    if len(REPs) <= 3:
        # fig_contact_maps already draws the three largest maps at full size
        RPT.skip('figS1_contact_maps_all', f'only {len(REPs)} interface(s): all already in fig_contact_maps')
        raise runlog.Skip()
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
with RPT.guard('fig_interaction_networks', ok_detail='written'):
    # One panel per SYMMETRY-UNIQUE interface, drawn from its representative chain pair, so
    # a 30-chain assembly with 105 contacts produces a few readable diagrams instead of a
    # hundred near-identical ones. Residues are nodes; every specific interaction is an
    # edge, coloured as in the footprint track. Covalent links are drawn whatever their
    # number, because one disulfide matters more than ten van der Waals contacts.
    LINKS = pd.read_csv('interface_links.csv') if os.path.exists('interface_links.csv') \
        else pd.DataFrame(columns=['kind', 'bonded', 'c1', 'resi1', 'resn1', 'c2', 'resi2', 'resn2'])
    COVAL = fs.COVALENT
    PHOBCOL = fs.INTCOL['phob']            # same colour as the footprint track and the renders
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
                    color=PHOBCOL, alpha=0.55, zorder=0, solid_capstyle='round')
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
        # Each column header has only half a panel of clear width — barely more than a word.
        # A short environment still fits there; a homo-oligomer's 36 chains do not, in any
        # wrapping, so the equivalence moves to the panel title, which has the full width.
        _env = env_of.get(c1, c1)
        _long = len(_env) > 6
        for x, ch in ((xl, c1), (xr, c2)):
            _h = ch if _long else f'{ch}  ({"/".join(env_of.get(ch, ch))})'
            ax.text(x, 1.012, _h, fontsize=6.0, ha='center', va='bottom', color=fs.INK,
                    fontweight='bold')
        _ttl = (f'{ctg}\n{area:.0f} Å$^2$  ·  {len(eds) + len(cov)} bonds'
                + (f'  ·  {len(phob)} apolar' if phob else ''))
        if _long:
            _ttl += f'\neach chain 1 of {len(_env)} equivalent copies'
        ax.set_title(_ttl, fontsize=6.5, loc='center', pad=12, color=fs.INK, linespacing=1.45)


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
                hh.append(mpl.patches.Patch(fc=PHOBCOL, ec='none', alpha=0.55,
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
with RPT.guard('ChimeraX files', ok_detail='model_interfaces.pdb + 2 .cxc scripts'):
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

        # Flat lighting with silhouettes: no shading gradients to confuse with colour, and
        # a dark outline on every edge, which is what makes a figure read on paper.
        PRESET = ['hide atoms', 'show cartoons', 'color all #D9D9D9',
                  'set bgColor white', 'lighting flat', 'lighting shadows false',
                  'graphics silhouettes true width 1.5 color black',
                  # licorice: every secondary-structure element drawn as the same thin round
                  # tube, no helix cylinders, no strand arrows. Same as the menu entry
                  # Presets > Cartoons/Nucleotides > Licorice/Ovals.
                  'cartoon style protein modeHelix default arrows false xsection oval '
                  'width 1 thick 1 sides 24',
                  'cartoon style nucleic xsection oval width 4 thick 1', '']
        colour_cmds = list(PRESET)
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
        # bound ligands: sticks, carbon in the ligand's colour, heteroatoms by element,
        # with the name given in --ligand-names as a label
        lig_cmds = []
        for code in CFG.get('ligand_resn') or []:
            lc = LIGCOL.get(code, '#8C6239')
            lig_cmds += [f'show :{code} atoms', f'style :{code} stick',
                         f'color :{code} {lc}', f'color :{code} byhetero']
            if code in LIGNAME:
                lig_cmds.append(f'label :{code} text "{LIGNAME[code]}" height 1.2')
        if lig_cmds:
            lig_cmds = ['', '# bound ligands (sticks; heteroatoms by element)'] + lig_cmds
            key += ['# ligands:'] + [f'#   {c:<5} {LIGNAME.get(c, c):<14} '
                                     f'{LIGCOL.get(c, "#8C6239")}'
                                     for c in CFG.get('ligand_resn') or []]
        open('chimerax_colour_only.cxc', 'w').write(
            '\n'.join(head2 + colour_cmds + lig_cmds + key) + '\n')

        # ---- second script: colour by the kind of interaction a residue makes ----
        # A residue can make several kinds at once, so one colour per residue has to be a
        # choice. The most specific and most structurally constraining wins: a salt bridge
        # over cation-pi, over stacking, over a hydrogen bond, over a hydrophobic contact
        # (>= --phobic-min apolar atom contacts with one partner residue). Residues that
        # bury surface but make none of these are left in a neutral tone - the
        # "no specific contact" class.
        PRIORITY = [('covalent', 'disulfide / covalent'), ('salt', 'salt bridge'),
                    ('catpi', 'cation-pi'), ('stack', 'aromatic stacking'),
                    ('hbond', 'hydrogen bond'), ('phob', 'hydrophobic contact')]
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
        icmds = list(PRESET)
        icmds[2] = 'color all #EDEDED'
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
            '\n'.join(ihead + icmds + lig_cmds + ikey) + '\n')
        print(f'wrote chimerax_colour_only.cxc and chimerax_colour_by_interaction.cxc '
              f'({len(sel)} residues >= {bsa_min:g} A^2 coloured)', flush=True)


    write_chimerax(args.bsa_min)


# ============================================================ whole-model renders
# The same flat, silhouetted licorice as fig_assemblies, painted residue by residue, for
# readers who want the 3D picture without opening ChimeraX. Every figure shows the model in
# THREE views, in the symmetry frame (z = symmetry axis): from the side with the axis
# vertical, tilted (--render-view, default 62,20), and from the top, down the axis.
#   proteins      each protein in its colour; neighbouring chains alternate in shade
#   budget        the protein that buries most of each residue (the legend of the
#                 "Interface budget" panel), grey where the residue is not at an interface
#   interactions  the most specific contact each residue makes, in the same priority as
#                 chimerax_colour_by_interaction.cxc: covalent > salt bridge > cation-pi >
#                 stacking > hydrogen bond > hydrophobic; buried without any of these in a
#                 mid grey, not at an interface in a light grey
# All use the --bsa-min cut-off of the ChimeraX scripts, so 2D, ChimeraX and the renders
# agree residue for residue.
if not args.render:
    RPT.skip('whole-model renders', 'not requested (--no-render)')
else:
  with RPT.guard('whole-model renders', ok_detail='fig_render_proteins, fig_render_budget, '
                                                  'fig_render_interactions (3 views each)'):
    from render_assembly import licorice
    try:
        _el, _az = (float(x) for x in args.render_view.split(','))
    except Exception:
        _el, _az = 62.0, 20.0
        RPT.default('render view', f'could not read --render-view {args.render_view!r}; '
                                   'used 62,20')
    # model -> symmetry frame (rows of frame_rotation are the frame axes; z = the axis)
    try:
        _FR = np.asarray(CFG['frame_rotation'], float)
        _FO = np.asarray(CFG['frame_origin'], float)
        assert _FR.shape == (3, 3) and _FO.shape == (3,)
    except Exception:
        _FR, _FO = np.eye(3), None
    _BUR = pd.read_csv('residue_burial_assembly.csv')
    _bur = {(r.chain, int(r.resi)): float(r.buried) for r in _BUR.itertuples()}
    # area each residue loses to each partner PROTEIN, summed over the partner chains
    # (by partner SUBUNIT in shades of the molecule colour, as in the budget figures)
    _pk = (lambda c: env_of[c]) if SUB_BUDGET else (lambda c: prot[c])
    RKEYS, RCOL, RLAB = (ENVORD, ENVCOL, ENVLAB) if SUB_BUDGET else (ENTS, PAL, {e: e for e in ENTS})
    _by = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in RES_ALL.itertuples():
        partner = r.c2 if r.chain == r.c1 else r.c1
        if partner in prot:
            _by[(r.chain, int(r.resi))][_pk(partner)] += float(r.dsasa)
    _cov = set()
    if len(LINKS):
        _Lk = LINKS[(LINKS.kind.isin(['disulfide', 'isopeptide'])) & (LINKS.bonded == True)]
        for r in _Lk.itertuples():
            _cov.add((r.c1, int(r.resi1))); _cov.add((r.c2, int(r.resi2)))
            # a bonded residue counts as buried by the protein it is bonded to, even if it
            # buries little surface, so both renders mark the same residues
            if r.c2 in prot:
                _by[(r.c1, int(r.resi1))][_pk(r.c2)] += 0.0
            if r.c1 in prot:
                _by[(r.c2, int(r.resi2))][_pk(r.c1)] += 0.0
    G_OUT, G_NONSPEC = '#E3E3E3', '#BDBDBD'
    _KIND_ORDER = ('salt', 'catpi', 'stack', 'hbond', 'phob')
    ICOL = {'covalent': fs.COVALENT, 'none': G_NONSPEC}
    ICOL.update({k: mpl.colors.to_hex(fs.INTCOL[k]) for k in _KIND_ORDER})
    ILAB = {'covalent': 'disulfide / covalent', 'salt': 'salt bridge', 'catpi': 'cation–π',
            'stack': 'aromatic stacking', 'hbond': 'hydrogen bond',
            'phob': 'hydrophobic contact', 'none': 'buried, no specific contact'}
    _CA, _cb, _ci, _rows = {}, {}, {}, []
    _nb, _ni = collections.Counter(), collections.Counter()
    for ch in model:
        cb, ci, pts = [], [], []
        for res in ch:
            # trace atom: CA for amino acids, C4' for nucleotides (DNA / RNA)
            _ta = 'CA' if 'CA' in res else ("C4'" if "C4'" in res else None)
            if _ta is None:
                continue
            key = (ch.id, res.id[1]); pts.append(res[_ta].coord)
            at_if = _bur.get(key, 0.0) >= args.bsa_min or key in _cov
            part = max(_by[key], key=_by[key].get) if (at_if and _by.get(key)) else None
            if key in _cov:
                cls = 'covalent'
            elif at_if:
                cnt = INTRES.get(key)
                cls = next((k for k in _KIND_ORDER if cnt and cnt.get(k)), 'none')
            else:
                cls = None
            if part is None and at_if:
                part = '_multi'      # buried only when several chains are present together
            cb.append(RCOL[part] if part in RCOL else (G_NONSPEC if part == '_multi' else G_OUT))
            ci.append(ICOL[cls] if cls else G_OUT)
            _nb[part] += 1; _ni[cls] += 1
            _rows.append((ch.id, res.id[1], prot.get(ch.id, ''),
                          {None: '', '_multi': 'several chains together'}.get(part, RLAB.get(part, part)),
                          cls or ''))
        if len(pts) >= 4:
            P_ = np.asarray(pts, float)
            if _FO is not None:
                P_ = (P_ - _FO) @ _FR.T
            _CA[ch.id] = P_; _cb[ch.id] = cb; _ci[ch.id] = ci
    pd.DataFrame(_rows, columns=['chain', 'resi', 'protein', 'buried_mainly_by',
                                 'interaction_class']).to_csv('render_residue_colours.csv',
                                                              index=False)

    # proteins: one colour per protein, neighbouring chains in three alternating shades so
    # the subunits can be told apart (ordered by height along the axis, then by angle)
    _cp = {}
    for e in ENTS:
        chs = [c for c in _CA if prot.get(c) == e]
        ctr = {c: _CA[c].mean(axis=0) for c in chs}
        zs = np.array([ctr[c][2] for c in chs]) if chs else np.array([])
        zstep = max(float(np.ptp(zs)) / 8.0, 4.0) if len(zs) else 1.0
        chs.sort(key=lambda c: (round(ctr[c][2] / zstep), np.arctan2(ctr[c][1], ctr[c][0])))
        base = mpl.colors.to_rgb(PAL[e])
        tones = [base, tuple(0.62 * np.array(base) + 0.38), tuple(0.72 * np.array(base))]
        for k, c in enumerate(chs):
            _cp[c] = mpl.colors.to_hex(tones[k % 3]) if len(chs) > 1 else PAL[e]

    VIEWS = [(0.0, _az, 'side · axis vertical'), (_el, _az, f'tilted · {_el:g}°'),
             (90.0, _az, 'top · down the axis')]

    def _render(cols, title, handles, fname, ncol=3, note=None):
        W_, PW = 7.2, 2.3                                # inches: figure, one panel
        nrow_leg = int(np.ceil(len(handles) / ncol))
        FH = 0.42 + PW + 0.22 + 0.15 * nrow_leg + 0.12
        fig = plt.figure(figsize=(W_, FH))
        gap = (W_ - 3 * PW) / 4
        for k, (el, az, lab) in enumerate(VIEWS):
            ax = fig.add_axes([(gap + k * (PW + gap)) / W_, (FH - 0.42 - PW) / FH,
                               PW / W_, PW / FH])
            licorice(ax, _CA, cols, elev=el, azim=az, dpi=600)
            fig.text((gap + k * (PW + gap) + PW / 2) / W_, (FH - 0.40) / FH, lab,
                     ha='center', va='bottom', fontsize=5.6, color='#555555')
        fig.text(0.5, 1 - 0.05 / FH, f'{len(_CA)} chain{"s" if len(_CA) != 1 else ""} · whole model · {title}',
                 ha='center', va='top', fontsize=6.8, color='#333333')
        ytop = (FH - 0.42 - PW - 0.06) / FH
        if note is None:
            note = f'residues burying ≥ {args.bsa_min:g} Å$^2$ coloured · symmetry axis = z'
        fig.text(0.5, ytop, note, ha='center', va='top', fontsize=4.8, color='#8A8A8A')
        fig.legend(handles=handles, frameon=False, fontsize=5.4, loc='upper center',
                   ncol=ncol, bbox_to_anchor=(0.5, ytop - 0.14 / FH), handlelength=1.0,
                   handletextpad=0.4, columnspacing=1.6, labelspacing=0.35)
        fig.savefig(fname + '.png', dpi=600); fig.savefig(fname + '.pdf'); plt.close(fig)
        print('wrote', fname, flush=True)

    _hp = [mpl.patches.Patch(fc=PAL[e], ec='#1E1E1E', lw=0.3,
                             label=(lambda n_: f'{e}  ({n_} chain{"s" if n_ != 1 else ""})')(sum(1 for c in _CA if prot.get(c) == e)))
           for e in ENTS if any(prot.get(c) == e for c in _CA)]
    _render(_cp, 'coloured by molecule (shades = individual chains)', _hp,
            'fig_render_proteins', note='symmetry axis = z')
    _hb = [mpl.patches.Patch(fc=RCOL[e], ec='#1E1E1E', lw=0.3,
                             label=f'buried mainly by {RLAB[e]}  ({_nb[e]:,})')
           for e in RKEYS if _nb[e]]
    if _nb['_multi']:
        _hb.append(mpl.patches.Patch(fc=G_NONSPEC, ec='#1E1E1E', lw=0.3,
                                     label=f'buried only by several chains together  '
                                           f'({_nb["_multi"]:,})'))
    _hb.append(mpl.patches.Patch(fc=G_OUT, ec='#1E1E1E', lw=0.3,
                                 label=f'not at an interface  ({_nb[None]:,})'))
    _render(_cb, 'coloured by burying partner (interface budget)', _hb, 'fig_render_budget')
    _hi = [mpl.patches.Patch(fc=ICOL[k], ec='#1E1E1E', lw=0.3,
                             label=f'{ILAB[k]}  ({_ni[k]:,})')
           for k in ('covalent',) + _KIND_ORDER + ('none',) if _ni[k]]
    _hi.append(mpl.patches.Patch(fc=G_OUT, ec='#1E1E1E', lw=0.3,
                                 label=f'not at an interface  ({_ni[None]:,})'))
    _ttl = 'coloured by interaction type'
    if LOWCONF:
        _ttl += f'  ({LOWTXT})'
    _render(_ci, _ttl, _hi, 'fig_render_interactions', ncol=4)

RPT.done()
