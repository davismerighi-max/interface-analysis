"""Step 02 — derived tables and statistics, for any assembly and any point group.

    python 02_derive_tables.py MODEL.{pdb,cif}

Reads step-01 output (run_config.json and its CSVs) and the model. Generalised from
the tail1-specific original:

- orbit sizes are whatever the detected group gives, not assumed to be 3;
- an interface present in only ONE copy gets no standard deviation, and the report
  says so instead of printing a misleading 0;
- statistical tests are attempted only when the design supports them;
- bond-level results are labelled with the confidence implied by the resolution;
- nucleic-acid residues are classified separately rather than silently dropped.

Outputs
  secondary_structure.csv, residue_burial_assembly.csv, footprints_by_environment.csv,
  subunit_budget.csv, table_interfaces_unique.csv, interface_bonds_unique.csv,
  aromatic_interactions.csv, interface_orbit_chemistry.csv,
  interface_composition_by_category.csv, interaction_density_by_category.csv,
  stats_summary.txt   (and interfaces_summary.csv is rewritten with a category column)
"""
import sys, os, json, glob, site, collections, importlib.util
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model as BModel
from Bio.PDB.SASA import ShrakeRupley

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'model.pdb'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog
RPT = runlog.Report('02', model=MODEL)
with RPT.guard('read step 01 outputs', ok_detail='run_config.json and tables found', fatal=True):
    if not os.path.exists('run_config.json'):
        raise FileNotFoundError('run_config.json is missing: run 01_interface_analysis.py '
                                'first, in this same folder')
    if not os.path.exists(MODEL):
        raise FileNotFoundError(f'{MODEL} not found in {os.getcwd()}')
    CFG = json.load(open('run_config.json'))
OUT = open('stats_summary.txt', 'w', buffering=1)
def say(*a):
    print(*a, flush=True); OUT.write(' '.join(map(str, a)) + '\n')

AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
      'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'MSE'}
CLASS = {**{k: 'charged' for k in ['ASP', 'GLU', 'LYS', 'ARG']},
         **{k: 'aromatic' for k in ['PHE', 'TYR', 'TRP', 'HIS']},
         **{k: 'polar' for k in ['SER', 'THR', 'ASN', 'GLN', 'CYS']},
         **{k: 'aliphatic' for k in ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'MSE']},
         **{k: 'Gly / Pro' for k in ['GLY', 'PRO']}}
for n in ['A', 'C', 'G', 'U', 'T', 'DA', 'DC', 'DG', 'DT', 'DU']:
    CLASS[n] = 'nucleotide'
# Bound ligands and cofactors get their own class, named by step 01 in run_config.json,
# so they never fall into 'other' and are visible in the chemistry figure.
for n in CFG.get('ligand_resn', []):
    CLASS[n] = 'ligand'

say(f'model {CFG["model"]} | point group {CFG["point_group"]} (order {CFG["group_order"]}) | '
    f'{CFG["n_chains"]} chains | resolution {CFG["resolution"]} A '
    f'(bond confidence: {CFG["bond_confidence"]})')

GEO = pd.read_csv('chain_geometry.csv').set_index('chain')
S = pd.read_csv('interfaces_summary.csv')
RES = pd.read_csv('interface_residues.csv')
B = pd.read_csv('interface_bonds.csv')
ORB = pd.read_csv('orbits.csv')
ent_name = dict(zip(GEO.index, GEO.entity_name))
env_of = dict(zip(GEO.index, GEO.environment))
orb_members = {int(r.orbit): [(m[0], m[1]) for m in r.members.split('/')] for _, r in ORB.iterrows()}

# ---------------------------------------------------------------- model
with RPT.guard('read model', ok_detail='parsed', fatal=True):
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

# ---------------------------------------------------------------- secondary structure
try:
    # Load pydssp's NumPy backend directly, without running the package __init__
    # (which imports torch). find_spec locates the package wherever pip put it — the
    # old guess at site-packages()[-1] missed it on Colab and on user installs.
    _ps = importlib.util.find_spec('pydssp')
    cand = [os.path.join(d, 'pydssp_numpy.py') for d in (_ps.submodule_search_locations or [])] \
        if _ps else []
    cand = [c for c in cand if os.path.exists(c)]
    if not cand:
        raise ImportError('pydssp is not installed (pip install pydssp)')
    spec = importlib.util.spec_from_file_location('pdn', cand[0])
    pdn = importlib.util.module_from_spec(spec); spec.loader.exec_module(pdn)
    ALPH = np.array(['-', 'H', 'E'])
    rows = []
    for c in model:
        ok = []
        for r in c:
            if r.get_resname().strip() not in AA:
                continue                                  # DSSP is undefined for nucleotides
            d = {a.get_id(): a.coord for a in r}
            if all(k in d for k in ('N', 'CA', 'C', 'O')):
                ok.append((r.id[1], [d['N'], d['CA'], d['C'], d['O']]))
        if len(ok) < 4:
            continue
        code = np.asarray(pdn.assign(np.array([o[1] for o in ok], dtype=float)))
        ss = ALPH[np.argmax(code, axis=-1)] if code.ndim == 2 else ALPH[code]
        for (ri, _), s in zip(ok, ss):
            rows.append(dict(chain=c.id, entity_name=ent_name[c.id], resi=ri, ss=str(s)))
    pd.DataFrame(rows).to_csv('secondary_structure.csv', index=False)
    RPT.ok('secondary structure', f'{len(rows)} residues assigned (pydssp)')
except Exception as e:
    say(f'WARNING: secondary structure unavailable ({type(e).__name__}: {e})')
    RPT.warn('secondary structure', f'unavailable ({type(e).__name__}: {e}); figures are '
                                  'drawn without the helix/strand track')
    pd.DataFrame(columns=['chain', 'entity_name', 'resi', 'ss']).to_csv('secondary_structure.csv', index=False)

# ---------------------------------------------------------------- burial in the assembly
with RPT.guard('burial in the assembly', ok_detail='residue_burial_assembly.csv'):
    sr = ShrakeRupley(probe_radius=1.40, n_points=CFG['sasa_points'])
    sr.compute(model, level='A')
    asm = {(c.id, r.id[1], a.get_id()): a.sasa for c in model for r in c for a in r}
    iso = {}
    for c in model:
        s = Structure('s'); m = BModel(0); s.add(m); m.add(c.copy()); sr.compute(s, level='A')
        for cc in m:
            for r in cc:
                for a in r:
                    iso[(cc.id, r.id[1], a.get_id())] = a.sasa
    rows = []
    for c in model:
        for r in c:
            k = [(c.id, r.id[1], a.get_id()) for a in r]
            i = sum(iso[x] for x in k); b = sum(asm[x] for x in k)
            rows.append(dict(chain=c.id, entity_name=ent_name[c.id], resi=r.id[1],
                             resn=r.get_resname().strip(), sasa_iso=i, sasa_asm=b,
                             buried=max(i - b, 0.0)))
    BUR = pd.DataFrame(rows)
    BUR.to_csv('residue_burial_assembly.csv', index=False)
    say('\nNOTE: "buried" is INTERMOLECULAR only (chain alone vs chain in the assembly). '
        'Residues buried inside their own fold are already buried when the chain is isolated '
        'and contribute ~0 here; run step 01 with --intra for fold-core burial.')

# ---------------------------------------------------------------- categories
with RPT.guard('interface categories', ok_detail='assigned', fatal=True):
    # Split a same-entity contact into "lateral" (same level on the symmetry axis) and
    # "axial" (different level) ONLY where both kinds actually occur. Labelling a set that
    # is entirely lateral as "X-X lateral" invents a distinction the data do not contain.
    S['_same_z'] = [abs(GEO.loc[r.c1, 'z'] - GEO.loc[r.c2, 'z']) < 5 for _, r in S.iterrows()]
    S['_pairkind'] = ['–'.join(sorted([r.e1, r.e2])) for _, r in S.iterrows()]
    split_pairs = {k for k, g in S.groupby('_pairkind')
                   if k.split('–')[0] == k.split('–')[1] and g._same_z.nunique() == 2}
    def cat_of(r):
        base = r._pairkind
        if base in split_pairs:
            return f'{base} lateral' if r._same_z else f'{base} axial'
        return base
    S['ctg'] = S.apply(cat_of, axis=1)
    S.to_csv('interfaces_summary.csv', index=False)
    pairkey = {(r.c1, r.c2): r.ctg for _, r in S.iterrows()}
    def ctg_of(a, b):
        return pairkey.get((a, b), pairkey.get((b, a)))
    CATS = list(S.groupby('ctg').area.sum().sort_values(ascending=False).index)

# ---------------------------------------------------------------- footprints
with RPT.guard('footprints', ok_detail='footprints_by_environment.csv'):
    RES['partner'] = np.where(RES.chain == RES.c1, RES.c2, RES.c1)
    RES['pent'] = RES.partner.map(ent_name)
    RES['penv'] = RES.partner.map(env_of)
    # Partner CATEGORY as well as partner entity: in a homo-oligomer every partner is the
    # same protein, so "buried by entity X" carries no information and the figures must
    # colour by which distinct interface the surface belongs to instead.
    RES['pctg'] = [ctg_of(a, b) for a, b in zip(RES.c1, RES.c2)]
    pw = RES.groupby(['chain', 'resi', 'pent', 'pctg']).dsasa.sum().reset_index()
    tot = pw.groupby(['chain', 'resi']).dsasa.sum().rename('pw').reset_index()
    pw = pw.merge(tot).merge(BUR[['chain', 'resi', 'resn', 'buried']], on=['chain', 'resi'])
    pw = pw[pw.pw > 0]
    pw['attr'] = pw.buried * pw.dsasa / pw.pw          # rescale pairwise onto true burial
    pw['env'] = pw.chain.map(env_of); pw['entity_name'] = pw.chain.map(ent_name)
    ratio = (pw.pw / pw.buried.clip(lower=.01))
    say(f'pairwise-sum / true-burial ratio: median {ratio.median():.3f}, 95th pct {ratio.quantile(.95):.3f}')
    # Per-monomer footprint = total over the chains of the environment divided by HOW MANY
    # chains it has -- not the mean of the rows present. A residue may contact a partner in
    # one copy of the environment and not in another; averaging only the non-zero rows would
    # silently divide by a smaller number and inflate the footprint.
    nchain = pw.groupby('env').chain.nunique()
    ENV = (pw.groupby(['entity_name', 'env', 'resi', 'resn', 'pent', 'pctg']).attr.sum()
           .reset_index().rename(columns={'pent': 'pfam'}))
    ENV['dsasa'] = ENV.attr / ENV.env.map(nchain)
    ENV = ENV.drop(columns='attr')
    ENV.to_csv('footprints_by_environment.csv', index=False)
    for e in sorted(set(ENV.entity_name)):
        sub = ENV[ENV.entity_name == e].groupby(['env', 'resi']).dsasa.sum().unstack(fill_value=0.0)
        line = ', '.join(f'{i}:{v:.0f}' for i, v in sub.sum(axis=1).items())
        if len(sub) > 1:
            C = np.corrcoef(sub.values); off = C[np.triu_indices(len(C), 1)]
            say(f'{e}: buried A2/monomer {line} | footprint r between environments '
                f'{off.min():.2f}-{off.max():.2f}')
        else:
            say(f'{e}: buried A2/monomer {line} | single environment (all copies equivalent)')

# ---------------------------------------------------------------- budgets
with RPT.guard('subunit budgets', ok_detail='subunit_budget.csv'):
    ents = sorted(set(GEO.entity_name))
    BUDG = pd.DataFrame([dict(chain=ch, env=env_of[ch], entity_name=ent_name[ch],
        **{k: S[((S.c1 == ch) & (S.e2 == k)) | ((S.c2 == ch) & (S.e1 == k))].area.sum() for k in ents})
        for ch in GEO.index])
    BUDG['total'] = BUDG[ents].sum(axis=1)
    BUDG.to_csv('subunit_budget.csv', index=False)
    say('\nburied area per subunit (A2):')
    say(BUDG.groupby('entity_name')[ents + ['total']].mean().round(0).to_string())

# ---------------------------------------------------------------- unique interfaces
with RPT.guard('unique interfaces', ok_detail='table_interfaces_unique.csv'):
    g = S.groupby('orbit')
    tab = g.agg(chainA=('c1', 'first'), chainB=('c2', 'first'), entA=('e1', 'first'),
                entB=('e2', 'first'), category=('ctg', 'first'), n_copies=('area', 'size'),
                area_mean=('area', 'mean'), area_sd=('area', 'std'),
                hbond_mean=('n_hbond', 'mean'), saltbridge_mean=('n_saltbridge', 'mean'),
                apolar_fraction=('apolar_frac', 'mean'), nres_A=('nres1', 'mean'),
                nres_B=('nres2', 'mean')).reset_index().sort_values('area_mean', ascending=False)
    tab['area_sd'] = np.where(tab.n_copies > 1, tab.area_sd, np.nan)   # no SD from one copy
    tab.round(3).to_csv('table_interfaces_unique.csv', index=False)
    multi = tab[tab.n_copies > 1]
    if len(multi):
        cv = 100 * (multi.area_sd / multi.area_mean)
        say(f'\nsymmetry-copy reproducibility: median CV of interface area {cv.median():.1f}% '
            f'(n={len(multi)} interfaces with >1 copy)')
    else:
        say('\nNO symmetry replicates: every interface occurs once, so areas carry no '
            'reproducibility estimate. Treat each value as a single measurement.')
    if (tab.n_copies == 1).any():
        say(f'{int((tab.n_copies == 1).sum())} of {len(tab)} interfaces have a single copy '
            '(no error bar).')

    # Spread among the symmetry-independent interfaces of each type.
    #
    # An ANOVA here tests whether interfaces that symmetry does NOT force to be equal are
    # nevertheless equal. A significant result is the DEFAULT expectation and means little
    # on its own; it is informative only where a pseudo-symmetry would predict equality
    # (e.g. a 12-subunit ring refined under 3-fold symmetry). So the spread is always
    # reported descriptively, and the test is run only where the interfaces are plausible
    # candidates for equivalence: same entity pair, and areas within a factor of two.
    say('\nspread among symmetry-independent interfaces of each type '
        '(descriptive; see note on when the ANOVA is meaningful):')
    for ctg, sub in S.groupby('ctg'):
        per = sub.groupby('orbit').area.mean()
        k = len(per)
        if k == 1:
            say(f'  {ctg}: 1 unique interface, {per.iloc[0]:.0f} A2'); continue
        ratio = per.max() / max(per.min(), 1e-9)
        line = (f'  {ctg}: {k} unique interfaces, {per.min():.0f}-{per.max():.0f} A2 '
                f'(max/min {ratio:.1f}x)')
        groups = [x.area.values for _, x in sub.groupby('orbit') if len(x) > 1]
        if k >= 3 and ratio <= 2.0 and len(groups) == k:
            F, pv = stats.f_oneway(*groups)
            line += f'; homogeneity ANOVA F({k-1},{len(sub)-k})={F:.2f}, p={pv:.4g}'
        say(line)

    # Lateral vs axial, within one entity pair.
    #
    # Deliberately NOT restricted to "major" contacts: any area threshold that keeps the
    # large axial interfaces and discards the small ones is chosen after seeing the data,
    # and a reviewer is entitled to ask where it came from. Axial areas are typically
    # bimodal (a few large interfaces plus several glancing ones), so the comparison uses
    # a rank test over ALL of them, and the largest interface of each kind is reported
    # separately as the concrete, non-selective statement.
    for base in sorted(split_pairs):
        lat = S[S.ctg == f'{base} lateral']
        axi = S[S.ctg == f'{base} axial']
        if lat.orbit.nunique() < 1 or axi.orbit.nunique() < 1:
            continue
        if len(lat) > 2 and len(axi) > 2:
            U, pv = stats.mannwhitneyu(lat.area.values, axi.area.values, alternative='two-sided')
            say(f'{base}: lateral median {np.median(lat.area):.0f} A2 (n={len(lat)}) vs '
                f'axial median {np.median(axi.area):.0f} A2 (n={len(axi)}); '
                f'Mann-Whitney U={U:.0f}, p={pv:.3g}')
        for tag, d in (('lateral', lat), ('axial', axi)):
            per = d.groupby('orbit').area
            if len(per):
                top = per.mean().idxmax(); v = d[d.orbit == top].area
                say(f'   largest {tag} interface: orbit {top}, '
                    f'{v.mean():.0f} +/- {v.std(ddof=1) if len(v)>1 else float("nan"):.0f} A2 '
                    f'(n={len(v)} symmetry copies)')


# ---------------------------------------------------------------- stacking geometry
with RPT.guard('stacking geometry', ok_detail='stacking_geometry.csv'):
    # For a stacked-ring assembly the ring-to-ring twist is defined only MODULO the
    # rotational symmetry of the ring itself: rotating a subunit by t, or by t +/- 360/n,
    # lands it on a different -- but equivalent -- neighbour in the ring above. Reporting
    # the smallest-magnitude representative is a trap, because it can name a glancing
    # contact rather than the one that actually stacks. The twist reported here is the one
    # belonging to the LARGEST interface of each axial category, which is the pair of
    # subunits that genuinely sit on top of each other.
    try:
        Rf = np.array(CFG.get('frame_rotation')) if CFG.get('frame_rotation') else None
        org = np.array(CFG.get('frame_origin')) if CFG.get('frame_origin') else None
        nfold = int(CFG.get('n_fold') or 1)
        axial = sorted({c for c in S.ctg.dropna().unique() if 'axial' in str(c)})
        if Rf is not None and axial:
            cau = {c.id: {r.id[1]: np.array(r['CA'].coord) for r in c if 'CA' in r} for c in model}
            say('\nstacking geometry, from the largest interface of each axial category:')
            rows_h = []
            for ctg in axial:
                sub = S[S.ctg == ctg].sort_values('area', ascending=False)
                a, b = sub.iloc[0].c1, sub.iloc[0].c2
                k = sorted(set(cau[a]) & set(cau[b]))
                if len(k) < 10:
                    continue
                P = (np.array([cau[a][i] for i in k]) - org) @ Rf.T
                Q = (np.array([cau[b][i] for i in k]) - org) @ Rf.T
                Pc, Qc = P - P.mean(0), Q - Q.mean(0)
                U, _, Vt = np.linalg.svd(Pc.T @ Qc)
                d = np.sign(np.linalg.det(Vt.T @ U.T))
                R = Vt.T @ np.diag([1, 1, d]) @ U.T
                tw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                rise = float(Q.mean(0)[2] - P.mean(0)[2])
                rows_h.append(dict(category=ctg, chainA=a, chainB=b, area=sub.iloc[0].area,
                                   twist_deg=round(tw, 2), rise_A=round(rise, 2)))
                say(f'  {ctg}: {a}->{b} ({sub.iloc[0].area:.0f} A2)  '
                    f'twist {tw:+.2f} deg, rise {rise:+.2f} A')
            if rows_h and nfold > 1:
                say(f'  the twist is defined modulo {360.0/nfold:.0f} deg (the ring is '
                    f'{nfold}-fold), so {rows_h[0]["twist_deg"]:+.2f} and '
                    f'{rows_h[0]["twist_deg"] - 360.0/nfold:+.2f} describe the same operation. '
                    'State the convention in the paper, and prefer the value above: it is the '
                    'one relating the subunits that actually bury surface against each other.')
            if rows_h:
                pd.DataFrame(rows_h).to_csv('stacking_geometry.csv', index=False)
    except Exception as e:
        say(f'\nstacking geometry skipped ({type(e).__name__}: {e})')

# ---------------------------------------------------------------- energetics
with RPT.guard('energetics', ok_detail='interface_energetics.csv'):
    # Solvation free energy from buried area (Eisenberg & McLachlan 1986). Read this as a
    # physically-grounded RANKING of interfaces, not as an affinity: there is no
    # conformational entropy, no translational/rotational entropy of association, no
    # explicit electrostatics and no strain term. It is not comparable to an ITC or SPR
    # number, and it is not PISA's dG_diss (which adds an entropy estimate).
    if 'dg_solv' in S.columns:
        E = S.groupby('orbit').agg(category=('ctg', 'first'), chainA=('c1', 'first'),
                                   chainB=('c2', 'first'), n_copies=('area', 'size'),
                                   area=('area', 'mean'), dg_solv=('dg_solv', 'mean'),
                                   dg_sd=('dg_solv', 'std'),
                                   dg_per_1000A2=('dg_solv_per_1000A2', 'mean'),
                                   unsat_polar=('n_buried_unsat_polar', 'mean')).reset_index()
        E['dg_sd'] = np.where(E.n_copies > 1, E.dg_sd, np.nan)
        E = E.sort_values('dg_solv')
        E.round(3).to_csv('interface_energetics.csv', index=False)
        say('\nsolvation free energy (Eisenberg-McLachlan ASP; a ranking, NOT an affinity):')
        say(f'  whole model: {S.dg_solv.sum():.0f} kcal/mol over {len(S)} interfaces')
        say('  most favourable interfaces:')
        for _, r in E.head(5).iterrows():
            sd = f' +/- {r.dg_sd:.1f}' if np.isfinite(r.dg_sd) else ''
            say(f'    {r.chainA}-{r.chainB} ({r.category}): {r.dg_solv:.1f}{sd} kcal/mol, '
                f'{r.dg_per_1000A2:.1f} per 1000 A2, {r.area:.0f} A2')
        say('  by category, kcal/mol per 1000 A2 buried (more negative = more hydrophobic burial):')
        say(S.groupby('ctg').dg_solv_per_1000A2.mean().sort_values().round(2).to_string())

# ---------------------------------------------------------------- packing quality
with RPT.guard('packing statistics', ok_detail='reported'):
    # Present unless step 01 was run with --no-shape.
    if 'sc' in S.columns and S.sc.notna().any():
        SHP = S.dropna(subset=['sc'])
        say('\nshape complementarity (Lawrence & Colman Sc) and gap index, by category:')
        T = SHP.groupby('ctg').agg(n=('sc', 'size'), area_min=('area', 'min'),
                                   area_max=('area', 'max'), sc_median=('sc', 'median'),
                                   sc_sd=('sc', 'std'),
                                   gap_index=('gap_index', 'median')
                                   ).sort_values('sc_median', ascending=False)
        say(T.round(3).to_string())
        cvs = SHP.groupby('orbit').sc.std() / SHP.groupby('orbit').sc.mean()
        if cvs.notna().any():
            say(f'  reproducibility across symmetry copies: median CV of Sc within an '
                f'orbit {100*float(cvs.median()):.1f} %')
        big = SHP[SHP.area >= 250]
        if len(big) > 8:
            rs = stats.spearmanr(big.area, big.sc)
            say(f'  Sc still rises with interface size among pairs >= 250 A2 '
                f'(Spearman r = {rs.statistic:.2f}, p = {rs.pvalue:.2g}, n = {len(big)}), '
                'so compare interfaces of similar area, not categories of different size.')
        say('  Sc is a SURFACE-FIT measure, not an energy: 1.0 is perfect interdigitation. '
            'This grid implementation reads about 0.05 low against the original sc program '
            '(0.65 here for trypsin-BPTI, 2PTC) and drifts by ~0.06 between 0.6 and 0.35 A '
            'grids, so use it to rank interfaces within one run at one spacing.')
        say('  gap index = trapped empty volume between the partners divided by the '
            'interface area; higher means the two surfaces fail to fill each other.')
        tot_unsat = S.n_buried_unsat_polar.sum()
        say(f'\nburied polar atoms with no cross-interface H-bond or salt bridge: {tot_unsat:.0f} '
            f'({tot_unsat/max(1,len(S)):.1f} per interface) — an energetic penalty and a '
            'model-quality flag; a high count at one interface is worth inspecting in the map')

# ---------------------------------------------------------------- B-factors at interfaces
with RPT.guard('B-factors at interfaces', ok_detail='reported'):
    try:
        bf = {}
        for c in model:
            for r in c:
                v = [a.get_bfactor() for a in r]
                if v:
                    bf[(c.id, r.id[1])] = float(np.mean(v))
        if bf and len(set(bf.values())) > 2:
            iface = {(r.chain, r.resi) for _, r in RES.iterrows()}
            inside = [v for k, v in bf.items() if k in iface]
            outside = [v for k, v in bf.items() if k not in iface]
            if inside and outside:
                mi, mo = float(np.median(inside)), float(np.median(outside))
                rel = (mi - mo) / mo if mo else 0.0
                # thousands of residues make any difference "significant"; judge the size, not p
                verdict = ('interfaces are the better-ordered part of the model, as expected'
                           if rel < -0.10 else
                           'interfaces are MORE mobile than the rest of the model — inspect the '
                           'density there before interpreting individual contacts' if rel > 0.10
                           else 'difference under 10%, i.e. interfaces are ordered like the rest '
                                'of the model; with this many residues a small difference is '
                                'statistically significant but not meaningful')
                u, pv = stats.mannwhitneyu(inside, outside, alternative='two-sided')
                say(f'\nB-factor / ADP: interface residues median {mi:.1f} vs non-interface '
                    f'{mo:.1f} ({rel:+.1%}, Mann-Whitney p={pv:.2g}) — {verdict}.')
        else:
            say('\nB-factor column is absent or constant — skipped.')
    except Exception as e:
        say(f'\nB-factor check skipped ({type(e).__name__}: {e})')

# ---------------------------------------------------------------- bond reproducibility
with RPT.guard('bond reproducibility', ok_detail='interface_bonds_unique.csv'):
    Bk = B.set_index(['c1', 'c2']).sort_index()
    rows = []
    for orbit, members in orb_members.items():
        sub = S[S.orbit == orbit]
        if sub.empty:
            continue
        rep = sub.sort_values('area', ascending=False).iloc[0]
        present, dist = collections.defaultdict(set), collections.defaultdict(list)
        for mi, (x, y) in enumerate(members):
            key = (x, y) if (x, y) in Bk.index else ((y, x) if (y, x) in Bk.index else None)
            if key is None:
                continue
            flip = key != (x, y)
            for _, bb in Bk.loc[[key]].iterrows():
                r1, i1 = (bb.rn2, bb.ri2) if flip else (bb.rn1, bb.ri1)
                r2, i2 = (bb.rn1, bb.ri1) if flip else (bb.rn2, bb.ri2)
                k = (bb.kind, i1, r1, i2, r2)
                present[k].add(mi); dist[k].append(bb.dist)
        for k, s in present.items():
            rows.append(dict(orbit=orbit, interface=f'{rep.c1}–{rep.c2}', category=rep.ctg,
                             area=round(rep.area), kind=k[0], res1=f'{k[2]}{k[1]}',
                             res2=f'{k[4]}{k[3]}', copies=len(s), n_copies_possible=len(members),
                             min_dist=round(min(dist[k]), 2),
                             confidence=CFG['bond_confidence']))
    BR = pd.DataFrame(rows)
    BR.to_csv('interface_bonds_unique.csv', index=False)
    if len(BR):
        full = BR[BR.copies == BR.n_copies_possible]
        say(f'\nunique residue-residue bonds: {len(BR)}; reproduced in every symmetry copy: '
            f'H-bonds {100*(full.kind=="hbond").sum()/max(1,(BR.kind=="hbond").sum()):.0f}%, '
            f'salt bridges {100*(full.kind=="saltbridge").sum()/max(1,(BR.kind=="saltbridge").sum()):.0f}%')
    if CFG['bond_confidence'] in ('low', 'unknown'):
        say(f'  bond confidence is "{CFG["bond_confidence"]}" at {CFG["resolution"]} A — '
            'treat individual H-bonds and salt bridges as tentative, not assignments.')

# ---------------------------------------------------------------- aromatic interactions
with RPT.guard('aromatic interactions', ok_detail='aromatic_interactions.csv'):
    RING = {'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TRP': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'], 'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2']}
    # Nucleobases are aromatic too. The whole fused purine system gives the centroid and
    # normal; both ring systems are planar, so the SVD normal is well defined. With these in
    # the table, base-base stacking AND protein-ring-to-base stacking fall out of exactly
    # the same geometry as protein ring pairs, and are labelled separately below.
    for _n in ('A', 'G', 'DA', 'DG', 'I'):
        RING[_n] = ['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8', 'N9']
    for _n in ('C', 'U', 'T', 'DC', 'DT', 'DU'):
        RING[_n] = ['N1', 'C2', 'N3', 'C4', 'C5', 'C6']
    NUCBASE = {'A', 'G', 'DA', 'DG', 'I', 'C', 'U', 'T', 'DC', 'DT', 'DU'}
    CATION = {'LYS': ['NZ'], 'ARG': ['NE', 'CZ', 'NH1', 'NH2']}
    rings, cats = [], []
    for c in model:
        for r in c:
            rn = r.get_resname().strip()
            if rn in RING and all(a in r for a in RING[rn]):
                P = np.array([r[a].coord for a in RING[rn]]); ctr = P.mean(0)
                rings.append((c.id, r.id[1], rn, ctr, np.linalg.svd(P - ctr)[2][2]))
            if rn in CATION and all(a in r for a in CATION[rn]):
                cats.append((c.id, r.id[1], rn, np.mean([r[a].coord for a in CATION[rn]], axis=0)))
    arows = []
    if rings:
        RC = np.array([x[3] for x in rings]); tr = cKDTree(RC)
        for i, j in tr.query_pairs(5.5):
            if rings[i][0] == rings[j][0]:
                continue
            ang = np.degrees(np.arccos(min(1, abs(rings[i][4] @ rings[j][4]))))
            kind = 'pi_stack_parallel' if ang <= 30 else ('pi_stack_T' if ang >= 60 else None)
            if kind:
                nb = (rings[i][2] in NUCBASE) + (rings[j][2] in NUCBASE)
                if nb == 2:                      # base on base
                    kind = kind.replace('pi_stack', 'base_stack')
                elif nb == 1:                    # protein ring on base
                    kind = kind.replace('pi_stack', 'base_amino_stack')
                arows.append((rings[i][0], rings[i][1], rings[i][2], rings[j][0], rings[j][1],
                              rings[j][2], kind, round(float(np.linalg.norm(RC[i] - RC[j])), 2), round(ang, 1)))
        for chn, ri_, rn_, cc in cats:
            for k in tr.query_ball_point(cc, 6.0):
                if chn == rings[k][0]:
                    continue
                v = cc - RC[k]; d = float(np.linalg.norm(v))
                off = np.degrees(np.arccos(min(1, abs(v @ rings[k][4]) / d)))
                if off <= 40:
                    arows.append((chn, ri_, rn_, rings[k][0], rings[k][1], rings[k][2],
                                  'cation_pi', round(d, 2), round(off, 1)))
    AR = pd.DataFrame(arows, columns=['c1', 'ri1', 'rn1', 'c2', 'ri2', 'rn2', 'kind', 'dist', 'angle'])
    if len(AR):
        AR['ctg'] = [ctg_of(a, b) for a, b in zip(AR.c1, AR.c2)]
        AR = AR.dropna(subset=['ctg'])
    AR.to_csv('aromatic_interactions.csv', index=False)
    if CFG.get('has_ligand') and os.path.exists('ligand_contacts.csv'):
        LC = pd.read_csv('ligand_contacts.csv')
        if len(LC):
            if 'name' not in LC.columns:
                LC['name'] = LC.resn
            g = LC.groupby(['chain', 'resi', 'resn', 'name']).agg(
                partners=('partner_chain', 'nunique'), buried=('dsasa', 'sum'),
                hbonds=('n_hbond', 'sum'), salt=('n_saltbridge', 'sum'))
            say('\nbound ligands at inter-chain interfaces (ligand_contacts.csv):')
            say(g.sort_values('buried', ascending=False).round(1).to_string())
            say('  a ligand with partners >= 2 bridges two chains and is part of the '
                'interface, not just bound to one subunit.')
            say('  ligand atoms carry no Eisenberg solvation parameter, so they add buried '
                'area but 0 to dG_solv.')
        else:
            say('\nbound ligands are present but none lies at an inter-chain interface')

    say('\ninter-chain aromatic interactions: ' +
        (AR.kind.value_counts().to_dict().__repr__() if len(AR) else 'none'))
    if CFG['has_nucleic']:
        nst = int(AR.kind.str.startswith('base_').sum()) if len(AR) else 0
        say(f'  (nucleobases are included: {nst} of these involve a base — '
            '"base_stack_*" is base on base, "base_amino_stack_*" is a protein ring on a base)')

# ---------------------------------------------------------------- chemistry
with RPT.guard('chemistry tables', ok_detail='composition and density tables'):
    RES['ctg'] = [ctg_of(a, b) for a, b in zip(RES.c1, RES.c2)]
    RES['cls'] = RES.resn.map(CLASS).fillna('other')
    COMP = RES.groupby(['ctg', 'cls']).dsasa.sum().unstack(fill_value=0)
    COMP = COMP.div(COMP.sum(axis=1), axis=0)
    COMP.to_csv('interface_composition_by_category.csv')
    ORBC = S.groupby('orbit').agg(ctg=('ctg', 'first'), area=('area', 'mean'),
                                  ap=('apolar_frac', 'mean'), hb=('n_hbond', 'mean'),
                                  sb=('n_saltbridge', 'mean')).reset_index()
    ORBC['bond_density'] = 1000 * (ORBC.hb + ORBC.sb) / ORBC.area
    ORBC['sb_density'] = 1000 * ORBC.sb / ORBC.area
    ORBC.to_csv('interface_orbit_chemistry.csv', index=False)
    n_units = max(1, CFG['group_order'])
    dens = []
    for c in CATS:
        sub = ORBC[ORBC.ctg == c]; A3 = sub.area.sum()
        nar = AR[AR.ctg == c] if len(AR) else AR
        dens.append(dict(ctg=c, salt=1000 * sub.sb.sum() / A3, hbond=1000 * sub.hb.sum() / A3,
                         stack=(1000 * nar.kind.str.startswith('pi_stack').sum() / n_units / A3) if len(nar) else 0.0,
                         catpi=(1000 * (nar.kind == 'cation_pi').sum() / n_units / A3) if len(nar) else 0.0))
    DEN = pd.DataFrame(dens).set_index('ctg')
    DEN.to_csv('interaction_density_by_category.csv')
    say('\ninteraction density per 1000 A2 buried:'); say(DEN.round(2).to_string())

    TOT = S.area.sum()
    share = (100 * S.groupby('ctg').area.sum() / TOT).sort_values(ascending=False)
    say(f'\ntotal buried surface in the model: {TOT:,.0f} A2'); say(share.round(1).to_string())
    json.dump({**CFG, 'categories': CATS, 'total_buried_area': float(TOT)},
              open('run_config.json', 'w'), indent=1)
    say('\nDONE')
RPT.done()
