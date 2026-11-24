"""Step 01 — contacts, symmetry and buried surface for ANY multi-chain assembly.

Generalised from the tail1-specific original: nothing is hard-coded about the
symmetry, the number of chains, or which protein is which.

    python 01_interface_analysis.py MODEL.{pdb,cif} [options]

Options
  --resolution R     map resolution in Angstrom. Gates how confidently bond-level
                     geometry is reported downstream. Omit only if truly unknown.
  --names a,b,c      friendly names for the entities, in order of first appearance
                     (e.g. tube,belt,fibre). Defaults to entity_1, entity_2, ...
  --intra            additionally compute INTRA-monomer burial (fold core), which the
                     inter-chain analysis deliberately does not see. Off by default.
  --contact-cut X    heavy-atom contact cutoff, default 5.0 A.
  --sasa-points N    Shrake-Rupley test points per atom, default 100.

Outputs
  run_config.json         symmetry, entities, resolution, confidence gating
  chain_geometry.csv      per chain: entity, cylindrical position in the symmetry frame
  chain_environments.csv  per chain: its symmetry orbit (independent environment)
  interfaces_summary.csv  one row per contacting chain pair
  interface_residues.csv  per-residue dSASA for every pair
  interface_bonds.csv     salt bridges, H-bonds (hydrophobic contacts are counted only)
  interface_links.csv     disulfides, isopeptide bonds, bridging metals, short contacts
  interface_hydrophobic.csv  apolar contact counts per residue pair, the non-specific glue
  orbits.csv              symmetry orbit membership of every pair
  residue_intra_burial.csv  (only with --intra) fold-core burial, RSA per residue
"""
import sys, os, json, argparse, collections, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model as BModel
from Bio.PDB.SASA import ShrakeRupley

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import symmetry as sym

T31 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'MSE': 'M'}
# Tien et al. 2013 theoretical maximum accessible surface, for relative accessibility
MAXASA = {'ALA': 129, 'ARG': 274, 'ASN': 195, 'ASP': 193, 'CYS': 167, 'GLU': 223,
          'GLN': 225, 'GLY': 104, 'HIS': 224, 'ILE': 197, 'LEU': 201, 'LYS': 236,
          'MET': 224, 'PHE': 240, 'PRO': 159, 'SER': 155, 'THR': 172, 'TRP': 285,
          'TYR': 263, 'VAL': 174, 'MSE': 224}
APOLAR = {'C', 'S'}
ACID = {('ASP', 'OD1'), ('ASP', 'OD2'), ('GLU', 'OE1'), ('GLU', 'OE2')}
BASE = {('ARG', 'NE'), ('ARG', 'NH1'), ('ARG', 'NH2'), ('LYS', 'NZ'),
        ('HIS', 'ND1'), ('HIS', 'NE2')}
PHOSPHATE = {'OP1', 'OP2', 'O1P', 'O2P', 'OP3'}      # nucleic backbone, also anionic
WATERS = {'HOH', 'WAT', 'DOD', 'H2O', 'TIP', 'SOL'}
# Crystallisation additives are kept but FLAGGED, never silently dropped: a sulfate or a
# glycerol is usually cryoprotectant, but occasionally it marks a real anion site.
ADDITIVES = {'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'PG4', 'PGE', 'MPD', 'ACT', 'FMT', 'DMS',
             'TRS', 'EPE', 'MES', 'IMD', 'CIT', 'TLA', 'BME', 'DTT', 'NO3', 'AZI', 'IOD',
             'BR', 'CL', 'F', 'FLC', 'ACY', 'BCT'}
LIGAND_RESN = set()                 # filled by load_model; used by atom_class
METAL_IONS = {'ZN', 'MG', 'CA', 'FE', 'FE2', 'MN', 'CU', 'CU1', 'NI', 'CO', 'CD',
              'NA', 'K', 'MO', 'W', 'HG', 'PT', 'AU', 'AG', 'SR', 'BA'}

# Atomic solvation parameters, Eisenberg & McLachlan (1986) Nature 319:199, in
# kcal mol^-1 A^-2. These are defined on EXPOSED area: dG_solv = sum(sigma_i * ASA_i),
# so sigma(C) > 0 means exposing carbon costs energy. The free energy change on
# ASSOCIATION is therefore dG = sum(sigma_i * (ASA_bound - ASA_free)) = -sum(sigma_i * dASA_i),
# i.e. the buried area enters with a MINUS sign. Burying apolar carbon is favourable
# (negative); burying charged nitrogen without compensation is unfavourable (positive).
# The result is a SOLVATION free energy,
# which is NOT a binding free energy: it contains no conformational entropy, no
# translational/rotational entropy of association, no explicit electrostatics and no
# strain. Treat it as a physically-grounded ranking of interfaces, not as an affinity.
# Eisenberg & McLachlan give no parameter for phosphorus, so a nucleic-acid backbone P
# is given sigma = 0 rather than being folded into one of the protein classes: its
# anionic phosphate oxygens already carry the charge through the 'O-' class.
ASP_SIGMA = {'C': 0.016, 'N/O': -0.006, 'O-': -0.024, 'N+': -0.050, 'S': 0.021,
             'P': 0.000, 'ligand': 0.000}
CATIONIC = {('ARG', 'NE'), ('ARG', 'NH1'), ('ARG', 'NH2'), ('LYS', 'NZ')}


def atom_class(resn, an, el):
    # Eisenberg & McLachlan parameterised the twenty amino acids only, so a cofactor
    # atom is given sigma = 0 rather than being forced into a protein class. Ligand
    # atoms therefore contribute buried AREA but no solvation energy, and the report
    # says so.
    if resn in LIGAND_RESN:
        return 'ligand'
    if el == 'C':
        return 'C'
    if el == 'S':
        return 'S'
    if el == 'P':
        return 'P'
    if el == 'O':
        return 'O-' if ((resn, an) in ACID or an in PHOSPHATE or an == 'OXT') else 'N/O'
    if el == 'N':
        return 'N+' if (resn, an) in CATIONIC else 'N/O'
    return 'N/O'


def log(*a):
    print(*a, flush=True)
    LOG.write(' '.join(map(str, a)) + '\n')


def load_model(path, keep_ligands=True, min_ligand_atoms=5):
    """Parse, drop waters/ligands, set hydrogens aside, and make residue numbers unique.

    Insertion codes matter: a crystal structure using Kabat or Chothia numbering can
    carry 100, 100A, 100B in one chain. Everything downstream keys residues by their
    integer number, so without this those three would silently collapse into one. Where
    insertion codes are present the chain is renumbered sequentially and the original
    labels are written to residue_renumbering.csv.
    """
    p = MMCIFParser(QUIET=True) if path.lower().endswith(('.cif', '.mmcif')) else PDBParser(QUIET=True)
    model = p.get_structure('m', path)[0]
    hyd, renum, metals, ligands = [], [], [], []
    for ch in model:
        for r in list(ch):
            if r.id[0] != ' ':                        # waters, ligands, ions
                nm = r.get_resname().strip().upper()
                if nm in WATERS:
                    ch.detach_child(r.id); continue
                if nm in METAL_IONS:                  # kept: they can bridge chains
                    for a in r:
                        metals.append((ch.id, r.id[1], nm, np.array(a.coord, float)))
                    ch.detach_child(r.id); continue
                heavy = [a for a in r if a.element != 'H']
                if keep_ligands and len(heavy) >= min_ligand_atoms:
                    # A cofactor at an interface is part of what holds the assembly
                    # together, so it STAYS in its chain and enters the surface,
                    # contact and bond calculations like any other residue.
                    LIGAND_RESN.add(nm)
                    ligands.append(dict(chain=ch.id, resi=int(r.id[1]), resn=nm,
                                        n_atoms=len(heavy),
                                        is_additive=nm in ADDITIVES))
                    for a in [a for a in r if a.element == 'H']:
                        hyd.append((ch.id, r.id[1], np.array(a.coord)))
                        r.detach_child(a.get_id())
                    continue
                ch.detach_child(r.id); continue
            for a in [a for a in r if a.element == 'H']:
                hyd.append((ch.id, r.id[1], np.array(a.coord)))
                r.detach_child(a.get_id())
    # A kept ligand may carry the same residue NUMBER as a polymer residue in the same
    # chain. Everything downstream keys residues by that integer, so a collision is
    # resolved here (ligand moved past the end of the chain) and recorded.
    for ch in model:
        poly = [r.id[1] for r in ch if r.id[0] == ' ']
        if not poly:
            continue
        taken = set(poly)
        top = max(poly)
        for r in [r for r in ch if r.id[0] != ' ']:
            if r.id[1] in taken:
                top += 1
                renum.append(dict(chain=ch.id, original=f'{r.get_resname()}{r.id[1]}',
                                  renumbered=top, reason='ligand number collided'))
                for lg in ligands:
                    if lg['chain'] == ch.id and lg['resi'] == r.id[1] \
                            and lg['resn'] == r.get_resname().strip().upper():
                        lg['resi'] = top
                ch.detach_child(r.id)
                r.id = (r.id[0], top, r.id[2])
                ch.add(r)
            taken.add(r.id[1])
    for ch in list(model):
        if len(list(ch)) == 0:
            model.detach_child(ch.id); continue
        res = list(ch)
        if any(r.id[2] != ' ' for r in res) or len({r.id[1] for r in res}) != len(res):
            for r in res:                             # detach first: ids must stay unique
                ch.detach_child(r.id)
            for i, r in enumerate(res, start=1):
                renum.append(dict(chain=ch.id, new_resi=i, orig_resseq=r.id[1],
                                  icode=r.id[2].strip(), resn=r.get_resname().strip()))
                r.id = (' ', i, ' ')
                ch.add(r)
    if renum:
        pd.DataFrame(renum).to_csv('residue_renumbering.csv', index=False)
    return model, hyd, bool(renum), metals, ligands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--resolution', type=float, default=None)
    ap.add_argument('--names', default=None)
    ap.add_argument('--intra', action='store_true')
    ap.add_argument('--contact-cut', type=float, default=5.0)
    ap.add_argument('--no-shape', dest='shape', action='store_false',
                    help='skip shape complementarity (Lawrence & Colman Sc) and gap volume '
                         '(Laskowski). They are computed by default for every contacting '
                         'pair: about a minute for a dimer, ten minutes or more for ~30 '
                         'chains. Results go to interface_shape.csv and the sc / gap_index '
                         'columns of interfaces_summary.csv.')
    ap.add_argument('--shape', dest='shape', action='store_true',
                    help=argparse.SUPPRESS)          # kept so older commands still work
    ap.set_defaults(shape=True)
    ap.add_argument('--shape-spacing', type=float, default=0.5, metavar='A',
                    help='grid step for the shape calculation (default 0.5 A). Finer grids '
                         'lower Sc slightly (0.58 at 0.6 A to 0.52 at 0.35 A on one test '
                         'interface), so keep it fixed when comparing structures.')
    ap.add_argument('--shape-workers', type=int, default=0, metavar='N',
                    help='processes for the shape calculation (default: half the cores).')
    ap.add_argument('--no-ligands', action='store_true',
                    help='discard bound ligands and cofactors. By default a '
                         'HETATM group with at least --min-ligand-atoms heavy '
                         'atoms is KEPT and takes part in the surface, contact '
                         'and bond analysis like any other residue; waters are '
                         'always discarded and metal ions follow their own path.')
    ap.add_argument('--min-ligand-atoms', type=int, default=5, metavar='N',
                    help='smallest HETATM group treated as a ligand (default 5 '
                         'heavy atoms). Common crystallisation additives are kept '
                         'but flagged is_additive in ligands.csv so you can '
                         'filter them.')
    ap.add_argument('--sasa-points', type=int, default=100)
    ap.add_argument('--proteins', default=None, metavar='FASTA',
                    help='names and colours of the proteins: a FASTA whose headers are '
                         '"name_#RRGGBB" (colour optional). Matched to the model by '
                         'sequence; anything that does not match runs with defaults.')
    ap.add_argument('--ligand-names', default=None, metavar='FILE',
                    help='names and colours of bound ligands: one per line, '
                         '"CODE  name  #RRGGBB" (CODE = the 3-letter residue name in the '
                         'model, e.g. "UMP  dUMP  #F0A030"; name and colour optional). '
                         'Used in the ligand tables, the residue labels of the figures and '
                         'the ChimeraX scripts.')
    args = ap.parse_args()

    global LOG
    LOG = open('analysis.log', 'w', buffering=1)
    t_start = time.time()
    import runlog
    for _f in ('run_report.json', 'run_report.txt', 'run_errors.log'):
        if os.path.exists(_f):          # step 01 starts a fresh report for a fresh run
            os.remove(_f)
    RPT = runlog.Report('01', model=args.model)

    with RPT.guard('read model', ok_detail=None, fatal=True):
        if not os.path.exists(args.model):
            raise FileNotFoundError(f'{args.model} not found in {os.getcwd()}')
        model, hyd, renumbered, metals, ligands = load_model(
            args.model, keep_ligands=not args.no_ligands,
            min_ligand_atoms=args.min_ligand_atoms)
        if len(model) == 0:
            raise ValueError('the file contains no chains with coordinates')
    chains = [c.id for c in model]
    RPT.ok('read model', f'{len(chains)} chains, {sum(len(list(c)) for c in model)} residues, '
                       f'{len(hyd)} hydrogens')
    if len(chains) < 2:
        RPT.warn('read model', 'only ONE chain in the file: there is no inter-chain interface '
                             'to measure. If this is a crystal entry, download its biological '
                             'assembly instead.')
    RPT.add('hydrogens', 'ok' if hyd else 'WARNING',
          'present: H-bonds use distance + angle' if hyd else
          'none in the model: H-bonds use distance only, which over-counts')
    _res_src = None
    if args.resolution is None:
        # Read it from the file header when not given: PDB REMARK 2, or the mmCIF refinement
        # (X-ray) / 3D-reconstruction (cryo-EM) items. Stops at the first coordinate record.
        try:
            import re
            _pat = re.compile(r'^(?:REMARK   2 RESOLUTION\.\s+([0-9.]+)\s+ANGSTROM'
                              r'|_refine\.ls_d_res_high\s+([0-9.]+)'
                              r'|_em_3d_reconstruction\.resolution\s+([0-9.]+)'
                              r'|_reflns\.d_resolution_high\s+([0-9.]+))')
            with open(args.model, encoding='utf-8', errors='replace') as _fh:
                for _ln in _fh:
                    if _ln.startswith(('ATOM', 'HETATM')):
                        break
                    _m = _pat.match(_ln)
                    if _m:
                        _v = float(next(g for g in _m.groups() if g))
                        if 0.3 < _v < 50:
                            args.resolution, _res_src = _v, 'file header'
                            break
        except Exception:
            pass
    if args.resolution is None:
        RPT.warn('resolution', 'not given (--resolution) and not in the file header: bonds '
                               'are reported as tentative')
    elif _res_src:
        RPT.ok('resolution', f'{args.resolution:g} Å, read from the file header '
                             '(--resolution overrides it)')
    else:
        RPT.ok('resolution', f'{args.resolution:g} Å')
    log(f'model {os.path.basename(args.model)}: {len(chains)} chains, '
        f'{sum(len(list(c)) for c in model)} residues, {len(hyd)} hydrogens (kept for H-bond geometry)')
    if renumbered:
        log('  NOTE: insertion codes or duplicate residue numbers found; affected chains were '
            'renumbered sequentially. Original labels are in residue_renumbering.csv.')
    if not hyd:
        log('  NOTE: no hydrogens in the model (usual for X-ray). Hydrogen bonds fall back to a '
            'donor-acceptor DISTANCE criterion with no angle check, which over-counts; '
            'add riding hydrogens first if bond geometry matters.')

    # ---------------------------------------------------------------- entities
    seqs = sym.chain_sequences(model, T31)
    ent_of, reps = sym.group_entities(seqs)
    names = [f'entity_{i+1}' for i in range(len(reps))]
    colours = {}
    if args.names:
        given = [x.strip() for x in args.names.split(',') if x.strip()]
        if len(given) == len(reps):
            names = given
            RPT.ok('protein names (--names)', ', '.join(names))
        else:
            log(f'WARNING: {len(given)} names given for {len(reps)} entities; using defaults')
            RPT.default('protein names (--names)',
                      f'{len(given)} names given for {len(reps)} proteins found; '
                      f'ran with {", ".join(names)}')
    if args.proteins:
        with RPT.guard('protein details (FASTA)', ok_detail=None):
            import proteins as prot_io
            recs = prot_io.read_proteins(args.proteins)
            if not recs:
                raise ValueError(f'{args.proteins} contains no FASTA records')
            hits = prot_io.match_entities(reps, recs)
            notes, n_hit, used = [], 0, {}
            for i, (rec, cov, note) in enumerate(hits):
                if rec is None:
                    notes.append(f'{names[i]}: {note}, kept default name and colour')
                    continue
                if rec['name'] in used:
                    notes.append(f'{names[i]}: also matches "{rec["name"]}", already used by '
                                 f'{used[rec["name"]]}; kept default name')
                    continue
                used[rec['name']] = names[i]
                names[i] = rec['name']; n_hit += 1
                if rec['colour']:
                    colours[rec['name']] = rec['colour']
                notes += [f'{rec["name"]}: {p}' for p in rec['problems']]
            unused = [r['name'] for r in recs if r['name'] not in used]
            if unused:
                notes.append('in the FASTA but not in the model: ' + ', '.join(unused))
            summary = (f'{n_hit} of {len(reps)} protein(s) matched by sequence'
                       + (f'; colours set for {len(colours)}' if colours else ''))
            for nline in notes:
                log('  proteins: ' + nline)
            serious = [x for x in notes if not x.startswith('in the FASTA but not')]
            if n_hit == len([r for r in reps if r[1] != 'ligand']) and not serious:
                RPT.ok('protein details (FASTA)', summary + ': ' + ', '.join(names)
                     + ('' if not notes else ' (' + '; '.join(notes) + ')'))
            elif n_hit:
                RPT.warn('protein details (FASTA)', summary + '. ' + '; '.join(notes))
            else:
                RPT.default('protein details (FASTA)', 'nothing matched, ran with default '
                          'names and colours. ' + '; '.join(notes))
    elif not args.names:
        RPT.default('protein details', 'no --proteins FASTA or --names given; proteins are '
                  'called ' + ', '.join(names))
    kinds = [k for _, k in reps]
    has_nucleic = any(k == 'nucleic' for k in kinds)
    for i, (s, k) in enumerate(reps):
        log(f'  entity {i} "{names[i]}": {k}, {len(s)} residues, '
            f'{sum(1 for c in ent_of if ent_of[c] == i)} copies')
    if ligands:
        LG = pd.DataFrame(ligands)
        LG.to_csv('ligands.csv', index=False)
        nadd = int(LG.is_additive.sum())
        log(f'  {len(LG)} bound ligand/cofactor group(s) kept: '
            + ', '.join(f'{n}x{k}' for k, n in LG.resn.value_counts().items())
            + (f'  ({nadd} flagged as common crystallisation additives)' if nadd else ''))
        log('  NOTE: ligand atoms contribute buried AREA and contacts, but carry no '
            'Eisenberg solvation parameter, so they add 0 to dG_solv.')
    lig_names, lig_cols = {}, {}
    if args.ligand_names:
        with RPT.guard('ligand names', ok_detail=None):
            import proteins as prot_io
            given, probs = prot_io.read_ligand_names(args.ligand_names)
            present = {lg['resn'] for lg in ligands}
            for code, (nm, col) in given.items():
                if code not in present:
                    probs.append(f'{code}: no ligand with this code in the model')
                    continue
                lig_names[code] = nm
                if col:
                    lig_cols[code] = col
            if ligands:
                LG['name'] = [lig_names.get(x, x) for x in LG.resn]
                LG.to_csv('ligands.csv', index=False)
            for p in probs:
                log('  ligand names: ' + p)
            summ = ', '.join(f'{k} = {v}' for k, v in lig_names.items()) or 'none applied'
            if probs:
                RPT.warn('ligand names', summ + '. ' + '; '.join(probs))
            else:
                RPT.ok('ligand names', summ)
    # a chain that is nothing but one kind of ligand takes the ligand's name
    for i, (s_, k_) in enumerate(reps):
        if k_ == 'ligand' and s_ in lig_names and names[i].startswith('entity_'):
            names[i] = lig_names[s_]
            if s_ in lig_cols:
                colours[names[i]] = lig_cols[s_]
    if has_nucleic:
        log('  NOTE: nucleic-acid chains present. Buried area, contacts, phosphate salt '
            'bridges and base stacking (step 02) are all handled; secondary structure is '
            'assigned for protein chains only, and phosphorus carries no Eisenberg '
            'solvation parameter, so it contributes 0 to dG_solv.')

    # ---------------------------------------------------------------- symmetry
    # One anchor atom per residue per chain: CA for protein, C1' (else P) for a
    # nucleotide. The fallback has to be applied PER CHAIN, not only when the whole
    # model lacks CA: in a mixed protein/nucleic assembly the nucleic chains would
    # otherwise be dropped here and the symmetry search would fail on them.
    ca = {}
    for c in model:
        d = {r.id[1]: np.array(r['CA'].coord) for r in c if 'CA' in r}
        if not d:
            for r in c:
                for an in ("C1'", 'P', "O5'"):
                    if an in r:
                        d[r.id[1]] = np.array(r[an].coord)
                        break
        if d:
            ca[c.id] = d
    cen = {c.id: np.mean([a.coord for r in c for a in r], axis=0) for c in model}
    centre = np.mean(list(cen.values()), axis=0)
    try:
        ops = sym.find_symmetry_operators(ca, cen, ent_of)
        grp = sym.classify(ops, centre=centre)
        RPT.ok('symmetry search', f'{grp["point_group"]}, order {grp["order"]}')
    except Exception as exc:
        # Without symmetry the analysis still works: every chain becomes its own
        # environment and every interface its own orbit. Slower to read, not wrong.
        RPT.fail('symmetry search', exc, sys.exc_info()[2])
        RPT.warn('symmetry fallback', 'treated as C1 (no symmetry): every chain and every '
               'interface is reported separately')
        ops = [(np.eye(3), np.zeros(3), {c: c for c in chains})]
        grp = sym.classify(ops, centre=centre)
    corb = sym.chain_orbits(ops, chains)
    env_of = {c: ''.join(corb[c]) for c in chains}
    log(f'point group {grp["point_group"]} (order {grp["order"]}); '
        f'{len(set(env_of.values()))} symmetry-independent chain environments')
    screw = sym.find_screw_operators(ca, cen, ent_of)
    if screw:
        b = sorted(screw, key=lambda x: -x[2])[0]
        log(f'  local/screw symmetry also present: maps {b[2]}/{b[3]} chains, '
            f'rise {b[4]:.1f} A, twist {b[5]:.1f} deg — NOT used to merge interfaces, '
            'because it runs off the ends of the model')

    Rf, origin = sym.frame(grp['axis'], grp['axis_point'] if grp['axis_point'] is not None else centre)

    # ---------------------------------------------------------------- geometry
    atoms = []
    for c in model:
        for r in c:
            for a in r:
                atoms.append((c.id, r.id[1], r.get_resname().strip(), a.get_id(),
                              *a.coord, (a.element or a.get_id()[0]).upper()))
    A = pd.DataFrame(atoms, columns=['ch', 'ri', 'rn', 'an', 'x', 'y', 'z', 'el'])
    XYZ = A[['x', 'y', 'z']].to_numpy()
    loc = (XYZ - origin) @ Rf.T                       # symmetry frame: axis is +z
    A['cr'] = np.hypot(loc[:, 0], loc[:, 1]); A['cz'] = loc[:, 2]
    A['ct'] = np.degrees(np.arctan2(loc[:, 1], loc[:, 0]))
    geo = []
    for c in chains:
        s = A[A.ch == c]
        d = (cen[c] - origin) @ Rf.T
        geo.append(dict(chain=c, entity=ent_of[c], entity_name=names[ent_of[c]],
                        kind=kinds[ent_of[c]], nres=s.ri.nunique(),
                        r=float(np.hypot(d[0], d[1])), z=float(d[2]),
                        theta=float(np.degrees(np.arctan2(d[1], d[0]))),
                        rmin=float(s.cr.min()), rmax=float(s.cr.max()),
                        environment=env_of[c]))
    pd.DataFrame(geo).to_csv('chain_geometry.csv', index=False)
    pd.DataFrame([dict(chain=c, entity_name=names[ent_of[c]], environment=env_of[c])
                  for c in chains]).to_csv('chain_environments.csv', index=False)

    # ---------------------------------------------------------------- contacts
    tree = cKDTree(XYZ)
    chv = A.ch.to_numpy()
    pr = tree.query_pairs(r=args.contact_cut, output_type='ndarray')
    pr = pr[chv[pr[:, 0]] != chv[pr[:, 1]]]
    cnt = collections.Counter(tuple(sorted((chv[i], chv[j]))) for i, j in pr)
    pairs = [k for k, v in cnt.items() if v >= 5]
    log(f'{len(pairs)} contacting chain pairs (>=5 heavy-atom pairs < {args.contact_cut} A)')
    porb = sym.pair_orbits(ops, pairs)
    uniq = sorted({v for v in porb.values()})
    log(f'{len(uniq)} symmetry-unique interfaces; orbit sizes '
        f'{dict(collections.Counter(len(o) for o in uniq))}')
    pd.DataFrame([dict(orbit=i, members='/'.join(f'{a}{b}' for a, b in o), size=len(o))
                  for i, o in enumerate(uniq)]).to_csv('orbits.csv', index=False)
    orb_id = {tuple(sorted(p)): i for i, o in enumerate(uniq) for p in o}

    # -------------------------------------------------- covalent and metal links
    # Buried area and hydrogen bonds do not describe every way two chains can be
    # joined. A disulfide, an isopeptide cross-link or a bridging metal ion is a far
    # stronger connection than anything else measured here, and it is easy to miss by
    # eye in a large assembly. Near-miss disulfides are reported too: when one copy of
    # a symmetric pair is bonded and another is not, that is a model inconsistency
    # worth resolving before publication.
    links = []
    with RPT.guard('covalent and metal links', ok_detail=None):
        links = []
        ani, rni, rii = A.an.to_numpy(), A.rn.to_numpy(), A.ri.to_numpy()
        heavy = np.where(A.el.to_numpy() != 'H')[0]

        sgi = np.where((rni == 'CYS') & (ani == 'SG'))[0]
        if len(sgi) > 1:
            kt = cKDTree(XYZ[sgi])
            for si, sj in kt.query_pairs(4.5):
                a, b = sgi[si], sgi[sj]
                if chv[a] == chv[b]:
                    continue
                d = float(np.linalg.norm(XYZ[a] - XYZ[b]))
                links.append(dict(kind='disulfide', bonded=bool(d <= 2.5), distance=round(d, 2),
                                  c1=chv[a], resi1=int(rii[a]), resn1='CYS', atom1='SG',
                                  c2=chv[b], resi2=int(rii[b]), resn2='CYS', atom2='SG'))

        don = np.where((rni == 'LYS') & (ani == 'NZ'))[0]
        acc = np.where(((rni == 'ASN') & (ani == 'CG')) | ((rni == 'ASP') & (ani == 'CG')) |
                       ((rni == 'GLN') & (ani == 'CD')) | ((rni == 'GLU') & (ani == 'CD')))[0]
        if len(don) and len(acc):
            kt = cKDTree(XYZ[acc])
            for di in don:
                for dj in kt.query_ball_point(XYZ[di], 1.8):
                    b = acc[dj]
                    if chv[di] == chv[b]:
                        continue
                    links.append(dict(kind='isopeptide', bonded=True,
                                      distance=round(float(np.linalg.norm(XYZ[di] - XYZ[b])), 2),
                                      c1=chv[di], resi1=int(rii[di]), resn1='LYS', atom1='NZ',
                                      c2=chv[b], resi2=int(rii[b]), resn2=rni[b], atom2=ani[b]))

        for mch, mri, mnm, mxyz in metals:
            near = tree.query_ball_point(mxyz, 3.0)
            coord = [mk for mk in near if A.el.to_numpy()[mk] in ('N', 'O', 'S')]
            by_ch = sorted({chv[mk] for mk in coord})
            if len(by_ch) >= 2:
                for mk in coord:
                    links.append(dict(kind=f'metal bridge ({mnm})', bonded=True,
                                      distance=round(float(np.linalg.norm(XYZ[mk] - mxyz)), 2),
                                      c1=f'{mnm}{mri}', resi1=int(mri), resn1=mnm, atom1=mnm,
                                      c2=chv[mk], resi2=int(rii[mk]), resn2=rni[mk], atom2=ani[mk]))

        known = {(l['c1'], l['resi1'], l['c2'], l['resi2']) for l in links}
        for li, lj in tree.query_pairs(r=2.0, output_type='ndarray'):
            if chv[li] == chv[lj] or A.el.to_numpy()[li] == 'H' or A.el.to_numpy()[lj] == 'H':
                continue
            if (chv[li], int(rii[li]), chv[lj], int(rii[lj])) in known:
                continue
            links.append(dict(kind='unassigned short contact', bonded=False,
                              distance=round(float(np.linalg.norm(XYZ[li] - XYZ[lj])), 2),
                              c1=chv[li], resi1=int(rii[li]), resn1=rni[li], atom1=ani[li],
                              c2=chv[lj], resi2=int(rii[lj]), resn2=rni[lj], atom2=ani[lj]))

        if links:
            LK = pd.DataFrame(links).sort_values(['kind', 'distance'])
            LK.to_csv('interface_links.csv', index=False)
            for lk_kind, lk_g in LK.groupby('kind'):
                lk_nb = int(lk_g.bonded.sum())
                log(f'  {lk_kind}: {len(lk_g)} inter-chain, {lk_nb} within bonding distance')
            lk_ss = LK[LK.kind == 'disulfide']
            if len(lk_ss) and 0 < int(lk_ss.bonded.sum()) < len(lk_ss):
                log('  NOTE: some CYS-CYS pairs are bonded and others of the same kind are not '
                    f'({lk_ss[lk_ss.bonded].distance.min():.2f}-{lk_ss[lk_ss.bonded].distance.max():.2f} A vs '
                    f'{lk_ss[~lk_ss.bonded].distance.min():.2f}-{lk_ss[~lk_ss.bonded].distance.max():.2f} A). '
                    'Either the cross-link is substoichiometric or the restraint was applied '
                    'unevenly across the symmetry copies — worth checking in the map.')
            lk_nu = int((LK.kind == 'unassigned short contact').sum())
            if lk_nu:
                log(f'  {lk_nu} inter-chain heavy-atom pair(s) closer than 2.0 A with no assignable '
                    'bond — usually a mis-placed side chain, not chemistry. See interface_links.csv')
        else:
            log('  no disulfides, isopeptide links or bridging metals between chains')

    if not RPT._has('covalent and metal links'):
        _ss = sum(1 for l in links if l.get('kind') == 'disulfide' and l.get('bonded'))
        _ot = sum(1 for l in links if l.get('kind') != 'disulfide' and l.get('bonded'))
        RPT.ok('covalent and metal links',
             f'{_ss} bonded inter-chain disulfide(s), {_ot} other link(s)'
             if (_ss or _ot) else 'none between chains')
    # ---------------------------------------------------------------- SASA
    sr = ShrakeRupley(probe_radius=1.40, n_points=args.sasa_points)

    def sasa_atoms(chain_ids):
        s = Structure('s'); m = BModel(0); s.add(m)
        for cid in chain_ids:
            m.add(model[cid].copy())
        sr.compute(s, level='A')
        return {(c.id, r.id[1], a.get_id()): a.sasa for c in m for r in c for a in r}

    iso = {}
    for i, c in enumerate(chains):
        iso[c] = sasa_atoms([c])
    log(f'isolated-chain SASA done ({time.time()-t_start:.0f}s)')

    # optional intra-monomer burial — the fold core, invisible to the inter-chain analysis
    if args.intra:
        rows = []
        for c in chains:
            for r in model[c]:
                rn = r.get_resname().strip()
                s_fold = sum(iso[c].get((c, r.id[1], a.get_id()), 0.0) for a in r)
                mx = MAXASA.get(rn)
                rows.append(dict(chain=c, entity_name=names[ent_of[c]], resi=r.id[1], resn=rn,
                                 sasa_in_folded_chain=s_fold, max_asa=mx,
                                 intra_buried=(mx - s_fold) if mx else np.nan,
                                 rsa=(s_fold / mx) if mx else np.nan))
        pd.DataFrame(rows).to_csv('residue_intra_burial.csv', index=False)
        log('intra-monomer burial written (max-ASA reference, Tien et al. 2013)')

    # ---------------------------------------------------------------- bonds
    by_chain = {c: g.reset_index(drop=True) for c, g in A.groupby('ch')}
    H = collections.defaultdict(list)
    for cid, ri, xyz in hyd:
        H[(cid, ri)].append(xyz)

    def hmap(c):
        hv = by_chain[c]; out = collections.defaultdict(list)
        P = hv[['x', 'y', 'z']].to_numpy()
        for i in range(len(hv)):
            for h in H.get((c, int(hv.ri[i])), []):
                if np.linalg.norm(P[i] - h) < 1.35:
                    out[i].append(h)
        return out
    HM = {c: hmap(c) for c in chains} if hyd else {c: {} for c in chains}

    def analyse(c1, c2):
        a1, a2 = by_chain[c1], by_chain[c2]
        X1 = a1[['x', 'y', 'z']].to_numpy(); X2 = a2[['x', 'y', 'z']].to_numpy()
        close = cKDTree(X1).query_ball_tree(cKDTree(X2), args.contact_cut)
        hb, sb, phob = [], [], 0
        # per RESIDUE PAIR, not just a total: a non-specific interface is still held
        # together somewhere, and only a residue-level count can show where.
        phob_pairs = collections.Counter()
        for i, js in enumerate(close):
            if not js:
                continue
            e1, n1, r1 = a1.el[i], a1.an[i], a1.rn[i]
            for j in js:
                e2, n2, r2 = a2.el[j], a2.an[j], a2.rn[j]
                d = float(np.linalg.norm(X1[i] - X2[j]))
                if e1 in APOLAR and e2 in APOLAR and d < 4.5:
                    phob += 1
                    phob_pairs[(int(a1.ri[i]), r1, int(a2.ri[j]), r2)] += 1
                anion1 = (r1, n1) in ACID or n1 in PHOSPHATE
                anion2 = (r2, n2) in ACID or n2 in PHOSPHATE
                if d < 4.0 and ((anion1 and (r2, n2) in BASE) or (anion2 and (r1, n1) in BASE)):
                    sb.append((a1.ri[i], r1, n1, a2.ri[j], r2, n2, round(d, 2)))
                if d < 3.5 and e1 in 'NO' and e2 in 'NO':
                    ok = not hyd                      # no riding H: distance criterion only
                    for src, dst, hs in ((X1[i], X2[j], HM[c1].get(i, [])),
                                         (X2[j], X1[i], HM[c2].get(j, []))):
                        for h in hs:
                            v1, v2 = src - h, dst - h
                            ang = np.degrees(np.arccos(np.clip(
                                v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
                            if np.linalg.norm(h - dst) < 2.6 and ang > 120:
                                ok = True
                    if ok:
                        hb.append((a1.ri[i], r1, n1, a2.ri[j], r2, n2, round(d, 2)))
        dim = sasa_atoms([c1, c2])
        rows, tot = [], {'polar': 0.0, 'apolar': 0.0}
        dg_total = 0.0
        unsat = 0
        bonded = {(c1, t[0], t[2]) for t in hb} | {(c2, t[3], t[5]) for t in hb} | \
                 {(c1, t[0], t[2]) for t in sb} | {(c2, t[3], t[5]) for t in sb}
        hbres, sbres = collections.Counter(), collections.Counter()
        for t in hb:
            hbres[(c1, t[0])] += 1; hbres[(c2, t[3])] += 1
        for t in sb:
            sbres[(c1, t[0])] += 1; sbres[(c2, t[3])] += 1
        for c, a in ((c1, a1), (c2, a2)):
            per = collections.defaultdict(lambda: [0.0, 0.0, 0.0, '', 0.0])
            for i in range(len(a)):
                key = (c, int(a.ri[i]), a.an[i])
                d = max(iso[c].get(key, 0.0) - dim.get(key, 0.0), 0.0)
                per[int(a.ri[i])][0] += d
                per[int(a.ri[i])][1 if a.el[i] in APOLAR else 2] += d
                per[int(a.ri[i])][3] = a.rn[i]
                per[int(a.ri[i])][4] -= d * ASP_SIGMA[atom_class(a.rn[i], a.an[i], a.el[i])]
                tot['apolar' if a.el[i] in APOLAR else 'polar'] += d
                dg_total -= d * ASP_SIGMA[atom_class(a.rn[i], a.an[i], a.el[i])]
                # a polar atom that the interface buries but that makes no cross-interface
                # hydrogen bond or salt bridge is an energetic penalty and a model-quality flag
                if a.el[i] in 'NO' and d > 10.0 and dim.get(key, 0.0) < 1.0 \
                        and (c, int(a.ri[i]), a.an[i]) not in bonded:
                    unsat += 1
            for ri, (dt, ap, po, rn, dg) in per.items():
                if dt > 0.1:
                    rows.append(dict(c1=c1, c2=c2, chain=c, entity_name=names[ent_of[c]],
                                     resi=ri, resn=rn, dsasa=dt, dsasa_apolar=ap,
                                     dsasa_polar=po, dg_solv=dg, nhb=hbres.get((c, ri), 0),
                                     nsb=sbres.get((c, ri), 0)))
        bsa = tot['polar'] + tot['apolar']
        summ = dict(c1=c1, c2=c2, e1=names[ent_of[c1]], e2=names[ent_of[c2]],
                    env1=env_of[c1], env2=env_of[c2],
                    orbit=orb_id[tuple(sorted((c1, c2)))],
                    natom_contacts=cnt[tuple(sorted((c1, c2)))], bsa_total=bsa, area=bsa / 2,
                    apolar_frac=tot['apolar'] / bsa if bsa else np.nan,
                    n_hbond=len(hb), n_saltbridge=len(sb), n_hydrophobic=phob,
                    dg_solv=dg_total, dg_solv_per_1000A2=1000 * dg_total / bsa if bsa else np.nan,
                    n_buried_unsat_polar=unsat,
                    nres1=sum(1 for r in rows if r['chain'] == c1),
                    nres2=sum(1 for r in rows if r['chain'] == c2))
        return summ, rows, hb, sb, phob_pairs

    summaries, allres, bondrows, phobrows = [], [], [], []
    bad_pairs = []
    for n, (c1, c2) in enumerate(pairs):
        try:
            s, rows, hb, sb, pp = analyse(c1, c2)
        except Exception as exc:              # one bad pair must not cost the whole run
            bad_pairs.append(f'{c1}-{c2}')
            RPT.fail(f'interface {c1}-{c2}', exc, sys.exc_info()[2])
            continue
        summaries.append(s); allres += rows
        for (ri1, rn1, ri2, rn2), k in pp.items():
            phobrows.append(dict(c1=c1, c2=c2, ri1=ri1, rn1=rn1, ri2=ri2, rn2=rn2,
                                 n_contacts=int(k)))
        for kind, lst in (('hbond', hb), ('saltbridge', sb)):
            for t in lst:
                bondrows.append(dict(c1=c1, c2=c2, kind=kind, ri1=t[0], rn1=t[1], at1=t[2],
                                     ri2=t[3], rn2=t[4], at2=t[5], dist=t[6]))
        if (n + 1) % 25 == 0 or n + 1 == len(pairs):
            log(f'  pair {n+1}/{len(pairs)}  {time.time()-t_start:.0f}s')
    SUMM = pd.DataFrame(summaries)
    if not len(SUMM):
        raise RuntimeError('no interface could be measured')
    RPT.add('buried surface and bonds', 'WARNING' if bad_pairs else 'ok',
          f'{len(SUMM)} of {len(pairs)} contacting pairs measured'
          + (f'; failed: {", ".join(bad_pairs)}' if bad_pairs else ''))

    # ---------------------------------------------------- shape complementarity / gaps
    # Area says how much surface is buried; Sc and the gap index say how WELL the two
    # surfaces fit. A large interface with a low Sc and a high gap index is a loose
    # contact, which is a different structural claim from a small tight one.
    with RPT.guard('shape complementarity', ok_detail=None):
        if args.shape:
            import interface_shape as ish
            if not ish.HAVE_SKIMAGE:
                log('WARNING: scikit-image is not installed, skipping shape complementarity')
                RPT.warn('shape complementarity', 'scikit-image not installed; Sc skipped')
            else:
                import concurrent.futures as _cf
                RAD = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
                       'SE': 1.90, 'F': 1.47}
                def _cr(c):
                    a = by_chain[c]
                    return (a[['x', 'y', 'z']].to_numpy(),
                            np.array([RAD.get(str(e).upper(), 1.80) for e in a.el]))
                amap = {(r['c1'], r['c2']): r['area'] for r in summaries}
                jobs = [((c1, c2), *_cr(c1), *_cr(c2), amap.get((c1, c2)),
                         args.shape_spacing) for c1, c2 in pairs]
                nw = args.shape_workers or max(1, (os.cpu_count() or 2) // 2)
                # Threads, not processes: the heavy steps (distance transform, marching
                # cubes, KD-tree queries) are compiled and release the GIL, and process
                # pools are not always permitted to spawn (sandboxes, some schedulers).
                log(f'shape complementarity and gap volume for {len(jobs)} pairs '
                    f'on {nw} threads ...')
                res, t0 = {}, time.time()
                try:
                    with _cf.ThreadPoolExecutor(max_workers=nw) as ex:
                        for n, (key, out) in enumerate(ex.map(ish._worker, jobs)):
                            res[key] = out
                            if (n + 1) % 25 == 0 or n + 1 == len(jobs):
                                log(f'  shape {n+1}/{len(jobs)}  {time.time()-t0:.0f}s')
                except Exception as exc:                      # fall back to one thread
                    log(f'  thread pool failed ({exc}); running serially')
                    for n, j in enumerate(jobs):
                        key, out = ish._worker(j); res[key] = out
                SH = pd.DataFrame([dict(c1=k[0], c2=k[1], **{kk: vv for kk, vv in v.items()
                                                             if kk != 'error'})
                                   for k, v in res.items()])
                SH.to_csv('interface_shape.csv', index=False)
                SUMM = SUMM.merge(SH[['c1', 'c2', 'sc', 'gap_volume', 'gap_index']],
                                  on=['c1', 'c2'], how='left')
                ok = SH.sc.notna().sum()
                log(f'  Sc obtained for {ok}/{len(SH)} pairs '
                    f'(median {SH.sc.median():.3f}); see interface_shape.csv')
                RPT.add('shape complementarity', 'ok' if ok == len(SH) else 'WARNING',
                      f'Sc for {ok} of {len(SH)} pairs, median {SH.sc.median():.3f}')

    if not args.shape:
        RPT.skip('shape complementarity', 'switched off (--no-shape)')
    SUMM.to_csv('interfaces_summary.csv', index=False)
    pd.DataFrame(allres).to_csv('interface_residues.csv', index=False)

    # ---------------------------------------------------------------- ligand contacts
    # Which chains does each bound ligand actually touch, and does it BRIDGE two of
    # them? A cofactor sitting in the seam is part of the interface, and this is the
    # table that says so.
    with RPT.guard('ligands', ok_detail=None):
        if ligands:
            lig_key = {(lg['chain'], lg['resi']) for lg in ligands}
            lrows = []
            for r in allres:
                if (r['chain'], r['resi']) in lig_key:
                    partner = r['c2'] if r['chain'] == r['c1'] else r['c1']
                    lrows.append(dict(chain=r['chain'], resi=r['resi'], resn=r['resn'],
                                      name=lig_names.get(r['resn'], r['resn']),
                                      partner_chain=partner, dsasa=r['dsasa'],
                                      n_hbond=r['nhb'], n_saltbridge=r['nsb']))
            LC = pd.DataFrame(lrows)
            LC.to_csv('ligand_contacts.csv', index=False)
            if len(LC):
                npart = LC.groupby(['chain', 'resi', 'resn']).partner_chain.nunique()
                br = npart[npart >= 2]
                log(f'  {len(npart)} ligand(s) lie at an inter-chain interface'
                    + (f'; {len(br)} BRIDGE two or more chains: '
                       + ', '.join(f'{r[2]}{r[1]}({r[0]})' for r in br.index)
                       if len(br) else ''))
            else:
                log('  no ligand lies at an inter-chain interface')
    if ligands:
        if not RPT._has('ligands'):
            RPT.ok('ligands', f'{len(ligands)} kept: '
                 + ', '.join(sorted({lig_names.get(lg["resn"], lg["resn"]) for lg in ligands})))
    else:
        RPT.skip('ligands', 'none in the model' if not args.no_ligands else '--no-ligands')
    pd.DataFrame(bondrows).to_csv('interface_bonds.csv', index=False)
    pd.DataFrame(phobrows).to_csv('interface_hydrophobic.csv', index=False)

    # ---------------------------------------------------------------- config
    res = args.resolution
    conf = ('unknown' if res is None else
            'high' if res <= 3.0 else 'moderate' if res <= 3.5 else 'low')
    cfg = dict(model=os.path.basename(args.model), resolution=res, bond_confidence=conf,
               point_group=grp['point_group'], group_order=grp['order'],
               n_fold=int(grp['n_fold']),
               axis=None if grp['axis'] is None else [float(x) for x in grp['axis']],
               axis_point=None if grp['axis_point'] is None else [float(x) for x in grp['axis_point']],
               frame_rotation=Rf.tolist(), frame_origin=[float(x) for x in origin],
               entities=[dict(index=i, name=names[i], kind=kinds[i], length=len(reps[i][0]),
                              copies=sum(1 for c in ent_of if ent_of[c] == i))
                         for i in range(len(reps))],
               has_nucleic=bool(has_nucleic), n_chains=len(chains),
               has_ligand=bool(ligands),
               ligand_resn=sorted({lg['resn'] for lg in ligands}),
               n_contacting_pairs=len(pairs), n_unique_interfaces=len(uniq),
               orbit_sizes=sorted({len(o) for o in uniq}),
               single_copy_orbits=int(sum(1 for o in uniq if len(o) == 1)),
               contact_cut=args.contact_cut, sasa_points=args.sasa_points,
               intra_monomer=bool(args.intra), hydrogens_present=bool(hyd),
               renumbered=bool(renumbered),
               hbond_criterion=('distance+angle' if hyd else 'distance only (no H in model)'),
               dg_model='Eisenberg-McLachlan 1986 atomic solvation parameters',
               colours=colours, proteins_fasta=args.proteins,
               ligand_names=lig_names, ligand_colours=lig_cols)
    json.dump(cfg, open('run_config.json', 'w'), indent=1)
    if conf in ('low', 'unknown'):
        log(f'NOTE: bond confidence "{conf}" — downstream steps will flag individual '
            'H-bonds and salt bridges as tentative rather than assignments.')
    if cfg['single_copy_orbits']:
        log(f'NOTE: {cfg["single_copy_orbits"]} interface(s) have only ONE copy in this '
            'assembly, so they carry no symmetry-replicate error estimate.')
    log(f'DONE in {time.time()-t_start:.0f}s')
    RPT.ok('interface inventory', f'{len(pairs)} contacting pairs, {len(uniq)} symmetry-unique '
                                f'interfaces, {len(chains)} chains')
    # Tables that are written only when there is something to put in them. If this model
    # has none, a copy left by an EARLIER model in the same folder would otherwise be read
    # by the later steps as if it belonged to this one (another model's disulfides in the
    # assembly verdicts, its ligands in the tables). Rewrite each as an empty table.
    _OPTIONAL = {'ligands.csv': 'chain,resi,resn,n_atoms,is_additive',
                 'ligand_contacts.csv': 'chain,resi,resn,partner_chain,dsasa,n_hbond,'
                                        'n_saltbridge',
                 'interface_links.csv': 'kind,bonded,distance,c1,resi1,resn1,atom1,c2,resi2,'
                                        'resn2,atom2',
                 'interface_shape.csv': 'c1,c2,sc,sc_n_dots,gap_volume,gap_index',
                 'residue_intra_burial.csv': 'chain,entity_name,resi,resn,'
                                             'sasa_in_folded_chain,max_asa,intra_buried,rsa',
                 'residue_renumbering.csv': 'chain,new_resi,orig_resseq,icode,resn'}
    _stale = [f for f in _OPTIONAL
              if os.path.exists(f) and os.path.getmtime(f) < t_start - 1]
    for f in _stale:
        with open(f, 'w', encoding='utf-8') as fh:
            fh.write(_OPTIONAL[f] + '\n')
    if _stale:
        RPT.ok('tables of an earlier run', f'emptied {len(_stale)} left in this folder by a '
                                           f'previous model: {", ".join(_stale)}')
    RPT.done()


if __name__ == '__main__':
    main()
