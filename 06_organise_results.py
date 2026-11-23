"""
06_organise_results.py — copy one run's outputs into a tidy folder, one folder per figure.

Steps 01-05 work in one flat working folder, because each step reads the tables the
previous one wrote. That folder is the pipeline's workbench; this step builds the folder
you keep and share: every figure in its own sub-folder together with the tables it was
drawn from, plus the report, the inputs and the code that produced them.

    python 06_organise_results.py MODEL.pdb               ->  MODEL_results/
    python 06_organise_results.py MODEL.pdb --out paper/fig_data

It only COPIES; the working folder is left as it is, so any step can still be re-run.
Which files belong to this run is read from run_manifest.json (each step records what it
wrote), so files of an earlier run or of your own are never swept in.
"""
import argparse, fnmatch, glob, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog

# (folder, description, patterns). The first folder whose pattern matches takes the file.
LAYOUT = [
    ('00_report', 'what ran and what it found: start here',
     ['run_report.txt', 'stats_summary.txt']),
    ('01_footprints', 'footprint figure per protein, and the per-residue tables behind it',
     ['fig*_footprints.*', 'footprints_by_environment.csv', 'interface_residues.csv',
      'secondary_structure.csv', 'residue_burial_assembly.csv', 'residue_intra_burial.csv']),
    ('02_contact_maps', 'residue-residue contact maps of the interfaces',
     ['fig_contact_maps.*', 'figS1_contact_maps_all.*']),
    ('03_architecture', 'contact topology (cylinder unrolled), chain positions and orbits',
     ['fig_architecture.*', 'chain_geometry.csv', 'chain_environments.csv', 'orbits.csv']),
    ('04_inventory', 'every symmetry-unique interface and its area',
     ['fig_inventory.*', 'interfaces_summary.csv', 'table_interfaces_unique.csv']),
    ('05_area_partition', 'how the buried surface is shared, per subunit (interface budget)',
     ['fig_area_partition.*', 'subunit_budget.csv']),
    ('06_chemistry', 'chemistry, energetics and stacking of the buried surface',
     ['fig_chemistry.*', 'interface_composition_by_category.csv',
      'interaction_density_by_category.csv', 'interface_orbit_chemistry.csv',
      'interface_energetics.csv', 'aromatic_interactions.csv', 'stacking_geometry.csv']),
    ('07_packing', 'shape complementarity and gap index',
     ['fig_packing.*', 'interface_shape.csv']),
    ('08_interaction_networks', 'residue-level bond networks and the bond tables',
     ['fig_interaction_networks*.*', 'interface_bonds.csv', 'interface_bonds_unique.csv',
      'interface_hydrophobic.csv', 'interface_links.csv']),
    ('09_3D_renders', 'whole-model renders by burying partner and by interaction type',
     ['fig_render_*.*', 'render_residue_colours.csv']),
    ('10_assemblies', 'candidate assemblies: verdict table, figure and one structure each',
     ['fig_assemblies.*', 'assembly_candidates.csv', 'assemblies.csv',
      'assembly_interfaces.csv', 'assemblies_summary.txt', 'assembly_*.pdb', 'assembly_*.cif']),
    ('11_chimerax', 'structure with interface values in the B-factor, plus colour scripts',
     ['model_interfaces.pdb', 'model_interfaces.cif', 'chimerax_*.cxc']),
    ('12_ligands', 'bound ligands and what they touch',
     ['ligands.csv', 'ligand_contacts.csv']),
    ('13_pisa', 'cross-check against PISA',
     ['fig_pisa_concordance.*', 'pisa_concordance.csv', 'pisa_*.json']),
    ('logs', 'settings and logs needed to reproduce or debug the run',
     ['run_config.json', 'run_report.json', 'run_errors.log', 'analysis.log',
      'residue_renumbering.csv', 'run_manifest.json']),
]
NEVER = {'run_report.json.tmp', 'run_manifest.json.tmp'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', help='the model file the run analysed (names the folder)')
    ap.add_argument('--out', default=None, help='output folder (default: <model>_results)')
    ap.add_argument('--inputs', default='', metavar='FILES',
                    help='extra input files to keep under inputs/, comma-separated '
                         '(the model, proteins.fasta and msa_* are found automatically)')
    a = ap.parse_args()

    RPT = runlog.Report('06', model=a.model)
    stem = os.path.splitext(os.path.basename(a.model))[0]
    out = a.out or f'{stem}_results'
    if os.path.abspath(out) == os.path.abspath('.'):
        sys.exit('--out must be a new folder, not the working folder itself')

    man = runlog.read_manifest()
    produced = sorted({f for fs in man.values() for f in fs} - NEVER)
    if not produced:
        RPT.default('files of this run', 'no run_manifest.json (run made with an older '
                    'version): took every file that matches a known output name')
        produced = sorted(f for f in os.listdir('.') if os.path.isfile(f) and
                          any(fnmatch.fnmatch(f, p) for _, _, ps in LAYOUT for p in ps))
    produced = [f for f in produced if os.path.isfile(f)]
    for f in ('run_report.txt', 'run_report.json', 'run_manifest.json'):
        if os.path.isfile(f) and f not in produced:
            produced.append(f)

    placed, other = {}, []
    for f in produced:
        for folder, _, pats in LAYOUT:
            if any(fnmatch.fnmatch(f, p) for p in pats):
                placed.setdefault(folder, []).append(f); break
        else:
            if not f.endswith(('.py', '.pyc', '.tmp')):
                other.append(f)

    inputs = [a.model] + [x for x in ('proteins.fasta',) if os.path.isfile(x)]
    inputs += sorted(glob.glob('msa_*'))
    inputs += [x.strip() for x in a.inputs.split(',') if x.strip()]
    inputs = [x for x in dict.fromkeys(inputs) if os.path.isfile(x) and x not in produced]

    here = os.path.dirname(os.path.abspath(__file__))
    code = sorted(glob.glob(os.path.join(here, '*.py')))

    with RPT.guard('copy files', ok_detail=None):
        os.makedirs(out, exist_ok=True)
        n = 0
        for folder, files in placed.items():
            os.makedirs(os.path.join(out, folder), exist_ok=True)
            for f in files:
                shutil.copy2(f, os.path.join(out, folder, f)); n += 1
        for sub, files in (('inputs', inputs), ('other', other), ('code', code)):
            if files:
                os.makedirs(os.path.join(out, sub), exist_ok=True)
                for f in files:
                    shutil.copy2(f, os.path.join(out, sub, os.path.basename(f))); n += 1
        RPT.ok('copy files', f'{n} files into {out}/ ({len(placed)} result folders)')

    with RPT.guard('index', ok_detail='README.txt'):
        desc = {f: d for f, d, _ in LAYOUT}
        desc.update(inputs='the model and the inputs you gave',
                    code='the exact scripts that produced these results',
                    other='outputs not linked to one figure')
        lines = [f'Interface analysis results for {os.path.basename(a.model)}', '',
                 'One folder per figure: the PNG (600 dpi), the PDF (vector, for editing)',
                 'and the tables the figure was drawn from. Start with 00_report/run_report.txt.',
                 '']
        for folder in sorted(set(placed) | {s for s, fs in (('inputs', inputs),
                                 ('other', other), ('code', code)) if fs}):
            lines.append(f'{folder}/  -  {desc.get(folder, "")}')
            names = sorted(os.listdir(os.path.join(out, folder)))
            for nm in names:
                lines.append(f'    {nm}')
            lines.append('')
        open(os.path.join(out, 'README.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
    RPT.done()
    # the report now has step 06 too: refresh the copy that goes with the results
    for f, folder in (('run_report.txt', '00_report'), ('run_report.json', 'logs')):
        if os.path.isfile(f) and os.path.isdir(os.path.join(out, folder)):
            shutil.copy2(f, os.path.join(out, folder, f))
    print(f'results organised in {out}/  (index: {out}/README.txt)')


if __name__ == '__main__':
    main()
