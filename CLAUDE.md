# CLAUDE.md — instructions for an AI assistant using this repository

This file is written for you, the assistant, not for the human user. Read it before
you run anything. It tells you how to run the pipeline and, more importantly, how to
explain its output to a scientist: topic by topic, with the numbers that matter, the
conventions behind them, and the claims you must not make.

The user is most likely a structural biologist with a refined model (cryo-EM or X-ray)
who wants to know what holds the assembly together. Explain like a careful colleague
reviewing their figures: precise, quantitative, honest about uncertainty, never more
confident than the method allows.

---

## 1. Running it

All steps run in ONE working directory, in order, on the same model file:

```bash
pip install -r requirements.txt
python 01_interface_analysis.py MODEL.cif --resolution 3.2 [--proteins proteins.fasta]
python 02_derive_tables.py      MODEL.cif
python 03_make_figures.py       MODEL.cif
python 05_assemblies.py         MODEL.cif --render
python 06_organise_results.py   MODEL.cif          # MODEL_results/, one folder per figure
```

- Run each model in ITS OWN working folder (the scripts can be called by path from the
  repository). Never analyse two models in one folder: per-protein figure names differ
  between runs, so old figures would sit next to new ones. Point the user to
  `MODEL_results/` (made by step 06), not the working folder, when handing results over.

- Ask for (or look up) the map/data resolution and pass `--resolution`. Without it every
  bond is reported as tentative.
- Shape complementarity runs by default and is the slow part: about a minute for a dimer,
  ten minutes or more for 30+ chains. Offer `--no-shape` for a first look; the packing
  figure is then skipped.
- `--proteins FASTA` names and colours the proteins. Headers are `>name_#RRGGBB` (colour
  optional); records are matched to the model BY SEQUENCE, so order does not matter and
  extra records are harmless. If the user gives names and colours in conversation,
  write this file for them. Keep names short (they label every axis).
  DNA/RNA records (bases only) are matched only to nucleic chains, also as the reverse
  complement: one record for a duplex names both strands, the second `<name>_complement`
  in a lighter shade. Protein records never match nucleic chains.
  Two chain groups that match the same record (pieces of one molecule, e.g. 4I6Z's DNA
  halves C = 1-12 and D = 13-24 of one 24-mer) both take its name; downstream code works by
  entity_name, so they share one footprint figure with one environment each.
- Bonded HETATM residues (MSE, SEP, ...) are kept in the polymer by load_model, never as
  ligands; the analysis log lists them. Fig_chemistry panel b now stacks hydrophobic
  residue pairs (>= 3 apolar atom contacts, column `phob` of
  interaction_density_by_category.csv) after the bonds; the label reads "bonds + pairs".
- `--ligand-names FILE` names and colours bound ligands: lines `CODE  name  #RRGGBB`
  (CODE = residue name in the model, as listed under "ligands" in run_report.txt). Names
  go into ligands.csv / ligand_contacts.csv, the figure labels and the ChimeraX scripts
  (sticks + label). Codes not in the model give a WARNING, not a failure.
- `04_pisa_concordance.py PDBID` is the only step that touches the network. Optional.
- Steps 01–03 and 05 need no internet; they run on air-gapped cluster nodes.
- Large assemblies (> ~50 chains, ribosomes, capsids) belong on a workstation or a
  cluster node, not a laptop or a free Colab session.

## 2. Before interpreting anything: read `run_report.txt`

Every step writes into `run_report.txt` (and `run_report.json`):

| status | meaning | what you do |
|---|---|---|
| ok | ran as asked | nothing |
| DEFAULT | ran with defaults instead of what was asked | say what was asked, what was used, why |
| WARNING | ran; the note matters | relay the note in plain words |
| SKIPPED | not requested, or not applicable | mention only if relevant |
| FAILED | that part did not run; the rest of the step did | open `run_errors.log`, read the traceback, explain cause and fix |

A step marked ABORTED stopped at the item shown. A FAILED item that says it "needs the
output of an earlier part that failed" is a consequence: trace back to the first
failure. Report failures FIRST, before any science, and never interpret a figure whose
item failed.

## 3. Explaining the results — topic by topic

Use this order. For each topic give the headline number, what it means, and the
caveat. Read every number from the file named; never estimate from a figure and never
state a number that is not in the output.

### 3.1 Symmetry and environments — `run_config.json`, `chain_environments.csv`
- Point group, number of chains, number of distinct proteins.
- **Symmetry-independent environments**: chains the symmetry maps onto each other share
  an environment and are true replicates. Chains of the same protein in different
  environments sit in genuinely different surroundings and are NEVER averaged.
- C1 means no symmetry replicates: no error bars, every interface is one measurement.
- A homo-oligomer is NOT reduced to an asymmetric unit: every chain is analysed; one
  environment only means every copy has identical surroundings.

### 3.2 Interface inventory — `table_interfaces_unique.csv`, `fig_inventory`, `fig_architecture`
- Contacting pairs and symmetry-unique interfaces (orbits). Each orbit is one interface
  type; its copies are replicates (reproducibility CV in `stats_summary.txt`).
- Largest interface; which subunit environment buries most.
- `fig_architecture` is the cylinder unrolled (angle around the axis vs height); line
  width ∝ buried area, with a key in the legend.

### 3.3 Buried surface — `interfaces_summary.csv`, `fig_area_partition`
- **Always state the convention:** pairwise in isolation (A alone, B alone, AB) and
  reported HALVED — the interface area, not the sum of both footprints. This is PISA's
  convention, so the numbers compare directly with PISA.
- Panel a: which interface categories carry the surface. Panel b: how one subunit
  spends its interface budget across partners.
- Burial inside the WHOLE assembly is in `residue_burial_assembly.csv`; use it for "how
  exposed is residue X really".

### 3.4 Footprints — `fig<N>_<protein>_footprints`, `footprints_by_environment.csv`
- Per residue: buried area coloured by partner (by interface in a homo-oligomer), contact
  counts by type (salt bridge, H-bond, stacking, cation–π and hydrophobic: one count per
  partner residue with >= --phobic-min apolar atom contacts), an interaction-density
  curve, and the ΔG strip (blue pays, orange costs). Hydrophobic counts are residue-pair
  contacts, not bonds: do not add them to the bond totals of fig_chemistry. Name the hotspot segments and the labelled residues, and point out where two
  environments of the same protein differ — those differences are real.

### 3.5 Chemistry — `fig_chemistry`, `interaction_density_by_category.csv`, `interface_bonds_unique.csv`
- Composition of the buried surface and directional interactions per 1000 Å² — the
  right way to compare interfaces of different size.
- Worse than ~3.0 Å, individual H-bonds and salt bridges are TENTATIVE: write
  "consistent with", not "forms". A bond reproduced in every symmetry copy
  (`stats_summary.txt`) is more credible than one seen once.
- Without hydrogens in the model, H-bonds are distance-only and over-counted.

### 3.6 Covalent links — `interface_links.csv`, yellow edges in `fig_interaction_networks`
- Inter-chain disulfides, isopeptides and bridging metals. If a disulfide is bonded in
  some symmetry copies and not others, say so: it is either real heterogeneity or a
  modelling inconsistency, and the map should be checked before publication.

### 3.7 Energetics — `interface_energetics.csv`, ΔG strip
- ΔG_solv is the Eisenberg–McLachlan atomic solvation term. NEGATIVE: burying that
  surface releases solvation free energy (hydrophobically driven). POSITIVE: burying it
  COSTS solvation energy; the interface is held by H-bonds and salt bridges.
- It is NOT a binding free energy (no entropy, no electrostatics, no explicit H-bond
  term). Nucleic-acid and ligand atoms add area but σ = 0. Never present it as K_d or
  as PISA's ΔG_diss.

### 3.8 Packing — `fig_packing`, `interface_shape.csv`
- Shape complementarity S_c (Lawrence & Colman 1993) and gap index (Å³ per Å²).
- **Always state the calibration:** this grid implementation reads about **0.10 below**
  the original `sc` program (trypsin–BPTI, 2PTC: 0.647; obligate histone-fold
  heterodimers of the nucleosome, 1KX5: 0.598–0.620; published ≈ 0.70). The class names
  on the figure ("loose", "very loose") are therefore pessimistic. Use S_c to RANK
  interfaces within one run; never call an interface "loose" without this caveat.
- S_c rises with interface size: compare categories at matched size.

### 3.9 Interaction networks — `fig_interaction_networks*`
- Residue-to-residue map of each symmetry-unique interface. Colour = interaction type;
  line width = bond length (thicker = shorter, key in the legend); yellow = covalent.
  Use it to name the residue pairs behind the numbers above.

### 3.9b 3D renders — `fig_render_proteins`, `fig_render_budget`, `fig_render_interactions`
- Each shows the whole model three times in the symmetry frame (z = axis): side, tilted,
  top. Colour per residue with the --bsa-min cut-off of the ChimeraX scripts; the
  interaction colouring uses the priority covalent > salt bridge > cation–π > stacking >
  H-bond > hydrophobic, and `render_residue_colours.csv` lists every residue's class.
  They are overviews: interfaces are mostly inside the assembly, so for any residue-level
  claim point to the footprints, networks or ChimeraX.

### 3.10 Ligands — `ligands.csv`, `ligand_contacts.csv`
- Cofactors are kept by default. A ligand touching two chains is part of the interface
  (a shared active site, for example). Additives (SO4, GOL, PEG…) are flagged, not
  dropped — say whether any interpretation depends on one.

### 3.11 Candidate assemblies — `assembly_candidates.csv`, `assemblies_summary.txt`, `fig_assemblies`
- The model is taken apart weakest interface first, whole symmetry orbits at a time;
  every piece that appears is a candidate. Read the `class` and `reason` columns and
  quote the reason.
- **kept** = buried area inside the piece ÷ (inside + across its boundary).
  ≥ 67 % strong candidate · ≥ 50 % candidate · < 50 % not supported by contacts alone ·
  covalent unit = every chain joined by disulfides (survives non-reducing conditions).
- **These cut-offs are a stated heuristic, not a published standard; PISA ranks by
  ΔG_diss.** Say so whenever you report a class.
- Disulfides are not weighed as area. A piece whose boundary crosses a bonded disulfide
  is "separable only under reducing conditions" — a testable prediction (reducing vs
  non-reducing SDS-PAGE).
- A positive boundary ΔG_solv marks a polar seam, plausibly pH- and salt-sensitive.
- For cryo-EM there is no crystal lattice; the question is whether the piece could exist
  on its own, which only experiment settles (SEC-MALS, mass photometry, 2D classes).
  Propose the experiment rather than asserting the assembly.
- Coordinates per candidate: `assembly_NN_*.pdb` (`.cif` for very large models).

### 3.12 Files for the user's own figures
- `model_interfaces.pdb` carries buried area in the B-factor column.
- `chimerax_colour_only.cxc` and `chimerax_colour_by_interaction.cxc`: open the model in
  ChimeraX, then open the script. Both set flat lighting, silhouettes and licorice.
  A ChimeraX session (.cxs) cannot be produced outside ChimeraX.
- Every figure exists as PNG (600 dpi) and as vector PDF.

## 4. Never
- Present a number you did not read from an output file.
- Call an S_c class, a "kept" class or ΔG_solv proof of a biological assembly or of
  binding strength.
- Average footprints across different symmetry environments.
- Drop the resolution caveat on individual bonds.
- Re-run with different settings without saying what changed and why.
- Fetch ConSurf or PISA on the user's behalf without asking: the pipeline is offline by
  design.

## 5. End every explanation with
1. The two or three findings that belong in a results paragraph, with their numbers.
2. What to check in the map before trusting them (tentative bonds, partially bonded
   disulfides, single-copy interfaces).
3. One experiment that would test the strongest structural claim.
4. Anything in `run_report.txt` that ran with defaults, warned or failed.
