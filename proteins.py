"""
proteins.py — read a FASTA that names and colours the proteins, and match it to the model.

    >YfdQ_#7FA8D4
    MSKLTRVEN...
    >belt #E8884A
    MAQ...

The header is the protein's name, optionally followed by a colour as #RRGGBB, joined by
"_", a space or "|". Records are matched to the entities found in the model BY SEQUENCE,
not by order, so the file can list the proteins in any order and may contain proteins
that are not in the model. A modelled chain usually covers only part of the full
sequence (missing loops, termini), so the test is how much of the MODELLED sequence is
found in the FASTA sequence, not overall identity.

Anything that does not fit — a bad colour, a sequence that matches nothing, two
proteins claiming the same record — falls back to the default and is reported, never
fatal.
"""
import re
from difflib import SequenceMatcher

HDR = re.compile(r'^\s*(.+?)(?:[\s_|]+(#[0-9A-Fa-f]{6}))?\s*$')
HEX = re.compile(r'^#[0-9A-Fa-f]{6}$')


def read_proteins(path):
    """[{name, colour, seq, problems}] from a FASTA file. Never raises on bad content."""
    recs, cur = [], None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                m = HDR.match(line[1:])
                name, col = (m.group(1), m.group(2)) if m else (line[1:].strip(), None)
                probs = []
                # a '#' that is not a valid colour is reported, and stripped from the name
                if col is None and '#' in name:
                    bad = name[name.index('#'):]
                    probs.append(f'colour "{bad}" is not #RRGGBB; default colour used')
                    name = name[:name.index('#')].rstrip(' _|')
                name = re.sub(r'\s+', '_', name.strip()) or f'protein_{len(recs) + 1}'
                cur = dict(name=name, colour=col.upper() if col else None, seq='',
                           problems=probs)
                recs.append(cur)
            elif cur is not None:
                cur['seq'] += re.sub(r'[^A-Za-z]', '', line).upper()
    for r in recs:
        if not r['seq']:
            r['problems'].append('no sequence under this header')
    return recs


def coverage(model_seq, ref_seq):
    """Fraction of the MODELLED residues found, in order, in the reference sequence."""
    a, b = model_seq.upper(), ref_seq.upper()
    if not a or not b:
        return 0.0
    sm = SequenceMatcher(None, a, b, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks()) / len(a)


def match_entities(reps, recs, min_cover=0.90):
    """For each entity (seq, kind) -> (record or None, coverage, note)."""
    out = []
    for seq, kind in reps:
        if kind == 'ligand' or not seq:
            out.append((None, 0.0, 'ligand-only chain, not matched')); continue
        best, cov = None, 0.0
        for r in recs:
            if not r['seq']:
                continue
            c = coverage(seq, r['seq'])
            if c > cov:
                best, cov = r, c
        if best is not None and cov >= min_cover:
            out.append((best, cov, f'matched "{best["name"]}" ({100 * cov:.0f} % of the '
                                   'modelled sequence)'))
        else:
            near = (f'; closest was "{best["name"]}" at {100 * cov:.0f} %' if best else '')
            out.append((None, cov, f'no FASTA record covers ≥{100 * min_cover:.0f} % of '
                                   f'this sequence{near}'))
    return out
