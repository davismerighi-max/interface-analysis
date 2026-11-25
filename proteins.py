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
import collections
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


NUC_LETTERS = set('ACGTUN')
_COMP = str.maketrans('ACGTUN', 'TGCAAN')


def is_nucleotide(seq):
    """True for a DNA/RNA sequence (only A C G T U N), as opposed to a protein."""
    s = seq.upper()
    return bool(s) and set(s) <= NUC_LETTERS


def _nt(s):
    return s.upper().replace('U', 'T')


def match_entities(reps, recs, min_cover=0.90):
    """For each entity (seq, kind) -> (record or None, coverage, note).

    Protein chains are only compared with protein records and DNA/RNA chains only with
    nucleotide records: A, C, G and T are amino-acid letters too, so a short DNA strand
    would otherwise "match" almost any long protein sequence. A DNA/RNA chain is also
    compared with the reverse complement of each record, so ONE line for a duplex names
    both strands: the second is called <name>_complement (same colour)."""
    out, used = [], collections.Counter()
    for seq, kind in reps:
        if kind == 'ligand' or not seq:
            out.append((None, 0.0, 'ligand-only chain, not matched')); continue
        nuc = kind == 'nucleic'
        cands = []                               # (coverage, is_reverse_complement, record)
        for r in recs:
            if not r['seq'] or is_nucleotide(r['seq']) != nuc:
                continue
            if nuc:
                fwd = coverage(_nt(seq), _nt(r['seq']))
                rev = coverage(_nt(seq), _nt(r['seq']).translate(_COMP)[::-1])
                cands.append((fwd, False, r) if fwd >= rev else (rev, True, r))
            else:
                cands.append((coverage(seq, r['seq']), False, r))
        what = 'nucleotide' if nuc else 'protein'
        best = max((c[0] for c in cands), default=0.0)
        if best >= min_cover:
            # among equally good records, take one no other chain group has used yet, so
            # two rows with the same sequence but different names name two groups
            tied = [c for c in cands if c[0] >= best - 1e-9]
            # (a complementary strand instead goes with the record its partner strand used)
            cov, rc, rec = min(tied, key=lambda c: -used[id(c[2])] if c[1] else used[id(c[2])])
            again = used[id(rec)] > 0
            used[id(rec)] += 1
            if rc:
                # same hue, lighter, so the two strands can be told apart as partners
                col = rec['colour']
                if col:
                    rgb = [int(col[k:k + 2], 16) for k in (1, 3, 5)]
                    col = '#' + ''.join(f'{int(v + 0.45 * (255 - v)):02X}' for v in rgb)
                rec = dict(rec, name=rec['name'] + '_complement', colour=col)
            note = (f'matched "{rec["name"]}" ({100 * cov:.0f} % of the modelled sequence'
                    + (', complementary strand' if rc else '') + ')')
            if again and not rc:
                note += ('; another part of the model matched the same record, so both '
                         'carry this name (one molecule modelled in pieces)')
            out.append((rec, cov, note))
        else:
            near = max(cands, key=lambda c: c[0]) if cands else None
            out.append((None, best, f'no {what} record covers ≥{100 * min_cover:.0f} % of '
                                    f'this sequence'
                                    + (f'; closest was "{near[2]["name"]}" at '
                                       f'{100 * best:.0f} %' if near else '')))
    return out


def read_ligand_names(path):
    """{CODE: (name, colour or None)} and a list of problems, from lines
    "CODE  name  #RRGGBB" (name and colour optional; also "CODE=name"). Never raises on
    bad content; '#' lines that are not a colour are treated as comments."""
    out, probs = {}, []
    with open(path, encoding='utf-8', errors='replace') as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tok = line.replace('=', ' ').replace(',', ' ').replace('\t', ' ').split()
            code, rest = tok[0].upper(), tok[1:]
            col = None
            if rest and rest[-1].startswith('#'):
                if HEX.match(rest[-1]):
                    col = rest[-1].upper()
                else:
                    probs.append(f'line {n}: colour "{rest[-1]}" is not #RRGGBB; default used')
                rest = rest[:-1]
            name = '_'.join(rest) or code
            if len(code) > 5:
                probs.append(f'line {n}: "{code}" does not look like a residue code; skipped')
                continue
            out[code] = (name, col)
    return out, probs
