"""Point-group detection and orbit construction for a multi-chain assembly.

Replaces the hard-coded C3 of the original tail1 pipeline. Nothing here assumes a
particular symmetry: the group is whatever set of rotations maps the chain set onto
itself, and C1 (no symmetry) is a first-class, supported answer.

Method
------
1. Chains are grouped into *entities* by sequence identity.
2. Every ordered pair of chains within an entity gives a candidate operator by Kabsch
   superposition.
3. A candidate is a genuine symmetry operator only if it maps *every* chain centroid
   onto some chain centroid of the same entity, within tolerance. This whole-assembly
   test is what makes the result trustworthy: a local pseudo-symmetry between two
   chains is rejected unless the entire assembly obeys it.
4. The surviving operators form the group. Its order, axes and angles classify it.

Caveats, deliberately explicit
------------------------------
- Only proper rotations are considered. Improper operations (mirror, inversion, S_n)
  do not occur in protein assemblies of one chirality and are not searched for.
- Helical/screw symmetry is NOT a point group. If a model is a helical segment, the
  screw operator maps interior subunits correctly but runs off the ends, so it fails
  the whole-assembly test and is reported separately by `find_screw_operators`.
- Classification falls back to an honest "unknown(order=N)" rather than guessing.
"""
import collections
import numpy as np

AA3 = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU',
       'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'MSE', 'SEC', 'PYL'}
NUC3 = {'A', 'C', 'G', 'U', 'T', 'DA', 'DC', 'DG', 'DT', 'DU', 'I', 'N'}


# ----------------------------------------------------------------- superposition
def kabsch(P, Q):
    """rotation R and translation t with Q ~= P @ R.T + t, plus the RMSD."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = qc - R @ pc
    rms = float(np.sqrt(((Q - (P @ R.T + t)) ** 2).sum(1).mean()))
    return R, t, rms


def axis_angle(R):
    """rotation axis (unit, sign-normalised) and angle in degrees, 0..180."""
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    w, v = np.linalg.eig(R)
    ax = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    n = np.linalg.norm(ax)
    ax = ax / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    if ax[np.argmax(np.abs(ax))] < 0:
        ax = -ax
    return ax, ang


def axis_point(R, t):
    """a point on the rotation axis: least-squares solution of (I - R) x = t."""
    return np.linalg.lstsq(np.eye(3) - R, t, rcond=None)[0]


# ----------------------------------------------------------------- entities
def chain_sequences(model, three2one=None):
    """{chain_id: (sequence_string, kind)} where kind is 'protein' | 'nucleic' | 'other'."""
    m = three2one or {}
    out = {}
    for ch in model:
        seq, n_aa, n_nuc = [], 0, 0
        het = []
        for r in ch:
            rn = r.get_resname().strip()
            if rn in AA3:
                n_aa += 1; seq.append(m.get(rn, 'X'))
            elif rn in NUC3:
                n_nuc += 1; seq.append(rn[-1].lower())
            else:
                # A bound ligand must NOT enter the sequence: otherwise two copies of
                # the same protein, one with the cofactor and one without, would look
                # like different entities and the symmetry search would miss the pair.
                het.append(rn)
        if n_aa or n_nuc:
            kind = 'protein' if n_aa >= max(1, n_nuc) else 'nucleic'
        else:                       # a chain that is nothing but ligand(s)
            kind = 'ligand'
            seq = ['|'.join(sorted(het))] if het else []
        out[ch.id] = (''.join(seq), kind)
    return out


def group_entities(seqs, min_identity=0.95):
    """Group chains whose sequences are identical or near-identical.

    Near-identity is judged on equal-length chains by position-wise match fraction;
    chains of different length are only grouped if one is a prefix-compatible subset
    of the other at >= min_identity over the overlap and lengths differ by < 10 %.
    """
    ids = list(seqs)
    ent = {}
    reps = []
    for c in ids:
        s, kind = seqs[c]
        placed = False
        for k, (rs, rkind) in enumerate(reps):
            if rkind != kind:
                continue
            if s == rs:
                ent[c] = k; placed = True; break
            la, lb = len(s), len(rs)
            if min(la, lb) == 0 or abs(la - lb) / max(la, lb) > 0.10:
                continue
            n = min(la, lb)
            idt = sum(1 for i in range(n) if s[i] == rs[i]) / n
            if idt >= min_identity:
                ent[c] = k; placed = True; break
        if not placed:
            reps.append((s, kind)); ent[c] = len(reps) - 1
    return ent, reps


# ----------------------------------------------------------------- the group
def _op_key(R, t, nd=1):
    return (tuple(np.round(R.flatten(), nd + 1)), tuple(np.round(t, nd)))


def _global_fit(ca, perm):
    """Best-fit operator for a chain permutation, using every matched CA atom.

    Once the permutation is known this is far better conditioned than the pairwise
    fit that suggested it: NCS is usually restrained rather than constrained, so
    individual chain-to-chain fits disagree in the second decimal and must never be
    used to decide whether two operators are the same group element.
    """
    P, Q = [], []
    for c, d in perm.items():
        src, dst = ca.get(c), ca.get(d)
        if not src or not dst:
            continue
        for i in src:
            if i in dst:
                P.append(src[i]); Q.append(dst[i])
    if len(P) < 10:
        return None
    return kabsch(np.array(P), np.array(Q))


def find_symmetry_operators(ca, centroids, entity_of, tol=2.0, rms_tol=3.0,
                            fit_tol=2.0):
    """Every rotation+translation that maps the whole chain set onto itself.

    Operators are identified by the chain PERMUTATION they induce, not by their
    numeric matrix, and the representative matrix is then re-fitted globally.
    Returns [(R, t, perm)] including the identity.
    """
    chains = list(centroids)
    C = np.array([centroids[c] for c in chains])

    def permutation_of(R, t):
        img = C @ R.T + t
        perm = {}
        for i, c in enumerate(chains):
            d = np.linalg.norm(C - img[i], axis=1)
            j = int(np.argmin(d))
            if d[j] > tol or entity_of[chains[j]] != entity_of[c]:
                return None
            perm[c] = chains[j]
        if len(set(perm.values())) != len(chains):      # must be a bijection
            return None
        return perm

    found = {}                                           # permutation -> (R, t, rms)
    ident = tuple((c, c) for c in chains)
    found[ident] = (np.eye(3), np.zeros(3), 0.0)

    by_ent = collections.defaultdict(list)
    for c in chains:
        by_ent[entity_of[c]].append(c)
    for members in by_ent.values():
        # a chain with no anchor atoms at all (a ligand-only chain, say) cannot take
        # part in the superposition search and is skipped rather than raising
        members = [c for c in members if ca.get(c)]
        for a in members:
            ka = sorted(ca[a])
            for b in members:
                if a == b:
                    continue
                k = [i for i in ka if i in ca[b]]
                if len(k) < 10:
                    continue
                R, t, rms = kabsch(np.array([ca[a][i] for i in k]),
                                   np.array([ca[b][i] for i in k]))
                if rms > rms_tol:
                    continue
                perm = permutation_of(R, t)
                if perm is None:
                    continue
                key = tuple(sorted(perm.items()))
                if key in found:
                    continue
                fit = _global_fit(ca, perm)
                if fit is None or fit[2] > fit_tol:
                    continue
                found[key] = fit
    return [(R, t, dict(k)) for k, (R, t, _) in found.items()]


def find_screw_operators(ca, centroids, entity_of, min_frac=0.5, rms_tol=3.0, tol=2.0):
    """Operators that map a large fraction (but not all) of the chains onto others.

    These are the helical/screw relations of a truncated filament — real local
    symmetry that must NOT be used to merge interfaces, because it runs off the ends.
    Returns [(R, t, n_mapped, n_total, rise, twist_deg)].
    """
    chains = list(centroids)
    C = np.array([centroids[c] for c in chains])
    out, seen = [], set()
    by_ent = collections.defaultdict(list)
    for c in chains:
        by_ent[entity_of[c]].append(c)
    for ent, members in by_ent.items():
        for a in members:
            ka = sorted(ca[a])
            for b in members:
                if a == b:
                    continue
                k = [i for i in ka if i in ca[b]]
                if len(k) < 10:
                    continue
                R, t, rms = kabsch(np.array([ca[a][i] for i in k]),
                                   np.array([ca[b][i] for i in k]))
                if rms > rms_tol:
                    continue
                key = _op_key(R, t)
                if key in seen:
                    continue
                seen.add(key)
                img = C @ R.T + t
                n_ok = 0
                for i, c in enumerate(chains):
                    d = np.linalg.norm(C - img[i], axis=1)
                    j = int(np.argmin(d))
                    if d[j] <= tol and entity_of[chains[j]] == entity_of[c]:
                        n_ok += 1
                if min_frac * len(chains) <= n_ok < len(chains):
                    ax, ang = axis_angle(R)
                    out.append((R, t, n_ok, len(chains), float(t @ ax), ang))
    return out


def classify(ops, tol_deg=6.0, centre=None):
    """Name the point group from its operator list.

    `centre` (the assembly centroid) is used only to pick an interpretable point on
    the symmetry axis: the fit determines the axis as a LINE, so the position along
    it is arbitrary and is normalised to the point closest to the assembly centre.
    """
    order = len(ops)
    if order == 1:
        return dict(point_group='C1', order=1, axis=None, axis_point=None, n_fold=1)
    info = []
    for R, t, _ in ops:
        if np.allclose(R, np.eye(3), atol=1e-6):
            continue
        ax, ang = axis_angle(R)
        info.append((ax, ang, R, t))
    # cluster rotation axes (antipodal directions are the same axis)
    axes = []
    for ax, ang, R, t in info:
        for grp in axes:
            if abs(abs(float(ax @ grp['dir'])) - 1.0) < 1 - np.cos(np.radians(tol_deg)) + 1e-3:
                grp['members'].append((ax, ang, R, t)); break
        else:
            axes.append(dict(dir=ax, members=[(ax, ang, R, t)]))
    # fold of an axis = number of ops about it + 1 (identity)
    for g in axes:
        g['fold'] = len(g['members']) + 1
    axes.sort(key=lambda g: -g['fold'])
    principal = axes[0]
    # use the member furthest from a null rotation: (I - R) is singular near 0 deg
    best = max(principal['members'], key=lambda m: min(m[1], 360.0 - m[1]))
    pt = axis_point(best[2], best[3])
    if centre is not None:                       # slide along the axis to the centre
        u = principal['dir']
        pt = pt + float((np.asarray(centre, float) - pt) @ u) * u

    if len(axes) == 1:
        pg = f"C{principal['fold']}"
        return dict(point_group=pg, order=order, axis=principal['dir'], axis_point=pt,
                    n_fold=principal['fold'])
    n = principal['fold']
    twofolds = [g for g in axes[1:] if g['fold'] == 2
                and abs(float(g['dir'] @ principal['dir'])) < np.cos(np.radians(90 - tol_deg))]
    if order == 2 * n and len(twofolds) == n:
        return dict(point_group=f"D{n}", order=order, axis=principal['dir'], axis_point=pt,
                    n_fold=n)
    for o, name in ((12, 'T'), (24, 'O'), (60, 'I')):
        if order == o:
            return dict(point_group=name, order=order, axis=principal['dir'],
                        axis_point=pt, n_fold=principal['fold'])
    return dict(point_group=f"unknown(order={order})", order=order,
                axis=principal['dir'], axis_point=pt, n_fold=principal['fold'])


# ----------------------------------------------------------------- orbits
def chain_orbits(ops, chains):
    """Orbit (symmetry-independent environment) of every chain."""
    perms = [p for _, _, p in ops if p]
    orb, seen = {}, set()
    for c in chains:
        if c in seen:
            continue
        o = sorted({p[c] for p in perms} | {c})
        for x in o:
            seen.add(x); orb[x] = tuple(o)
    return orb


def pair_orbits(ops, pairs):
    """Orbit of every contacting chain pair, as ORDERED pairs.

    Orientation matters: member i of an orbit is the image of the representative
    under group element i, so (a, b) maps to (g(a), g(b)) in that order. Sorting the
    members would silently swap some pairs end-for-end, which breaks any downstream
    comparison of residue i of one chain against residue j of the other -- e.g.
    counting how many symmetry copies reproduce a given salt bridge.
    """
    perms = [p for _, _, p in ops if p]
    pairset = {tuple(sorted(p)) for p in pairs}
    out, seen = {}, set()
    for pr in pairs:
        key = tuple(sorted(pr))
        if key in seen:
            continue
        ordered, have = [], set()
        for p in perms:
            im = (p[key[0]], p[key[1]])
            k = tuple(sorted(im))
            if k in pairset and k not in have:
                ordered.append(im); have.add(k)
        if not ordered:
            ordered, have = [key], {key}
        for k in have:
            seen.add(k)
        for k in have:
            out[k] = tuple(ordered)
    return out


def frame(axis, point):
    """Rotation matrix taking `axis` to +z, plus the origin shift, for cylindrical plots."""
    if axis is None:
        return np.eye(3), np.zeros(3)
    z = np.asarray(axis, float); z = z / np.linalg.norm(z)
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z]), np.asarray(point, float)
