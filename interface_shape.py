"""
interface_shape.py - shape complementarity and gap volume for a pair of chains.

Sc   Lawrence & Colman (1993) J Mol Biol 234:946. For every point of one partner's
     solvent-excluded surface that faces the other partner, the dot product of the two
     outward normals is weighted by exp(-w d^2) and the median is taken over both
     directions. 1.0 = perfect interdigitation, ~0.7-0.8 = typical oligomer interface,
     ~0.4-0.5 = a loose or flat contact.

gap  Laskowski (1995) J Mol Graph 13:323. The volume of empty space trapped between the
     two partners, i.e. outside both van der Waals volumes and unreachable by a solvent
     probe coming from bulk. Divided by the interface area it gives the gap index (A),
     a measure of how badly the two surfaces fail to fill each other.

Both are computed on a grid restricted to the interface region, so cost scales with the
size of the interface and not with the size of the assembly. No network, no external
program: NumPy + SciPy + scikit-image only.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    from skimage.measure import marching_cubes
    HAVE_SKIMAGE = True
except ImportError:                                    # pragma: no cover
    HAVE_SKIMAGE = False

PROBE = 1.40
SPACING = 0.50          # grid step, A
SEP = 8.0               # a surface point faces the partner if a partner point is nearer
TRIM = 1.5              # peripheral band discarded, as in the original Sc program
W = 0.5                 # A^-2, the exp(-w d^2) weight of Lawrence & Colman
NEAR_CUT = 12.0         # atoms kept around the interface
GAP_REACH = 5.0         # a gap voxel must be this close to BOTH partners


def _paint(shape, origin, spacing, coords, radii, extra=0.0):
    """Boolean mask of the union of spheres, painted one local box at a time."""
    m = np.zeros(shape, bool)
    for c, r in zip(coords, radii):
        rr = r + extra
        lo = np.maximum(((c - rr - origin) / spacing).astype(int), 0)
        hi = np.minimum(((c + rr - origin) / spacing).astype(int) + 2, shape)
        if np.any(lo >= hi):
            continue
        g = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d2 = sum((origin[k] + spacing * g[k] - c[k]) ** 2 for k in range(3))
        m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= d2 <= rr * rr
    return m


def _ses(mask_probe, origin, spacing, probe=PROBE):
    """Solvent-excluded surface: vertices (A) and OUTWARD unit normals.

    mask_probe marks every voxel within r+probe of an atom, so its complement is where a
    probe centre may sit. The SES is the probe-radius level set of the distance to that
    accessible set; marching cubes returns normals pointing into the molecule, so they
    are negated on the way out.
    """
    acc = ~mask_probe
    if not acc.any() or acc.all():
        return None, None
    g = ndimage.distance_transform_edt(~acc, sampling=spacing)
    if g.max() <= probe:
        return None, None
    verts, _, normals, _ = marching_cubes(g, level=probe, spacing=(spacing,) * 3)
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-9)
    return verts + origin, -n


def sc_and_gap(cA, rA, cB, rB, area=None, spacing=SPACING, probe=PROBE,
               near_cut=NEAR_CUT, want_gap=True):
    """Shape complementarity and gap volume for one pair of chains.

    cA/cB are (n,3) heavy-atom coordinates, rA/rB the matching van der Waals radii.
    Returns a dict; sc is None when the interface is too small to mesh.
    """
    out = {'sc': None, 'sc_n_dots': 0, 'gap_volume': None, 'gap_index': None}
    if not HAVE_SKIMAGE or len(cA) == 0 or len(cB) == 0:
        return out
    cA, cB = np.asarray(cA, float), np.asarray(cB, float)
    rA, rB = np.asarray(rA, float), np.asarray(rB, float)

    # keep only the neighbourhood of the contact: cost follows interface size
    ia = cKDTree(cB).query_ball_point(cA, near_cut)
    ib = cKDTree(cA).query_ball_point(cB, near_cut)
    ka = np.array([len(x) > 0 for x in ia])
    kb = np.array([len(x) > 0 for x in ib])
    if ka.sum() < 4 or kb.sum() < 4:
        return out
    cA, rA, cB, rB = cA[ka], rA[ka], cB[kb], rB[kb]

    pad = probe + max(rA.max(), rB.max()) + 2.0
    allc = np.vstack([cA, cB])
    origin = allc.min(0) - pad
    shape = tuple((((allc.max(0) + pad) - origin) / spacing).astype(int) + 2)
    if np.prod(shape) > 60e6:                    # refuse to blow up memory
        return out

    pA = _paint(shape, origin, spacing, cA, rA, probe)
    pB = _paint(shape, origin, spacing, cB, rB, probe)
    vA, nA = _ses(pA, origin, spacing, probe)
    vB, nB = _ses(pB, origin, spacing, probe)
    if vA is None or vB is None:
        return out

    # --- Sc: keep only BURIED surface points, match them, trim the peripheral band
    # A point of A's isolated surface is buried by B when a probe rolling on B could
    # touch it, i.e. it lies within 2 probe radii of B's van der Waals surface. Without
    # this test the median is taken over the whole neighbourhood and collapses to zero.
    def buried(v, c, r):
        dd, jj = cKDTree(c).query(v, k=min(12, len(c)))
        return (dd - r[jj]).min(axis=1) < 2 * probe

    tB, tA = cKDTree(vB), cKDTree(vA)
    dAB, jAB = tB.query(vA)
    dBA, jBA = tA.query(vB)
    mA = buried(vA, cB, rB) & (dAB < SEP)
    mB = buried(vB, cA, rA) & (dBA < SEP)
    if mA.sum() < 20 or mB.sum() < 20:
        return out
    for v, m in ((vA, mA), (vB, mB)):              # peripheral trim
        if (~m).any() and m.any():
            edge = cKDTree(v[~m]).query(v[m])[0] < TRIM
            m[np.where(m)[0][edge]] = False
    if mA.sum() < 20 or mB.sum() < 20:
        return out
    sA = -np.einsum('ij,ij->i', nA[mA], nB[jAB[mA]]) * np.exp(-W * dAB[mA] ** 2)
    sB = -np.einsum('ij,ij->i', nB[mB], nA[jBA[mB]]) * np.exp(-W * dBA[mB] ** 2)
    out['sc'] = float(0.5 * (np.median(sA) + np.median(sB)))
    out['sc_n_dots'] = int(mA.sum() + mB.sum())

    # --- gap volume: empty, buried, and flanked by both partners
    if want_gap:
        vdwA = _paint(shape, origin, spacing, cA, rA)
        vdwB = _paint(shape, origin, spacing, cB, rB)
        acc = ~(pA | pB)                                    # probe centres in the complex
        lab, nlab = ndimage.label(acc)
        if nlab:
            border = set(np.unique(np.concatenate([
                lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))) - {0}
            bulk = np.isin(lab, list(border)) if border else np.zeros_like(acc)
            solvent = ndimage.distance_transform_edt(~bulk, sampling=spacing) <= probe
            dA = ndimage.distance_transform_edt(~vdwA, sampling=spacing)
            dB = ndimage.distance_transform_edt(~vdwB, sampling=spacing)
            gap = (~(vdwA | vdwB)) & (~solvent) & (dA < GAP_REACH) & (dB < GAP_REACH)
            vol = float(gap.sum()) * spacing ** 3
            out['gap_volume'] = vol
            if area:
                out['gap_index'] = vol / float(area)
    return out


def _worker(job):
    """Top-level so it can be pickled to a process pool."""
    key, cA, rA, cB, rB, area, spacing = job
    try:
        return key, sc_and_gap(cA, rA, cB, rB, area=area, spacing=spacing)
    except Exception as exc:                                   # never kill the run
        return key, {'sc': None, 'sc_n_dots': 0, 'gap_volume': None,
                     'gap_index': None, 'error': str(exc)}
