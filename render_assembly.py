"""
render_assembly.py — flat, silhouetted licorice renders of a structure, from Python.

The backbone is drawn as one continuous round tube per chain, in a single flat colour,
with a thin dark outline wherever the depth jumps — the same rule ChimeraX uses for
`graphics silhouettes`. Because outlines come from DEPTH, not from drawing order, a chain
reads as one smooth worm: there are no seams at residue joints, and an outline appears
only where one part of the structure passes in front of another, or against the
background.

How: the Cα trace is splined, sampled every ~0.4 Å, and each sample is rasterised as a
sphere into a z-buffer (orthographic projection). The nearest sphere wins each pixel and
gives it its chain colour; a pixel whose neighbour is more than `edge` Å deeper, or is
background, becomes outline. No GPU, no ray tracing, plain NumPy.

    from render_assembly import load_ca, licorice
    CA = load_ca('model.cif')
    licorice(ax, CA, {'A': '#7FA8D4'}, elev=15, azim=30)

A colour may also be given PER RESIDUE: pass a list of colours, one per Cα in the order
of `chains[id]`, instead of one colour for the chain. The colour then changes along the
tube without breaking it: outlines still come from depth only, never from colour.

    python render_assembly.py model.cif out.png --chains ABCDEF --elev 15 --azim 30
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy import ndimage


def load_ca(path, atom='CA'):
    """{chain id: (n_res, 3) Cα coordinates} from a .pdb or .cif file."""
    from Bio.PDB import PDBParser, MMCIFParser
    par = MMCIFParser(QUIET=True) if path.lower().endswith(('.cif', '.mmcif')) \
        else PDBParser(QUIET=True)
    model = par.get_structure('m', path)[0]
    out = {}
    for ch in model:
        pts = [r[atom].coord for r in ch if atom in r]
        if len(pts) >= 4:
            out[ch.id] = np.asarray(pts, float)
    return out


def _pieces(P, gap=4.6):
    """Split a trace at chain breaks (Cα–Cα > gap Å) so a missing loop is not bridged."""
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cut = np.where(d > gap)[0] + 1
    return [q for q in np.split(P, cut) if len(q) >= 2]


def smooth(P, step=0.4, with_index=False):
    """Cubic spline through the Cα positions, resampled every `step` Å.

    with_index=True also returns, for every sample, the index of the nearest Cα along
    the chain, so a per-residue colour can follow the spline."""
    out = []
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    starts = np.concatenate([[0], np.where(d > 4.6)[0] + 1])
    for off, Q in zip(starts, np.split(P, starts[1:])):
        if len(Q) < 2:
            continue
        S, idx = Q, np.arange(len(Q))
        if len(Q) >= 4:
            try:
                tck, u0 = splprep(Q.T, s=0, k=3)
                L = np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()
                u = np.linspace(0, 1, max(int(L / step), 8))
                S = np.asarray(splev(u, tck)).T
                idx = np.rint(np.interp(u, u0, np.arange(len(Q)))).astype(int)
            except Exception:
                pass
        out.append((S, idx + off) if with_index else S)
    return out


def _rot(elev, azim):
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    # tilt the camera so z (the symmetry axis) points up the page at elev = 0
    Rx = np.array([[1, 0, 0], [0, np.sin(e), np.cos(e)], [0, -np.cos(e), np.sin(e)]])
    return Rx @ Rz


def licorice(ax, chains, colours, elev=15.0, azim=0.0, radius=1.0, edge=1.2,
             outline_pt=0.12, outline_colour='#1E1E1E', dpi=600, margin=1.04, zoom=1.0,
             bg=None):
    """Render `chains` ({id: (n,3)}) into `ax` as continuous flat tubes with silhouettes.

    radius      tube radius in Å (1.0 is close to ChimeraX licorice)
    edge        depth jump in Å that counts as an outline
    outline_pt  silhouette width in points
    dpi         internal raster resolution; match the dpi you save at
    """
    R = _rot(elev, azim)
    pts, ids = [], []
    keys = {}                                   # colour -> palette index

    def code(c):
        return keys.setdefault(mpl.colors.to_hex(c), len(keys))

    for cid in chains:
        c = colours.get(cid, '#9A9A9A')
        per_res = not isinstance(c, str) and len(c) == len(chains[cid])
        codes = np.array([code(x) for x in c]) if per_res else None
        for S, idx in smooth(chains[cid], with_index=True):
            pts.append(S @ R.T)
            ids.append(codes[np.clip(idx, 0, len(codes) - 1)] if per_res
                       else np.full(len(S), code(c)))
    P = np.vstack(pts); I = np.concatenate(ids)

    # raster size from the axes' real size, so `radius` and `outline_pt` are physical
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    w_in, h_in = bb.width / ax.figure.dpi, bb.height / ax.figure.dpi
    W, H = max(int(w_in * dpi), 32), max(int(h_in * dpi), 32)
    lo, hi = P[:, :2].min(0) - radius, P[:, :2].max(0) + radius
    span = np.maximum(hi - lo, 1e-6)
    scale = min(W / span[0], H / span[1]) / margin * zoom      # px per Å
    cx, cy = (lo + hi) / 2
    X = (P[:, 0] - cx) * scale + W / 2
    Y = (cy - P[:, 1]) * scale + H / 2                         # image rows grow downward
    Z = P[:, 2]

    rp = radius * scale
    r_i = int(np.ceil(rp))
    oy, ox = np.mgrid[-r_i:r_i + 1, -r_i:r_i + 1]
    inside = ox ** 2 + oy ** 2 <= rp ** 2
    ox, oy = ox[inside], oy[inside]
    bulge = np.sqrt(np.maximum(rp ** 2 - ox ** 2 - oy ** 2, 0)) / scale  # Å toward viewer

    zbuf = np.full(H * W, -np.inf)
    xi0, yi0 = np.round(X).astype(int), np.round(Y).astype(int)
    for dx, dy, b in zip(ox, oy, bulge):
        xi, yi = xi0 + dx, yi0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        np.maximum.at(zbuf, yi[ok] * W + xi[ok], Z[ok] + b)
    owner = np.full(H * W, -1)
    for dx, dy, b in zip(ox, oy, bulge):
        xi, yi = xi0 + dx, yi0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        flat = yi[ok] * W + xi[ok]
        hit = (Z[ok] + b) >= zbuf[flat] - 1e-9
        owner[flat[hit]] = I[ok][hit]
    zbuf = zbuf.reshape(H, W); owner = owner.reshape(H, W)
    fg = owner >= 0

    # silhouettes: background boundary, or a depth jump larger than `edge`
    zf = np.where(fg, zbuf, -1e9)
    jump = np.zeros_like(fg)
    for sh in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = np.roll(zf, sh, axis=(0, 1))
        jump |= fg & ((zf - nb) > edge)            # this pixel is IN FRONT of a far neighbour
    jump |= fg & ~ndimage.binary_erosion(fg)
    k = max(int(round(outline_pt / 72 * dpi)), 1)
    if k > 1:
        jump = ndimage.binary_dilation(jump, iterations=k - 1) & fg | jump

    pal = np.array([mpl.colors.to_rgba(k) for k in keys])
    img = np.zeros((H, W, 4))
    img[fg] = pal[owner[fg]]
    img[jump] = mpl.colors.to_rgba(outline_colour)
    if bg is not None:
        img[~fg] = mpl.colors.to_rgba(bg)
    ax.imshow(img, interpolation='antialiased', extent=(0, W, H, 0))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal'); ax.set_axis_off()
    return scale


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model'); ap.add_argument('out')
    ap.add_argument('--chains', default=None, help='chain ids to draw, e.g. ABCDEF')
    ap.add_argument('--elev', type=float, default=15.0)
    ap.add_argument('--azim', type=float, default=0.0)
    ap.add_argument('--radius', type=float, default=1.0)
    ap.add_argument('--colour', default='#7FA8D4')
    ap.add_argument('--size', type=float, default=4.0, help='figure side, inches')
    ap.add_argument('--dpi', type=int, default=600)
    a = ap.parse_args()
    CA = load_ca(a.model)
    if a.chains:
        CA = {k: v for k, v in CA.items() if k in set(a.chains)}
    fig, ax = plt.subplots(figsize=(a.size, a.size))
    fig.subplots_adjust(0, 0, 1, 1)
    licorice(ax, CA, {k: a.colour for k in CA}, elev=a.elev, azim=a.azim,
             radius=a.radius, dpi=a.dpi)
    fig.savefig(a.out, dpi=a.dpi)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
