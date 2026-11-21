"""Shared drawing helpers: pastel palette, squircle bars, helix cylinders,
soft-edged pentagon strand arrows, density strips and gradient density plots."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Ellipse, Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# ---- pastel palette -------------------------------------------------------
# belt hues are true orange (hue ~18-22 deg), not yellow-brown
PAL = {'tube': '#8CB4DC', 'belt': '#F79E72', 'fibre': '#95C892'}
DARK = {'tube': '#4A7DB0', 'belt': '#DD5F2C', 'fibre': '#4F9A55'}
LIGHT = {'tube': '#C5DCF0', 'belt': '#FDD6C3', 'fibre': '#CFE7CE'}
SHADES = {'tube': ['#C5DCF0', '#6FA3D2', '#3C6F9E'],
          'belt': ['#FFD5C0', '#FBA887', '#F17F51', '#D45B2B'],
          'fibre': ['#CFE7CE', '#8CC489', '#4E9A55']}
INK = '#333333'
BOND = '#1A1A1A'

CLSCOL = {'charged': '#DD6150', 'polar': '#7FB685', 'aromatic': '#B48BC4',
          'aliphatic': '#F0D264', 'Gly / Pro': '#C2C2C2',
          # nucleotides: a blue-violet that is distinct from every amino-acid class
          'nucleotide': '#6C8EBF', 'ligand': '#8A6A4B', 'other': '#E3E3E3'}
# Grey is reserved for "buried but making no specific interaction", in the figures and
# in the ChimeraX scripts alike, so no interaction type may use it. Salt bridges take
# blue to echo the blue nitrogen of the 3D view; hydrogen bonds take the rose that salt
# bridges vacated, a tone already proven legible beside the charged red of fig_chemistry.
INTCOL = {'salt': '#2E6FB7',
          'hbond': '#D9698F',
          'stack': '#3FA796',
          'catpi': '#9B6BC9'}


def ramp(colour, light='#FFFFFF'):
    return LinearSegmentedColormap.from_list('r', [light, colour])


def blend(c1, c2, f=0.5):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a * (1 - f) + b * f)


# ---- geometry helpers -----------------------------------------------------
def _rad(ax, pts):
    """radius in points -> (dx, dy) in data units"""
    bb = ax.get_window_extent()
    px = pts / 72.0 * ax.figure.dpi
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return px * (x1 - x0) / bb.width, px * (y1 - y0) / bb.height


def rounded_polygon(ax, pts, r_pt):
    """Closed polygon with corners rounded by r_pt POINTS (circular under any aspect)."""
    sx, sy = _rad(ax, 1.0)
    P = np.array([[p[0] / sx, p[1] / sy] for p in pts], float)
    n = len(P)
    verts, codes = [], []
    for i in range(n):
        v = P[i]
        u1, u2 = P[(i - 1) % n] - v, P[(i + 1) % n] - v
        l1, l2 = np.linalg.norm(u1), np.linalg.norm(u2)
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        u1, u2 = u1 / l1, u2 / l2
        d = min(r_pt, 0.48 * l1, 0.48 * l2)
        a, b = v + u1 * d, v + u2 * d
        first = not verts
        verts.append(a); codes.append(Path.MOVETO if first else Path.LINETO)
        verts += [v, b]; codes += [Path.CURVE3, Path.CURVE3]
    verts.append(verts[0]); codes.append(Path.CLOSEPOLY)
    return Path([[x * sx, y * sy] for x, y in verts], codes)


def _capsule(x, y, w, h, rx, ry, round_lo=True, round_hi=True, vertical=True):
    """rectangle with the low and/or high end rounded; vertical=True rounds top/bottom."""
    rx, ry = min(rx, abs(w) / 2.0), min(ry, abs(h) / 2.0)
    v, c = [], []
    if vertical:
        v += [(x, y + ry)] if round_lo else [(x, y)]
        c += [Path.MOVETO]
        if round_hi:
            v += [(x, y + h - ry)]; c += [Path.LINETO]
            v += [(x, y + h), (x + rx, y + h)]; c += [Path.CURVE3, Path.CURVE3]
            v += [(x + w - rx, y + h)]; c += [Path.LINETO]
            v += [(x + w, y + h), (x + w, y + h - ry)]; c += [Path.CURVE3, Path.CURVE3]
        else:
            v += [(x, y + h), (x + w, y + h)]; c += [Path.LINETO, Path.LINETO]
        if round_lo:
            v += [(x + w, y + ry)]; c += [Path.LINETO]
            v += [(x + w, y), (x + w - rx, y)]; c += [Path.CURVE3, Path.CURVE3]
            v += [(x + rx, y)]; c += [Path.LINETO]
            v += [(x, y), (x, y + ry)]; c += [Path.CURVE3, Path.CURVE3]
        else:
            v += [(x + w, y), (x, y)]; c += [Path.LINETO, Path.LINETO]
    else:
        v += [(x + rx, y)] if round_lo else [(x, y)]
        c += [Path.MOVETO]
        if round_hi:
            v += [(x + w - rx, y)]; c += [Path.LINETO]
            v += [(x + w, y), (x + w, y + ry)]; c += [Path.CURVE3, Path.CURVE3]
            v += [(x + w, y + h - ry)]; c += [Path.LINETO]
            v += [(x + w, y + h), (x + w - rx, y + h)]; c += [Path.CURVE3, Path.CURVE3]
        else:
            v += [(x + w, y), (x + w, y + h)]; c += [Path.LINETO, Path.LINETO]
        if round_lo:
            v += [(x + rx, y + h)]; c += [Path.LINETO]
            v += [(x, y + h), (x, y + h - ry)]; c += [Path.CURVE3, Path.CURVE3]
            v += [(x, y + ry)]; c += [Path.LINETO]
            v += [(x, y), (x + rx, y)]; c += [Path.CURVE3, Path.CURVE3]
        else:
            v += [(x, y + h), (x, y)]; c += [Path.LINETO, Path.LINETO]
    v.append(v[0]); c.append(Path.CLOSEPOLY)
    return Path(v, c)


def sbar(ax, xs, heights, bottoms=None, width=1.0, color='#8CB4DC',
         round_pt=1.2, gap_pt=0.0, **kw):
    """vertical bars; every segment rounded at BOTH ends."""
    xs = np.atleast_1d(np.asarray(xs, float))
    heights = np.atleast_1d(np.asarray(heights, float))
    bottoms = np.zeros_like(heights) if bottoms is None else np.atleast_1d(np.asarray(bottoms, float))
    rx, ry = _rad(ax, round_pt)
    _, gy = _rad(ax, gap_pt)
    for x, h, b in zip(xs, heights, bottoms):
        if h <= 0:
            continue
        hh = max(h - gy, h * 0.45)
        ax.add_patch(PathPatch(_capsule(x - width / 2, b + (h - hh) / 2, width, hh, rx, ry),
                               fc=color, ec='none', **kw))


def sbarh(ax, y, width, left=0.0, height=0.6, color='#8CB4DC', round_pt=2.0,
          gap_pt=1.1, round_left=None, **kw):
    """single horizontal bar segment, rounded at BOTH ends, inset by gap_pt."""
    rx, ry = _rad(ax, round_pt)
    gx, _ = _rad(ax, gap_pt)
    w = max(width - gx, width * 0.35)
    x = left + (width - w) / 2
    rx, ry = min(rx, abs(w) / 2), min(ry, height / 2)
    ax.add_patch(PathPatch(_capsule(x, y - height / 2, w, height, rx, ry, vertical=False),
                           fc=color, ec='none', **kw))


# ---- secondary structure --------------------------------------------------
def ss_ribbon(ax, resi, ss, y, h, helix_c='#6E7B8B', strand_c='#AFB7C0',
              line_c='#C8CCD0', round_pt=0.9, fuse_pt=4.0):
    """Draw the secondary structure as a track: helices as cylinders, strands as arrows.

    Helices broken by one or two residues of coil are drawn as one cylinder when the
    break would be narrower than `fuse_pt` points on the page. A real 2-residue break
    carries no information at this scale, and drawing it produces a row of touching
    end-caps that reads as noise. Fusing is decided in POINTS, not residues, so the
    same model wrapped onto a wider or narrower axis still looks right.
    """
    resi = np.asarray(resi)
    span = resi.max() - resi.min() + 1
    ax.add_patch(Rectangle((resi.min() - 0.5, y + h / 2 - h * 0.055), span, h * 0.11,
                           fc=line_c, ec='none', zorder=1))
    runs, cur = [], None
    for r, s_ in zip(resi, ss):
        if cur and cur[0] == s_ and r == cur[2] + 1:
            cur = (s_, cur[1], r)
        else:
            if cur:
                runs.append(cur)
            cur = (s_, r, r)
    if cur:
        runs.append(cur)
    rx, ry = _rad(ax, 1.0)                       # data units per point
    runs = [r for r in runs if r[0] in ('H', 'E')]
    fused = []
    for s_, a, b in runs:
        if fused and fused[-1][0] == s_ and (a - fused[-1][2] - 1) <= fuse_pt * rx:
            fused[-1] = (s_, fused[-1][1], b)
        else:
            fused.append([s_, a, b])
    for s_, a, b in fused:
        L = b - a + 1
        if s_ == 'H' and L >= 3:
            cw = min(L * 0.32, max(h * 0.30 / max(ry, 1e-12) * rx, 1.2))
            ax.add_patch(Rectangle((a - 0.5 + cw / 2, y), L - cw, h,
                                   fc=helix_c, ec='none', zorder=2))
            for cx in (a - 0.5 + cw / 2, a - 0.5 + L - cw / 2):
                ax.add_patch(Ellipse((cx, y + h / 2), cw, h, fc=helix_c, ec='none', zorder=3))
            body = max(L - cw, 0.0)
            body_pt = body / max(rx, 1e-12)      # only ring a cylinder wide enough to read
            nring = int(body_pt // 7)
            for i in range(1, nring + 1):
                cx = a - 0.5 + cw / 2 + body * i / (nring + 1)
                ax.add_patch(Ellipse((cx, y + h / 2), cw, h, fc='none', ec='white',
                                     lw=0.35, zorder=4))
        elif s_ == 'E' and L >= 2:
            x0, x1 = a - 0.5, a - 0.5 + L
            head = min(max(L * 0.38, 0.8), L * 0.55)
            xb = x1 - head
            yc, hb = y + h / 2, h * 0.56
            pts = [(x0, yc - hb / 2), (xb, yc - hb / 2), (xb, y), (x1, yc),
                   (xb, y + h), (xb, yc + hb / 2), (x0, yc + hb / 2)]
            ax.add_patch(PathPatch(rounded_polygon(ax, pts, round_pt),
                                   fc=strand_c, ec='none', zorder=2))

def density_strip(ax, resi, values, y, h, colour, vmax=None, sigma=3.0):
    from scipy.ndimage import gaussian_filter1d
    sm = gaussian_filter1d(np.asarray(values, float), sigma)
    vmax = vmax if vmax else sm.max()
    ax.imshow(sm[None, :], aspect='auto', cmap=ramp(colour), vmin=0, vmax=vmax,
              extent=(resi.min() - 0.5, resi.max() + 0.5, y, y + h), zorder=2,
              interpolation='bilinear')
    ax.add_patch(Rectangle((resi.min() - 0.5, y), resi.max() - resi.min() + 1, h,
                           fc='none', ec='white', lw=0.5, zorder=3))
    return sm, vmax


def density_plot(ax, resi, values, colour, vmax=None, sigma=3.0, lw=0.7):
    from scipy.ndimage import gaussian_filter1d
    resi = np.asarray(resi, float)
    sm = gaussian_filter1d(np.asarray(values, float), sigma)
    vmax = vmax if vmax else max(sm.max(), 1e-9)
    ax.set_ylim(0, vmax * 1.12)
    im = ax.imshow(sm[None, :], aspect='auto', cmap=ramp(colour), vmin=0, vmax=vmax,
                   extent=(resi.min() - 0.5, resi.max() + 0.5, 0, vmax * 1.12),
                   zorder=2, interpolation='bilinear')
    poly = Polygon(np.column_stack([np.r_[resi[0], resi, resi[-1]], np.r_[0, sm, 0]]),
                   closed=True, transform=ax.transData)
    im.set_clip_path(poly)
    ax.plot(resi, sm, color=colour, lw=lw, zorder=3, solid_capstyle='round')
    return sm, vmax


# ---- palettes for an arbitrary number of entities -------------------------
FAMILIES = [
    dict(name='blue',   pal='#8CB4DC', dark='#4A7DB0', light='#C5DCF0',
         shades=['#C5DCF0', '#6FA3D2', '#3C6F9E', '#27506F']),
    dict(name='orange', pal='#F79E72', dark='#DD5F2C', light='#FDD6C3',
         shades=['#FFD5C0', '#FBA887', '#F17F51', '#D45B2B']),
    dict(name='green',  pal='#95C892', dark='#4F9A55', light='#CFE7CE',
         shades=['#CFE7CE', '#8CC489', '#4E9A55', '#356E3C']),
    dict(name='purple', pal='#B9A3D6', dark='#7B5FA8', light='#E2D8F0',
         shades=['#E2D8F0', '#BFA8DC', '#9376BE', '#6B4E93']),
    dict(name='teal',   pal='#8FCBC8', dark='#3F8F8A', light='#CDE8E6',
         shades=['#CDE8E6', '#92CDC9', '#4FA39D', '#2F726E']),
    dict(name='rose',   pal='#E8A6B8', dark='#C25C7C', light='#F6D9E1',
         shades=['#F6D9E1', '#E9A8BA', '#D27592', '#A94E6B']),
]


def assign_palettes(names):
    """Map entity names onto distinct colour families, in order."""
    return {n: FAMILIES[i % len(FAMILIES)] for i, n in enumerate(names)}


def shade_of(fam, i, n):
    """Pick shade i of n from a family, spread across its range."""
    s = fam['shades']
    if n <= 1:
        return fam['pal']
    idx = int(round(i * (len(s) - 1) / max(1, n - 1)))
    return s[min(idx, len(s) - 1)]


def family_from_hex(hexcol, name=None):
    """Build a full colour family from one user-supplied hex colour.

    Lets a user pin a protein to the exact colour their other figures use, while the
    pipeline still has the lighter and darker shades it needs to separate symmetry
    copies and to draw density tracks.
    """
    import matplotlib.colors as mc
    c = hexcol.strip()
    c = c if c.startswith('#') else '#' + c
    mc.to_rgb(c)                                   # raises on a malformed colour
    hexof = (lambda t: mc.to_hex(t))
    return {'name': name or c,
            'pal': c,
            'dark': hexof(blend(c, '#1A1A1A', 0.38)),
            'light': hexof(blend(c, '#FFFFFF', 0.55)),
            'shades': [hexof(blend(c, '#FFFFFF', 0.55)), c,
                       hexof(blend(c, '#1A1A1A', 0.22)), hexof(blend(c, '#1A1A1A', 0.45))]}


CONSURF = ['#10C8D1', '#8CFFFF', '#D7FFFF', '#EAFFFF', '#FFFFFF',
           '#FCEDF4', '#FAC9DE', '#F07DAB', '#A02560']


# Diverging pair for per-residue solvation. Deliberately desaturated and dark so it
# cannot be mistaken for an interaction-type colour or a protein family colour.
SOLV_GOOD = '#2C6FAD'    # burial pays for itself (apolar surface hidden from water)
SOLV_BAD = '#E08214'     # burial costs (polar or charged surface desolvated)


def solv_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('solv', [SOLV_GOOD, '#F5F3EF', SOLV_BAD])


# Sulfur yellow, so a disulfide in the 2D network matches the S atom in ChimeraX.
COVALENT = '#E6B800'
