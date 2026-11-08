"""pcbrisk_graph.py - local consequence engine (v3).

Fixes vs the notebook-05 engine:
 1. LOCAL evidence. v2 asked "does this template net split ANYWHERE / does this test component merge nets ANYWHERE";
    any box touching such a net was flagged. v3 only looks at (template-net, test-component) pairs that sit next to
    the missing / excess copper inside the box.
 2. Alignment tolerance: missing = template & ~dilate(test, TOL); excess = test & ~dilate(template, TOL).
 3. Minimum evidence: tiny missing/excess blobs (< MIN_PX) are ignored.
 4. Copper polarity is decided on the TEMPLATE and reused for the test image (v2 decided per image).
"""
import numpy as np, cv2
from scipy import ndimage as ndi

S8 = np.ones((3, 3), bool)
H = W = 640
DEFAULTS = dict(tol=1, min_px=6, reach=4, sig=8, pad=6, win=12)


def _otsu(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t > 0


def drop_small(m, min_px):
    lab, n = ndi.label(m, structure=S8)
    if n == 0:
        return m
    keep = np.bincount(lab.ravel(), minlength=n + 1) >= min_px
    keep[0] = False
    return keep[lab]


def polarity_invert(template_bgr, copper='auto'):
    if copper == 'dark':
        return True
    if copper == 'light':
        return False
    return bool(_otsu(template_bgr).mean() > 0.5)      # copper = minority class


def copper_mask(img, invert):
    m = _otsu(img)
    if invert:
        m = ~m
    return drop_small(m, 30)


def base(template_bgr, test_bgr, copper='auto'):
    """Parameter-free part (cache this when tuning)."""
    inv = polarity_invert(template_bgr, copper)
    mt, mx = copper_mask(template_bgr, inv), copper_mask(test_bgr, inv)
    lt, nt = ndi.label(mt, structure=S8)
    lx, nx = ndi.label(mx, structure=S8)
    return dict(mt=mt, mx=mx, lt=lt.astype(np.int32), lx=lx.astype(np.int32))


def _dil(m, r):
    if r <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def derive(b, tol=DEFAULTS['tol'], min_px=DEFAULTS['min_px'], **_):
    """Parameter-dependent maps."""
    a = dict(b)
    a['miss'] = drop_small(b['mt'] & ~_dil(b['mx'], tol), min_px)
    a['exc'] = drop_small(b['mx'] & ~_dil(b['mt'], tol), min_px)
    return a


def analyse(template_bgr, test_bgr, copper='auto', **kw):
    return derive(base(template_bgr, test_bgr, copper), **kw)


def _split_pairs(lt, lx, region, key, sig):
    """Among pixels of `region` that are copper in BOTH images, count (template-net, test-component) pairs.
    key='lt': return max over template nets of the 2nd-largest number of pixels in a different test component
              (>=sig => net is split here)  -> open strength
    key='lx': same for a test component covering >=2 template nets -> short strength"""
    t, x = lt[region], lx[region]
    ok = (t > 0) & (x > 0)
    if not ok.any():
        return 0
    t, x = t[ok].astype(np.int64), x[ok].astype(np.int64)
    code = t * (x.max() + 1) + x
    u, c = np.unique(code, return_counts=True)
    ut, ux = u // (x.max() + 1), u % (x.max() + 1)
    grp = ut if key == 'lt' else ux
    best = 0
    for g in np.unique(grp[c >= 1]):
        cs = np.sort(c[grp == g])[::-1]
        if len(cs) >= 2:
            best = max(best, int(cs[1]))
    return best


def query(a, box, pad=DEFAULTS['pad'], reach=DEFAULTS['reach'], sig=DEFAULTS['sig'], win=DEFAULTS['win'],
          min_px=DEFAULTS['min_px'], **_):
    """-> dict(g_open, g_short, missing, excess, s_open, s_short)"""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    ext = pad + win
    ya, yb = max(0, y1 - ext), min(H, y2 + ext + 1)
    xa, xb = max(0, x1 - ext), min(W, x2 + ext + 1)
    inner = np.zeros((yb - ya, xb - xa), bool)
    inner[max(0, y1 - pad) - ya:min(H, y2 + pad + 1) - ya, max(0, x1 - pad) - xa:min(W, x2 + pad + 1) - xa] = True
    miss, exc = a['miss'][ya:yb, xa:xb] & inner, a['exc'][ya:yb, xa:xb] & inner
    lt, lx = a['lt'][ya:yb, xa:xb], a['lx'][ya:yb, xa:xb]
    both = a['mt'][ya:yb, xa:xb] & a['mx'][ya:yb, xa:xb]
    ms, ex = int(miss.sum()), int(exc.sum())
    s_open = _split_pairs(lt, lx, _dil(miss, reach) & both, 'lt', sig) if ms >= min_px else 0
    s_short = _split_pairs(lt, lx, _dil(exc, reach) & both, 'lx', sig) if ex >= min_px else 0
    return dict(g_open=s_open >= sig, g_short=s_short >= sig, missing=ms, excess=ex, s_open=s_open, s_short=s_short)


def verdict(go, gs):
    return np.select([go & gs, go, gs], ['both', 'open', 'short'], 'none')
