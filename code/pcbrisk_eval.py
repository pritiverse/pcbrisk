"""pcbrisk_eval.py - leave-one-board-out flagging experiment (v3).

Fixes vs notebook 05:
 * every board is a test board (LOBO, 11 boards) instead of 4 folds
 * score model for a calibration board is cross-fitted (never trained on it or on the test board)
 * calibration pools ALL other boards (not one fold) and offers board-aware calibrators:
     pooled   : classic split-conformal (what v2 did, but with 10 boards)
     balanced : every board gets equal weight
     robust   : choose tau so that  mean_board_miss + Z * sd_board_miss <= alpha  (accounts for board-to-board drift)
 * paired bootstrap over BOARDS for B-A, plus a calibration-free matched-miss comparison
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ---------------------------------------------------------------- features
def add_features(df, crit=(0, 1)):
    df = df.copy()
    df['crit'] = df.cls.isin(crit).astype(int)
    df['conf_crit'] = df.conf * df.crit
    df['logarea'] = np.log((df.x2 - df.x1).clip(1) * (df.y2 - df.y1).clip(1))
    go, gs = df.g_open.astype(int), df.g_short.astype(int)
    df['g_open'], df['g_short'] = go, gs
    df['g_agree'] = (((df.cls == 0) & (go == 1)) | ((df.cls == 1) & (gs == 1))).astype(int)
    df['g_cross'] = (((df.cls == 0) & (gs == 1)) | ((df.cls == 1) & (go == 1))).astype(int)
    df['lm'], df['le'] = np.log1p(df.missing), np.log1p(df.excess)
    df['lso'], df['lss'] = np.log1p(df.s_open), np.log1p(df.s_short)
    df['conf_agree'] = df.conf * df.g_agree
    return df


F_IMG = ['conf', 'crit', 'conf_crit', 'logarea']
F_GRAPH = F_IMG + ['g_open', 'g_short', 'g_agree', 'g_cross', 'lm', 'le', 'lso', 'lss', 'conf_agree']


# ---------------------------------------------------------------- scores
def _fit(df, feats, mask):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    m.fit(df.loc[mask, feats], df.loc[mask, 'y'])
    return m


def gt_scores(sc, gt_sub):
    """best score of any matched box per critical GT defect; -1 = no box at all (always a miss)"""
    return np.array([sc[i].max() if len(i) else -1.0 for i in gt_sub.pred_idx])


# ---------------------------------------------------------------- calibrators (miss = score < tau)
def tau_pooled(s, alpha):
    s = np.sort(s); k = int(np.floor(alpha * (len(s) + 1)))
    return -np.inf if k < 1 else s[k - 1]


def tau_weighted(s, w, alpha):
    o = np.argsort(s); s, w = s[o], w[o]
    p = np.cumsum(w) / (w.sum() + w.mean())
    k = int((p <= alpha).sum())
    return -np.inf if k < 1 else s[k - 1]


def tau_robust(per_board, alpha, z=1.0):
    """largest tau with mean_b miss_b(tau) + z*sd_b miss_b(tau) <= alpha (miss_b = share of board-b GT with score<tau)"""
    allv = np.unique(np.concatenate(per_board))
    best = -np.inf
    for t in allv:
        m = np.array([(v < t).mean() for v in per_board])
        if m.mean() + z * m.std(ddof=1) <= alpha:
            best = t
        else:
            break                                   # miss(tau) is monotone in tau
    return best


CALIBRATORS = ('pooled', 'balanced', 'robust')


def calibrate(kind, cal, alpha, z=1.0):
    """cal: dict board -> array of GT scores"""
    if kind == 'pooled':
        return tau_pooled(np.concatenate(list(cal.values())), alpha)
    if kind == 'balanced':
        s = np.concatenate(list(cal.values())); w = np.concatenate([np.full(len(v), 1.0 / len(v)) for v in cal.values()])
        return tau_weighted(s, w, alpha)
    if kind == 'robust':
        return tau_robust(list(cal.values()), alpha, z)
    raise ValueError(kind)


# ---------------------------------------------------------------- LOBO
def lobo(df, gt, imgs_per_board, policies, alphas, calibrators=CALIBRATORS, z=1.0):
    """df: boxes (needs board, y, features); gt: critical GT (board, pred_idx).
    policies: {name: feature list or None (raw conf*crit)}.  Returns (per-board table, LOBO box scores dict)."""
    boards = sorted(df.board.unique()); rows = []; scores = {p: np.zeros(len(df)) for p in policies}
    for b in boards:
        others = [x for x in boards if x != b]
        for pol, feats in policies.items():
            if feats is None:
                sc_all = (df.conf * df.crit).values
                sc_te = sc_all; cal = {o: gt_scores(sc_all, gt[gt.board == o]) for o in others}
            else:
                sc_te = _fit(df, feats, df.board.isin(others)).predict_proba(df[feats])[:, 1]
                cal = {}
                for o in others:                     # cross-fit: model never saw o or b
                    m = _fit(df, feats, ~df.board.isin([o, b]))
                    idx = np.where(df.board == o)[0]
                    sc_o = np.zeros(len(df)); sc_o[idx] = m.predict_proba(df.iloc[idx][feats])[:, 1]
                    cal[o] = gt_scores(sc_o, gt[gt.board == o])
            mte = (df.board == b).values; scores[pol][mte] = sc_te[mte]
            s_te = gt_scores(sc_te, gt[gt.board == b])
            for kind in calibrators:
                for al in alphas:
                    tau = calibrate(kind, cal, al, z)
                    flag = mte & (sc_te >= tau)
                    rows.append(dict(board=b, policy=pol, calib=kind, alpha=al, n_gt=len(s_te),
                                     n_miss=int(((s_te < tau) | (s_te < 0)).sum()), flagged=int(flag.sum()),
                                     n_img=int(imgs_per_board[b]), tau=tau))
    return pd.DataFrame(rows), scores


def summarise(R):
    out = []
    for (c, a, p), g in R.groupby(['calib', 'alpha', 'policy']):
        mr = g.n_miss / g.n_gt
        out.append(dict(calib=c, alpha=a, policy=p, miss_pooled=g.n_miss.sum() / g.n_gt.sum(), miss_board_mean=mr.mean(),
                        miss_board_max=mr.max(), boards_over_alpha=(mr > a).mean(),
                        flagged_per_img=g.flagged.sum() / g.n_img.sum()))
    return pd.DataFrame(out).set_index(['calib', 'alpha', 'policy']).round(4)


# ---------------------------------------------------------------- paired bootstrap over boards
def paired_bootstrap(R, pol_a, pol_b, calib, alpha, n_boot=5000, seed=0):
    ra = R[(R.policy == pol_a) & (R.calib == calib) & (R.alpha == alpha)].set_index('board').sort_index()
    rb = R[(R.policy == pol_b) & (R.calib == calib) & (R.alpha == alpha)].set_index('board').sort_index()
    nb = len(ra); rng = np.random.default_rng(seed)
    def stats(ix):
        la, lb = ra.flagged.values[ix].sum() / ra.n_img.values[ix].sum(), rb.flagged.values[ix].sum() / rb.n_img.values[ix].sum()
        ma, mb = ra.n_miss.values[ix].sum() / ra.n_gt.values[ix].sum(), rb.n_miss.values[ix].sum() / rb.n_gt.values[ix].sum()
        return lb - la, mb - ma
    full = stats(np.arange(nb))
    bs = np.array([stats(rng.integers(0, nb, nb)) for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    wins = int((rb.flagged / rb.n_img < ra.flagged / ra.n_img).sum())
    return dict(calib=calib, alpha=alpha, d_load=full[0], d_load_lo=lo[0], d_load_hi=hi[0], d_miss=full[1],
                d_miss_lo=lo[1], d_miss_hi=hi[1], boards_B_flags_fewer=wins, n_boards=nb)


# ---------------------------------------------------------------- calibration-free, matched-miss comparison
def matched_miss(df, gt, imgs_per_board, scores, pol_a, pol_b, alpha, n_boot=2000, seed=0):
    """Both policies are thresholded to the SAME pooled miss rate alpha (threshold taken on the LOBO test scores
    themselves, identical treatment for both) => isolates ranking quality from calibration noise.
    Bootstrap resamples boards (multiplicity weights)."""
    boards = sorted(df.board.unique()); bidx = {b: i for i, b in enumerate(boards)}
    img = np.array([imgs_per_board[b] for b in boards], float)
    box_b = df.board.map(bidx).values
    gt_b = gt.board.map(bidx).values
    out = {}
    for pol in (pol_a, pol_b):
        sc = scores[pol]; s_gt = gt_scores(sc, gt)
        o = np.argsort(-sc); out[pol] = (sc[o], box_b[o], s_gt)
    rng = np.random.default_rng(seed)

    def load(pol, m):
        sc_sorted, bsorted, s_gt = out[pol]
        w_gt = m[gt_b]; order = np.argsort(s_gt); sg, wg = s_gt[order], w_gt[order]
        k = np.searchsorted(np.cumsum(wg), alpha * wg.sum(), side='right')     # misses allowed
        tau = -np.inf if k < 1 else sg[k - 1]
        if tau < 0:
            return np.inf
        flagged = (m[bsorted] * (sc_sorted >= tau)).sum()
        return flagged / (m * img).sum()
    m1 = np.ones(len(boards))
    la, lb = load(pol_a, m1), load(pol_b, m1)
    d = []
    for _ in range(n_boot):
        m = np.bincount(rng.integers(0, len(boards), len(boards)), minlength=len(boards)).astype(float)
        x, y = load(pol_a, m), load(pol_b, m)
        if np.isfinite(x) and np.isfinite(y):
            d.append(y - x)
    lo, hi = np.percentile(d, [2.5, 97.5]) if d else (np.nan, np.nan)
    return dict(alpha=alpha, load_A=la, load_B=lb, d_load=lb - la, lo=lo, hi=hi)
