import numpy as np
import scipy.sparse as sp


NONMALIGNANT = {
    "Macrophages":       ["CD14", "AIF1", "FCER1G", "FCGR3A", "TYROBP", "CSF1R"],
    "T_cells":           ["CD2", "CD3D", "CD3E", "CD3G"],
    "Oligodendrocytes":  ["MBP", "TF", "PLP1", "MAG", "MOG", "CLDN11"],
}
NONMALIGNANT_THRESHOLD = 4.0


META_ORDER = ["MES1", "MES2", "AC", "OPC", "NPC1", "NPC2", "G1S", "G2M"]


def load_metamodules(csv_path):
    import csv
    from collections import defaultdict
    out = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mod = (row.get("module") or row.get("meta_module") or "").strip()
            gen = (row.get("gene") or "").strip()
            if mod and gen:
                out[mod].append(gen)
    return {k: v for k, v in out.items()}


def _as_csc(X):
    return X.tocsc() if sp.issparse(X) and not sp.isspmatrix_csc(X) else X


def score_sets(X, var_names, gene_sets, n_bins=30, n_ctrl=100, seed=0,
               return_coverage=False):
    rng = np.random.default_rng(seed)
    X = _as_csc(X)
    n_cells, n_genes = X.shape
    idx = {g: i for i, g in enumerate(var_names)}


    if sp.issparse(X):
        mu = np.asarray(X.mean(axis=0)).ravel()
    else:
        mu = X.mean(axis=0)


    order = np.argsort(mu, kind="mergesort")
    bin_of = np.empty(n_genes, dtype=np.int32)
    edges = np.linspace(0, n_genes, n_bins + 1).astype(int)
    for b in range(n_bins):
        bin_of[order[edges[b]:edges[b + 1]]] = b
    members = [order[edges[b]:edges[b + 1]] for b in range(n_bins)]


    def _col_mean(cols, weights=None):
        if len(cols) == 0:
            return np.zeros(n_cells)
        sub = X[:, cols]
        if weights is None:
            s = np.asarray(sub.sum(axis=1)).ravel() if sp.issparse(sub) else sub.sum(axis=1)
            return s / len(cols)
        w = np.asarray(weights, dtype=float)
        s = (sub @ w) if not sp.issparse(sub) else np.asarray(sub @ w).ravel()
        return s / w.sum()


    scores, coverage = {}, {}
    for name, genes in gene_sets.items():
        cols = np.array([idx[g] for g in genes if g in idx], dtype=int)
        coverage[name] = (len(cols), len(genes))
        if len(cols) == 0:
            scores[name] = np.full(n_cells, np.nan)
            continue


        ctrl_counts = np.zeros(n_genes, dtype=np.int64)
        for c in cols:
            pool = members[bin_of[c]]
            pick = rng.choice(pool, size=min(n_ctrl, len(pool)), replace=False)
            np.add.at(ctrl_counts, pick, 1)
        ctrl_cols = np.nonzero(ctrl_counts)[0]
        ctrl_w = ctrl_counts[ctrl_cols].astype(float)


        set_term  = _col_mean(cols) - mu[cols].mean()
        ctrl_term = _col_mean(ctrl_cols, ctrl_w) - np.average(mu[ctrl_cols], weights=ctrl_w)
        scores[name] = set_term - ctrl_term


    return (scores, coverage) if return_coverage else scores


def neftel_plane(sc, use_literal_lower=False):
    sc_ac  = np.asarray(sc["AC"])
    sc_opc = np.asarray(sc["OPC"])
    sc_mes = np.maximum(np.asarray(sc["MES1"]), np.asarray(sc["MES2"]))
    sc_npc = np.maximum(np.asarray(sc["NPC1"]), np.asarray(sc["NPC2"]))


    y = np.maximum(sc_opc, sc_npc) - np.maximum(sc_ac, sc_mes)
    up = y > 0


    d_up  = sc_opc - sc_npc
    d_dn  = sc_ac - sc_mes
    x = np.empty_like(y)
    x[up]  = np.sign(d_up[up]) * np.log2(np.abs(d_up[up]) + 1.0)
    if use_literal_lower:
        with np.errstate(divide="ignore"):
            x[~up] = np.sign(d_dn[~up]) * np.log2(np.abs(d_dn[~up]))
    else:
        x[~up] = np.sign(d_dn[~up]) * np.log2(np.abs(d_dn[~up]) + 1.0)


    state = np.where(up,
                     np.where(d_up > 0, "OPC", "NPC"),
                     np.where(d_dn > 0, "AC", "MES"))
    return dict(x=x, y=y, state=state,
                sc_ac=sc_ac, sc_mes=sc_mes, sc_opc=sc_opc, sc_npc=sc_npc)


def classify_nonmalignant(sc, threshold=NONMALIGNANT_THRESHOLD):
    keys = [k for k in NONMALIGNANT if k in sc]
    M = np.vstack([np.asarray(sc[k]) for k in keys])
    over = M > threshold
    n_over = over.sum(axis=0)
    lab = np.full(M.shape[1], "unassigned", dtype=object)
    lab[n_over > 1] = "ambiguous"
    one = n_over == 1
    if one.any():
        lab[one] = np.array(keys, dtype=object)[over[:, one].argmax(axis=0)]
    import collections
    return lab, dict(collections.Counter(lab))
