from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import warnings as _warnings

_warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
_warnings.filterwarnings("ignore", message=".*observed=False is deprecated.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from config import DATA_DIR, RESULTS_DIR, PACKAGE_DATA
except Exception as exc:
    raise SystemExit(f"Cannot import config.py. Run from the repository root. Error: {exc}")

DATA_DIR = Path(DATA_DIR)
RESULTS_DIR = Path(RESULTS_DIR)
PACKAGE_DATA = Path(PACKAGE_DATA)
ROBUST_DIR = RESULTS_DIR / "robustness"
CACHE_DIR = RESULTS_DIR / "cache"

COUNTS_H5AD = Path(os.environ.get("GBM_COUNTS_H5AD", DATA_DIR / "gbm_snrna_counts.h5ad"))
STATES_H5AD = Path(os.environ.get("GBM_STATES_H5AD", DATA_DIR / "gbm_snrna_states.h5ad"))
CNV_CALLS = RESULTS_DIR / "cnv" / "infercnvpy_cell_level_calls.csv"
CNV_META = RESULTS_DIR / "cnv" / "infercnvpy_sample_metadata.csv"
GENE_POS = RESULTS_DIR / "cnv" / "gene_positions.csv"
COHORT_CSV = PACKAGE_DATA / "snrna_cohort.csv"

MIN_CELLS = 30
MIN_REF = 20
REF_CELLTYPES = ["Microglia_CG", "Oligodendrocytes"]
GENES3 = ["CHST11", "XYLT1", "DSEL"]
SKIP_EXIT_CODE = 3


def outdir(name: str) -> Path:
    p = ROBUST_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def skip(msg: str):
    print(f"[SKIPPED] {msg}", flush=True)
    sys.exit(SKIP_EXIT_CODE)


def require(paths, what="this analysis"):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        skip(f"{what} needs: " + ", ".join(missing))


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def signflip(x, n_mc: int = 200_000, seed: int = 42):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, "none"
    obs = float(x.mean())
    if n <= 22:
        null = np.empty(2 ** n)
        step = 1 << 16
        for s in range(0, 2 ** n, step):
            e = min(s + step, 2 ** n)
            signs = ((np.arange(s, e)[:, None] >> np.arange(n)) & 1) * 2 - 1
            null[s:e] = (signs * x).mean(axis=1)
        p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
        return obs, p, 2.0 / 2 ** n, "exact"
    rng = np.random.default_rng(seed)
    b = 0
    done = 0
    while done < n_mc:
        k = min(20_000, n_mc - done)
        null = (rng.choice([-1.0, 1.0], size=(k, n)) * x).mean(axis=1)
        b += int((np.abs(null) >= abs(obs) - 1e-12).sum())
        done += k
    return obs, (b + 1) / (n_mc + 1), 1.0 / (n_mc + 1), "monte_carlo"


def sign_test(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) == 0:
        return np.nan, 0, 0
    k = int((x > 0).sum())
    return float(stats.binomtest(k, len(x), 0.5).pvalue), k, len(x)


def wilcoxon_p(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.allclose(x, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def bootstrap_ci(x, n_boot=100_000, seed=42, alpha=0.05, stat=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    done = 0
    while done < n_boot:
        k = min(10_000, n_boot - done)
        idx = rng.integers(0, len(x), size=(k, len(x)))
        vals.append(stat(x[idx], axis=1))
        done += k
    v = np.concatenate(vals)
    lo, hi = np.quantile(v, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm(p):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def bh(p):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def describe_effect(x, prefix="", n_boot=20_000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m, p, pmin, branch = signflip(x)
    ps, k, n = sign_test(x)
    lo, hi = bootstrap_ci(x, n_boot=n_boot)
    return {f"{prefix}n": int(len(x)), f"{prefix}mean": m,
            f"{prefix}median": float(np.median(x)) if len(x) else np.nan,
            f"{prefix}positive": int((x > 0).sum()),
            f"{prefix}p_signflip": p, f"{prefix}min_p": pmin,
            f"{prefix}p_sign_test": ps, f"{prefix}p_wilcoxon": wilcoxon_p(x),
            f"{prefix}ci_low": lo, f"{prefix}ci_high": hi}


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3g}"


def load_cohort():
    c = pd.read_csv(COHORT_CSV)
    return c[["gsm", "patient_pair", "progression"]].drop_duplicates("gsm")


def load_cnv_calls():
    require([CNV_CALLS, CNV_META], "CNV calls")
    ic = pd.read_csv(CNV_CALLS).set_index("cell")
    im = pd.read_csv(CNV_META)
    tier1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"].astype(str))
    return ic, im, tier1


def load_states_obs(columns=None):
    import anndata as ad
    require([STATES_H5AD], "the processed discovery H5AD")
    S = ad.read_h5ad(STATES_H5AD, backed="r")
    obs = S.obs.copy() if columns is None else S.obs[[c for c in columns if c in S.obs]].copy()
    var = pd.Index(np.asarray(S.var_names))
    S.file.close()
    for c in ["gsm", "patient", "progression", "neftel_state", "cluster_call", "leiden"]:
        if c in obs:
            obs[c] = obs[c].astype(str)
    if "malignant" in obs:
        obs["malignant"] = obs["malignant"].astype(str).str.lower().isin(["true", "1"])
    return obs, var


def analysis_cells(obs: pd.DataFrame, pred_col="pred", state="OPC", min_cells=MIN_CELLS,
                   samples_allowed=None):
    base = (obs["neftel_state"].astype(str) == state) & obs[pred_col].isin(["aneuploid", "diploid"])
    if samples_allowed is not None:
        base &= obs["gsm"].astype(str).isin(set(samples_allowed))
    nn = obs[base].groupby(["gsm", pred_col], observed=True).size()
    keep = {(g, p) for (g, p), v in nn.items() if v >= min_cells}
    gsms = sorted({g for g, _ in keep if (g, "aneuploid") in keep and (g, "diploid") in keep})
    pair_ok = pd.Series(list(zip(obs["gsm"].astype(str), obs[pred_col]))).isin(keep).to_numpy()
    mask = base.to_numpy() & obs["gsm"].astype(str).isin(gsms).to_numpy() & pair_ok
    return mask, gsms


def _backed(path):
    import anndata as ad
    return ad.read_h5ad(path, backed="r")


def read_rows(path, rows, chunk=8000):
    import scipy.sparse as sp
    A = _backed(path)
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    srt = rows[order]
    blocks = []
    for s in range(0, len(srt), chunk):
        blk = A.X[srt[s:s + chunk]]
        blocks.append(sp.csr_matrix(blk))
    var = pd.Index(np.asarray(A.var_names))
    A.file.close()
    X = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, len(var)))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return X[inv], var


def read_gene_columns(path, rows, genes, chunk=10000, with_totals=False):
    import scipy.sparse as sp
    A = _backed(path)
    var = pd.Index(np.asarray(A.var_names))
    genes = [g for g in genes if g in var]
    cols = var.get_indexer(genes)
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    srt = rows[order]
    out = np.zeros((len(rows), len(cols)), dtype=np.float32)
    tot = np.zeros(len(rows), dtype=np.float64)
    for s in range(0, len(srt), chunk):
        blk = sp.csr_matrix(A.X[srt[s:s + chunk]])
        out[s:s + len(blk.indptr) - 1] = blk[:, cols].toarray()
        if with_totals:
            tot[s:s + blk.shape[0]] = np.asarray(blk.sum(1)).ravel()
    A.file.close()
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out = out[inv]
    tot = tot[inv]
    return (out, genes, tot) if with_totals else (out, genes)


def lognorm_counts(counts, totals):
    counts = np.asarray(counts, float)
    t = np.asarray(totals, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = counts * (1e4 / np.where(t > 0, t, np.nan))[:, None] if counts.ndim == 2 \
            else counts * 1e4 / np.where(t > 0, t, np.nan)
    return np.log1p(np.nan_to_num(v))


def per_sample_delta(values, gsm, pred, hi="aneuploid", lo="diploid"):
    v = np.asarray(values, float)
    if v.ndim == 1:
        v = v[:, None]
    df = pd.DataFrame(v)
    df["gsm"] = np.asarray(gsm).astype(str)
    df["pred"] = np.asarray(pred).astype(str)
    m = df.groupby(["gsm", "pred"], observed=True).mean()
    try:
        d = m.xs(hi, level="pred") - m.xs(lo, level="pred")
    except KeyError:
        return pd.DataFrame()
    return d.dropna(how="all")


def depth_adjusted_coef(y, an, logdepth, gsm):
    out = {}
    gsm = np.asarray(gsm).astype(str)
    for g in pd.unique(gsm):
        i = gsm == g
        if i.sum() < 5 or len(np.unique(an[i])) < 2:
            continue
        D = np.column_stack([np.ones(i.sum()), an[i], logdepth[i]])
        out[g] = float(np.linalg.lstsq(D, y[i], rcond=None)[0][1])
    return pd.Series(out)


def score_sets_fixed(X, var_names, gene_sets, mu, n_bins=30, n_ctrl=100, seed=0):
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    X = X.tocsc() if sp.issparse(X) else np.asarray(X)
    n_cells, n_genes = X.shape
    idx = {g: i for i, g in enumerate(var_names)}
    mu = np.asarray(mu, float)
    order = np.argsort(mu, kind="mergesort")
    bin_of = np.empty(n_genes, dtype=np.int32)
    edges = np.linspace(0, n_genes, n_bins + 1).astype(int)
    for b in range(n_bins):
        bin_of[order[edges[b]:edges[b + 1]]] = b
    members = [order[edges[b]:edges[b + 1]] for b in range(n_bins)]

    def col_mean(cols, w=None):
        if len(cols) == 0:
            return np.zeros(n_cells)
        sub = X[:, cols]
        if w is None:
            s = np.asarray(sub.sum(axis=1)).ravel() if sp.issparse(sub) else sub.sum(axis=1)
            return s / len(cols)
        w = np.asarray(w, float)
        s = np.asarray(sub @ w).ravel()
        return s / w.sum()

    scores = {}
    for name, genes in gene_sets.items():
        cols = np.array([idx[g] for g in genes if g in idx], dtype=int)
        if len(cols) == 0:
            scores[name] = np.full(n_cells, np.nan)
            continue
        cc = np.zeros(n_genes, dtype=np.int64)
        for c in cols:
            pool = members[bin_of[c]]
            pick = rng.choice(pool, size=min(n_ctrl, len(pool)), replace=False)
            np.add.at(cc, pick, 1)
        cw = np.nonzero(cc)[0]
        w = cc[cw].astype(float)
        scores[name] = (col_mean(cols) - mu[cols].mean()) - (col_mean(cw, w) - np.average(mu[cw], weights=w))
    return scores


def pkg_version(name):
    from importlib import metadata
    try:
        return metadata.version(name)
    except Exception:
        return None


def versions():
    v = {"python": platform.python_version()}
    for m in ["numpy", "pandas", "scipy", "anndata", "scanpy", "infercnvpy", "decoupler",
              "statsmodels", "matplotlib"]:
        v[m] = pkg_version(m)
    return v


def rel_path(p) -> str:
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def write_json(path: Path, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")


def provenance(out: Path, script: str, args=None, extra=None):
    write_json(out / "provenance.json", {
        "script": rel_path(script),
        "args": vars(args) if args is not None else None,
        "data_dir": rel_path(DATA_DIR), "results_dir": rel_path(RESULTS_DIR),
        "versions": versions(), **(extra or {})})


def load_analysis_set_matrix(state="OPC", pred: pd.Series | None = None, allowed=None):
    import anndata as ad
    import scipy.sparse as sp
    require([COUNTS_H5AD, STATES_H5AD, CNV_CALLS, CNV_META], "the compartment analysis")
    obs, _ = load_states_obs(["gsm", "patient", "progression", "neftel_state", "cluster_call"])
    ic, im, tier1 = load_cnv_calls()
    obs["pred"] = (pred if pred is not None else ic.icnv_pred).reindex(obs.index)
    mask, gsms = analysis_cells(obs, "pred", state=state,
                                samples_allowed=tier1 if allowed is None else allowed)
    C = ad.read_h5ad(COUNTS_H5AD, backed="r")
    row_of = pd.Series(np.arange(C.n_obs), index=C.obs_names)
    C.file.close()
    names = obs.index[mask]
    rows = row_of.reindex(names).to_numpy()
    if np.isnan(rows.astype(float)).any():
        raise SystemExit("analysis-set nuclei missing from the counts H5AD")
    X, var = read_rows(COUNTS_H5AD, rows.astype(int))
    tot = np.asarray(X.sum(1)).ravel().astype(float)
    Xn = sp.diags(1e4 / tot) @ X.astype(np.float64)
    Xn = sp.csr_matrix(Xn)
    Xn.data = np.log1p(Xn.data)
    o = obs.loc[names, ["gsm", "pred"]].copy()
    o = o.reset_index().rename(columns={"index": "cell"})
    if "cell" not in o:
        o = o.rename(columns={o.columns[0]: "cell"})
    o = o.merge(load_cohort(), on="gsm", how="left")
    det = np.asarray((X > 0).sum(0)).ravel() / X.shape[0] * 100
    return dict(X=X, Xn=Xn, var=var, obs=o, totals=tot, logdepth=np.log10(tot), det_pct=det, gsms=gsms)


def group_means(Xn, keys):
    import scipy.sparse as sp
    k = pd.Series(keys)
    codes, uniq = pd.factorize(k)
    G = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                      shape=(len(uniq), len(codes)))
    n = np.asarray(G.sum(1)).ravel()
    M = np.asarray((sp.diags(1 / n) @ G @ Xn).todense()) if sp.issparse(Xn) else (sp.diags(1 / n) @ G @ Xn)
    return pd.DataFrame(np.asarray(M), index=pd.Index(uniq))


def compartment_delta_all(Xn, gsm, pred, var):
    keys = [f"{g}||{p}" for g, p in zip(gsm, pred)]
    M = group_means(Xn, keys)
    M.columns = list(var)
    idx = pd.MultiIndex.from_tuples([tuple(i.split("||")) for i in M.index], names=["gsm", "pred"])
    M.index = idx
    return (M.xs("aneuploid", level="pred") - M.xs("diploid", level="pred")).dropna(how="all")


def depth_adjusted_all(Xn, gsm, pred, logdepth, var):
    import scipy.sparse as sp
    gsm = np.asarray(gsm).astype(str)
    an = (np.asarray(pred) == "aneuploid").astype(float)
    out = {}
    for g in pd.unique(gsm):
        i = np.where(gsm == g)[0]
        D = np.column_stack([np.ones(len(i)), an[i], logdepth[i]])
        DtD_inv = np.linalg.pinv(D.T @ D)
        DtY = np.asarray((sp.csr_matrix(D.T) @ Xn[i]).todense()) if sp.issparse(Xn) else D.T @ Xn[i]
        out[g] = (DtD_inv @ DtY)[1]
    return pd.DataFrame(out, index=list(var)).T


def matched_null_p(det_pct, j, stat_vec, obs, n_match=500, tol=2.0, rng=None):
    rng = rng or np.random.default_rng(42)
    ar = np.arange(len(det_pct))
    pool = np.where((np.abs(det_pct - det_pct[j]) <= tol) & (ar != j) & np.isfinite(stat_vec))[0]
    if len(pool) < 50:
        pool = np.where((np.abs(det_pct - det_pct[j]) <= tol * 3) & (ar != j) & np.isfinite(stat_vec))[0]
    if len(pool) == 0:
        return np.nan, np.nan, 0
    pick = rng.choice(pool, size=min(n_match, len(pool)), replace=False)
    null = stat_vec[pick]
    return (int((null >= obs).sum()) + 1) / (len(null) + 1), float(np.median(null)), len(pick)


def state_axes(force=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / "state_axes_malignant.csv.gz"
    if f.exists() and not force:
        return pd.read_csv(f, index_col=0)
    import anndata as ad
    import scanpy as sc
    require([STATES_H5AD], "state axes")
    log("computing state axes (loads gbm_snrna_states.h5ad into memory once; cached afterwards)")
    gs = json.load(open(PACKAGE_DATA / "metabolic_genesets.json", encoding="utf-8"))
    A = ad.read_h5ad(STATES_H5AD)
    vs = set(A.var_names)
    for k, genes in gs["neftel_metamodules"].items():
        sc.tl.score_genes(A, gene_list=[g for g in genes if g in vs], score_name=f"sg_{k}",
                          ctrl_size=50, n_bins=25, random_state=0, use_raw=False)
    o = A.obs
    o["axis_scanpy"] = (np.maximum(o.sg_OPC, np.maximum(o.sg_NPC1, o.sg_NPC2))
                        - np.maximum(o.sg_AC, np.maximum(o.sg_MES1, o.sg_MES2)))
    need = [f"sc_{k}" for k in ["OPC", "NPC1", "NPC2", "AC", "MES1", "MES2"]]
    if all(c in o for c in need):
        o["axis_neftel"] = (np.maximum(o.sc_OPC, np.maximum(o.sc_NPC1, o.sc_NPC2))
                            - np.maximum(o.sc_AC, np.maximum(o.sc_MES1, o.sc_MES2)))
    else:
        log("WARNING: sc_* columns missing; axis_neftel will be NaN")
        o["axis_neftel"] = np.nan
    mal = o.malignant.astype(str).str.lower().isin(["true", "1"])
    keep = ["gsm", "patient", "neftel_state", "axis_scanpy", "axis_neftel"] + \
           [c for c in o if c.startswith("sg_")]
    D = o.loc[mal, keep].copy()
    del A
    D.to_csv(f, compression="gzip")
    return D
