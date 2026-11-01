import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR
import json, gc, itertools, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

MIN = 30
GENES = ["CHST11", "XYLT1", "DSEL"]
PROTEOGLYCANS = ["CHST11", "CSPG4", "VCAN", "BCAN", "NCAN", "ACAN"]
STATES = ["OPC", "AC", "MES", "NPC"]


def perm_p(d):
    d = np.asarray(d, float)
    obs, n = d.mean(), len(d)
    if n < 3:
        return obs, np.nan, np.nan
    if n <= 22:
        a = np.array([(d * np.array(s)).mean()
                      for s in itertools.product([1, -1], repeat=n)])
        return obs, float((np.abs(a) >= abs(obs)).mean()), 2 / len(a)
    rng = np.random.default_rng(42)
    a = (rng.choice([1, -1], size=(200000, n)) * d).mean(1)
    return obs, (int((np.abs(a) >= abs(obs)).sum()) + 1) / 200001, 1 / 200001


def contrast(vals, group, split, lo, hi):
    m = (pd.DataFrame({"v": vals, "g": group, "s": split})
         .groupby(["g", "s"], observed=True).v.mean().unstack())
    if lo not in m.columns or hi not in m.columns:
        return None, None
    return m, (m[hi] - m[lo]).dropna()


def analysis_mask(obs, state):
    base = obs.tier1 & (obs.neftel_state == state) & obs.pred.isin(["aneuploid", "diploid"])
    nn = obs[base].groupby(["gsm", "pred"], observed=True).size()
    ok = {(g, p) for (g, p), v in nn.items() if v >= MIN}
    gsms = sorted({g for g, _ in ok if (g, "aneuploid") in ok and (g, "diploid") in ok})
    m = (base & obs.gsm.isin(gsms)).values
    return m & np.array([(g, p) in ok for g, p in zip(obs.gsm, obs.pred)]), gsms


def main():
    C = ad.read_h5ad(f"{DATA_DIR}/gbm_snrna_counts.h5ad")
    S = ad.read_h5ad(f"{DATA_DIR}/gbm_snrna_states.h5ad", backed="r")
    for c in ["gsm", "patient", "progression", "neftel_state"]:
        C.obs[c] = S.obs[c].astype(str).values
    ic = pd.read_csv(os.path.join(RESULTS_DIR, "cnv", "infercnvpy_cell_level_calls.csv")).set_index("cell")
    im = pd.read_csv(os.path.join(RESULTS_DIR, "cnv", "infercnvpy_sample_metadata.csv"))
    tier1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"])
    C.obs["pred"] = ic.icnv_pred.reindex(C.obs_names).values
    C.obs["tier1"] = C.obs.gsm.isin(tier1)

    genes = [g for g in dict.fromkeys(GENES + PROTEOGLYCANS) if g in C.var_names]
    A = C[(C.obs.neftel_state.isin(STATES) & C.obs.tier1).values].copy()
    del C
    gc.collect()
    col = {g: A.var_names.get_loc(g) for g in genes}
    raw = sp.csr_matrix(A.X)
    R = np.column_stack([np.asarray(raw[:, col[g]].todense()).ravel() for g in genes])
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    obs = A.obs
    Xs = sp.csr_matrix(A.X)
    X = np.column_stack([np.asarray(Xs[:, col[g]].todense()).ravel() for g in genes])
    del A, Xs, raw
    gc.collect()
    gi = {g: i for i, g in enumerate(genes)}

    per_rows, scope_rows = [], []
    for st in STATES:
        m, gsms = analysis_mask(obs, st)
        if len(gsms) < 4:
            continue
        for gene in [g for g in GENES if g in gi]:
            tab, d = contrast(X[m, gi[gene]], obs.gsm.values[m], obs.pred.values[m],
                              "diploid", "aneuploid")
            o, p, mn = perm_p(d.values)
            scope_rows.append(dict(state=st, gene=gene, n_samples=int(len(d)),
                                   n_nuclei=int(m.sum()), delta_log=round(float(o), 4),
                                   p_perm=round(float(p), 5), min_p=float(mn),
                                   positive=f"{int((d > 0).sum())}/{len(d)}"))
            if st == "OPC":
                t = tab.reset_index().rename(columns={"g": "gsm"})
                t["gene"] = gene
                per_rows.append(t)
            print(f"{st:<4} {gene:<8} n={len(d):2d}  delta={o:+.4f}  p={p:.5f}", flush=True)
    pd.concat(per_rows).to_csv(os.path.join(RESULTS_DIR, "targets", "per_tumor_expression.csv"), index=False)
    pd.DataFrame(scope_rows).to_csv(os.path.join(RESULTS_DIR, "targets", "cell_state_scope.csv"), index=False)

    m, gsms = analysis_mask(obs, "OPC")
    an = (obs.pred.values == "aneuploid") & m
    di = (obs.pred.values == "diploid") & m
    rows, deltas = [], {}
    for gene in [g for g in PROTEOGLYCANS if g in gi]:
        _, d = contrast(X[m, gi[gene]], obs.gsm.values[m], obs.pred.values[m],
                        "diploid", "aneuploid")
        o, p, _ = perm_p(d.values)
        deltas[gene] = d
        r = R[:, gi[gene]]
        rows.append(dict(gene=gene, n_samples=int(len(d)), delta_log=round(float(o), 4),
                         p_perm=round(float(p), 5), positive=f"{int((d > 0).sum())}/{len(d)}",
                         detection_pct=round(100 * float((r[m] > 0).mean()), 2),
                         detection_aneuploid_pct=round(100 * float((r[an] > 0).mean()), 2),
                         detection_diploid_pct=round(100 * float((r[di] > 0).mean()), 2)))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "targets", "cspg_pathway_consistency.csv"), index=False)
    ref = deltas["CHST11"]
    corr = {}
    for gene, d in deltas.items():
        if gene == "CHST11":
            continue
        j = pd.concat([ref, d], axis=1, join="inner").dropna()
        s = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])
        corr[gene] = dict(spearman_r=round(float(s.statistic), 3), p=round(float(s.pvalue), 4), n=int(len(j)))
    json.dump(dict(genes=list(deltas), n_samples=len(gsms),
                   per_sample_delta_vs_CHST11_spearman=corr),
              open(os.path.join(RESULTS_DIR, "targets", "cspg_pathway_consistency.json"), "w"), indent=1)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps(corr, indent=1), flush=True)


if __name__ == "__main__":
    main()
