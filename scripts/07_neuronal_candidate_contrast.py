import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR
import itertools, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad, scanpy as sc
warnings.filterwarnings("ignore")

MIN = 30
TARGETS = [("GRM5", "line 2"), ("XYLT1", "line 1"), ("CACNA1B", "line 2"), ("GABRB2", "line 2")]


def perm_p(d):
    d = np.asarray(d, float); obs = d.mean(); n = len(d)
    a = np.array([(d * np.array(s)).mean() for s in itertools.product([1, -1], repeat=n)])
    return obs, float((np.abs(a) >= abs(obs)).mean())


def main():
    C = ad.read_h5ad(f"{DATA_DIR}/gbm_snrna_counts.h5ad")
    S = ad.read_h5ad(f"{DATA_DIR}/gbm_snrna_states.h5ad", backed="r")
    for c in ["gsm", "patient", "progression", "neftel_state"]:
        C.obs[c] = S.obs[c].astype(str).values
    ic = pd.read_csv(f"{RESULTS_DIR}/cnv/infercnvpy_cell_level_calls.csv").set_index("cell")
    im = pd.read_csv(f"{RESULTS_DIR}/cnv/infercnvpy_sample_metadata.csv")
    tier1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"])
    C.obs["pred"] = ic.icnv_pred.reindex(C.obs_names).values
    C.obs["tier1"] = C.obs.gsm.isin(tier1)

    sel = (C.obs.tier1 & (C.obs.neftel_state == "OPC")
           & C.obs.pred.isin(["aneuploid", "diploid"])).values
    A = C[sel].copy()
    nn = A.obs.groupby(["gsm", "pred"], observed=True).size()
    keep = {(g, p_) for (g, p_), v in nn.items() if v >= MIN}
    gs = sorted({g for g, _ in keep if (g, "aneuploid") in keep and (g, "diploid") in keep})
    A = A[[(g in gs) and ((g, p_) in keep) for g, p_ in zip(A.obs.gsm, A.obs.pred)]].copy()
    Xr = sp.csr_matrix(A.X)
    det_all = {}
    for g_, line in TARGETS:
        if g_ not in A.var_names: continue
        j = A.var_names.get_loc(g_)
        col = Xr[:, j].toarray().ravel()
        det_all[g_] = dict(detection_pct=round(100 * (col > 0).mean(), 2),
                           detection_aneuploid_pct=round(100 * (col[(A.obs.pred == "aneuploid").values] > 0).mean(), 2),
                           detection_diploid_pct=round(100 * (col[(A.obs.pred == "diploid").values] > 0).mean(), 2))
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)
    trow = []
    for g_, line in TARGETS:
        if g_ not in A.var_names: continue
        v = np.asarray(A[:, g_].X.todense()).ravel()
        df = pd.DataFrame({"v": v, "gsm": A.obs.gsm.values, "pred": A.obs.pred.values})
        m = df.groupby(["gsm", "pred"], observed=True).v.mean().unstack()
        d2 = (m.aneuploid - m.diploid).dropna()
        o2, p2 = perm_p(d2.values)
        trow.append(dict(gene=g_, drug_line=line, n_samples=len(d2),
                         delta_log=round(float(o2), 4), p_perm=round(float(p2), 4),
                         positive=f"{int((d2>0).sum())}/{len(d2)}",
                         reliable=det_all[g_]["detection_pct"] >= 5.0, **det_all[g_]))
        print(f"{g_:<9} delta={o2:+.4f}  p={p2:.4f}  {int((d2>0).sum())}/{len(d2)}  "
              f"detection={det_all[g_]['detection_pct']}%", flush=True)
    pd.DataFrame(trow).to_csv(f"{RESULTS_DIR}/targets/target_enrichment.csv", index=False)


if __name__ == "__main__":
    main()
