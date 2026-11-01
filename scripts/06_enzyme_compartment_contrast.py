import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR, PACKAGE_DATA
import os, io, sys, json, gc, itertools, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import decoupler as dc
import gseapy
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
BASE = DATA_DIR
OUT = RESULTS_DIR
TRACT = os.path.join(PACKAGE_DATA, "target_tractability.csv")
MIN = 30
NEG = "Translation (GO:0006412)"
N_NULL = 500
SEED = 42


def perm_p(d):
    d = np.asarray(d, float)
    obs = d.mean()
    n = len(d)
    a = np.array([(d * np.array(s)).mean() for s in itertools.product([1, -1], repeat=n)])
    return obs, float((np.abs(a) >= abs(obs)).mean()), 2 / len(a)


def main():
    tract = pd.read_csv(TRACT)
    gag = sorted(tract.loc[tract.line == 1, "gene"].dropna().unique())
    print(f"glycosylation / GAG-biosynthesis genes nominated: {len(gag)}", flush=True)
    C = ad.read_h5ad(f"{BASE}/gbm_snrna_counts.h5ad")
    S = ad.read_h5ad(f"{BASE}/gbm_snrna_states.h5ad", backed="r")
    for c in ["gsm", "neftel_state"]:
        C.obs[c] = S.obs[c].astype(str).values
    ic = pd.read_csv(os.path.join(OUT, "cnv", "infercnvpy_cell_level_calls.csv")).set_index("cell")
    im = pd.read_csv(os.path.join(OUT, "cnv", "infercnvpy_sample_metadata.csv"))
    tier1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"])
    C.obs["pred"] = ic.icnv_pred.reindex(C.obs_names).values
    sel = (C.obs.gsm.isin(tier1) & (C.obs.neftel_state == "OPC")
           & C.obs.pred.isin(["aneuploid", "diploid"])).values
    A = C[sel].copy()
    del C
    gc.collect()
    nn = A.obs.groupby(["gsm", "pred"], observed=True).size()
    keep = {(g, p) for (g, p), v in nn.items() if v >= MIN}
    gsms = sorted({g for g, _ in keep
                   if (g, "aneuploid") in keep and (g, "diploid") in keep})
    A = A[[(g in gsms) and ((g, p) in keep)
           for g, p in zip(A.obs.gsm, A.obs.pred)]].copy()
    print(f"analysis set: {A.n_obs} nuclei, {len(gsms)} samples", flush=True)
    present = [g for g in gag if g in A.var_names]
    Xr = sp.csr_matrix(A.X)
    det = {}
    an = (A.obs.pred == "aneuploid").values
    for g in present:
        col = np.asarray(Xr[:, A.var_names.get_loc(g)].todense()).ravel()
        det[g] = (100 * (col > 0).mean(), 100 * (col[an] > 0).mean(),
                  100 * (col[~an] > 0).mean())
    print(f"genes present in the matrix: {len(present)} of {len(gag)}", flush=True)
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    rows = []
    for g in present:
        v = np.asarray(A[:, g].X.todense()).ravel()
        df = pd.DataFrame({"v": v, "gsm": A.obs.gsm.values, "pred": A.obs.pred.values})
        m = df.groupby(["gsm", "pred"], observed=True).v.mean().unstack()
        d = (m["aneuploid"] - m["diploid"]).dropna()
        o, p, mn = perm_p(d.values)
        rows.append(dict(gene=g, n_samples=int(len(d)), delta_log=round(float(o), 4),
                         p_perm=float(p), min_p=float(mn),
                         positive=f"{int((d > 0).sum())}/{len(d)}",
                         detection_pct=round(det[g][0], 2),
                         detection_aneuploid_pct=round(det[g][1], 2),
                         detection_diploid_pct=round(det[g][2], 2),
                         interpretable=det[g][0] >= 5.0))
    G = pd.DataFrame(rows).sort_values("delta_log", ascending=False)
    G["p_holm"] = multipletests(G.p_perm, method="holm")[1].round(5)
    G["p_perm"] = G.p_perm.round(5)
    G.to_csv(os.path.join(OUT, "targets", "enzyme_compartment_contrast.csv"), index=False)
    sig = G[(G.p_holm < 0.05) & G.interpretable]
    print(f"\ngenes with Holm p < 0.05 and detection >= 5%: {len(sig)} of {len(G)}",
          flush=True)
    print(G.head(12)[["gene", "delta_log", "p_perm", "p_holm", "positive",
                      "detection_pct", "interpretable"]].to_string(index=False), flush=True)
    gl = gseapy.get_library("GO_Biological_Process_2023")
    neg = sorted({x.split(",")[0] for x in gl[NEG]} & set(A.var_names))
    net = pd.concat([pd.DataFrame({"source": "GAG_biosynthesis", "target": present}),
                     pd.DataFrame({"source": "NEG_CONTROL_translation", "target": neg})])
    dc.mt.aucell(A, net=net, verbose=False)
    s = A.obsm["score_aucell"].copy()
    s["gsm"] = A.obs.gsm.values
    s["pred"] = A.obs.pred.values
    g2 = s.groupby(["gsm", "pred"], observed=True).mean()
    delta = (g2.xs("aneuploid", level="pred") - g2.xs("diploid", level="pred")).dropna()
    delta.to_csv(os.path.join(OUT, "targets", "enzyme_set_per_sample.csv"))
    res = {}
    print("\nset level (AUCell, full-transcriptome background):", flush=True)
    for c in delta.columns:
        o, p, mn = perm_p(delta[c].values)
        res[c] = dict(delta=round(float(o), 6), p_perm=round(float(p), 6),
                      min_p=float(mn), positive=int((delta[c] > 0).sum()),
                      n=int(len(delta)))
        print(f"  {c:<26} delta={o:+.5f}  p={p:.5f}  "
              f"{int((delta[c] > 0).sum())}/{len(delta)}", flush=True)
    print(f"\ndetection-matched null, {N_NULL} sets:", flush=True)
    d_all = np.asarray((sp.csr_matrix(A.layers["counts"]) > 0).sum(0)).ravel() / A.n_obs \
        if "counts" in A.layers else None
    if d_all is None:
        d_all = np.asarray((A.X > 0).sum(0)).ravel() / A.n_obs
    order = np.argsort(d_all)
    rank = {A.var_names[i]: k for k, i in enumerate(order)}
    pool = [A.var_names[i] for i in order]
    rng = np.random.default_rng(SEED)
    tgt = [rank[g] for g in present]
    nulls = []
    for i in range(N_NULL):
        pick = [pool[min(len(pool) - 1, max(0, int(rng.normal(r, 60))))] for r in tgt]
        nulls.append(pd.DataFrame({"source": f"R{i}", "target": sorted(set(pick))}))
    dc.mt.aucell(A, net=pd.concat(nulls), verbose=False)
    ns = A.obsm["score_aucell"].copy()
    ns["gsm"] = A.obs.gsm.values
    ns["pred"] = A.obs.pred.values
    ng = ns.groupby(["gsm", "pred"], observed=True).mean()
    nd = (ng.xs("aneuploid", level="pred") - ng.xs("diploid", level="pred")).dropna()
    null_means = nd.mean(0).values
    obs = res["GAG_biosynthesis"]["delta"]
    b = int((null_means >= obs).sum())
    res["detection_matched_null"] = dict(
        n_sets=N_NULL, null_median=round(float(np.median(null_means)), 6),
        null_97_5=round(float(np.percentile(null_means, 97.5)), 6),
        observed=obs, p_add_one=round((b + 1) / (N_NULL + 1), 6))
    print(f"  null median={np.median(null_means):+.5f}  "
          f"97.5th={np.percentile(null_means, 97.5):+.5f}  observed={obs:+.5f}  "
          f"p={(b + 1) / (N_NULL + 1):.5f}", flush=True)
    res["n_samples"] = len(gsms)
    res["n_nuclei"] = int(A.n_obs)
    res["genes_nominated"] = len(gag)
    res["genes_present"] = len(present)
    res["genes_significant_and_interpretable"] = int(len(sig))
    io.open(os.path.join(OUT, "targets", "enzyme_set_summary.json"), "w", encoding="utf-8").write(
        json.dumps(res, indent=1))
    print(f"\n[OK] three files written to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
