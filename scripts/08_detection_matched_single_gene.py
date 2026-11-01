import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR
import os, io, sys, json, gc, itertools, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")


BASE = DATA_DIR
OUT = RESULTS_DIR
MIN = 30
N_MATCH = 500
TOL = 2.0
SEED = 42
TARGETS = ["XYLT1", "CHST11", "DSEL", "GALNT13"]
PRESPEC = ["XYLT1", "GRM5", "CACNA1B", "GABRB2"]


def perm_p(d):
    d = np.asarray(d, float)
    obs = d.mean()
    a = np.array([(d * np.array(s)).mean()
                  for s in itertools.product([1, -1], repeat=len(d))])
    return obs, float((np.abs(a) >= abs(obs)).mean())


def main():
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


    Xr = sp.csr_matrix(A.X)
    det_all = np.asarray((Xr > 0).sum(0)).ravel() / A.n_obs * 100
    gsm = A.obs.gsm.values
    pred = A.obs.pred.values
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    Xn = sp.csr_matrix(A.X)


    def delta_of(j):
        v = np.asarray(Xn[:, j].todense()).ravel()
        m = (pd.DataFrame({"v": v, "gsm": gsm, "pred": pred})
             .groupby(["gsm", "pred"], observed=True).v.mean().unstack())
        return (m["aneuploid"] - m["diploid"]).dropna()


    rng = np.random.default_rng(SEED)
    rows, res = [], {}
    for g in TARGETS:
        if g not in A.var_names:
            print(f"  {g}: not in matrix"); continue
        j = A.var_names.get_loc(g)
        d = delta_of(j)
        obs, p = perm_p(d.values)
        dg = det_all[j]
        pool = np.where((np.abs(det_all - dg) <= TOL) & (np.arange(len(det_all)) != j))[0]
        if len(pool) < 50:
            print(f"  {g}: only {len(pool)} matched genes, widening tolerance", flush=True)
            pool = np.where((np.abs(det_all - dg) <= TOL * 3)
                            & (np.arange(len(det_all)) != j))[0]
        pick = rng.choice(pool, size=min(N_MATCH, len(pool)), replace=False)
        null = np.array([delta_of(k).mean() for k in pick])
        b = int((null >= obs).sum())
        p_match = (b + 1) / (len(null) + 1)
        rows.append(dict(gene=g, detection_pct=round(float(dg), 2),
                         delta_log=round(float(obs), 4), p_perm=round(float(p), 5),
                         positive=f"{int((d > 0).sum())}/{len(d)}",
                         n_matched_genes=int(len(null)),
                         matched_pool_size=int(len(pool)),
                         null_median=round(float(np.median(null)), 4),
                         null_97_5=round(float(np.percentile(null, 97.5)), 4),
                         p_detection_matched=round(float(p_match), 5)))
        print(f"  {g:<9} detection={dg:5.2f}%  delta={obs:+.4f}  perm p={p:.5f}  "
              f"matched-null p={p_match:.5f}  (null median {np.median(null):+.4f}, "
              f"97.5th {np.percentile(null, 97.5):+.4f}, {len(null)} genes)", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "targets", "detection_matched_single_gene.csv"), index=False)


    te = pd.read_csv(os.path.join(OUT, "targets", "target_enrichment.csv")).set_index("gene")
    fam = [g for g in PRESPEC if g in te.index]
    ps = te.loc[fam, "p_perm"].values
    holm = multipletests(ps, method="holm")[1]
    print("\npre-specified family of four candidate targets, Holm:", flush=True)
    for g, p0, p1 in zip(fam, ps, holm):
        print(f"  {g:<9} raw p={p0:.4f}  Holm p={p1:.4f}  "
              f"{'survives' if p1 < 0.05 else 'does not survive'}", flush=True)
    res["prespecified_family"] = {g: dict(raw_p=float(p0), holm_p=round(float(p1), 5),
                                          survives=bool(p1 < 0.05))
                                  for g, p0, p1 in zip(fam, ps, holm)}
    gg = pd.read_csv(os.path.join(OUT, "targets", "enzyme_compartment_contrast.csv")).set_index("gene")
    res["post_hoc_28_gene_family"] = {
        g: dict(holm_p=float(gg.loc[g, "p_holm"]),
                survives=bool(gg.loc[g, "p_holm"] < 0.05))
        for g in TARGETS if g in gg.index}
    res["matched_null"] = {r["gene"]: r for r in rows}
    io.open(os.path.join(OUT, "targets", "detection_matched_single_gene.json"), "w", encoding="utf-8").write(
        json.dumps(res, indent=1))
    print(f"\n[OK] two files written to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
