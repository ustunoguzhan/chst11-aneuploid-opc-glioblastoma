import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR
import os, json, time, warnings
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import infercnvpy as cnv
warnings.filterwarnings("ignore")


REF = ["Microglia_CG", "Oligodendrocytes"]
MINREF, PCT, NJOBS = 20, 95, 1


def main():
    gp = pd.read_csv(f"{RESULTS_DIR}/cnv/gene_positions.csv", index_col=0)
    C = ad.read_h5ad(f"{DATA_DIR}/gbm_snrna_counts.h5ad")
    S = ad.read_h5ad(f"{DATA_DIR}/gbm_snrna_states.h5ad", backed="r")
    for c in ["gsm", "patient", "progression", "cluster_call", "neftel_state"]:
        C.obs[c] = S.obs[c].astype(str).values
    gsms = sorted(C.obs.gsm.unique())
    print(f"sample: {len(gsms)}  cell: {C.n_obs}", flush=True)


    call, meta = [], []
    t0 = time.time()
    for i, g in enumerate(gsms, 1):
        A = C[(C.obs.gsm == g).values].copy()
        A.var = A.var.join(gp[["chromosome", "start", "end"]])
        A = A[:, A.var.chromosome.notna()].copy()
        sc.pp.normalize_total(A, target_sum=1e4)
        sc.pp.log1p(A)
        A.obs["ref"] = np.where(A.obs.cluster_call.isin(REF), "reference", "test")
        nref = int((A.obs.ref == "reference").sum())
        tier1 = nref >= MINREF
        try:
            if tier1:
                cnv.tl.infercnv(A, reference_key="ref", reference_cat=["reference"],
                                window_size=100, step=10,
                                exclude_chromosomes=("chrX", "chrY"), n_jobs=NJOBS)
            else:
                cnv.tl.infercnv(A, reference_key=None, window_size=100, step=10,
                                exclude_chromosomes=("chrX", "chrY"), n_jobs=NJOBS)
            X = A.obsm["X_cnv"]
            score = np.asarray(np.abs(X).mean(1)).ravel()
            baseline = score[(A.obs.ref == "reference").values] if tier1 else score
            threshold = float(np.percentile(baseline, PCT))
            pred = np.where(score > threshold, "aneuploid", "diploid")


            ch = A.uns["cnv"]["chr_pos"]
            order = sorted(ch.items(), key=lambda x: x[1])
            bounds = {k: (v, order[j + 1][1] if j + 1 < len(order) else X.shape[1])
                     for j, (k, v) in enumerate(order)}


            def chrom_mean(c):
                if c not in bounds:
                    return np.full(X.shape[0], np.nan)
                a, b = bounds[c]
                return np.asarray(X[:, a:b].mean(1)).ravel()


            call.append(pd.DataFrame({"cell": A.obs_names, "gsm": g, "icnv_pred": pred,
                                       "icnv_score": score, "chr7_minus_chr10": chrom_mean("chr7") - chrom_mean("chr10")}))
            meta.append(dict(gsm=g, n_cells=A.n_obs, n_reference=nref, tier=1 if tier1 else 2,
                             threshold=threshold, aneuploid_pct=float(100 * (pred == "aneuploid").mean()),
                             n_genes=A.n_vars, state="ok"))
            print(f"[{i:2d}/{len(gsms)}] {g} n={A.n_obs:5d} ref={nref:5d} tier={1 if tier1 else 2} "
                  f"aneuploid={100*(pred=='aneuploid').mean():5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            meta.append(dict(gsm=g, n_cells=A.n_obs, n_reference=nref, tier=1 if tier1 else 2,
                             state=f"ERROR: {type(e).__name__}: {e}"))
            print(f"[{i:2d}/{len(gsms)}] {g} ERROR: {type(e).__name__}: {e}", flush=True)


    MT = pd.DataFrame(meta)
    MT.to_csv(f"{RESULTS_DIR}/cnv/infercnvpy_sample_metadata.csv", index=False)
    if call:
        CG = pd.concat(call, ignore_index=True)
        CG.to_csv(f"{RESULTS_DIR}/cnv/infercnvpy_cell_level_calls.csv", index=False)
        print(f"\ncalled nuclei: {len(CG)}  sample: {CG.gsm.nunique()}", flush=True)
    print(f"total elapsed: {time.time()-t0:.0f} s", flush=True)
    ok = MT[MT.state == "ok"]
    print(f"successful: {len(ok)}/{len(MT)}  |  Tier 1: {(ok.tier==1).sum()}  Tier 2: {(ok.tier==2).sum()}", flush=True)
    if len(ok):
        print(f"aneuploid median (Tier 1): {ok[ok.tier==1].aneuploid_pct.median():.1f}%", flush=True)


if __name__ == "__main__":
    main()
