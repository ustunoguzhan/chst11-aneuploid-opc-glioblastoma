import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PACKAGE_DATA
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, json, sys, time
sys.path.insert(0, PACKAGE_DATA)
import neftel

IN  = os.path.join(DATA_DIR, "gbm_snrna_qc.h5ad")
OUT = os.path.join(DATA_DIR, "gbm_snrna_states.h5ad")
N_HVG, N_PCS, LEIDEN_RES, SEED = 3000, 50, 1.0, 0
t0 = time.time()

A = ad.read_h5ad(IN)
sc.pp.normalize_total(A, target_sum=1e4)
sc.pp.log1p(A)

ct = json.load(open(os.path.join(PACKAGE_DATA, "celltype_genesets.json"),
                    encoding="utf-8"))
thr = json.load(open(os.path.join(PACKAGE_DATA, "nonmalignant_thresholds.json"),
                     encoding="utf-8"))

meta = neftel.load_metamodules(os.path.join(PACKAGE_DATA, "neftel_metamodules.csv"))
nm_primary = ct["neftel_primary"]
all_ct = {**nm_primary, **ct["guard_beyond_neftel"], **ct["cellguide_crosscheck"]}

sc_all, cov = neftel.score_sets(A.X, list(A.var_names), {**meta, **all_ct},
                                seed=SEED, return_coverage=True)
for k, v in sc_all.items():
    A.obs[f"sc_{k}"] = v
pd.DataFrame([{"set": k, "measured": c[0], "requested": c[1],
               "coverage": c[0] / c[1]} for k, c in cov.items()]
             ).sort_values("coverage").to_csv(os.path.join(DATA_DIR, "geneset_coverage.csv"), index=False)
print(f"scoring done {time.time()-t0:.0f}s", flush=True)

sc.pp.highly_variable_genes(A, n_top_genes=N_HVG, flavor="seurat", batch_key="gsm")
Ah = A[:, A.var.highly_variable].copy()
sc.pp.scale(Ah, max_value=10)
sc.tl.pca(Ah, n_comps=N_PCS, svd_solver="arpack", random_state=SEED)
sc.pp.neighbors(Ah, n_neighbors=15, n_pcs=N_PCS, random_state=SEED)
sc.tl.leiden(Ah, resolution=LEIDEN_RES, key_added="leiden",
             flavor="igraph", n_iterations=2, directed=False, random_state=SEED)
A.obs["leiden"] = Ah.obs["leiden"].values
A.obsm["X_pca"] = Ah.obsm["X_pca"]
print(f"clustering done {time.time()-t0:.0f}s | {A.obs.leiden.nunique()} clusters", flush=True)

def shannon(x):
    p = x.value_counts(normalize=True).values
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0

rows = []
for cl, idx in A.obs.groupby("leiden", observed=True).groups.items():
    o = A.obs.loc[idx]
    r = {"leiden": cl, "n": len(o),
         "patient_entropy": shannon(o.patient),
         "n_patients": o.patient.nunique(),
         "frac_recurrent": float((o.progression == "Recurrent").mean())}
    for k in all_ct:
        r[f"med_{k}"] = float(o[f"sc_{k}"].median())
    for k in meta:
        r[f"med_{k}"] = float(o[f"sc_{k}"].median())
    rows.append(r)
CL = pd.DataFrame(rows).set_index("leiden")

call_sets = list(nm_primary) + list(ct["guard_beyond_neftel"])
def cluster_call(r):
    hits = []
    for k in call_sets:
        t = thr.get(k, {}).get("q999")
        if t is None:
            t = 3 * np.std([r[f"med_{x}"] for x in meta])
        if r[f"med_{k}"] > t:
            hits.append((r[f"med_{k}"], k))
    if not hits:
        return "malignant"
    return max(hits)[1]
CL["call"] = CL.apply(cluster_call, axis=1)
A.obs["cluster_call"] = A.obs.leiden.map(CL["call"]).astype(str)

for k in nm_primary:
    A.obs[f"pass_{k}"] = A.obs[f"sc_{k}"] > thr[k]["q999"]
A.obs["malignant"] = A.obs.cluster_call == "malignant"

CL.to_csv(os.path.join(DATA_DIR, "cluster_summary.csv"))
print("\n--- cluster summary ---", flush=True)
cols = ["n", "n_patients", "patient_entropy", "frac_recurrent", "call"]
print(CL[cols].sort_values("n", ascending=False).to_string(), flush=True)
print("\nnuclei per cluster call:", flush=True)
print(A.obs.cluster_call.value_counts().to_string(), flush=True)
print("\nmalignant nuclei: %d (%.1f%%)" % (A.obs.malignant.sum(), 100*A.obs.malignant.mean()), flush=True)
print("median patient entropy, malignant clusters: %.3f | non-malignant: %.3f" % (
    CL.loc[CL.call == "malignant", "patient_entropy"].median(),
    CL.loc[CL.call != "malignant", "patient_entropy"].median()), flush=True)

M = A[A.obs.malignant]
pl = neftel.neftel_plane({k: M.obs[f"sc_{k}"].values for k in meta})
A.obs["neftel_x"] = np.nan; A.obs["neftel_y"] = np.nan
A.obs["neftel_state"] = pd.Series("nonmalignant", index=A.obs_names, dtype=object)
A.obs.loc[M.obs_names, "neftel_x"] = pl["x"]
A.obs.loc[M.obs_names, "neftel_y"] = pl["y"]
A.obs.loc[M.obs_names, "neftel_state"] = pl["state"]
cyc = (M.obs["sc_G1S"].values > 0) | (M.obs["sc_G2M"].values > 0)
A.obs["cycling"] = False
A.obs.loc[M.obs_names, "cycling"] = cyc

print("\n--- malignant state distribution ---", flush=True)
d = A.obs.loc[A.obs.malignant, "neftel_state"].value_counts(normalize=True).mul(100).round(1)
print(d.to_string(), flush=True)
print("cycling malignant nuclei: %.1f%%" % (100*A.obs.loc[A.obs.malignant,"cycling"].mean()), flush=True)

A.uns["assign"] = dict(n_hvg=N_HVG, n_pcs=N_PCS, leiden_res=LEIDEN_RES, seed=SEED,
                       thresholds=thr, call_sets=call_sets)
A.write_h5ad(OUT, compression="gzip")
print(f"\nWROTE {OUT} | total {time.time()-t0:.0f}s", flush=True)
