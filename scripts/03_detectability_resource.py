import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR


import json, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import gseapy


warnings.filterwarnings("ignore")


SETS = {"GO_synaptic": "Chemical Synaptic Transmission (GO:0007268)",
        "GO_translation_reference": "Translation (GO:0006412)"}
REPORTED_NEURONAL = ["GABRB2", "GALNTL6", "TENM2", "WNK2", "RYR2",
                     "CACNA1B", "CNTN3", "CNTN5", "PRKG1", "SNAP91"]


def main():
    C = ad.read_h5ad(os.path.join(DATA_DIR, "gbm_snrna_counts.h5ad"))
    S = ad.read_h5ad(os.path.join(DATA_DIR, "gbm_snrna_states.h5ad"), backed="r")
    C.obs["neftel_state"] = S.obs["neftel_state"].astype(str).values
    C.obs["gsm"] = S.obs["gsm"].astype(str).values
    ic = pd.read_csv(os.path.join(RESULTS_DIR, "cnv",
                                  "infercnvpy_cell_level_calls.csv")).set_index("cell")
    im = pd.read_csv(os.path.join(RESULTS_DIR, "cnv", "infercnvpy_sample_metadata.csv"))
    tier1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"])
    C.obs["pred"] = ic.icnv_pred.reindex(C.obs_names).values


    sel = (C.obs.gsm.isin(tier1) & (C.obs.neftel_state == "OPC")
           & C.obs.pred.isin(["aneuploid", "diploid"])).values
    A = C[sel].copy()
    del C
    pred = A.obs.pred.values
    X = sp.csr_matrix(A.X)
    idx = {g: i for i, g in enumerate(A.var_names)}
    an = pred == "aneuploid"
    dip = pred == "diploid"
    print(f"OPC nuclei {A.n_obs}  aneuploid {int(an.sum())}  diploid {int(dip.sum())}",
          flush=True)


    gl = gseapy.get_library("GO_Biological_Process_2023")
    members = {k: sorted({x.split(",")[0] for x in gl[v]} & set(A.var_names))
               for k, v in SETS.items()}
    members["reported_neuronal_10"] = [g for g in REPORTED_NEURONAL if g in A.var_names]


    def profile(genes, label):
        j = [idx[g] for g in genes]
        sub = X[:, j]
        return pd.DataFrame({
            "gene": genes,
            "gene_set": label,
            "detection_pct": (np.asarray((sub > 0).sum(0)).ravel() / A.n_obs * 100).round(2),
            "mean_counts_per_nucleus": np.asarray(sub.mean(0)).ravel().round(4),
            "detection_aneuploid_pct": (np.asarray((X[an][:, j] > 0).sum(0)).ravel()
                                        / an.sum() * 100).round(2),
            "detection_diploid_pct": (np.asarray((X[dip][:, j] > 0).sum(0)).ravel()
                                      / dip.sum() * 100).round(2)})


    P = pd.concat([profile(members["reported_neuronal_10"], "reported_neuronal_10"),
                   profile(members["GO_synaptic"],
                           f"GO_synaptic_{len(members['GO_synaptic'])}"),
                   profile(members["GO_translation_reference"],
                           "GO_translation_reference")], ignore_index=True)
    P.to_csv(os.path.join(RESULTS_DIR, "expression_programs",
                          "detectability_resource.csv"), index=False)


    med = P.groupby("gene_set").detection_pct.agg(["count", "median", "min", "max"])
    print("\ndetection rate by gene set", flush=True)
    print(med.round(2).to_string(), flush=True)


    out = {"n_nuclei": int(A.n_obs), "n_aneuploid": int(an.sum()),
           "n_diploid": int(dip.sum()),
           "by_gene_set": {k: dict(n=int(v["count"]), median_pct=float(v["median"]),
                                   min_pct=float(v["min"]), max_pct=float(v["max"]))
                           for k, v in med.iterrows()}}
    with open(os.path.join(RESULTS_DIR, "expression_programs",
                           "detectability_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[OK] results written to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
