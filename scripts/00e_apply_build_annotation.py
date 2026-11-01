import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PACKAGE_DATA, RESULTS_DIR
sys.path.insert(0, PACKAGE_DATA)
import numpy as np, pandas as pd, anndata as ad
import neftel

ap = argparse.ArgumentParser()
ap.add_argument("--build", type=int, choices=[1, 2], required=True)
ap.add_argument("--states", default=os.path.join(DATA_DIR, "gbm_snrna_states.h5ad"),
                help="matrix written by 00c_state_assignment.py")
args = ap.parse_args()

BUILDS = os.path.join(PACKAGE_DATA, "builds")
OUT = {1: os.path.join(DATA_DIR, "gbm_snrna_states.h5ad"),
       2: os.path.join(DATA_DIR, "gbm_snrna_states_build2.h5ad")}[args.build]

A = ad.read_h5ad(args.states)
b1 = pd.read_csv(os.path.join(BUILDS, "build1_annotation.csv.gz"), index_col=0)
assert set(b1.index) == set(A.obs_names), "barcodes of the matrix and of the Build 1 annotation differ"
b1 = b1.reindex(A.obs_names)

meta = neftel.load_metamodules(os.path.join(PACKAGE_DATA, "neftel_metamodules.csv"))
sc_cols = [c for c in b1.columns if c.startswith("sc_")]
dev = max(float(np.nanmax(np.abs(A.obs[c].to_numpy(float) - b1[c].to_numpy(float)))) for c in sc_cols)
print(f"maximum difference of the meta-module scores from the deposited values: {dev:.2e}", flush=True)
assert dev < 1e-4, "meta-module scores differ from the deposited ones; the input is not the 00c matrix"

if args.build == 1:
    for c in ["leiden", "cluster_call", "malignant", "neftel_state", "neftel_x", "neftel_y", "cycling"] + sc_cols:
        A.obs[c] = b1[c].to_numpy()
    A.obs["leiden"] = A.obs["leiden"].astype(str)
    A.obs["malignant"] = A.obs["malignant"].astype(bool)
    A.obs["cycling"] = A.obs["cycling"].astype(bool)
else:
    b2 = pd.read_csv(os.path.join(BUILDS, "build2_annotation.csv.gz"), index_col=0)
    assert b2.index.isin(A.obs_names).all(), "Build 2 barcodes missing from the matrix"
    mal = A.obs_names.isin(b2.index)
    A.obs["malignant"] = mal
    call = b1["cluster_call"].astype(str).to_numpy(dtype=object)
    call[mal] = "malignant"
    call[(~mal) & (call == "malignant")] = "nonmalignant_unresolved"
    A.obs["cluster_call"] = call
    M = A.obs_names[mal]
    pl = neftel.neftel_plane({k: b1.loc[M, f"sc_{k}"].to_numpy() for k in meta})
    st = pd.Series(pl["state"], index=M).astype(str)
    agree = float((st == b2.loc[M, "neftel_state"].astype(str)).mean())
    print(f"Neftel state recomputed from the scores agrees with the deposited state: {agree:.4%}", flush=True)
    assert agree == 1.0, "recomputed Neftel states differ from the deposited Build 2 states"
    A.obs["neftel_state"] = pd.Series("nonmalignant", index=A.obs_names, dtype=object)
    A.obs.loc[M, "neftel_state"] = st.to_numpy()
    A.obs["neftel_x"] = np.nan; A.obs["neftel_y"] = np.nan
    A.obs.loc[M, "neftel_x"] = pl["x"]; A.obs.loc[M, "neftel_y"] = pl["y"]
    cache = os.path.join(RESULTS_DIR, "cache")
    os.makedirs(cache, exist_ok=True)
    ax = b2.copy()
    ax.insert(1, "patient", A.obs.loc[ax.index, "patient"].astype(str).to_numpy())
    ax.to_csv(os.path.join(cache, "state_axes_malignant.csv.gz"), compression="gzip")
    print("wrote results/cache/state_axes_malignant.csv.gz (Build 2 state scores)", flush=True)

A.uns["build"] = dict(build=args.build,
                      annotation=f"data/builds/build{args.build}_annotation.csv.gz")
opc = int(((A.obs.neftel_state == "OPC") & A.obs.malignant).sum())
print(f"Build {args.build}: malignant {int(A.obs.malignant.sum()):,} | OPC-like {opc:,}", flush=True)
A.write_h5ad(OUT, compression="gzip")
print(f"WROTE {OUT}", flush=True)
