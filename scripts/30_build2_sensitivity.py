#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (COUNTS_H5AD, PACKAGE_DATA, STATES_H5AD, analysis_cells, compartment_delta_all,
                    depth_adjusted_all, load_cnv_calls, load_cohort, load_states_obs, log, matched_null_p,
                    outdir, provenance, read_rows, require, signflip, skip, write_json)

GENES = ["CHST11", "XYLT1", "DSEL", "PTPRZ1", "VCAN", "BCAN", "CSPG4", "UST", "CHST3"]
MARKERS = ["MBP", "PLP1", "MOG", "MOBP", "ST18", "PDGFRA", "OLIG2", "SOX10"]


def load_set(names, obs, row_of):
    rows = row_of.reindex(names).to_numpy()
    if np.isnan(rows.astype(float)).any():
        raise SystemExit("analysis-set nuclei missing from the counts H5AD")
    X, var = read_rows(COUNTS_H5AD, rows.astype(int))
    tot = np.asarray(X.sum(1)).ravel().astype(float)
    Xn = sp.csr_matrix(sp.diags(1e4 / tot) @ X.astype(np.float64))
    Xn.data = np.log1p(Xn.data)
    o = obs.loc[names, ["gsm", "pred"]].reset_index(drop=True)
    det = np.asarray((X > 0).sum(0)).ravel() / X.shape[0] * 100
    return Xn, var, o, np.log10(tot), det


def evaluate(label, names, obs, row_of, pat_of, n_match, hallmark):
    log(f"{label}: {len(names)} nuclei")
    Xn, var, o, logdepth, det = load_set(names, obs, row_of)
    D = compartment_delta_all(Xn, o.gsm, o.pred, var)
    Dadj = depth_adjusted_all(Xn, o.gsm.to_numpy(), o.pred.to_numpy(), logdepth, var)
    mean_raw, mean_adj = D.mean(0).to_numpy(), Dadj.mean(0).to_numpy()
    pat = pat_of.reindex(D.index)
    hm = pd.DataFrame({"v": hallmark.reindex(names).to_numpy(), "gsm": o.gsm, "pred": o.pred})
    hm = hm.groupby(["gsm", "pred"]).v.mean().unstack()
    hd = (hm["aneuploid"] - hm["diploid"]).dropna()
    rng = np.random.default_rng(42)
    rows = []
    for g in GENES:
        j = var.get_loc(g)
        d, a = D[g], Dadj[g]
        m, p, _, _ = signflip(d.values)
        pm, pp, _, _ = signflip(d.groupby(pat).mean().values)
        am, apv, _, _ = signflip(a.values)
        pnull, _, nm = matched_null_p(det, j, mean_raw, m, n_match, rng=rng)
        pnull_adj, _, nm_adj = matched_null_p(det, j, mean_adj, am, n_match, rng=rng)
        rows.append(dict(set=label, gene=g, n_nuclei=len(names), n_samples=len(d),
                         n_patients=int(pat.nunique()), detection_pct=det[j], delta=m, positive=int((d > 0).sum()),
                         p_signflip=p, patient_delta=pm, p_patient=pp, depth_adj=am, p_depth_adj=apv,
                         p_matched_null=pnull, n_matched=nm, p_matched_null_depth_adj=pnull_adj,
                         n_matched_depth_adj=nm_adj, hallmark_delta=float(hd.mean()),
                         hallmark_positive=int((hd > 0).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-match", type=int, default=500)
    args = ap.parse_args()
    b2_path = os.path.join(PACKAGE_DATA, "builds", "build2_annotation.csv.gz")
    require([COUNTS_H5AD, STATES_H5AD, b2_path], "30_build2_sensitivity")
    out = outdir("build2_sensitivity")

    obs, _ = load_states_obs(["gsm", "patient", "neftel_state", "cluster_call", "malignant"])
    ic, _, tier1 = load_cnv_calls()
    obs["pred"] = ic.icnv_pred.reindex(obs.index)
    b2 = pd.read_csv(b2_path, index_col=0, usecols=["cell", "neftel_state"])
    if not b2.index.isin(obs.index).any():
        skip("the Build 2 annotation shares no barcodes with the states matrix")
    obs2 = obs.copy()
    obs2["neftel_state"] = b2.neftel_state.reindex(obs.index).fillna("nonmalignant").astype(str)

    m1, g1 = analysis_cells(obs, samples_allowed=tier1)
    m2, g2 = analysis_cells(obs2, samples_allowed=tier1)
    n1, n2 = obs.index[m1], obs.index[m2]
    added, lost = n2.difference(n1), n1.difference(n2)
    shared = n2.intersection(n1)
    log(f"Build 1 set {len(n1)} | Build 2 set {len(n2)} | added in Build 2 {len(added)} | Build 1 only {len(lost)}")

    import anndata as ad
    C = ad.read_h5ad(COUNTS_H5AD, backed="r")
    row_of = pd.Series(np.arange(C.n_obs), index=C.obs_names)
    C.file.close()

    prof = []
    groups = {"added_in_build2": added,
              "shared_diploid": shared[obs2.loc[shared, "pred"].to_numpy() == "diploid"],
              "shared_aneuploid": shared[obs2.loc[shared, "pred"].to_numpy() == "aneuploid"]}
    for name, ix in groups.items():
        Xn, var, _, _, _ = load_set(ix, obs2, row_of)
        r = dict(group=name, n_nuclei=len(ix))
        for g in MARKERS + GENES:
            if g in var:
                v = np.asarray(Xn[:, var.get_loc(g)].todense()).ravel()
                r[f"{g}_mean"] = float(v.mean())
                r[f"{g}_detection_pct"] = float(100 * (v > 0).mean())
        prof.append(r)
    P = pd.DataFrame(prof)
    P.to_csv(out / "added_nuclei_profile.csv", index=False)

    dip = obs2.loc[n2][obs2.loc[n2, "pred"] == "diploid"]
    share = pd.Series(dip.index.isin(added).astype(float), index=dip.index)
    per_sample = share.groupby(dip.gsm.to_numpy()).mean()
    summary = dict(
        build1_set=len(n1), build2_set=len(n2), added_in_build2=len(added), build1_only=len(lost),
        build1_samples=len(g1), build2_samples=len(g2), samples_identical=sorted(g1) == sorted(g2),
        added_build1_cluster_call=obs.loc[added, "cluster_call"].value_counts().to_dict(),
        added_compartment=obs2.loc[added, "pred"].value_counts().to_dict(),
        added_share_of_build2_diploid=float(share.mean()),
        added_share_of_build2_diploid_per_sample_median=float(per_sample.median()),
        added_share_of_build2_diploid_per_sample_max=float(per_sample.max()),
    )
    log(str(summary))

    pat_of = load_cohort().set_index("gsm").patient_pair
    hallmark = ic.chr7_minus_chr10
    R = pd.concat([evaluate("build1", n1, obs, row_of, pat_of, args.n_match, hallmark),
                   evaluate("build2", n2, obs2, row_of, pat_of, args.n_match, hallmark),
                   evaluate("build2_shared", shared, obs2, row_of, pat_of, args.n_match, hallmark)])
    R.to_csv(out / "build_comparison.csv", index=False)
    write_json(out / "summary.json", summary)
    provenance(out, __file__, args,
               extra=dict(inputs=["gbm_snrna_states.h5ad (Build 1)", "data/builds/build2_annotation.csv.gz",
                                  "gbm_snrna_counts.h5ad", "results/cnv/infercnvpy_cell_level_calls.csv"]))
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
