#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import (COUNTS_H5AD, PACKAGE_DATA, STATES_H5AD, analysis_cells, bh, compartment_delta_all,
                    depth_adjusted_all, holm, load_analysis_set_matrix, load_cnv_calls, load_cohort,
                    load_states_obs, lognorm_counts, log, matched_null_p, outdir, per_sample_delta,
                    provenance, read_gene_columns, require, sign_test, signflip, state_axes, write_json)

warnings.filterwarnings("ignore")
CORE = ["CHST11", "CHST12", "CHST13", "CHST14", "CHST15", "CHST3", "CHST7", "UST"]
EXTENDED = ([f"CHST{i}" for i in range(1, 16)] + ["UST", "HS2ST1", "HS3ST1", "HS3ST2", "HS3ST3A1",
            "HS3ST3B1", "HS3ST4", "HS3ST5", "HS3ST6", "HS6ST1", "HS6ST2", "HS6ST3", "NDST1", "NDST2",
            "NDST3", "NDST4", "GAL3ST1", "GAL3ST2", "GAL3ST3", "GAL3ST4"])
STATES = ["OPC", "NPC", "AC", "MES"]


def axis_assoc(axes, genes, min_cells=100, min_expr=20):
    import anndata as ad
    S = ad.read_h5ad(STATES_H5AD, backed="r")
    row_of = pd.Series(np.arange(S.n_obs), index=S.obs_names)
    S.file.close()
    rows = row_of.reindex(axes.index).dropna().astype(int)
    vals, present = read_gene_columns(STATES_H5AD, rows.to_numpy(), genes)
    ax = axes.loc[rows.index]
    gsm = ax.gsm.astype(str).to_numpy()
    patient_of = ax.groupby("gsm").patient.first().astype(str).to_dict()
    out = []
    for col in ("axis_scanpy", "axis_neftel"):
        a = ax[col].to_numpy(float)
        if not np.isfinite(a).any():
            continue
        for j, g in enumerate(present):
            rh = {}
            for s in pd.unique(gsm):
                m = gsm == s
                if m.sum() < min_cells or (vals[m, j] > 0).sum() < min_expr:
                    continue
                r = stats.spearmanr(vals[m, j], a[m]).statistic
                if np.isfinite(r):
                    rh[s] = r
            r = np.array(list(rh.values()))
            pc = pd.Series(rh).groupby(pd.Series({s: patient_of[s] for s in rh})).mean() if rh else pd.Series(dtype=float)
            out.append(dict(gene=g, axis=col, n_samples=len(r), median_rho=np.median(r) if len(r) else np.nan,
                            positive=int((r > 0).sum()),
                            p=stats.wilcoxon(r).pvalue if len(r) >= 6 else np.nan,
                            n_patients=len(pc), median_rho_patient=pc.median() if len(pc) else np.nan,
                            p_patient=stats.wilcoxon(pc.values).pvalue if len(pc) >= 6 else np.nan,
                            meets_min_samples=len(r) >= 20))
    return pd.DataFrame(out)


def state_scope(genes):
    import anndata as ad
    obs, _ = load_states_obs(["gsm", "neftel_state"])
    ic, im, tier1 = load_cnv_calls()
    obs["pred"] = ic.icnv_pred.reindex(obs.index)
    C = ad.read_h5ad(COUNTS_H5AD, backed="r")
    row_of = pd.Series(np.arange(C.n_obs), index=C.obs_names)
    C.file.close()
    cand = obs[obs.neftel_state.isin(STATES) & obs.gsm.isin(tier1) & obs.pred.isin(["aneuploid", "diploid"])]
    rows = row_of.reindex(cand.index).astype(int).to_numpy()
    cnt, present, tot = read_gene_columns(COUNTS_H5AD, rows, genes, with_totals=True)
    V = pd.DataFrame(lognorm_counts(cnt, tot), index=cand.index, columns=present)
    out = []
    for st in STATES:
        m, gsms = analysis_cells(cand, "pred", state=st, samples_allowed=tier1)
        if len(gsms) < 4:
            continue
        sub = cand[m]
        D = per_sample_delta(V.loc[sub.index].to_numpy(), sub.gsm, sub.pred)
        D.columns = present
        for g in present:
            mm, p, _, _ = signflip(D[g].values)
            out.append(dict(state=st, gene=g, n_samples=len(D), delta=mm, p=p, positive=int((D[g] > 0).sum())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-match", type=int, default=500)
    ap.add_argument("--skip-axis", action="store_true", help="skip the state-axis layer (needs RAM)")
    args = ap.parse_args()
    require([COUNTS_H5AD, STATES_H5AD], "21_sulfotransferase_family")
    out = outdir("sulfotransferase_family")

    log("loading the OPC-like analysis set (all genes)")
    M = load_analysis_set_matrix()
    var, o = M["var"], M["obs"]
    log(f"analysis set: {len(o)} nuclei, {len(M['gsms'])} samples")
    log("compartment differences and depth-adjusted coefficients for all genes")
    D = compartment_delta_all(M["Xn"], o.gsm, o.pred, var)
    Dadj = depth_adjusted_all(M["Xn"], o.gsm.to_numpy(), o.pred.to_numpy(), M["logdepth"], var)
    mean_raw, mean_adj = D.mean(0).to_numpy(), Dadj.mean(0).to_numpy()
    coh = load_cohort().set_index("gsm")
    pat = coh.patient_pair.reindex(D.index)
    prog = coh.progression.reindex(D.index)
    det = M["det_pct"]
    genes = [g for g in dict.fromkeys(CORE + EXTENDED) if g in var]
    rng = np.random.default_rng(42)
    rows = []
    for g in genes:
        j = var.get_loc(g)
        d, a = D[g], Dadj[g]
        m, p, _, _ = signflip(d.values)
        ps, k, n = sign_test(d.values)
        pm, pp, _, _ = signflip(d.groupby(pat).mean().values)
        am, apv, _, _ = signflip(a.values)
        pnull, nmed, nm = matched_null_p(det, j, mean_raw, m, args.n_match, rng=rng)
        pnull_adj, nmed_adj, _ = matched_null_p(det, j, mean_adj, am, args.n_match, rng=rng)
        prm, prp, _, _ = signflip(d[prog == "Primary"].values)
        rcm, rcp, _, _ = signflip(d[prog == "Recurrent"].values)
        rows.append(dict(gene=g, core=g in CORE, detection_pct=det[j], n_samples=len(d), delta=m, p_signflip=p,
                         positive=k, p_sign_test=ps, patient_delta=pm, p_patient=pp,
                         p_matched_null=pnull, matched_null_median=nmed, n_matched=nm,
                         depth_adj=am, p_depth_adj=apv, p_matched_null_depth_adj=pnull_adj,
                         primary_delta=prm, p_primary=prp, recurrent_delta=rcm, p_recurrent=rcp,
                         rank_of_delta_among_all_genes=int((mean_raw > m).sum() + 1),
                         percentile_among_detection_matched=float(100 * (1 - pnull)) if np.isfinite(pnull) else np.nan))
    R = pd.DataFrame(rows)
    for col in ("p_signflip", "p_matched_null", "p_depth_adj", "p_matched_null_depth_adj", "p_patient"):
        R[f"{col}_holm_core"] = np.nan
        R.loc[R.core, f"{col}_holm_core"] = holm(R.loc[R.core, col].values)
        R[f"{col}_bh_extended"] = bh(R[col].values)
    D[genes].reset_index().to_csv(out / "per_sample_delta_family.csv", index=False)

    log("cell-state scope (OPC/NPC/AC/MES)")
    SS = state_scope(genes)
    SS.to_csv(out / "cell_state_scope_family.csv", index=False)
    wide = SS.pivot_table(index="gene", columns="state", values="delta")
    wide.columns = [f"delta_{c}" for c in wide.columns]
    R = R.merge(wide, left_on="gene", right_index=True, how="left")

    if not args.skip_axis:
        log("state-axis association (both scorings)")
        AX = axis_assoc(state_axes(), genes)
        AX.to_csv(out / "state_axis_family.csv", index=False)
        for col in ("axis_scanpy", "axis_neftel"):
            sub = AX[AX.axis == col].set_index("gene")
            R[f"{col}_median_rho"] = R.gene.map(sub.median_rho)
            R[f"{col}_p"] = R.gene.map(sub.p)
            R[f"{col}_n_samples"] = R.gene.map(sub.n_samples)

    tr = pd.read_csv(PACKAGE_DATA / "target_tractability.csv")
    tr = tr[[c for c in ["gene", "sm_best", "sm_evidence", "ab_evidence", "n_drugs", "max_phase"] if c in tr]]
    R = R.merge(tr.drop_duplicates("gene"), on="gene", how="left")
    R = R.sort_values(["core", "delta"], ascending=[False, False])
    R.to_csv(out / "chst_family_evidence_matrix.csv", index=False)
    show = ["gene", "detection_pct", "delta", "p_signflip", "p_matched_null", "depth_adj", "p_depth_adj",
            "patient_delta", "p_patient", "primary_delta", "recurrent_delta"]
    print(R[R.core][show].to_string(index=False))

    write_json(out / "key_results.json", {"core": R[R.core].to_dict(orient="records")})
    provenance(out, __file__, args)
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
