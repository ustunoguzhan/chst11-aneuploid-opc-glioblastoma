#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import (PACKAGE_DATA, RESULTS_DIR, STATES_H5AD, bh, log, outdir, provenance,
                    read_gene_columns, require, state_axes, write_json)

KEGG = ["hsa00514", "hsa00512", "hsa00515", "hsa00532", "hsa00534", "hsa00531", "hsa00604"]
MIN_CELLS, MIN_EXPR, MIN_SAMPLES = 100, 20, 20
FOCUS_GENES = ["CHST11", "XYLT1", "GALNT13", "DSEL"]


def axis_test(rhos, patient_of):
    rhos = {s: r for s, r in rhos.items() if np.isfinite(r)}
    if len(rhos) < MIN_SAMPLES or len(rhos) < 2:
        return None
    r = np.array(list(rhos.values()))
    pc = pd.Series(rhos).groupby(pd.Series({s: patient_of[s] for s in rhos})).mean().values
    return dict(median_rho=float(np.median(r)), positive=int((r > 0).sum()), n_samples=len(r),
                p=stats.wilcoxon(r).pvalue, n_patients=len(pc), median_rho_patient=float(np.median(pc)),
                p_patient=stats.wilcoxon(pc).pvalue)


def screens(axes, axis_col, F, cells_flux, X, genes, gsm_all):
    patient_of = axes.groupby("gsm", observed=True).patient.first().astype(str).to_dict()
    rows = []
    a = axes.loc[cells_flux, axis_col].to_numpy(float)
    g = axes.loc[cells_flux, "gsm"].astype(str).to_numpy()
    groups = {s: np.where(g == s)[0] for s in pd.unique(g)}
    for m in F.columns:
        v = F[m].to_numpy(float)
        res = axis_test({s: stats.spearmanr(v[i], a[i]).statistic for s, i in groups.items()}, patient_of)
        if res:
            rows.append(dict(module=m, **res))
    Mo = pd.DataFrame(rows, columns=None if rows else ["module", "median_rho", "positive", "n_samples", "p",
                                                       "n_patients", "median_rho_patient", "p_patient"])
    if len(Mo):
        Mo["q"] = bh(Mo.p)
        Mo["rank"] = Mo.median_rho.rank(ascending=False).astype(int)
    ax = axes[axis_col].to_numpy(float)
    samples = [s for s, n in pd.Series(gsm_all).value_counts().items() if n >= MIN_CELLS]
    masks = {s: gsm_all == s for s in samples}
    rows = []
    for j, gname in enumerate(genes):
        vals = {}
        for s, m in masks.items():
            x = X[m, j]
            if (x > 0).sum() >= MIN_EXPR:
                vals[s] = stats.spearmanr(x, ax[m]).statistic
        res = axis_test(vals, patient_of)
        if res:
            rows.append(dict(gene=gname, detection_pct=100 * float((X[:, j] > 0).mean()), **res))
    En = pd.DataFrame(rows, columns=None if rows else ["gene", "detection_pct", "median_rho", "positive",
                                                       "n_samples", "p", "n_patients", "median_rho_patient",
                                                       "p_patient"])
    if len(En):
        En["q"] = bh(En.p)
        En["carried_forward"] = (En.q < .05) & (En.median_rho > 0) & (En.detection_pct >= 5)
        En["rank"] = En.median_rho.rank(ascending=False).astype(int)
    return Mo, En


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute-axes", action="store_true")
    ap.add_argument("--min-samples", type=int, default=int(os.environ.get("GBM_AXIS_MIN_SAMPLES", MIN_SAMPLES)),
                    help="minimum samples with a finite per-sample rho (script 05 uses 20)")
    args = ap.parse_args()
    globals()["MIN_SAMPLES"] = args.min_samples
    flux = PACKAGE_DATA / "scfea_flux_5986_malignant_nuclei.csv.gz"
    require([STATES_H5AD, flux], "24_state_scoring_harmonization")
    out = outdir("state_scoring_harmonization")
    axes = state_axes(force=args.recompute_axes)
    if not np.isfinite(axes.axis_neftel).any():
        raise SystemExit("axis_neftel unavailable (sc_* columns missing from the states H5AD)")
    r_all = stats.spearmanr(axes.axis_scanpy, axes.axis_neftel).statistic
    per = axes.groupby("gsm", observed=True).apply(lambda d: stats.spearmanr(d.axis_scanpy, d.axis_neftel).statistic if len(d) > 20 else np.nan)
    sign_agree = float((np.sign(axes.axis_scanpy) == np.sign(axes.axis_neftel)).mean())
    log(f"per-nucleus Spearman between the two coordinates: {r_all:.3f} (median per sample {per.median():.3f}); "
        f"same side of zero for {100 * sign_agree:.1f}% of malignant nuclei")

    F = pd.read_csv(flux, index_col=0)
    cells_flux = F.index.intersection(axes.index)
    if len(cells_flux) < len(F):
        log(f"WARNING: {len(F) - len(cells_flux)} flux nuclei not found among malignant nuclei")
    F = F.loc[cells_flux]
    gs = json.load(open(PACKAGE_DATA / "metabolic_genesets.json", encoding="utf-8"))
    union = sorted({g for k in KEGG for g in gs["kegg_metabolism"][k]["genes"]})
    import anndata as ad
    S = ad.read_h5ad(STATES_H5AD, backed="r")
    row_of = pd.Series(np.arange(S.n_obs), index=S.obs_names)
    S.file.close()
    rows = row_of.reindex(axes.index).astype(int).to_numpy()
    log(f"reading {len(union)} KEGG genes for {len(rows):,} malignant nuclei")
    X, genes = read_gene_columns(STATES_H5AD, rows, union)
    gsm_all = axes.gsm.astype(str).to_numpy()

    res = {}
    for col in ("axis_scanpy", "axis_neftel"):
        log(f"axis screens with {col}")
        res[col] = screens(axes, col, F, cells_flux, X, genes, gsm_all)
    Mo = res["axis_scanpy"][0].merge(res["axis_neftel"][0], on="module", suffixes=("_scanpy", "_neftel"))
    En = res["axis_scanpy"][1].merge(res["axis_neftel"][1], on="gene", how="outer", suffixes=("_scanpy", "_neftel"))
    Mo.to_csv(out / "module_axis_both_scorings.csv", index=False)
    En.to_csv(out / "enzyme_axis_both_scorings.csv", index=False)

    reference = {}
    f1 = RESULTS_DIR / "metabolic_flux" / "scfea_module_axis.csv"
    if f1.exists():
        s1 = pd.read_csv(f1).set_index("module").median_rho
        x = Mo.set_index("module").median_rho_scanpy.reindex(s1.index)
        reference["module_median_rho_max_abs_diff_vs_reference"] = float(np.nanmax(np.abs(x - s1)))
    f2 = RESULTS_DIR / "metabolic_flux" / "enzyme_axis_screen.csv"
    cf_ref = set()
    if f2.exists():
        s2 = pd.read_csv(f2)
        cf_ref = set(s2.gene[s2.carried_forward])
        x = En.set_index("gene").median_rho_scanpy.reindex(s2.gene)
        reference["enzyme_median_rho_max_abs_diff_vs_reference"] = float(np.nanmax(np.abs(x.values - s2.median_rho.values)))
    cf_a = set(En.gene[En.carried_forward_scanpy.fillna(False).astype(bool)])
    cf_b = set(En.gene[En.carried_forward_neftel.fillna(False).astype(bool)])
    jac = len(cf_a & cf_b) / max(1, len(cf_a | cf_b))
    focus = En[En.gene.isin(FOCUS_GENES)][["gene", "median_rho_scanpy", "q_scanpy", "rank_scanpy",
                                            "median_rho_neftel", "q_neftel", "rank_neftel",
                                            "carried_forward_scanpy", "carried_forward_neftel"]]
    m129 = Mo[Mo.module == "M_129"]
    rM = stats.spearmanr(Mo.median_rho_scanpy, Mo.median_rho_neftel).statistic
    rE = stats.spearmanr(En.median_rho_scanpy, En.median_rho_neftel, nan_policy="omit").statistic
    print(focus.to_string(index=False))
    print(m129.to_string(index=False))
    log(f"module median-rho agreement rho={rM:.3f}; enzyme rho={rE:.3f}; carried-forward Jaccard={jac:.2f} "
        f"({len(cf_a)} vs {len(cf_b)} genes); reference check {reference}")

    write_json(out / "key_results.json", {
        "axis_spearman_all_nuclei": r_all, "axis_spearman_median_per_sample": float(per.median()),
        "axis_same_sign_fraction": sign_agree, "module_rho_agreement": rM, "enzyme_rho_agreement": rE,
        "carried_forward_scanpy": sorted(cf_a), "carried_forward_neftel": sorted(cf_b),
        "carried_forward_jaccard": jac, "carried_forward_reference": sorted(cf_ref),
        "M_129": m129.to_dict(orient="records"), "focus_genes": focus.to_dict(orient="records"),
        "reference_reproduction": reference})
    provenance(out, __file__, args)
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
