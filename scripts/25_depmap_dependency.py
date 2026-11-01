#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from common import (PACKAGE_DATA, log, outdir, provenance, require, write_json)

EXP, DEP = "CHST11_Expression_TPM_Logp1_24Q4", "CHST11_Chronos_GeneEffect_24Q4"


def corr_block(df, label, n_boot):
    d = df.dropna(subset=[EXP, DEP])
    x, y = d[EXP].to_numpy(float), d[DEP].to_numpy(float)
    rs = stats.spearmanr(x, y)
    rp = stats.pearsonr(x, y)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(stats.spearmanr(x[i], y[i]).statistic)
    lo, hi = np.nanquantile(boots, [.025, .975])
    slope = stats.linregress(x, y)
    return dict(group=label, n=len(d), spearman_rho=rs.statistic, spearman_p=rs.pvalue,
                spearman_ci_low=lo, spearman_ci_high=hi, pearson_r=rp.statistic, pearson_p=rp.pvalue,
                ols_slope=slope.slope, ols_slope_p=slope.pvalue,
                interpretation=("higher expression -> weaker (less negative) dependency" if rs.statistic > 0
                                else "higher expression -> stronger (more negative) dependency"))


def partial_rank_lineage(df):
    d = df.dropna(subset=[EXP, DEP, "OncotreeLineage"]).copy()
    rx, ry = stats.rankdata(d[EXP]), stats.rankdata(d[DEP])
    L = pd.get_dummies(d.OncotreeLineage, drop_first=True).to_numpy(float)
    Z = np.column_stack([np.ones(len(d)), L])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = stats.pearsonr(ex, ey)
    return dict(group="All lines, lineage-adjusted (partial rank)", n=len(d), spearman_rho=r.statistic,
                spearman_p=r.pvalue, n_lineages=int(d.OncotreeLineage.nunique()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    f = PACKAGE_DATA / "depmap_chst11_dependency_by_cellline.csv"
    require([f], "25_depmap_dependency")
    out = outdir("depmap_dependency")
    dm = pd.read_csv(f)
    dm = dm.dropna(subset=[DEP])
    for c in ("is_GBM", "is_Glioma_any"):
        dm[c] = dm[c].astype(str).str.lower().isin(["true", "1"])
    gbm, gli, oth = dm[dm.is_GBM], dm[dm.is_Glioma_any], dm[~dm.is_Glioma_any]

    C = pd.DataFrame([corr_block(gbm, "GBM", args.n_boot), corr_block(gli, "All diffuse glioma", args.n_boot),
                      corr_block(dm, "All lines", args.n_boot), partial_rank_lineage(dm)])
    C.to_csv(out / "depmap_expression_vs_dependency.csv", index=False)
    print(C.to_string(index=False))

    g = []
    for nm, d in [("GBM", gbm), ("Other diffuse glioma", gli[~gli.is_GBM]), ("Non-glioma", oth), ("All", dm)]:
        v = d[DEP].to_numpy(float)
        g.append(dict(group=nm, n=len(v), median=np.median(v), mean=v.mean(),
                      n_below_minus_0_5=int((v < -0.5).sum()), n_below_minus_1=int((v < -1).sum()),
                      frac_below_minus_0_5=float((v < -0.5).mean())))
    G = pd.DataFrame(g)
    mw = stats.mannwhitneyu(gbm[DEP], oth[DEP])
    G.to_csv(out / "depmap_dependency_by_group.csv", index=False)
    print(G.to_string(index=False))
    print(f"GBM vs non-glioma Mann-Whitney P = {mw.pvalue:.3g}")

    write_json(out / "key_results.json", {"correlations": C.to_dict(orient="records"),
                                          "groups": G.to_dict(orient="records"),
                                          "gbm_vs_nonglioma_mannwhitney_p": mw.pvalue})
    provenance(out, __file__, args)
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
