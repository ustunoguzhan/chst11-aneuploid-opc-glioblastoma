#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import RESULTS_DIR, bh, holm, log, outdir, provenance, require, write_json


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "ns"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args()
    out = outdir("multiplicity_corrections")
    res = {}

    f = RESULTS_DIR / "targets" / "cell_state_scope.csv"
    require([f], "26_multiplicity_corrections")
    S = pd.read_csv(f)
    S["p_holm_all12"] = holm(S.p_perm)
    S["p_bh_all12"] = bh(S.p_perm)
    S["p_holm_within_state"] = S.groupby("state").p_perm.transform(lambda p: holm(p.values))
    S["stars_nominal"] = S.p_perm.map(stars)
    S["stars_holm_all12"] = S.p_holm_all12.map(stars)
    S.to_csv(out / "cell_state_scope_holm_bh.csv", index=False)
    print(S.to_string(index=False))
    res["cell_state_scope"] = S.to_dict(orient="records")

    f = RESULTS_DIR / "targets" / "cspg_pathway_consistency.csv"
    j = RESULTS_DIR / "targets" / "cspg_pathway_consistency.json"
    if f.exists() and j.exists():
        P = pd.read_csv(f)
        P["p_holm_genes"] = holm(P.p_perm)
        P["p_bh_genes"] = bh(P.p_perm)
        corr = json.load(open(j, encoding="utf-8"))["per_sample_delta_vs_CHST11_spearman"]
        Cc = pd.DataFrame([dict(gene=g, **v) for g, v in corr.items()])
        Cc["p_holm"] = holm(Cc.p)
        Cc["p_bh"] = bh(Cc.p)
        P.to_csv(out / "cspg_genes_holm_bh.csv", index=False)
        Cc.to_csv(out / "cspg_correlations_holm_bh.csv", index=False)
        print(P.to_string(index=False))
        print(Cc.to_string(index=False))
        res["cspg_genes"] = P.to_dict(orient="records")
        res["cspg_correlations"] = Cc.to_dict(orient="records")

    f = RESULTS_DIR / "independent_cohorts" / "survival_covariate_adjusted.csv"
    if f.exists():
        V = pd.read_csv(f)
        V["p_bh_all_models"] = bh(V.p)
        V.to_csv(out / "survival_bh.csv", index=False)
        res["survival_min_p_bh"] = float(np.nanmin(V.p_bh_all_models))
        print(f"survival: min BH-adjusted P across {len(V)} models = {res['survival_min_p_bh']:.3g}")

    write_json(out / "key_results.json", res)
    provenance(out, __file__, args)
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
