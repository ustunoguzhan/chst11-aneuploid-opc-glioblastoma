#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from common import (COHORT_CSV, GENES3, RESULTS_DIR, describe_effect, holm, log, outdir, provenance,
                    require, sign_test, signflip, write_json)


def load_effects(genes):
    f = RESULTS_DIR / "targets" / "per_tumor_expression.csv"
    require([f, COHORT_CSV], "19_patient_level_robustness")
    E = pd.read_csv(f)
    E["delta"] = E.aneuploid - E.diploid
    c = pd.read_csv(COHORT_CSV)[["gsm", "patient_pair", "progression"]]
    E = E.merge(c, on="gsm", how="left", validate="many_to_one")
    if E.patient_pair.isna().any():
        raise SystemExit("some GSMs in per_tumor_expression.csv are missing from snrna_cohort.csv")
    return E[E.gene.isin(genes)].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes", nargs="+", default=GENES3)
    ap.add_argument("--n-boot", type=int, default=100_000)
    args = ap.parse_args()
    out = outdir("patient_level")
    E = load_effects(args.genes)

    summary, lopo, loso = [], [], []
    for gene in args.genes:
        s = E[E.gene == gene]
        if s.empty:
            continue
        for unit, d in [("sample", s.set_index("gsm").delta),
                        ("patient", s.groupby("patient_pair").delta.mean())]:
            r = dict(gene=gene, unit=unit)
            r.update(describe_effect(d.values, "", n_boot=args.n_boot))
            summary.append(r)
        P = s.groupby("patient_pair").delta.mean().sort_index()
        for om in P.index:
            d = P.drop(om).values
            m, p, _, _ = signflip(d)
            ps, k, n = sign_test(d)
            lopo.append(dict(gene=gene, omitted_patient=om, omitted_delta=P[om], n=n, mean=m,
                             positive=k, p_signflip=p, p_sign_test=ps))
        Ss = s.set_index("gsm").delta.sort_index()
        for om in Ss.index:
            d = Ss.drop(om).values
            m, p, _, _ = signflip(d)
            loso.append(dict(gene=gene, omitted_sample=om, omitted_delta=Ss[om], n=len(d), mean=m,
                             positive=int((d > 0).sum()), p_signflip=p))
    S = pd.DataFrame(summary).rename(columns={"n": "n_units"})
    for unit in ("sample", "patient"):
        m = S.unit == unit
        for col in ("p_signflip", "p_sign_test", "p_wilcoxon"):
            S.loc[m, f"{col}_holm3"] = holm(S.loc[m, col].values)
    L, LS = pd.DataFrame(lopo), pd.DataFrame(loso)
    S.to_csv(out / "summary.csv", index=False)
    L.to_csv(out / "leave_one_patient_out.csv", index=False)
    LS.to_csv(out / "leave_one_sample_out.csv", index=False)
    print(S[["gene", "unit", "n_units", "mean", "ci_low", "ci_high", "positive", "p_signflip",
             "p_sign_test", "p_wilcoxon", "p_signflip_holm3"]].to_string(index=False))

    lopo_rng = L.groupby("gene").agg(min_mean=("mean", "min"), max_mean=("mean", "max"),
                                     max_p=("p_signflip", "max"), all_positive_mean=("mean", lambda v: bool((v > 0).all())))
    loso_rng = LS.groupby("gene").agg(min_mean=("mean", "min"), max_mean=("mean", "max"), max_p=("p_signflip", "max"))
    print("\nLOPO ranges\n", lopo_rng.to_string())
    print("\nLOSO ranges\n", loso_rng.to_string())

    ch = S[S.gene == "CHST11"].set_index("unit")
    write_json(out / "key_results.json", {
        "CHST11_sample": ch.loc["sample"].to_dict(), "CHST11_patient": ch.loc["patient"].to_dict(),
        "lopo": lopo_rng.reset_index().to_dict(orient="records"),
        "loso": loso_rng.reset_index().to_dict(orient="records")})
    provenance(out, __file__, args)
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
