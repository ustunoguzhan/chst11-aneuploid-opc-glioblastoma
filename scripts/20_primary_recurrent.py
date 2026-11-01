#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from common import (GENES3, describe_effect, log, outdir, provenance, signflip, sign_test, write_json)

import importlib
_m19 = importlib.import_module("19_patient_level_robustness")


def cluster_boot(s, n_boot, seed=42):
    w = s.pivot_table(index="patient_pair", columns="progression", values="delta", aggfunc="mean")
    pr = w.get("Primary", pd.Series(np.nan, index=w.index)).to_numpy(float)
    rc = w.get("Recurrent", pd.Series(np.nan, index=w.index)).to_numpy(float)
    rng = np.random.default_rng(seed)
    out = []
    done = 0
    while done < n_boot:
        k = min(10_000, n_boot - done)
        idx = rng.integers(0, len(w), size=(k, len(w)))
        with np.errstate(all="ignore"):
            v = np.nanmean(rc[idx], 1) - np.nanmean(pr[idx], 1)
        out.append(v[np.isfinite(v)])
        done += k
    v = np.concatenate(out)
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


def label_perm(s, n_perm=20_000, seed=7):
    rng = np.random.default_rng(seed)
    s = s.reset_index(drop=True)
    obs = s.loc[s.progression == "Recurrent", "delta"].mean() - s.loc[s.progression == "Primary", "delta"].mean()
    cnt = s.patient_pair.value_counts()
    paired = s[s.patient_pair.isin(cnt[cnt == 2].index)].sort_values(["patient_pair", "progression"])
    single = s[s.patient_pair.isin(cnt[cnt == 1].index)]
    pv = paired.delta.to_numpy().reshape(-1, 2) if len(paired) else np.zeros((0, 2))
    sv, sl = single.delta.to_numpy(), (single.progression == "Recurrent").to_numpy()
    b = 0
    for _ in range(n_perm):
        flip = rng.random(len(pv)) < .5
        rec_p = np.where(flip, pv[:, 0], pv[:, 1])
        pri_p = np.where(flip, pv[:, 1], pv[:, 0])
        lab = rng.permutation(sl)
        rec = np.r_[rec_p, sv[lab]]
        pri = np.r_[pri_p, sv[~lab]]
        if abs(rec.mean() - pri.mean()) >= abs(obs) - 1e-12:
            b += 1
    return float(obs), (b + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes", nargs="+", default=GENES3)
    ap.add_argument("--n-boot", type=int, default=50_000)
    args = ap.parse_args()
    out = outdir("primary_recurrent")
    E = _m19.load_effects(args.genes)

    rows, paired = [], []
    for gene in args.genes:
        s = E[E.gene == gene].copy()
        for prog in ("Primary", "Recurrent"):
            r = dict(gene=gene, group=prog)
            r.update(describe_effect(s.loc[s.progression == prog, "delta"].values, "", n_boot=args.n_boot))
            rows.append(r)
        a, b = s.loc[s.progression == "Primary", "delta"], s.loc[s.progression == "Recurrent", "delta"]
        lo, hi = cluster_boot(s, args.n_boot)
        obs, pperm = label_perm(s)
        rows.append(dict(gene=gene, group="Recurrent_minus_Primary", n=len(s), mean=obs,
                         p_mannwhitney=float(stats.mannwhitneyu(b, a).pvalue) if len(a) and len(b) else np.nan,
                         p_label_permutation=pperm, ci_low=lo, ci_high=hi))
        w = s.pivot_table(index="patient_pair", columns="progression", values="delta").dropna()
        if len(w):
            d = (w["Recurrent"] - w["Primary"]).to_numpy()
            m, p, pmin, _ = signflip(d)
            ps, k, n = sign_test(d)
            paired.append(dict(gene=gene, n_paired_patients=len(d), mean_change=m, increased=k,
                               p_signflip=p, min_attainable_p=pmin, p_sign_test=ps,
                               patients=";".join(w.index)))
    R, PL = pd.DataFrame(rows), pd.DataFrame(paired)
    R.to_csv(out / "primary_recurrent_summary.csv", index=False)
    PL.to_csv(out / "paired_longitudinal_summary.csv", index=False)
    E.to_csv(out / "per_sample_effects_with_progression.csv", index=False)
    print(R.to_string(index=False))
    print(PL.to_string(index=False))

    write_json(out / "key_results.json", {"summary": R.to_dict(orient="records"), "paired": PL.to_dict(orient="records")})
    provenance(out, __file__, args)
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
