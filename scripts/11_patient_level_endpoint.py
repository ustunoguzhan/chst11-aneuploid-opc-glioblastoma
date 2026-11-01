import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PACKAGE_DATA, RESULTS_DIR
import io, json, itertools
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests

sys.stdout.reconfigure(encoding="utf-8")

OUT = RESULTS_DIR
GENES = ["CHST11", "XYLT1", "DSEL"]


def exact_sign_flip(d):
    d = np.asarray(d, float)
    res = permutation_test((d,), np.mean, permutation_type="samples",
                           alternative="two-sided", n_resamples=np.inf)
    return float(d.mean()), float(res.pvalue), 2 / 2 ** len(d)


def local_enumeration(d):
    d = np.asarray(d, float)
    obs = d.mean()
    a = np.array([(d * np.array(s)).mean()
                  for s in itertools.product([1, -1], repeat=len(d))])
    return obs, float((np.abs(a) >= abs(obs)).mean())


def main():
    E = pd.read_csv(os.path.join(OUT, "targets", "per_tumor_expression.csv"))
    C = pd.read_csv(os.path.join(PACKAGE_DATA, "snrna_cohort.csv"))

    E["delta"] = E.aneuploid - E.diploid
    E = E.merge(C[["gsm", "patient_pair", "progression"]], on="gsm", how="left")
    assert E.patient_pair.notna().all(), "sample missing from the cohort table"

    samples = sorted(E.gsm.unique())
    pat = C.loc[C.gsm.isin(samples), ["gsm", "patient_pair", "progression"]]
    counts = pat.patient_pair.value_counts()

    structure = {
        "n_samples": int(len(samples)),
        "n_patients": int(pat.patient_pair.nunique()),
        "patients_contributing_two_samples": int((counts == 2).sum()),
        "patients_contributing_one_sample": int((counts == 1).sum()),
        "n_primary_samples": int((pat.progression == "Primary").sum()),
        "n_recurrent_samples": int((pat.progression == "Recurrent").sum()),
        "paired_patients": sorted(counts[counts == 2].index.tolist()),
    }
    print(f"samples {structure['n_samples']}   patients {structure['n_patients']}   "
          f"patients with both timepoints {structure['patients_contributing_two_samples']}")
    print(f"primary {structure['n_primary_samples']}   "
          f"recurrent {structure['n_recurrent_samples']}")

    rows, res = [], {"analysis_set": structure, "sample_level": {}, "patient_level": {}}

    for unit in ("sample", "patient"):
        print(f"\n--- {unit} as unit of analysis ---")
        p_raw = []
        for g in GENES:
            s = E[E.gene == g]
            d = (s.delta.values if unit == "sample"
                 else s.groupby("patient_pair").delta.mean().values)
            obs, p, minp = exact_sign_flip(d)
            obs_loc, p_loc = local_enumeration(d)
            assert abs(obs - obs_loc) < 1e-12, "mean disagrees between routines"
            assert abs(p - p_loc) < 1e-12, "p disagrees between scipy and local enumeration"
            p_raw.append(p)
            rows.append(dict(unit=unit, gene=g, n=len(d), delta=round(obs, 6),
                             p_perm=round(p, 6), min_attainable_p=minp,
                             positive=int((d > 0).sum())))
            print(f"  {g:<8} n={len(d):<3} delta={obs:+.4f}  p={p:.6f}  "
                  f"{(d > 0).sum()}/{len(d)}  (smallest attainable {minp:.2e})")
        holm = multipletests(p_raw, method="holm")[1]
        for r, ph in zip(rows[-len(GENES):], holm):
            r["p_holm_three_enzymes"] = round(float(ph), 6)
            res[f"{unit}_level"][r["gene"]] = {k: v for k, v in r.items()
                                               if k not in ("unit", "gene")}
        print("  Holm across the three enzymes: " +
              "  ".join(f"{g}={ph:.4f}" for g, ph in zip(GENES, holm)))

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "targets", "patient_level_endpoint.csv"), index=False)

    ch_s = res["sample_level"]["CHST11"]
    ch_p = res["patient_level"]["CHST11"]
    res["verdict"] = (
        "CHST11 retains significance with the patient as the unit of analysis "
        f"(sample level p={ch_s['p_perm']}, patient level p={ch_p['p_perm']})."
        if ch_p["p_perm"] < 0.05 else
        "CHST11 does not retain significance with the patient as the unit of "
        f"analysis (sample level p={ch_s['p_perm']}, patient level "
        f"p={ch_p['p_perm']}).")

    io.open(os.path.join(OUT, "targets", "patient_level_endpoint.json"),
            "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print(f"\n{res['verdict']}")
    print(f"[OK] two files written to {OUT}/targets/")


if __name__ == "__main__":
    main()
