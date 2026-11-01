import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR
import json, itertools, warnings
import numpy as np
import pandas as pd
import anndata as ad

warnings.filterwarnings("ignore")
MIN = 30


def perm_p(d, seed=42):
    d = np.asarray(d, float)
    obs = d.mean()
    n = len(d)
    if n <= 22:
        a = np.array([(d * np.array(s)).mean()
                      for s in itertools.product([1, -1], repeat=n)])
        return obs, float((np.abs(a) >= abs(obs)).mean()), 2 / len(a), "exact"
    rng = np.random.default_rng(seed)
    a = (rng.choice([1, -1], size=(1_000_000, n)) * d).mean(1)
    b = int((np.abs(a) >= abs(obs)).sum())
    return obs, (b + 1) / 1_000_001, 1 / 1_000_001, "MonteCarlo"


def main():
    S = ad.read_h5ad(os.path.join(DATA_DIR, "gbm_snrna_states.h5ad"), backed="r")
    obs = S.obs[["gsm", "patient", "progression", "neftel_state"]].astype(
        {"gsm": str, "neftel_state": str})
    ic = pd.read_csv(os.path.join(RESULTS_DIR, "cnv",
                                  "infercnvpy_cell_level_calls.csv")).set_index("cell")
    im = pd.read_csv(os.path.join(RESULTS_DIR, "cnv", "infercnvpy_sample_metadata.csv"))
    tier1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"])
    obs["icnv_pred"] = ic.icnv_pred.reindex(obs.index).values
    obs["chr7_minus_chr10"] = ic.chr7_minus_chr10.reindex(obs.index).values
    obs["tier1"] = obs.gsm.isin(tier1)
    print(f"called nuclei {obs.icnv_pred.notna().sum()} of {len(obs)}", flush=True)
    print(f"samples with a within-sample reference {len(tier1)} of {im.gsm.nunique()}", flush=True)

    o = obs[(obs.neftel_state == "OPC") & obs.tier1
            & obs.icnv_pred.isin(["aneuploid", "diploid"])]
    t = o.groupby(["gsm", "icnv_pred"]).size().unstack(fill_value=0)
    ok = t[(t.get("aneuploid", 0) >= MIN) & (t.get("diploid", 0) >= MIN)]
    samples = sorted(ok.index)
    print(f"analysis set {len(samples)} samples", flush=True)

    w = o[o.gsm.isin(samples)]
    g = w.groupby(["gsm", "icnv_pred"]).chr7_minus_chr10.mean().unstack()
    d = (g["aneuploid"] - g["diploid"]).dropna()
    delta, p, minp, branch = perm_p(d.values)
    g.round(4).to_csv(os.path.join(RESULTS_DIR, "expression_programs",
                                   "positive_control_chr7_chr10.csv"))
    print(f"chr7 gain minus chr10 loss: delta {delta:+.6f}  p {p:.8f}  "
          f"{int((d > 0).sum())}/{len(d)} positive", flush=True)

    out = dict(
        n_samples_cohort=int(im.gsm.nunique()),
        n_samples_within_sample_reference=len(tier1),
        n_samples_analysis_set=len(samples),
        n_nuclei_analysis_set=int(len(w)),
        samples=samples,
        positive_control=dict(delta=float(delta), p_perm=float(p), min_p=float(minp),
                              branch=branch, positive=int((d > 0).sum()), n=int(len(d))))
    with open(os.path.join(RESULTS_DIR, "expression_programs",
                           "compartment_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
