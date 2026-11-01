import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CGGA_DIR, RESULTS_DIR
import os, io, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")


CGGA = CGGA_DIR
OUT = RESULTS_DIR
GENES = ["CHST11", "XYLT1", "DSEL", "GALNT13", "GRM5"]
EXPR = "CGGA.mRNAseq_693.RSEM-genes.20200506.txt"
CLIN = "CGGA.mRNAseq_693_clinical.20200506.txt"


def main():
    E = pd.read_csv(os.path.join(CGGA, EXPR), sep="\t", index_col=0)
    K = pd.read_csv(os.path.join(CGGA, CLIN), sep="\t", index_col=0)
    print(f"expression {E.shape}, clinical {K.shape}", flush=True)


    K = K[K.Histology.isin(["GBM", "rGBM"]) & K.PRS_type.isin(["Primary", "Recurrent"])]
    ids = [i for i in K.index if i in E.columns]
    K = K.loc[ids]
    print(f"glioblastoma samples with expression: {len(ids)} "
          f"({int((K.PRS_type == 'Primary').sum())} primary, "
          f"{int((K.PRS_type == 'Recurrent').sum())} recurrent)", flush=True)


    present = [g for g in GENES if g in E.index]
    X = np.log2(E.loc[present, ids].astype(float) + 1).T
    X.columns = present
    D = X.join(K[["PRS_type", "OS", "Censor (alive=0; dead=1)"]].rename(
        columns={"Censor (alive=0; dead=1)": "event"}))
    D.to_csv(os.path.join(OUT, "independent_cohorts", "cgga_targets_expression.csv"))


    rows = []
    for g in present:
        a = D.loc[D.PRS_type == "Primary", g].dropna()
        b = D.loc[D.PRS_type == "Recurrent", g].dropna()
        u = stats.mannwhitneyu(b, a)
        rows.append(dict(gene=g, n_primary=len(a), n_recurrent=len(b),
                         median_primary=round(float(a.median()), 3),
                         median_recurrent=round(float(b.median()), 3),
                         log2FC_rec_vs_prim=round(float(b.median() - a.median()), 4),
                         mannwhitney_p=float(u.pvalue)))
    P = pd.DataFrame(rows)
    P["p_holm"] = multipletests(P.mannwhitney_p, method="holm")[1].round(5)
    P["mannwhitney_p"] = P.mannwhitney_p.round(5)
    P.to_csv(os.path.join(OUT, "independent_cohorts", "cgga_targets_primary_recurrent.csv"), index=False)
    print("\nprimary versus recurrent (bulk):", flush=True)
    print(P.to_string(index=False), flush=True)


    S = D.dropna(subset=["OS", "event"]).copy()
    S["OS"] = pd.to_numeric(S.OS, errors="coerce")
    S["event"] = pd.to_numeric(S.event, errors="coerce")
    S = S.dropna(subset=["OS", "event"])
    S = S[S.OS > 0]
    print(f"\nsurvival set: {len(S)} samples, {int(S.event.sum())} events", flush=True)


    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        print("lifelines not available; survival not run", flush=True)
        return


    srows = []
    for g in present:
        d = S[[g, "OS", "event"]].dropna().copy()
        d["z"] = (d[g] - d[g].mean()) / d[g].std(ddof=0)
        cph = CoxPHFitter()
        cph.fit(d[["z", "OS", "event"]], duration_col="OS", event_col="event")
        hr = float(np.exp(cph.params_["z"]))
        lo, hi = np.exp(cph.confidence_intervals_.loc["z"].values)
        p = float(cph.summary.loc["z", "p"])
        try:
            cph.check_assumptions(d[["z", "OS", "event"]], show_plots=False,
                                  advice=False)
            ph_ok = "checked"
        except Exception:
            ph_ok = "not evaluable"
        hi_g = d[d[g] > d[g].median()]
        lo_g = d[d[g] <= d[g].median()]
        lr = logrank_test(hi_g.OS, lo_g.OS, hi_g.event, lo_g.event)
        srows.append(dict(gene=g, n=len(d), n_event=int(d.event.sum()),
                          cox_HR_per_SD=round(hr, 4),
                          cox_CI_low=round(float(lo), 4), cox_CI_high=round(float(hi), 4),
                          cox_p=float(p), median_split_logrank_p=float(lr.p_value),
                          ph_assumption=ph_ok))
    R = pd.DataFrame(srows)
    R["cox_p_holm"] = multipletests(R.cox_p, method="holm")[1].round(5)
    R["cox_p"] = R.cox_p.round(6)
    R["median_split_logrank_p"] = R.median_split_logrank_p.round(6)
    R.to_csv(os.path.join(OUT, "independent_cohorts", "cgga_targets_survival.csv"), index=False)
    print("\noverall survival, Cox on z-scored expression:", flush=True)
    print(R.to_string(index=False), flush=True)
    print(f"\n[OK] three files written to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
