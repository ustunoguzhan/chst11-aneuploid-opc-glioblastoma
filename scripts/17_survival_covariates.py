import os, sys, io, json, zipfile, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EXT_DIR, RESULTS_DIR
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
EXT = EXT_DIR
IC = os.path.join(RESULTS_DIR, "independent_cohorts")


def fit(d, gene, covars, T, E):
    d = d[[gene, T, E] + covars].dropna().copy()
    d["z"] = (d[gene] - d[gene].mean()) / d[gene].std(ddof=0)
    X = pd.get_dummies(d[["z", T, E] + covars], drop_first=True, dtype=float)
    cph = CoxPHFitter().fit(X, duration_col=T, event_col=E)
    lo, hi = np.exp(cph.confidence_intervals_.loc["z"].values)
    ph = proportional_hazard_test(cph, X, time_transform="rank").summary
    return dict(n=int(len(d)), n_event=int(d[E].sum()),
                HR_per_SD=round(float(np.exp(cph.params_["z"])), 4),
                CI_low=round(float(lo), 4), CI_high=round(float(hi), 4),
                p=float(cph.summary.loc["z", "p"]),
                ph_p_expression=round(float(ph.loc["z", "p"]), 4),
                ph_min_p_any_term=round(float(ph.p.min()), 4),
                ph_flag=bool(ph.p.min() < 0.05),
                covariates=";".join(c for c in X.columns if c not in ("z", T, E)))


def run(D, genes, models, T, E, cohort):
    rows = []
    for mname, (sub, cov) in models.items():
        part = D[sub] if sub is not None else D
        for g in genes:
            r = fit(part, g, cov, T, E)
            rows.append(dict(cohort=cohort, model=mname, gene=g, **r))
    R = pd.DataFrame(rows)
    R["p_holm"] = R.groupby("model").p.transform(lambda p: multipletests(p, method="holm")[1])
    return R


def main():
    T = pd.read_csv(os.path.join(IC, "tcga_targets_expression.csv")).set_index("patientId")
    cl = pd.DataFrame(json.load(open(os.path.join(EXT, "tcga_patient_clinical.json"))))
    C = cl.pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    T = T.join(C[["AGE", "SEX", "SUBTYPE"]])
    T["age10"] = pd.to_numeric(T.AGE, errors="coerce") / 10
    T["sex"] = T.SEX.str.lower()
    T["idh"] = T.SUBTYPE.map({"GBM_IDHwt": "wt", "GBM_IDHmut-non-codel": "mut"})
    tg = ["CHST11", "XYLT1", "DSEL", "GALNT13", "ST6GALNAC3"]
    for g in tg:
        T[g] = np.log2(T[g].clip(lower=0) + 1)
    RT = run(T, tg, {"univariate": (None, []),
                     "adjusted_age_sex": (None, ["age10", "sex"]),
                     "IDHwt_only_age_sex": (T.idh == "wt", ["age10", "sex"])},
             "OS_MONTHS", "event", "TCGA")
    G = pd.read_csv(os.path.join(IC, "cgga_targets_expression.csv"), index_col=0)
    z = zipfile.ZipFile(os.path.join(EXT, "cgga693_clinical.zip"))
    K = pd.read_csv(z.open("CGGA.mRNAseq_693_clinical.20200506.txt"), sep="\t", index_col=0)
    G = G.join(K[["Age", "Gender", "IDH_mutation_status", "MGMTp_methylation_status"]])
    G["OS"] = pd.to_numeric(G.OS, errors="coerce")
    G["event"] = pd.to_numeric(G.event, errors="coerce")
    G = G.dropna(subset=["OS", "event"])
    G = G[G.OS > 0]
    G["age10"] = pd.to_numeric(G.Age, errors="coerce") / 10
    G["sex"] = G.Gender.str.strip().str.lower()
    G["idh"] = G.IDH_mutation_status.str.strip().str.lower()
    G["mgmt"] = G.MGMTp_methylation_status.str.strip().str.lower()
    G["recurrent"] = G.PRS_type.str.strip()
    cg = ["CHST11", "XYLT1", "DSEL", "GALNT13", "GRM5"]
    RC = run(G, cg, {"univariate": (None, []),
                     "adjusted_age_sex_IDH_MGMT_PRS": (None, ["age10", "sex", "idh", "mgmt", "recurrent"]),
                     "IDHwt_only_age_sex_MGMT_PRS": (G.idh == "wildtype", ["age10", "sex", "mgmt", "recurrent"])},
             "OS", "event", "CGGA")
    R = pd.concat([RT, RC])
    R["p"] = R.p.round(6)
    R["p_holm"] = R.p_holm.round(5)
    R.to_csv(os.path.join(IC, "survival_covariate_adjusted.csv"), index=False)
    ref_hr = {"TCGA": pd.read_csv(os.path.join(IC, "tcga_targets_survival.csv")).set_index("gene").cox_HR_per_SD,
            "CGGA": pd.read_csv(os.path.join(IC, "cgga_targets_survival.csv")).set_index("gene").cox_HR_per_SD}
    u = R[R.model == "univariate"]
    check = {f"{r.cohort}:{r.gene}": bool(abs(r.HR_per_SD - ref_hr[r.cohort][r.gene]) < 5e-4) for r in u.itertuples()}
    nom = R[(R.model != "univariate") & R.gene.isin(["CHST11", "XYLT1", "DSEL"]) & (R.p_holm < 0.05)]
    both = [g for g in ["CHST11", "XYLT1", "DSEL"]
            if set(nom[nom.gene == g].cohort) == {"TCGA", "CGGA"}
            and len(set(np.sign(np.log(nom[nom.gene == g].HR_per_SD)))) == 1]
    res = dict(univariate_reproduces_reference=check,
               idh_counts_tcga=T.idh.value_counts(dropna=False).to_dict(),
               idh_counts_cgga=G.idh.value_counts(dropna=False).to_dict(),
               mgmt_counts_cgga=G.mgmt.value_counts(dropna=False).to_dict(),
               mgmt_in_tcga="not available in cBioPortal PanCan patient clinical data; not imputed",
               nominated_enzymes_significant_both_cohorts=both,
               decision=("no nominated enzyme significant in both cohorts" if not both
                         else "significant in both cohorts: " + ", ".join(both)),
               ph_test="lifelines proportional_hazard_test, rank time transform")
    io.open(os.path.join(IC, "survival_covariate_adjusted.json"), "w", encoding="utf-8").write(
        json.dumps(res, indent=1))
    print(R.to_string(index=False))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
