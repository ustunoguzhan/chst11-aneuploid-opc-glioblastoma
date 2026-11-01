import os, sys, json, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.stdout.reconfigure(encoding="utf-8")
PDC = "https://pdc.cancer.gov/graphql"
STUDY = "PDC000204"
CACHE = os.path.join(DATA_DIR, "external_cache")
OUT = os.path.join(RESULTS_DIR, "independent_cohorts")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
PARTNERS = ["CSPG4", "VCAN", "BCAN"]


def graphql(query, cache_name):
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    url = PDC + "?" + urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}),
                                timeout=600) as r:
        data = json.loads(r.read())["data"]
    json.dump(data, open(path, "w", encoding="utf-8"))
    return data


bio = graphql('{ biospecimenPerStudy(pdc_study_id: "%s" acceptDUA: true) '
              '{ aliquot_id aliquot_submitter_id case_submitter_id sample_type aliquot_is_ref } }' % STUDY,
              f"pdc_{STUDY}_biospecimen.json")["biospecimenPerStudy"]
B = pd.DataFrame(bio).drop_duplicates("aliquot_submitter_id").set_index("aliquot_submitter_id")
mat = graphql('{ quantDataMatrix(pdc_study_id: "%s" data_type: "log2_ratio" acceptDUA: true) }' % STUDY,
              f"pdc_{STUDY}_log2_ratio.json")["quantDataMatrix"]
D = pd.DataFrame(mat[1:], columns=mat[0]).set_index("Gene/Aliquot")
D.columns = [c.split(":")[1] for c in D.columns]
D = D.apply(pd.to_numeric, errors="coerce")
stype = B.sample_type.reindex(D.columns)
assert stype.notna().all(), "aliquots without biospecimen record"
tum = (stype == "Primary Tumor").to_numpy()
nor = (stype == "Solid Tissue Normal").to_numpy()
cases = B.case_submitter_id.reindex(D.columns)
assert not cases[tum].duplicated().any() and not cases[nor].duplicated().any(), "more than one aliquot per case"
print(f"{STUDY}: {D.shape[0]:,} genes x {D.shape[1]} aliquots | primary tumor {tum.sum()}, "
      f"solid-tissue normal {nor.sum()}", flush=True)

ch = D.loc["CHST11"]
t, n = ch[tum].dropna(), ch[nor].dropna()
mw = stats.mannwhitneyu(t, n, alternative="two-sided")
res = {"study": STUDY, "source": "PDC Common Data Analysis Pipeline, gene-level log2 ratio",
       "aliquots": int(D.shape[1]), "primary_tumor": int(tum.sum()), "solid_tissue_normal": int(nor.sum()),
       "chst11_quantified": int(ch.notna().sum()),
       "tumor_vs_normal": {"n_tumor": int(len(t)), "n_normal": int(len(n)),
                           "median_tumor": float(t.median()), "median_normal": float(n.median()),
                           "difference_in_medians": float(t.median() - n.median()),
                           "difference_in_means": float(t.mean() - n.mean()),
                           "mann_whitney_U": float(mw.statistic), "p_two_sided": float(mw.pvalue)},
       "tumor_correlations": []}
for g in PARTNERS:
    x = pd.concat([ch, D.loc[g]], axis=1)[tum].dropna()
    r, p = stats.spearmanr(x.iloc[:, 0], x.iloc[:, 1])
    res["tumor_correlations"].append({"gene": g, "n": int(len(x)), "spearman_r": float(r), "p": float(p)})
holm = multipletests([c["p"] for c in res["tumor_correlations"]], method="holm")[1]
for c, h in zip(res["tumor_correlations"], holm):
    c["p_holm"] = float(h)

tv = res["tumor_vs_normal"]
print(f"CHST11 quantified in {res['chst11_quantified']}/{res['aliquots']} aliquots "
      f"(tumor {tv['n_tumor']}, normal {tv['n_normal']})")
print(f"tumor - normal: difference in medians {tv['difference_in_medians']:+.3f}, in means "
      f"{tv['difference_in_means']:+.3f}; Mann-Whitney P = {tv['p_two_sided']:.4g}")
for c in res["tumor_correlations"]:
    print(f"CHST11 vs {c['gene']} in tumors: Spearman r = {c['spearman_r']:.3f}, P = {c['p']:.3g}, "
          f"Holm P = {c['p_holm']:.3g}, n = {c['n']}")

json.dump(res, open(os.path.join(OUT, "cptac_chst11_proteome.json"), "w", encoding="utf-8"), indent=1)
V = pd.DataFrame({"case": cases, "sample_type": stype}).join(D.loc[["CHST11"] + PARTNERS].T)
V.index.name = "aliquot"
V.to_csv(os.path.join(OUT, "cptac_chst11_proteome_values.csv"))
print("wrote results/independent_cohorts/cptac_chst11_proteome.json and _values.csv", flush=True)
