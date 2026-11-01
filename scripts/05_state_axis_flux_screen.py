import os, sys, json
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PACKAGE_DATA, DATA_DIR, RESULTS_DIR

H5AD = os.environ.get("GBM_STATES_H5AD", os.path.join(DATA_DIR, "gbm_snrna_states.h5ad"))
OUT = os.path.join(RESULTS_DIR, "metabolic_flux")
os.makedirs(OUT, exist_ok=True)
SEED, MIN_CELLS, MIN_EXPR, MIN_SAMPLES = 0, 100, 20, 20
KEGG = {"hsa00514": "Other types of O-glycan biosynthesis",
        "hsa00512": "Mucin type O-glycan biosynthesis",
        "hsa00515": "Mannose type O-glycan biosynthesis",
        "hsa00532": "Glycosaminoglycan biosynthesis - chondroitin sulfate / dermatan sulfate",
        "hsa00534": "Glycosaminoglycan biosynthesis - heparan sulfate / heparin",
        "hsa00531": "Glycosaminoglycan degradation",
        "hsa00604": "Glycosphingolipid biosynthesis - ganglio series"}


def axis_test(values_by_sample, patient_of):
    rhos = {s: r for s, r in values_by_sample.items() if np.isfinite(r)}
    if len(rhos) < MIN_SAMPLES:
        return None, rhos
    r = np.array(list(rhos.values()))
    pc = pd.Series(rhos).groupby(pd.Series({s: patient_of[s] for s in rhos})).mean().values
    return dict(median_rho=float(np.median(r)), positive=int((r > 0).sum()), n_samples=len(r),
                p=stats.wilcoxon(r).pvalue, n_patients=len(pc),
                median_rho_patient=float(np.median(pc)), p_patient=stats.wilcoxon(pc).pvalue), rhos


def fdr(df):
    df["q"] = multipletests(df.p, method="fdr_bh")[1]
    df["q_patient"] = multipletests(df.p_patient, method="fdr_bh")[1]
    return df


A = ad.read_h5ad(H5AD)
gs = json.load(open(os.path.join(PACKAGE_DATA, "metabolic_genesets.json"), encoding="utf-8"))
vs = set(A.var_names)
for k, genes in gs["neftel_metamodules"].items():
    sc.tl.score_genes(A, gene_list=[g for g in genes if g in vs], score_name=f"sg_{k}",
                      ctrl_size=50, n_bins=25, random_state=SEED, use_raw=False)
o = A.obs
o["state_axis"] = (np.maximum(o.sg_OPC, np.maximum(o.sg_NPC1, o.sg_NPC2))
                   - np.maximum(o.sg_AC, np.maximum(o.sg_MES1, o.sg_MES2)))
patient_of = o.groupby("gsm", observed=True).patient.first().astype(str).to_dict()

F = pd.read_csv(os.path.join(PACKAGE_DATA, "scfea_flux_5986_malignant_nuclei.csv.gz"), index_col=0)
cells = o.loc[F.index, ["gsm", "malignant", "state_axis"]]
assert cells.malignant.all() and len(cells) == 5986
info = pd.read_csv(os.path.join(PACKAGE_DATA, "scfea_m168", "Human_M168_information.symbols.csv"),
                   index_col=0)
mg = gs["scfea_m168"]
groups = cells.groupby("gsm", observed=True).groups
rows, per = [], []
for m in F.columns:
    res, rhos = axis_test({s: stats.spearmanr(F.loc[ix, m], cells.loc[ix, "state_axis"]).statistic
                           for s, ix in groups.items()}, patient_of)
    per += [dict(module=m, gsm=s, rho=r) for s, r in rhos.items()]
    if res:
        rows.append(dict(module=m, compound_in=info.loc[m, "Compound_IN_name"],
                         compound_out=info.loc[m, "Compound_OUT_name"],
                         genes=";".join(mg[m]["genes"]), **res))
M = fdr(pd.DataFrame(rows)).sort_values("median_rho", ascending=False)
M.to_csv(os.path.join(OUT, "scfea_module_axis.csv"), index=False)
pd.DataFrame(per).to_csv(os.path.join(OUT, "scfea_module_axis_per_sample.csv"), index=False)

member = {}
for pid, name in KEGG.items():
    for g in gs["kegg_metabolism"][pid]["genes"]:
        member.setdefault(g, []).append(pid)
union = sorted(member)
present = [g for g in union if g in vs]
Mal = A[A.obs.malignant.values, present]
X = Mal.X.toarray() if hasattr(Mal.X, "toarray") else np.asarray(Mal.X)
ax_m = Mal.obs.state_axis.values.astype(float)
gsm = Mal.obs.gsm.astype(str).values
samples = [s for s, n in pd.Series(gsm).value_counts().items() if n >= MIN_CELLS]
masks = {s: gsm == s for s in samples}
rows = []
for j, g in enumerate(present):
    vals = {}
    for s, m in masks.items():
        x = X[m, j]
        if (x > 0).sum() >= MIN_EXPR:
            vals[s] = stats.spearmanr(x, ax_m[m]).statistic
    res, _ = axis_test(vals, patient_of)
    if res:
        rows.append(dict(gene=g, kegg=";".join(member[g]), detection_pct=100 * float((X[:, j] > 0).mean()),
                         **res))
E = fdr(pd.DataFrame(rows))
E["carried_forward"] = (E.q < 0.05) & (E.median_rho > 0) & (E.detection_pct >= 5)
E = E.sort_values("median_rho", ascending=False)
E.to_csv(os.path.join(OUT, "enzyme_axis_screen.csv"), index=False)

tract = pd.read_csv(os.path.join(PACKAGE_DATA, "target_tractability.csv"))
family = set(tract.loc[tract.line == 1, "gene"])
summary = dict(
    n_nuclei_flux=len(cells), n_samples=int(cells.gsm.nunique()),
    nuclei_per_sample=[int(cells.groupby("gsm", observed=True).size().min()),
                       int(cells.groupby("gsm", observed=True).size().max())],
    modules_tested=len(M), modules_q05=int((M.q < 0.05).sum()),
    modules_q05_positive=int(((M.q < 0.05) & (M.median_rho > 0)).sum()),
    kegg_genes=len(union), kegg_genes_present=len(present), enzymes_tested=len(E),
    enzymes_q05_positive=int(((E.q < 0.05) & (E.median_rho > 0)).sum()),
    enzymes_carried_forward=int(E.carried_forward.sum()),
    carried_forward_equals_28_enzyme_family=set(E.gene[E.carried_forward]) == family)
json.dump(summary, open(os.path.join(OUT, "axis_screen_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
