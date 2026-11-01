import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR
import gc, itertools, json, warnings
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import decoupler as dc, gseapy
warnings.filterwarnings("ignore")

MIN = 30
SETS = {"primary": "Chemical Synaptic Transmission (GO:0007268)",
        "trans_synaptic_GO0098916": "Anterograde Trans-Synaptic Signaling (GO:0098916)",
        "synaptic_modulation_GO0050804": "Modulation Of Chemical Synaptic Transmission (GO:0050804)",
        "negative_control_translation": "Translation (GO:0006412)"}


def perm_p(d):
    d = np.asarray(d, float)
    obs = d.mean()
    a = np.array([(d * np.array(s)).mean() for s in itertools.product([1, -1], repeat=len(d))])
    return obs, float((np.abs(a) >= abs(obs)).mean()), 2 / len(a)


def main():
    C = ad.read_h5ad(f"{DATA_DIR}/gbm_snrna_counts.h5ad")
    S = ad.read_h5ad(f"{DATA_DIR}/gbm_snrna_states.h5ad", backed="r")
    for c in ["gsm", "neftel_state"]:
        C.obs[c] = S.obs[c].astype(str).values
    ic = pd.read_csv(os.path.join(RESULTS_DIR, "cnv", "infercnvpy_cell_level_calls.csv")).set_index("cell")
    im = pd.read_csv(os.path.join(RESULTS_DIR, "cnv", "infercnvpy_sample_metadata.csv"))
    tier1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"])
    C.obs["pred"] = ic.icnv_pred.reindex(C.obs_names).values
    sel = (C.obs.gsm.isin(tier1) & (C.obs.neftel_state == "OPC")
           & C.obs.pred.isin(["aneuploid", "diploid"])).values
    A = C[sel].copy()
    del C
    gc.collect()
    nn = A.obs.groupby(["gsm", "pred"], observed=True).size()
    keep = {(g, p) for (g, p), v in nn.items() if v >= MIN}
    gsms = sorted({g for g, _ in keep if (g, "aneuploid") in keep and (g, "diploid") in keep})
    A = A[[(g in gsms) and ((g, p) in keep) for g, p in zip(A.obs.gsm, A.obs.pred)]].copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)

    gl = gseapy.get_library("GO_Biological_Process_2023")
    members = {k: sorted({x.split(",")[0] for x in gl[v]} & set(A.var_names)) for k, v in SETS.items()}
    net = pd.concat([pd.DataFrame({"source": k, "target": g}) for k, g in members.items()])
    dc.mt.aucell(A, net=net, verbose=False)
    s = A.obsm["score_aucell"].copy()
    s["gsm"] = A.obs.gsm.values
    s["pred"] = A.obs.pred.values
    g2 = s.groupby(["gsm", "pred"], observed=True).mean()
    delta = (g2.xs("aneuploid", level="pred") - g2.xs("diploid", level="pred")).dropna()

    res = {}
    for k in SETS:
        o, p, mn = perm_p(delta[k].values)
        res[k] = dict(delta=round(float(o), 6), p_perm=round(float(p), 6), min_p=float(mn),
                      positive=int((delta[k] > 0).sum()), n=int(len(delta)), set_size=len(members[k]))
        print(f"{k:<32} delta={o:+.6f}  p={p:.6f}  {int((delta[k] > 0).sum())}/{len(delta)}", flush=True)
    out = dict(delta=res["primary"]["delta"], p_perm_exact=res["primary"]["p_perm"],
               min_p=res["primary"]["min_p"], positive=f"{res['primary']['positive']}/{len(delta)}",
               n_samples=len(gsms), set_size=res["primary"]["set_size"],
               secondary={k: res[k] for k in ["trans_synaptic_GO0098916", "synaptic_modulation_GO0050804"]},
               negative_control_translation=res["negative_control_translation"])
    delta.to_csv(os.path.join(RESULTS_DIR, "expression_programs", "synaptic_program_per_sample.csv"))
    json.dump(out, open(os.path.join(RESULTS_DIR, "expression_programs", "synaptic_program_contrast.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
