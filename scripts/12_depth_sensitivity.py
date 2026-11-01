import os, sys, io, json, itertools, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR, PACKAGE_DATA
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

warnings.filterwarnings("ignore")
MIN, TOL, N_MATCH, SEED = 30, 2.0, 500, 42
GENES = ["CHST11", "XYLT1", "DSEL"]
REF = ["Microglia_CG", "Oligodendrocytes"]


def signflip(d):
    d = np.asarray(d, float)
    n = len(d)
    S = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    a = (S * d).mean(1)
    obs = d.mean()
    return float(obs), float(np.mean(np.abs(a) >= abs(obs) - 1e-12)), 2.0 / 2 ** n


def main():
    S = ad.read_h5ad(os.path.join(DATA_DIR, "gbm_snrna_states.h5ad"), backed="r")
    obs = S.obs[["gsm", "neftel_state", "cluster_call"]].astype(str).copy()
    S.file.close()
    ic = pd.read_csv(os.path.join(RESULTS_DIR, "cnv", "infercnvpy_cell_level_calls.csv")).set_index("cell")
    im = pd.read_csv(os.path.join(RESULTS_DIR, "cnv", "infercnvpy_sample_metadata.csv"))
    tier1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"])
    obs["pred"] = ic.icnv_pred.reindex(obs.index).values

    sel = obs.gsm.isin(tier1) & (obs.neftel_state == "OPC") & obs.pred.isin(["aneuploid", "diploid"])
    nn = obs[sel].groupby(["gsm", "pred"], observed=True).size()
    keep = {(g, p) for (g, p), v in nn.items() if v >= MIN}
    gsms = sorted({g for g, _ in keep if (g, "aneuploid") in keep and (g, "diploid") in keep})
    refc = obs[obs.gsm.isin(gsms) & obs.cluster_call.isin(REF)].cluster_call.value_counts().to_dict()

    idx = np.where(sel & obs.gsm.isin(gsms).values
                   & np.array([(g, p) in keep for g, p in zip(obs.gsm, obs.pred)]))[0]
    C = ad.read_h5ad(os.path.join(DATA_DIR, "gbm_snrna_counts.h5ad"), backed="r")
    X = sp.csr_matrix(C.X[idx])
    var = np.asarray(C.var_names)
    C.file.close()
    o = obs.iloc[idx].reset_index(drop=True)
    print(f"analysis set: {X.shape[0]} nuclei, {len(gsms)} samples", flush=True)

    tot = np.asarray(X.sum(1)).ravel()
    det = np.asarray((X > 0).sum(0)).ravel() / X.shape[0] * 100
    Xn = sp.diags(1e4 / tot) @ X
    Xn.data = np.log1p(Xn.data)
    Xn = Xn.tocsc()
    an = (o.pred == "aneuploid").values.astype(float)
    ld = np.log10(tot)
    groups = {g: np.where(o.gsm.values == g)[0] for g in gsms}
    coh = pd.read_csv(os.path.join(PACKAGE_DATA, "snrna_cohort.csv")).set_index("gsm")
    pat = coh.loc[gsms, "patient_pair"].values

    depth = pd.DataFrame({g: [np.median(tot[i][an[i] == 1]), np.median(tot[i][an[i] == 0])]
                          for g, i in groups.items()}, index=["aneuploid", "diploid"]).T
    ldr = np.log2(depth.aneuploid / depth.diploid)
    d_obs, d_p, d_min = signflip(ldr.values)

    def coefs(cols):
        Y = Xn[:, cols].toarray()
        raw = np.zeros((len(gsms), len(cols)))
        adj = np.zeros_like(raw)
        for k, g in enumerate(gsms):
            i = groups[g]
            y, a = Y[i], an[i]
            raw[k] = y[a == 1].mean(0) - y[a == 0].mean(0)
            D = np.column_stack([np.ones(len(i)), a, ld[i]])
            adj[k] = np.linalg.lstsq(D, y, rcond=None)[0][1]
        return raw, adj

    def patient_mean(v):
        return pd.Series(v, index=pat).groupby(level=0).mean().values

    rng = np.random.default_rng(SEED)
    rows = []
    for gname in GENES:
        j = int(np.where(var == gname)[0][0])
        raw, adj = coefs([j])
        r_obs, r_p, _ = signflip(raw[:, 0])
        a_obs, a_p, a_min = signflip(adj[:, 0])
        pa_obs, pa_p, pa_min = signflip(patient_mean(adj[:, 0]))
        pool = np.where((np.abs(det - det[j]) <= TOL) & (np.arange(len(det)) != j))[0]
        if len(pool) < 50:
            pool = np.where((np.abs(det - det[j]) <= TOL * 3) & (np.arange(len(det)) != j))[0]
        pick = rng.choice(pool, size=min(N_MATCH, len(pool)), replace=False)
        nraw, nadj = coefs(list(pick))
        null_raw, null_adj = nraw.mean(0), nadj.mean(0)
        rows.append(dict(
            gene=gname, n_samples=len(gsms), detection_pct=round(float(det[j]), 2),
            delta_raw=round(r_obs, 4), p_raw=round(r_p, 6),
            coef_depth_adjusted=round(a_obs, 4), p_depth_adjusted=round(a_p, 6),
            positive_depth_adjusted=f"{int((adj[:, 0] > 0).sum())}/{len(gsms)}",
            coef_depth_adjusted_patient=round(pa_obs, 4), p_depth_adjusted_patient=round(pa_p, 6),
            n_patients=int(len(set(pat))),
            matched_pool_size=int(len(pool)), n_matched_genes=int(len(pick)),
            null_raw_median=round(float(np.median(null_raw)), 4),
            p_matched_raw=round(float((np.sum(null_raw >= r_obs) + 1) / (len(pick) + 1)), 5),
            null_adj_median=round(float(np.median(null_adj)), 4),
            null_adj_97_5=round(float(np.percentile(null_adj, 97.5)), 4),
            p_matched_depth_adjusted=round(float((np.sum(null_adj >= a_obs) + 1) / (len(pick) + 1)), 5)))
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULTS_DIR, "targets"), exist_ok=True)
    T.to_csv(os.path.join(RESULTS_DIR, "targets", "depth_sensitivity.csv"), index=False)
    res = dict(
        n_samples=len(gsms), n_nuclei=int(X.shape[0]), samples=gsms,
        depth_median_counts=depth.round(1).to_dict(orient="index"),
        depth_log2_ratio_aneuploid_vs_diploid=dict(mean=round(d_obs, 4), p_perm=round(d_p, 6),
                                                   min_p=d_min, positive=int((ldr > 0).sum())),
        reference_nuclei_in_analysis_samples=refc,
        model="per-sample OLS log1p(CP10k) ~ 1 + aneuploid + log10(total counts); "
              "exact two-sided sign-flip over samples/patients; matched null one-sided, add-one",
        genes=rows)
    io.open(os.path.join(RESULTS_DIR, "targets", "depth_sensitivity.json"), "w", encoding="utf-8").write(
        json.dumps(res, indent=1))
    print("[OK] depth_sensitivity written", flush=True)


if __name__ == "__main__":
    main()
