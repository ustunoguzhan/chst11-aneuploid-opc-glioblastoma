import os, sys, io, json, itertools, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR, PACKAGE_DATA
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import decoupler as dc
import gseapy

warnings.filterwarnings("ignore")
MIN, SEED, B = 30, 42, 1000
SYN = "Chemical Synaptic Transmission (GO:0007268)"
NEG = "Translation (GO:0006412)"
CHUNK = 250


def signflip(d):
    d = np.asarray(d, float)
    n = len(d)
    S = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    a = (S * d).mean(1)
    return float(d.mean()), float(np.mean(np.abs(a) >= abs(d.mean()) - 1e-12)), 2.0 / 2 ** n


def load():
    S = ad.read_h5ad(os.path.join(DATA_DIR, "gbm_snrna_states.h5ad"), backed="r")
    obs = S.obs[["gsm", "neftel_state"]].astype(str).copy()
    S.file.close()
    ic = pd.read_csv(os.path.join(RESULTS_DIR, "cnv", "infercnvpy_cell_level_calls.csv")).set_index("cell")
    im = pd.read_csv(os.path.join(RESULTS_DIR, "cnv", "infercnvpy_sample_metadata.csv"))
    tier1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"])
    obs["pred"] = ic.icnv_pred.reindex(obs.index).values
    sel = obs.gsm.isin(tier1) & (obs.neftel_state == "OPC") & obs.pred.isin(["aneuploid", "diploid"])
    nn = obs[sel].groupby(["gsm", "pred"], observed=True).size()
    keep = {(g, p) for (g, p), v in nn.items() if v >= MIN}
    gsms = sorted({g for g, _ in keep if (g, "aneuploid") in keep and (g, "diploid") in keep})
    idx = np.where(sel.values & obs.gsm.isin(gsms).values
                   & np.array([(g, p) in keep for g, p in zip(obs.gsm, obs.pred)]))[0]
    C = ad.read_h5ad(os.path.join(DATA_DIR, "gbm_snrna_counts.h5ad"), backed="r")
    X = sp.csr_matrix(C.X[idx])
    var = pd.Index(np.asarray(C.var_names))
    C.file.close()
    A = ad.AnnData(X, obs=obs.iloc[idx].copy(), var=pd.DataFrame(index=var))
    return A, gsms


def set_deltas(A, net):
    out, per = {}, {}
    srcs = net.source.unique()
    for k in range(0, len(srcs), CHUNK):
        sub = net[net.source.isin(srcs[k:k + CHUNK])]
        B_ = A.copy()
        dc.mt.aucell(B_, net=sub, verbose=False, tmin=1)
        s = B_.obsm["score_aucell"].copy()
        s.index = B_.obs_names
        s["gsm"] = B_.obs.gsm.values
        s["pred"] = B_.obs.pred.values
        g = s.groupby(["gsm", "pred"], observed=True).mean()
        d = (g.xs("aneuploid", level="pred") - g.xs("diploid", level="pred")).dropna()
        for c in d.columns:
            out[c] = float(d[c].mean())
            per[c] = d[c]
        del B_
    return out, per


def summarize(obs, null, label):
    null = np.asarray(null)
    b = int((null >= obs).sum())
    return dict(construction=label, n_sets=int(len(null)),
                null_median=round(float(np.median(null)), 6),
                null_95=round(float(np.percentile(null, 95)), 6),
                null_97_5=round(float(np.percentile(null, 97.5)), 6),
                b_at_or_above=b, p_add_one=round((b + 1) / (len(null) + 1), 6),
                gate_exceeds_95th=bool(obs > np.percentile(null, 95)))


def main():
    A, gsms = load()
    assert len(gsms) == 20 and A.n_obs == 13196, (len(gsms), A.n_obs)
    det_raw = np.asarray((A.X > 0).sum(0)).ravel() / A.n_obs
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    names = A.var_names.to_numpy()
    idx = {g: i for i, g in enumerate(names)}

    gl = gseapy.get_library("GO_Biological_Process_2023")
    syn = sorted({x.split(",")[0] for x in gl[SYN]} & set(names))
    neg = sorted({x.split(",")[0] for x in gl[NEG]} & set(names))
    tract = pd.read_csv(os.path.join(PACKAGE_DATA, "target_tractability.csv"))
    gag = sorted(g for g in tract.loc[tract.line == 1, "gene"].dropna().unique() if g in idx)
    print(f"synaptic {len(syn)}, enzymes {len(gag)}, translation {len(neg)}", flush=True)

    base = pd.concat([pd.DataFrame({"source": "SYN", "target": syn}),
                      pd.DataFrame({"source": "GAG", "target": gag}),
                      pd.DataFrame({"source": "NEG", "target": neg})])
    obs_d, per = set_deltas(A, base)
    checks = dict(synaptic_set_size=len(syn), enzyme_set_size=len(gag),
                  synaptic_observed=round(obs_d["SYN"], 8), enzyme_observed=round(obs_d["GAG"], 8),
                  synaptic_matches_reference=abs(obs_d["SYN"] - 0.00800854517821015) < 1e-6,
                  enzyme_matches_reference=abs(obs_d["GAG"] - 0.010547179648263438) < 1e-6)
    print(checks, flush=True)
    syn_obs, syn_p, syn_min = signflip(per["SYN"].values)

    ranked_det = sorted([(det_raw[i], g) for i, g in enumerate(names) if det_raw[i] > 0])
    dvals = np.array([d for d, _ in ranked_det])
    dgen = [g for _, g in ranked_det]
    syn_set = set(syn)
    syn_det = np.array([det_raw[idx[g]] for g in syn])

    def rank_window_null(rng):
        out = []
        for d0 in syn_det:
            j = int(np.searchsorted(dvals, d0))
            lo, hi = max(0, j - 50), min(len(dgen), j + 50)
            out.append(dgen[rng.integers(lo, hi)])
        return list(dict.fromkeys(out))

    def size_preserving_null(rng):
        out, used = [], set()
        for d0 in syn_det:
            j = int(np.searchsorted(dvals, d0))
            lo, hi = max(0, j - 50), min(len(dgen), j + 50)
            for _ in range(200):
                g = dgen[rng.integers(lo, hi)]
                if g not in used and g not in syn_set:
                    break
            used.add(g)
            out.append(g)
        return out

    rng = np.random.default_rng(SEED)
    netA = pd.concat([pd.DataFrame({"source": f"P{i}", "target": size_preserving_null(rng)}) for i in range(B)])
    rng = np.random.default_rng(SEED)
    netA2 = pd.concat([pd.DataFrame({"source": f"O{i}", "target": rank_window_null(rng)}) for i in range(B)])
    sizes_P = netA.groupby("source").size()
    sizes_O = netA2.groupby("source").size()
    nA, _ = set_deltas(A, pd.concat([netA, netA2]))
    nullP = [nA[f"P{i}"] for i in range(B)]
    nullO = [nA[f"O{i}"] for i in range(B)]
    resA = dict(endpoint="primary synaptic set (GO:0007268), aneuploid minus diploid AUCell",
                observed=round(syn_obs, 8), p_perm_exact=round(syn_p, 6), min_p=syn_min,
                positive=f"{int((per['SYN'] > 0).sum())}/{len(per['SYN'])}",
                primary=dict(**summarize(obs_d["SYN"], nullP, "size-preserving, members excluded, +/-50 detection ranks"),
                             set_size_min=int(sizes_P.min()), set_size_max=int(sizes_P.max())),
                sensitivity=dict(**summarize(obs_d["SYN"], nullO, "rank-window null (duplicates dropped)"),
                                 set_size_min=int(sizes_O.min()), set_size_median=float(sizes_O.median())),
                checks=checks, seed=SEED)
    resA["decision"] = ("gate passes: observed contrast exceeds the 95th percentile of the matched null"
                        if resA["primary"]["gate_exceeds_95th"]
                        else "gate fails: observed contrast does not exceed the 95th percentile of the matched null")
    io.open(os.path.join(RESULTS_DIR, "expression_programs", "synaptic_detection_matched_null_1000.json"),
            "w", encoding="utf-8").write(json.dumps(resA, indent=1))
    print("A", json.dumps({k: resA[k] for k in ["observed", "p_perm_exact", "decision"]}),
          resA["primary"], resA["sensitivity"], flush=True)

    order = np.argsort(det_raw)
    rank = {names[i]: k for k, i in enumerate(order)}
    pool = [names[i] for i in order]
    tgt = [rank[g] for g in gag]
    gag_set = set(gag)
    rng = np.random.default_rng(SEED)
    rel = [pd.DataFrame({"source": f"R{i}",
                         "target": sorted(set(pool[min(len(pool) - 1, max(0, int(rng.normal(r, 60))))] for r in tgt))})
           for i in range(500)]
    rng = np.random.default_rng(SEED)
    sp_ = []
    for i in range(B):
        used = []
        for r in tgt:
            for _ in range(200):
                g = pool[min(len(pool) - 1, max(0, int(rng.normal(r, 60))))]
                if g not in used and g not in gag_set:
                    break
            used.append(g)
        sp_.append(pd.DataFrame({"source": f"S{i}", "target": used}))
    netB = pd.concat(rel + sp_)
    sizes_R = pd.concat(rel).groupby("source").size()
    nB, _ = set_deltas(A, netB)
    nullR = [nB[f"R{i}"] for i in range(500)]
    nullS = [nB[f"S{i}"] for i in range(B)]
    resB = dict(observed=round(obs_d["GAG"], 8),
                primary=summarize(obs_d["GAG"], nullS, "size-preserving, 28 enzymes excluded, rank jitter SD 60"),
                rank_jitter_500=dict(**summarize(obs_d["GAG"], nullR, "rank-jitter null (500 sets, duplicates dropped)"),
                                   set_size_min=int(sizes_R.min()), set_size_median=float(sizes_R.median()),
                                   reference=dict(null_median=0.00784, null_97_5=0.0191, p_add_one=0.331)),
                checks=checks, seed=SEED)
    resB["decision"] = ("matched p >= 0.05: set-level contrast not distinguishable from the matched null"
                        if resB["primary"]["p_add_one"] >= 0.05
                        else "matched p < 0.05: set-level contrast exceeds the matched null")
    io.open(os.path.join(RESULTS_DIR, "targets", "enzyme_set_null_size_preserving.json"),
            "w", encoding="utf-8").write(json.dumps(resB, indent=1))
    print("B", resB["primary"], resB["rank_jitter_500"], resB["decision"], flush=True)


if __name__ == "__main__":
    main()
