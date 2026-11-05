#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from common import (CACHE_DIR, COUNTS_H5AD, GENES3, PACKAGE_DATA, REF_CELLTYPES, STATES_H5AD,
                    analysis_cells, describe_effect, lognorm_counts, load_cnv_calls, load_states_obs,
                    log, outdir, per_sample_delta, provenance, read_gene_columns, read_rows, require,
                    score_sets_fixed, write_json)

sys.path.insert(0, str(PACKAGE_DATA))
import neftel

META = ["MES1", "MES2", "AC", "OPC", "NPC1", "NPC2", "G1S", "G2M"]


def gene_means(path, chunk=20000):
    import anndata as ad
    import scipy.sparse as sp
    A = ad.read_h5ad(path, backed="r")
    n, g = A.shape
    s = np.zeros(g)
    for i in range(0, n, chunk):
        s += np.asarray(sp.csr_matrix(A.X[i:min(n, i + chunk)]).sum(0)).ravel()
    var = pd.Index(np.asarray(A.var_names))
    A.file.close()
    return s / n, var


def calibrate(Xs, var, mu, sets, n_perm, seed, n_bins=30):
    order = np.argsort(mu, kind="mergesort")
    bin_of = np.empty(len(mu), dtype=np.int32)
    edges = np.linspace(0, len(mu), n_bins + 1).astype(int)
    for b in range(n_bins):
        bin_of[order[edges[b]:edges[b + 1]]] = b
    members = [order[edges[b]:edges[b + 1]] for b in range(n_bins)]
    idx = {g: i for i, g in enumerate(var)}
    rng = np.random.default_rng(seed)
    out = {}
    for name, genes in sets.items():
        cols = [idx[g] for g in genes if g in idx]
        if not cols:
            continue
        own = set(cols)
        pools = {c: np.array([x for x in members[bin_of[c]] if x not in own]) for c in cols}
        rand = {f"R{b}": [var[int(rng.choice(pools[c]))] for c in cols] for b in range(n_perm)}
        vals = []
        keys = list(rand)
        for k in range(0, len(keys), 50):
            sc = score_sets_fixed(Xs, list(var), {q: rand[q] for q in keys[k:k + 50]}, mu, seed=seed + k)
            vals.append(np.concatenate(list(sc.values())))
        v = np.concatenate(vals)
        out[name] = dict(q999=float(np.quantile(v, .999)), q9999=float(np.quantile(v, .9999)),
                         null_sd=float(np.std(v)), n_genes=len(cols), n_perm=n_perm)
        log(f"  {name:<18} q999={out[name]['q999']:.4f}  ({len(cols)} genes)")
    return out


def cluster_calls(obs, call_sets, thr, use_fallback):
    rows = []
    for cl, d in obs.groupby("leiden", observed=True):
        r = {"leiden": cl, "n": len(d)}
        for k in call_sets + META:
            c = f"sc_{k}"
            r[f"med_{k}"] = float(d[c].median()) if c in d else np.nan
        rows.append(r)
    CL = pd.DataFrame(rows).set_index("leiden")

    def call(r):
        hits = []
        for k in call_sets:
            t = thr.get(k, {}).get("q999")
            if t is None:
                if not use_fallback:
                    continue
                t = 3 * np.std([r[f"med_{x}"] for x in META])
            if r[f"med_{k}"] > t:
                hits.append((r[f"med_{k}"], k))
        return "malignant" if not hits else max(hits)[1]
    CL["call"] = CL.apply(call, axis=1)
    return CL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--n-cells", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    require([STATES_H5AD], "threshold_calibration")
    out = outdir("threshold_calibration")

    ct = json.load(open(PACKAGE_DATA / "celltype_genesets.json", encoding="utf-8"))
    reference_thr = json.load(open(PACKAGE_DATA / "nonmalignant_thresholds.json", encoding="utf-8"))
    nm_primary, guard = ct["neftel_primary"], ct["guard_beyond_neftel"]
    call_sets = list(nm_primary) + list(guard)
    sets = {**nm_primary, **guard}

    log("gene means over all nuclei (chunked)")
    mu, var = gene_means(STATES_H5AD)
    import anndata as ad
    S = ad.read_h5ad(STATES_H5AD, backed="r")
    n = S.n_obs
    S.file.close()
    rng = np.random.default_rng(args.seed)
    sub = np.sort(rng.choice(n, size=min(args.n_cells, n), replace=False))
    log(f"reading {len(sub):,} random nuclei for the permutation null")
    Xs, _ = read_rows(STATES_H5AD, sub)
    Xs = Xs.tocsc()
    log(f"calibrating {len(sets)} marker sets x {args.n_perm} permutations")
    cal = calibrate(Xs, var, mu, sets, args.n_perm, args.seed)
    T = pd.DataFrame([dict(set=k, **v, reference_q999=reference_thr.get(k, {}).get("q999"),
                           uses_fallback_in_00c=k not in reference_thr) for k, v in cal.items()])
    T["ratio_to_reference"] = T.q999 / T.reference_q999
    T.to_csv(out / "calibrated_thresholds.csv", index=False)
    write_json(out / "nonmalignant_thresholds_recalibrated.json",
               {k: {kk: v[kk] for kk in ("q999", "q9999", "null_sd")} for k, v in cal.items()})
    print(T.to_string(index=False))

    log("re-deriving cluster calls")
    obs, _ = load_states_obs()
    if "leiden" not in obs:
        raise SystemExit("obs['leiden'] missing in states H5AD")
    CL0 = cluster_calls(obs, call_sets, reference_thr, use_fallback=True)
    CL1 = cluster_calls(obs, call_sets, cal, use_fallback=False)
    rep_ok = float((obs.leiden.map(CL0.call) == obs.cluster_call).mean())
    log(f"stored thresholds reproduce obs.cluster_call for {100 * rep_ok:.2f}% of nuclei")
    CL = CL0[["n", "call"]].rename(columns={"call": "call_primary"}).join(CL1[["call"]].rename(columns={"call": "call_recalibrated"}))
    CL["changed"] = CL.call_primary != CL.call_recalibrated
    CL.to_csv(out / "cluster_calls_primary_vs_recalibrated.csv")
    print(CL[CL.changed].to_string())
    obs["cluster_call_recal"] = obs.leiden.map(CL.call_recalibrated)
    obs["malignant_recal"] = obs.cluster_call_recal == "malignant"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    obs[["cluster_call", "cluster_call_recal"]].to_csv(CACHE_DIR / "cluster_call_recalibrated.csv.gz", compression="gzip")

    trans = pd.crosstab(obs.cluster_call, obs.cluster_call_recal)
    trans.to_csv(out / "nucleus_call_transitions.csv")

    newly = obs.index[obs.malignant_recal & ~obs.malignant]
    obs["neftel_state_recal"] = np.where(obs.malignant_recal, obs.neftel_state, "nonmalignant")
    if len(newly) and all(f"sc_{k}" in obs for k in META):
        pl = neftel.neftel_plane({k: obs.loc[newly, f"sc_{k}"].to_numpy() for k in META})
        obs.loc[newly, "neftel_state_recal"] = pl["state"]

    ic, im, tier1 = load_cnv_calls()
    obs["pred"] = ic.icnv_pred.reindex(obs.index)
    refn = pd.DataFrame({
        "n_reference_primary": obs[obs.cluster_call.isin(REF_CELLTYPES)].groupby("gsm").size(),
        "n_reference_recalibrated": obs[obs.cluster_call_recal.isin(REF_CELLTYPES)].groupby("gsm").size()}).fillna(0).astype(int)
    refn["tier_changes"] = (refn.n_reference_primary >= 20) != (refn.n_reference_recalibrated >= 20)
    refn.to_csv(out / "cnv_reference_size_per_sample.csv")

    m0, g0 = analysis_cells(obs.assign(neftel_state=obs.neftel_state), "pred", samples_allowed=tier1)
    m1, g1 = analysis_cells(obs.assign(neftel_state=obs.neftel_state_recal), "pred", samples_allowed=tier1)
    union = obs.index[m0 | m1]
    C = ad.read_h5ad(COUNTS_H5AD, backed="r")
    row_of = pd.Series(np.arange(C.n_obs), index=C.obs_names)
    C.file.close()
    cnt, genes, tot = read_gene_columns(COUNTS_H5AD, row_of.reindex(union).astype(int).to_numpy(), GENES3, with_totals=True)
    V = pd.DataFrame(lognorm_counts(cnt, tot), index=union, columns=genes)
    res = []
    for lab, m in (("primary", m0), ("recalibrated", m1)):
        idx = obs.index[m]
        D = per_sample_delta(V.loc[idx].to_numpy(), obs.loc[idx, "gsm"], obs.loc[idx, "pred"])
        D.columns = genes
        for g in genes:
            r = dict(gate=lab, gene=g, n_nuclei=int(m.sum()))
            r.update(describe_effect(D[g].values, ""))
            res.append(r)
    RES = pd.DataFrame(res)
    RES.to_csv(out / "CHST11_contrast_primary_vs_recalibrated_gate.csv", index=False)
    print(RES.to_string(index=False))
    write_json(out / "key_results.json", {
        "thresholds": T.to_dict(orient="records"), "reproduction_of_cluster_calls": rep_ok,
        "clusters_changed": int(CL.changed.sum()), "nuclei_changed": int((obs.cluster_call != obs.cluster_call_recal).sum()),
        "opc_analysis_nuclei_primary": int(m0.sum()), "opc_analysis_nuclei_recalibrated": int(m1.sum()),
        "samples_primary": g0, "samples_recalibrated": g1,
        "samples_with_cnv_tier_change": refn.index[refn.tier_changes].tolist(),
        "contrast": RES.to_dict(orient="records")})
    provenance(out, __file__, args)
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
