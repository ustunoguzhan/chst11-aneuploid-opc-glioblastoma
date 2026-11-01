#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import (PACKAGE_DATA, bh, compartment_delta_all, describe_effect, load_analysis_set_matrix,
                    log, outdir, provenance, signflip, write_json)

warnings.filterwarnings("ignore")
CSPG_CORE = ["VCAN", "BCAN", "NCAN", "ACAN", "CSPG4", "PTPRZ1", "CSPG5"]
NEG = "Translation (GO:0006412)"
CHUNK = 250


def aucell_deltas(Xn, var, obs, sets):
    import anndata as ad
    import decoupler as dc
    net = pd.concat([pd.DataFrame({"source": k, "target": v}) for k, v in sets.items() if len(v)],
                    ignore_index=True)
    srcs = list(dict.fromkeys(net.source))
    scores = []
    for k in range(0, len(srcs), CHUNK):
        A = ad.AnnData(X=Xn, obs=pd.DataFrame(index=obs.cell.astype(str).values), var=pd.DataFrame(index=list(var)))
        dc.mt.aucell(A, net=net[net.source.isin(srcs[k:k + CHUNK])], verbose=False, tmin=1)
        scores.append(A.obsm["score_aucell"].copy())
    Sc = pd.concat(scores, axis=1)
    Sc.index = obs.cell.values
    g = Sc.assign(gsm=obs.gsm.values, pred=obs.pred.values).groupby(["gsm", "pred"]).mean()
    D = (g.xs("aneuploid", level="pred") - g.xs("diploid", level="pred")).dropna(how="all")
    return Sc, D


def null_sets(det, names, target, B, seed):
    order = np.argsort(det)
    pool = names[order]
    rank = {g: i for i, g in enumerate(pool)}
    tset = set(target)
    rng = np.random.default_rng(seed)
    out = {}
    for b in range(B):
        used = []
        for gname in target:
            r = rank[gname]
            for _ in range(200):
                q = pool[min(len(pool) - 1, max(0, int(rng.normal(r, 60))))]
                if q not in used and q not in tset:
                    break
            used.append(q)
        out[f"NULL{b}"] = used
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = outdir("cs_pathway_coherence")

    log("loading the OPC-like analysis set (all genes)")
    M = load_analysis_set_matrix()
    var, o, Xn = M["var"], M["obs"], M["Xn"]
    names = np.asarray(var)
    det = M["det_pct"] / 100
    present = set(var)
    gs = json.load(open(PACKAGE_DATA / "metabolic_genesets.json", encoding="utf-8"))
    bios = [g for g in gs["kegg_metabolism"]["hsa00532"]["genes"] if g in present]
    core = [g for g in CSPG_CORE if g in present]
    sets = {"CS_DS_biosynthesis": bios,
            "CS_DS_biosynthesis_without_CHST11": [g for g in bios if g != "CHST11"],
            "CSPG_core": core,
            "combined_without_CHST11": sorted(set(bios + core) - {"CHST11"})}
    try:
        import gseapy
        gl = gseapy.get_library("GO_Biological_Process_2023")
        sets["NEG_translation"] = sorted({x.split(",")[0] for x in gl[NEG]} & present)
    except Exception as e:
        log(f"negative-control set unavailable ({type(e).__name__}: {str(e)[:120]}); "
            "needs internet access to Enrichr - continuing without it")

    log("AUCell set scores")
    Sc, D = aucell_deltas(Xn, var, o, sets)
    D.to_csv(out / "per_sample_set_deltas.csv")
    rows = []
    for name in sets:
        r = dict(set=name, n_genes=len(sets[name]))
        r.update(describe_effect(D[name].values, ""))
        rows.append(r)
    R = pd.DataFrame(rows)

    log(f"detection-matched null ({args.n_null} sets per tested set)")
    nulls = {}
    for t in ["CS_DS_biosynthesis", "CS_DS_biosynthesis_without_CHST11", "CSPG_core", "combined_without_CHST11"]:
        ns = null_sets(det, names, sets[t], args.n_null, args.seed)
        _, ND = aucell_deltas(Xn, var, o, ns)
        means = ND.mean(0).to_numpy()
        obs = float(D[t].mean())
        nulls[t] = means
        R.loc[R.set == t, "null_median"] = float(np.median(means))
        R.loc[R.set == t, "null_97_5"] = float(np.percentile(means, 97.5))
        R.loc[R.set == t, "p_matched_null"] = (int((means >= obs).sum()) + 1) / (len(means) + 1)
        pd.DataFrame({"null_mean_delta": means}).to_csv(out / f"null_{t}.csv", index=False)
    R.to_csv(out / "set_level_summary.csv", index=False)
    print(R[["set", "n_genes", "mean", "positive", "n", "p_signflip", "p_matched_null"]].to_string(index=False))

    log("gene-level leading edge")
    allg = sorted(set(bios + core))
    GD = compartment_delta_all(Xn[:, [var.get_loc(g) for g in allg]], o.gsm, o.pred, allg)
    gl_rows = []
    for g in allg:
        m, p, _, _ = signflip(GD[g].values)
        gl_rows.append(dict(gene=g, member=("KEGG_hsa00532" if g in bios else "") + (";CSPG_core" if g in core else ""),
                            detection_pct=float(M["det_pct"][var.get_loc(g)]), delta=m, p_signflip=p,
                            positive=int((GD[g] > 0).sum()), n=len(GD)))
    G = pd.DataFrame(gl_rows)
    G["q_bh"] = bh(G.p_signflip)
    ch = GD["CHST11"] if "CHST11" in GD else None
    if ch is not None:
        G["spearman_with_CHST11_delta"] = [stats.spearmanr(ch, GD[g]).statistic if g != "CHST11" else 1.0 for g in allg]
        G["p_spearman_with_CHST11_delta"] = [stats.spearmanr(ch, GD[g]).pvalue if g != "CHST11" else np.nan for g in allg]
        G["q_spearman_bh"] = bh(G.p_spearman_with_CHST11_delta)
    G = G.sort_values("delta", ascending=False)
    G.to_csv(out / "gene_level_leading_edge.csv", index=False)
    print(G.head(15).to_string(index=False))

    log("single-nucleus co-expression of CHST11 with the CHST11-free proteoglycan score")
    j = var.get_loc("CHST11")
    chv = np.asarray(Xn[:, j].todense()).ravel()
    sc_rows = []
    for (g, p), idx in o.groupby(["gsm", "pred"]).groups.items():
        ii = o.index.get_indexer(idx)
        if len(ii) < 30:
            continue
        for setname in ("CSPG_core", "combined_without_CHST11"):
            y = Sc[setname].to_numpy()[ii]
            x = chv[ii]
            z = M["logdepth"][ii]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            rxy, rxz, ryz = (stats.spearmanr(x, y).statistic, stats.spearmanr(x, z).statistic,
                             stats.spearmanr(y, z).statistic)
            sc_rows.append(dict(gsm=g, compartment=p, set=setname, n=len(ii), rho=rxy,
                                partial_rho_depth=(rxy - rxz * ryz) / np.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))))
    SCC = pd.DataFrame(sc_rows)
    SCC.to_csv(out / "single_nucleus_coexpression_per_sample.csv", index=False)
    scs = []
    for (setname, comp), d in SCC.groupby(["set", "compartment"]):
        for col in ("rho", "partial_rho_depth"):
            r = dict(set=setname, compartment=comp, measure=col)
            r.update(describe_effect(d[col].values, ""))
            scs.append(r)
    SCS = pd.DataFrame(scs)
    SCS.to_csv(out / "single_nucleus_coexpression_summary.csv", index=False)
    print(SCS[["set", "compartment", "measure", "n", "median", "positive", "p_signflip"]].to_string(index=False))

    z = pd.DataFrame({"CHST11_delta": GD["CHST11"], "set_delta": D["combined_without_CHST11"].reindex(GD.index)}).dropna()
    sr = stats.spearmanr(z.CHST11_delta, z.set_delta)
    rng = np.random.default_rng(1)
    perm = [stats.spearmanr(z.CHST11_delta, rng.permutation(z.set_delta)).statistic for _ in range(10000)]
    coh = dict(n=len(z), spearman_r=sr.statistic, p_asymptotic=sr.pvalue,
               p_permutation=(np.sum(np.abs(perm) >= abs(sr.statistic)) + 1) / 10001)
    z.to_csv(out / "CHST11_vs_set_delta.csv")
    print("CHST11 delta vs CHST11-free pathway delta:", coh)

    write_json(out / "key_results.json", {"sets": R.to_dict(orient="records"), "coherence": coh,
                                          "single_nucleus": SCS.to_dict(orient="records")})
    provenance(out, __file__, args)
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
