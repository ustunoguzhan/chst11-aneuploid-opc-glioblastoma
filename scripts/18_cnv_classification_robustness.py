#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import (CACHE_DIR, COUNTS_H5AD, GENE_POS, GENES3, MIN_REF, REF_CELLTYPES, STATES_H5AD,
                    analysis_cells, depth_adjusted_coef, describe_effect, fmt_p, lognorm_counts,
                    load_cnv_calls, load_cohort, load_states_obs, log, outdir, per_sample_delta,
                    provenance, read_gene_columns, read_rows, require, signflip, write_json)

warnings.filterwarnings("ignore")

RECOMPUTE_REFS = {
    "ref_oligo": lambda o: o.cluster_call == "Oligodendrocytes",
    "ref_microglia": lambda o: o.cluster_call == "Microglia_CG",
    "ref_allnm": lambda o: o.cluster_call != "malignant",
    "ref_primary": lambda o: o.cluster_call.isin(REF_CELLTYPES),
    "ref_recalibrated": lambda o: o.cluster_call_recal.isin(REF_CELLTYPES),
}
RECAL_FILE = CACHE_DIR / "cluster_call_recalibrated.csv.gz"
LABELS = {
    "primary": "Primary calls",
    "p95_rederived": "Primary, re-derived P95",
    "p97_5": "Stricter cut-off P97.5",
    "p99": "Stricter cut-off P99",
    "gap_p90_p99": "High-confidence (<=P90 vs >P99)",
    "hallmark_7_10": "chr7+/chr10- hallmark >P95",
    "concordant": "Score AND hallmark concordant",
    "ref_oligo": "Reference: oligodendrocytes only",
    "ref_microglia": "Reference: microglia only",
    "ref_allnm": "Reference: all non-malignant",
    "ref_primary": "Reference: primary (re-run)",
    "ref_recalibrated": "Reference: recalibrated gate",
}


def derive_from_primary(cells: pd.DataFrame, tier1: set) -> pd.DataFrame:
    out = {k: pd.Series(np.nan, index=cells.index, dtype=object)
           for k in ["p95_rederived", "p97_5", "p99", "gap_p90_p99", "hallmark_7_10", "concordant"]}
    out["primary"] = cells["icnv_pred"].where(cells.gsm.isin(tier1))
    rows = []
    for g, d in cells[cells.gsm.isin(tier1)].groupby("gsm", observed=True):
        ref = d.cluster_call.isin(REF_CELLTYPES)
        if ref.sum() < MIN_REF:
            continue
        s, h = d.icnv_score.to_numpy(float), d.chr7_minus_chr10.to_numpy(float)
        rs, rh = s[ref.to_numpy()], h[ref.to_numpy()]
        q = {p: np.nanpercentile(rs, p) for p in (90, 95, 97.5, 99)}
        h95 = np.nanpercentile(rh, 95)
        idx = d.index
        out["p95_rederived"].loc[idx] = np.where(s > q[95], "aneuploid", "diploid")
        out["p97_5"].loc[idx] = np.where(s > q[97.5], "aneuploid", "diploid")
        out["p99"].loc[idx] = np.where(s > q[99], "aneuploid", "diploid")
        gap = np.full(len(s), "ambiguous", dtype=object)
        gap[s <= q[90]] = "diploid"
        gap[s > q[99]] = "aneuploid"
        out["gap_p90_p99"].loc[idx] = gap
        out["hallmark_7_10"].loc[idx] = np.where(h > h95, "aneuploid", "diploid")
        con = np.full(len(s), "ambiguous", dtype=object)
        con[(s > q[95]) & (h > h95)] = "aneuploid"
        con[(s <= q[95]) & (h <= h95)] = "diploid"
        out["concordant"].loc[idx] = con
        rows.append(dict(gsm=g, n_reference=int(ref.sum()), p90=q[90], p95=q[95], p97_5=q[97.5],
                         p99=q[99], hallmark_p95=h95))
    return pd.DataFrame(out), pd.DataFrame(rows)


def recompute(cells: pd.DataFrame, refs: list[str], n_jobs: int, gp: pd.DataFrame,
              max_samples: int | None = None):
    import anndata as ad
    import infercnvpy as cnv
    import scanpy as sc

    C = ad.read_h5ad(COUNTS_H5AD, backed="r")
    row_of = pd.Series(np.arange(C.n_obs), index=C.obs_names)
    C.file.close()
    calls, qc = [], []
    ocols = ["cluster_call"] + (["cluster_call_recal"] if "cluster_call_recal" in cells else [])
    gsms = sorted(cells.gsm.unique())[:max_samples] if max_samples else sorted(cells.gsm.unique())
    t0 = time.time()
    for i, g in enumerate(gsms, 1):
        d = cells[cells.gsm == g]
        rows = row_of.reindex(d.index).dropna().astype(int).to_numpy()
        X, var = read_rows(COUNTS_H5AD, rows)
        A0 = ad.AnnData(X=X.astype(np.float32), obs=d.loc[row_of.index[rows]][ocols].copy(),
                        var=pd.DataFrame(index=var))
        A0.var = A0.var.join(gp[["chromosome", "start", "end"]])
        A0 = A0[:, A0.var.chromosome.notna()].copy()
        sc.pp.normalize_total(A0, target_sum=1e4)
        sc.pp.log1p(A0)
        for r in refs:
            isref = RECOMPUTE_REFS[r](A0.obs.astype(str)).to_numpy()
            nref = int(isref.sum())
            if nref < MIN_REF:
                qc.append(dict(gsm=g, reference=r, n_reference=nref, state="insufficient_reference"))
                continue
            A = A0.copy()
            A.obs["refclass"] = np.where(isref, "reference", "test")
            try:
                cnv.tl.infercnv(A, reference_key="refclass", reference_cat=["reference"],
                                window_size=100, step=10, exclude_chromosomes=("chrX", "chrY"),
                                n_jobs=n_jobs)
                Xc = A.obsm["X_cnv"]
                score = np.asarray(np.abs(Xc).mean(1)).ravel()
                thr = float(np.percentile(score[isref], 95))
                order = sorted(A.uns["cnv"]["chr_pos"].items(), key=lambda x: x[1])
                bounds = {k: (v, order[j + 1][1] if j + 1 < len(order) else Xc.shape[1])
                          for j, (k, v) in enumerate(order)}

                def cm(c):
                    if c not in bounds:
                        return np.full(Xc.shape[0], np.nan)
                    a, b = bounds[c]
                    return np.asarray(Xc[:, a:b].mean(1)).ravel()
                calls.append(pd.DataFrame(dict(cell=A.obs_names, gsm=g, reference=r,
                                               pred=np.where(score > thr, "aneuploid", "diploid"),
                                               score=score, chr7_minus_chr10=cm("chr7") - cm("chr10"))))
                qc.append(dict(gsm=g, reference=r, n_reference=nref, state="ok", threshold=thr,
                               aneuploid_pct=float(100 * (score > thr).mean())))
            except Exception as e:
                qc.append(dict(gsm=g, reference=r, n_reference=nref, state=f"ERROR: {type(e).__name__}: {e}"))
            del A
            gc.collect()
        del A0, X
        gc.collect()
        el = time.time() - t0
        log(f"[{i}/{len(gsms)}] {g}: {len(d)} nuclei  elapsed {el / 60:.1f} min  "
            f"ETA {el / i * (len(gsms) - i) / 60:.1f} min")
    calls = pd.concat(calls, ignore_index=True) if calls else pd.DataFrame(
        columns=["cell", "gsm", "reference", "pred", "score", "chr7_minus_chr10"])
    return calls, pd.DataFrame(qc)


def evaluate(name, pred: pd.Series, pc: pd.Series, cells, expr, logdepth, cohort, allowed):
    o = cells[["gsm", "neftel_state"]].copy()
    o["pred"] = pred.reindex(o.index)
    mask, gsms = analysis_cells(o, "pred", samples_allowed=allowed)
    mask &= o.index.isin(expr.index)
    if not gsms:
        return [], pd.DataFrame()
    idx = o.index[mask]
    g = o.loc[idx, "gsm"].to_numpy()
    p = o.loc[idx, "pred"].to_numpy()
    E = expr.loc[idx]
    D = per_sample_delta(E.to_numpy(), g, p)
    D.columns = list(E.columns)
    D["chr7_minus_chr10"] = per_sample_delta(pc.reindex(idx).to_numpy(float), g, p)[0]
    an = (p == "aneuploid").astype(float)
    ld = logdepth.loc[idx].to_numpy(float)
    adj = {gene: depth_adjusted_coef(E[gene].to_numpy(float), an, ld, g) for gene in E.columns}
    D = D.reset_index().rename(columns={"index": "gsm"})
    if "gsm" not in D.columns:
        D = D.rename(columns={D.columns[0]: "gsm"})
    n_ct = pd.crosstab(g, p)
    D["n_aneuploid"] = D.gsm.map(n_ct.get("aneuploid", pd.Series(dtype=int))).astype(int)
    D["n_diploid"] = D.gsm.map(n_ct.get("diploid", pd.Series(dtype=int))).astype(int)
    D = D.merge(cohort, on="gsm", how="left")
    D.insert(0, "scheme", name)
    rows = []
    for gene in list(E.columns) + ["chr7_minus_chr10"]:
        d = D.set_index("gsm")[gene]
        pat = D.groupby("patient_pair")[gene].mean()
        r = dict(scheme=name, label=LABELS.get(name, name), gene=gene, n_samples=len(d),
                 n_patients=int(D.patient_pair.nunique()), n_nuclei=int(mask.sum()))
        r.update(describe_effect(d.values, "sample_"))
        r.update(describe_effect(pat.values, "patient_"))
        if gene in adj:
            a = adj[gene].reindex(d.index)
            am, ap, _, _ = signflip(a.values)
            apm, app, _, _ = signflip(a.groupby(D.set_index("gsm").patient_pair).mean().values)
            r.update(depth_adj_mean=am, depth_adj_p=ap, depth_adj_patient_mean=apm,
                     depth_adj_patient_p=app, depth_adj_positive=int((a > 0).sum()))
        rows.append(r)
    return rows, D


def threshold_free(cells, expr, logdepth, score_col, allowed, min_cells=60):
    rows = []
    o = cells[(cells.neftel_state == "OPC") & cells.gsm.isin(allowed)]
    o = o[o.index.isin(expr.index) & o[score_col].notna()]
    for g, d in o.groupby("gsm", observed=True):
        if len(d) < min_cells:
            continue
        x = d[score_col].to_numpy(float)
        z = logdepth.loc[d.index].to_numpy(float)
        rec = dict(gsm=g, n=len(d))
        for gene in expr.columns:
            y = expr.loc[d.index, gene].to_numpy(float)
            if np.std(y) == 0:
                continue
            rxy = stats.spearmanr(x, y).statistic
            rxz = stats.spearmanr(x, z).statistic
            ryz = stats.spearmanr(y, z).statistic
            den = np.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
            rec[f"{gene}_rho"] = rxy
            rec[f"{gene}_partial_rho_depth"] = (rxy - rxz * ryz) / den
        rec["score_vs_depth_rho"] = stats.spearmanr(x, z).statistic
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--references", default="ref_oligo,ref_microglia,ref_allnm,ref_primary,ref_recalibrated",
                    help="comma list of re-run references, or 'none' to skip the inferCNV re-run. "
                         "ref_recalibrated is used only if script 23 has been run.")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--reuse-cache", action="store_true",
                    help="reuse previously re-computed calls (cnv_recomputed_calls.csv.gz) if present")
    ap.add_argument("--max-samples", type=int, default=None, help="debug: limit re-run to N samples")
    args = ap.parse_args()
    require([COUNTS_H5AD, STATES_H5AD, GENE_POS], "18_cnv_classification_robustness")
    out = outdir("cnv_classification")

    log("loading obs and stored CNV calls")
    obs, _ = load_states_obs(["gsm", "patient", "progression", "neftel_state", "cluster_call",
                              "malignant", "total_counts"])
    ic, im, tier1 = load_cnv_calls()
    cells = obs.join(ic[["icnv_pred", "icnv_score", "chr7_minus_chr10"]], how="left")
    cohort = load_cohort()

    A, thr = derive_from_primary(cells, tier1)
    thr.to_csv(out / "primary_score_thresholds_per_sample.csv", index=False)
    agree = (A["p95_rederived"].dropna() == A["primary"].reindex(A["p95_rederived"].dropna().index)).mean()
    log(f"re-derived P95 calls agree with stored calls for {100 * agree:.2f}% of nuclei")

    refs = [] if args.references.strip().lower() == "none" else \
        [r.strip() for r in args.references.split(",") if r.strip()]
    if "ref_recalibrated" in refs:
        if RECAL_FILE.exists():
            rc = pd.read_csv(RECAL_FILE, index_col=0)
            cells["cluster_call_recal"] = rc.cluster_call_recal.reindex(cells.index).astype(str)
            if (cells.cluster_call_recal == cells.cluster_call).all():
                log("recalibrated gate identical to the primary gate; dropping ref_recalibrated")
                refs.remove("ref_recalibrated")
        else:
            log("script 23 not run (no recalibrated calls); dropping ref_recalibrated")
            refs.remove("ref_recalibrated")
    bad = [r for r in refs if r not in RECOMPUTE_REFS]
    if bad:
        raise SystemExit(f"unknown references {bad}; choose from {list(RECOMPUTE_REFS)}")
    rc_path = out / "cnv_recomputed_calls.csv.gz"
    recalls, rqc = pd.DataFrame(), pd.DataFrame()
    if refs:
        if args.reuse_cache and rc_path.exists():
            log(f"reusing {rc_path}")
            recalls = pd.read_csv(rc_path)
            rqc = pd.read_csv(out / "cnv_recomputed_sample_qc.csv")
            refs = [r for r in refs if r in set(recalls.reference)]
        else:
            gp = pd.read_csv(GENE_POS, index_col=0)
            log(f"re-running infercnvpy for references {refs} (n_jobs={args.n_jobs}); this is the slow step")
            recalls, rqc = recompute(cells, refs, args.n_jobs, gp, args.max_samples)
            recalls.to_csv(rc_path, index=False, compression="gzip")
            rqc.to_csv(out / "cnv_recomputed_sample_qc.csv", index=False)

    log("reading CHST11/XYLT1/DSEL counts of all OPC-like nuclei")
    import anndata as ad
    C = ad.read_h5ad(COUNTS_H5AD, backed="r")
    row_of = pd.Series(np.arange(C.n_obs), index=C.obs_names)
    C.file.close()
    opc_idx = cells.index[(cells.neftel_state == "OPC").to_numpy()]
    rows = row_of.reindex(opc_idx).dropna().astype(int).to_numpy()
    cnt, genes, tot = read_gene_columns(COUNTS_H5AD, rows, GENES3, with_totals=True)
    names = row_of.index[rows]
    expr = pd.DataFrame(lognorm_counts(cnt, tot), index=names, columns=genes)
    logdepth = pd.Series(np.log10(np.maximum(tot, 1)), index=names)

    schemes = {k: (A[k], cells.chr7_minus_chr10, tier1) for k in
               ["primary", "p95_rederived", "p97_5", "p99", "gap_p90_p99", "hallmark_7_10", "concordant"]}
    for r in refs:
        x = recalls[recalls.reference == r].set_index("cell")
        ok = set(rqc.loc[(rqc.reference == r) & (rqc.state == "ok"), "gsm"].astype(str))
        schemes[r] = (x.pred, x.chr7_minus_chr10, ok)
    summary, per = [], []
    for name, (pred, pc, allowed) in schemes.items():
        rws, D = evaluate(name, pred, pc, cells, expr, logdepth, cohort, allowed)
        summary += rws
        if len(D):
            per.append(D)
        ch = [r for r in rws if r["gene"] == "CHST11"]
        if ch:
            c = ch[0]
            log(f"{name:<15} n={c['n_samples']:2d} samples/{c['n_patients']} patients  "
                f"CHST11 d={c['sample_mean']:+.3f} P={fmt_p(c['sample_p_signflip'])} "
                f"{c['sample_positive']}/{c['n_samples']} | patient P={fmt_p(c['patient_p_signflip'])} "
                f"| depth-adj {c.get('depth_adj_mean', np.nan):+.3f} P={fmt_p(c.get('depth_adj_p', np.nan))}")
    S = pd.DataFrame(summary)
    P = pd.concat(per, ignore_index=True) if per else pd.DataFrame()
    base = set(P.loc[P.scheme == "primary", "gsm"]) if len(P) else set()
    if len(P):
        ov = P.groupby("scheme").gsm.apply(lambda s: len(set(s) & base) / max(1, len(set(s) | base)))
        S["sample_jaccard_vs_primary"] = S.scheme.map(ov)
    opc = cells.index[cells.neftel_state == "OPC"]
    agr = {}
    for name, (pred, _, _) in schemes.items():
        a = pred.reindex(opc)
        b = cells.loc[opc, "icnv_pred"]
        ok = a.isin(["aneuploid", "diploid"]) & b.isin(["aneuploid", "diploid"])
        agr[name] = float((a[ok] == b[ok]).mean()) if ok.any() else np.nan
    S["opc_nucleus_agreement_with_primary"] = S.scheme.map(agr)
    S.to_csv(out / "cnv_robustness_summary.csv", index=False)
    P.to_csv(out / "cnv_robustness_per_sample.csv", index=False)

    log("threshold-free dose-response (Spearman CNV score vs expression within OPC-like nuclei)")
    TF = threshold_free(cells, expr, logdepth, "icnv_score", tier1)
    TF = TF.merge(cohort, on="gsm", how="left")
    TF.to_csv(out / "threshold_free_per_sample.csv", index=False)
    tf_rows = []
    for col in [c for c in TF.columns if c.endswith("_rho") or c.endswith("_depth")]:
        if col == "score_vs_depth_rho":
            continue
        r = dict(measure=col)
        r.update(describe_effect(TF[col].values, "sample_"))
        r.update(describe_effect(TF.groupby("patient_pair")[col].mean().values, "patient_"))
        tf_rows.append(r)
    TFS = pd.DataFrame(tf_rows)
    TFS.to_csv(out / "threshold_free_summary.csv", index=False)
    print(TFS[["measure", "sample_n", "sample_median", "sample_positive", "sample_p_signflip",
               "patient_p_signflip"]].to_string(index=False))

    ch = S[S.gene == "CHST11"].reset_index(drop=True)


    write_json(out / "key_results.json", {
        "p95_rederived_agreement_with_primary": agree,
        "CHST11": ch[["scheme", "n_samples", "n_patients", "sample_mean", "sample_positive",
                      "sample_p_signflip", "patient_mean", "patient_p_signflip",
                      "depth_adj_mean", "depth_adj_p"]].to_dict(orient="records"),
        "threshold_free": TFS.to_dict(orient="records"),
        "recompute_qc": rqc.groupby(["reference", "state"]).size().reset_index(name="n").to_dict(orient="records")
        if len(rqc) else []})
    provenance(out, __file__, args)
    print(ch[["scheme", "n_samples", "sample_mean", "sample_positive", "sample_p_signflip",
              "patient_p_signflip", "depth_adj_mean", "depth_adj_p"]].to_string(index=False))
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
