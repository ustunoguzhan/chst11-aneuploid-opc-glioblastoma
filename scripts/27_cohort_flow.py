#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from common import (RESULTS_DIR, CNV_META, COHORT_CSV, DATA_DIR, MIN_CELLS, MIN_REF, STATES_H5AD,
                    CNV_CALLS, log, outdir, provenance, write_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args()
    out = outdir("cohort_flow")
    coh = pd.read_csv(COHORT_CSV)
    im = pd.read_csv(CNV_META)
    T = coh[["gsm", "sample", "patient_pair", "progression"]].copy()
    T["in_processed_cohort"] = T.gsm.isin(im.gsm)
    T = T.merge(im[["gsm", "n_cells", "n_reference", "tier", "state"]], on="gsm", how="left")
    T["cnv_reference_ok"] = (T.tier == 1) & (T.state == "ok")
    extra = {}

    qcw = DATA_DIR / "qc_waterfall.csv"
    if qcw.exists():
        extra["qc_waterfall"] = pd.read_csv(qcw).to_dict(orient="records")
    qch = DATA_DIR / "gbm_snrna_qc.h5ad"
    if qch.exists():
        try:
            import anndata as ad
            Q = ad.read_h5ad(qch, backed="r")
            extra["qc_uns"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in dict(Q.uns.get("qc", {})).items()}
            Q.file.close()
        except Exception as e:
            extra["qc_uns_error"] = str(e)
    raw = DATA_DIR / "gbm_snrna_raw.h5ad"
    if raw.exists():
        try:
            import anndata as ad
            R = ad.read_h5ad(raw, backed="r")
            n = R.obs.groupby(R.obs.gsm.astype(str), observed=True).size()
            T["n_nuclei_raw_ge200genes"] = T.gsm.map(n)
            if "pct_counts_mt" in R.obs:
                mt = R.obs[R.obs.pct_counts_mt < 5].groupby(R.obs.gsm.astype(str), observed=True).size()
                T["n_nuclei_after_mt_filter"] = T.gsm.map(mt)
            R.file.close()
        except Exception as e:
            extra["raw_error"] = str(e)

    if STATES_H5AD.exists():
        import anndata as ad
        S = ad.read_h5ad(STATES_H5AD, backed="r")
        obs = S.obs[["gsm", "neftel_state", "malignant"]].copy()
        S.file.close()
        obs["gsm"] = obs.gsm.astype(str)
        ic = pd.read_csv(CNV_CALLS, usecols=["cell", "icnv_pred"]).set_index("cell")
        obs["pred"] = ic.icnv_pred.reindex(obs.index)
        opc = obs[obs.neftel_state.astype(str) == "OPC"]
        ct = opc.groupby(["gsm", "pred"], observed=True).size().unstack(fill_value=0)
        T["n_malignant"] = T.gsm.map(obs.groupby("gsm").malignant.apply(lambda s: s.astype(str).str.lower().isin(["true", "1"]).sum()))
        T["n_opc_like"] = T.gsm.map(opc.groupby("gsm").size())
        T["n_opc_aneuploid"] = T.gsm.map(ct.get("aneuploid", pd.Series(dtype=int)))
        T["n_opc_diploid"] = T.gsm.map(ct.get("diploid", pd.Series(dtype=int)))
        for c in ("n_opc_like", "n_opc_aneuploid", "n_opc_diploid"):
            T[c] = T[c].fillna(0).astype(int)

    def reason(r):
        if not r.in_processed_cohort:
            return "removed in QC (00b: <100 nuclei after filtering or pairing broken)"
        if not r.cnv_reference_ok:
            return f"no within-sample CNV reference (<{MIN_REF} reference nuclei)"
        if "n_opc_aneuploid" in r and pd.notna(r.get("n_opc_aneuploid")):
            if r.n_opc_aneuploid < MIN_CELLS and r.n_opc_diploid < MIN_CELLS:
                return f"<{MIN_CELLS} aneuploid and <{MIN_CELLS} diploid OPC-like nuclei"
            if r.n_opc_aneuploid < MIN_CELLS:
                return f"<{MIN_CELLS} aneuploid OPC-like nuclei"
            if r.n_opc_diploid < MIN_CELLS:
                return f"<{MIN_CELLS} diploid OPC-like nuclei"
            return "in analysis set"
        if r.gsm in primary_set:
            return "in analysis set"
        return f"<{MIN_CELLS} OPC-like nuclei in one compartment (details need gbm_snrna_states.h5ad)"
    pte = RESULTS_DIR / "targets" / "per_tumor_expression.csv"
    primary_set = set(pd.read_csv(pte).gsm) if pte.exists() else set()
    T["status"] = T.apply(reason, axis=1)
    T.to_csv(out / "per_sample_accounting.csv", index=False)

    flow = [("Deposited (GSE174554, snrna_cohort.csv)", len(T), T.patient_pair.nunique())]
    a = T[T.in_processed_cohort]
    flow.append(("After QC and pairing (processed cohort)", len(a), a.patient_pair.nunique()))
    b = a[a.cnv_reference_ok]
    flow.append((f"Within-sample CNV reference (>={MIN_REF} nuclei)", len(b), b.patient_pair.nunique()))
    if "n_opc_aneuploid" in T or primary_set:
        c = T[T.status == "in analysis set"]
        flow.append((f">={MIN_CELLS} OPC-like nuclei in both compartments", len(c), c.patient_pair.nunique()))
    F = pd.DataFrame(flow, columns=["step", "samples", "patients"])
    F.to_csv(out / "analysis_flow.csv", index=False)
    print(F.to_string(index=False))
    print("\nExcluded samples:")
    print(T[T.status != "in analysis set"][["gsm", "sample", "patient_pair", "progression", "status"]].to_string(index=False))
    dropped = T[~T.in_processed_cohort]
    if len(dropped):
        print(f"\nPairs dropped in QC: {sorted(dropped.patient_pair.unique())}  "
              f"(see qc_waterfall.csv / 00b log for the exact filter)")

    write_json(out / "key_results.json", {"flow": F.to_dict(orient="records"),
                                          "dropped_in_qc": dropped[["gsm", "sample", "patient_pair", "progression"]].to_dict(orient="records"),
                                          **extra})
    provenance(out, __file__, args)
    log(f"WROTE {out}")


if __name__ == "__main__":
    main()
