#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd

from common import (CNV_CALLS, CNV_META, COHORT_CSV, COUNTS_H5AD, DATA_DIR, GENE_POS, PACKAGE_DATA,
                    RESULTS_DIR, STATES_H5AD, log, outdir, pkg_version, rel_path, signflip, sign_test,
                    versions, write_json)

REQUIRED_PKGS = {"numpy": "2.3.3", "pandas": "2.3.2", "scipy": "1.17.1", "anndata": "0.12.13",
                 "scanpy": "1.12.1", "infercnvpy": "0.6.1", "decoupler": "2.2.0",
                 "statsmodels": "0.14.6", "matplotlib": "3.10.6"}
OPTIONAL_PKGS = ["igraph", "leidenalg", "openpyxl", "gseapy", "lifelines"]
BUILD1_REFERENCE = dict(sample_mean=0.4586, sample_positive=18, sample_n=20, sample_p="6.0e-4",
                        patient_mean=0.475, patient_p=0.0016)


def main():
    out = outdir("build_concordance")
    rep = {"versions": versions(), "packages": {}, "files": {}, "checks": {}}
    ok = True

    log("Python packages")
    for m, want in REQUIRED_PKGS.items():
        try:
            importlib.import_module(m)
            have = pkg_version(m)
            flag = "OK" if have == want else "version differs"
        except Exception as e:
            have, flag = None, f"MISSING ({type(e).__name__})"
        rep["packages"][m] = dict(required=want, installed=have, status=flag)
        print(f"  {m:<12} required {want:<8} installed {str(have):<10} {flag}")
    for m in OPTIONAL_PKGS:
        try:
            importlib.import_module(m)
            have = pkg_version(m) or "present"
        except Exception:
            have = None
        rep["packages"][m] = dict(required="optional", installed=have)
        print(f"  {m:<12} optional            installed {str(have)}")

    log("Input files")
    files = {
        "counts_h5ad": COUNTS_H5AD, "states_h5ad": STATES_H5AD,
        "cnv_calls": CNV_CALLS, "cnv_meta": CNV_META, "gene_positions": GENE_POS,
        "cohort": COHORT_CSV,
        "per_tumor_expression": RESULTS_DIR / "targets" / "per_tumor_expression.csv",
        "cell_state_scope": RESULTS_DIR / "targets" / "cell_state_scope.csv",
        "cspg_json": RESULTS_DIR / "targets" / "cspg_pathway_consistency.json",
        "scfea_flux": PACKAGE_DATA / "scfea_flux_5986_malignant_nuclei.csv.gz",
        "depmap": PACKAGE_DATA / "depmap_chst11_dependency_by_cellline.csv",
        "tractability": PACKAGE_DATA / "target_tractability.csv",
        "qc_waterfall": DATA_DIR / "qc_waterfall.csv",
    }
    for k, p in files.items():
        e = p.exists()
        size = p.stat().st_size / 1e6 if e else 0
        rep["files"][k] = dict(path=rel_path(p), exists=e, size_mb=round(size, 2))
        print(f"  {'OK ' if e else '-- '} {k:<22} {rel_path(p)}" + (f"  ({size:,.1f} MB)" if e else ""))

    if files["states_h5ad"].exists():
        import anndata as ad
        S = ad.read_h5ad(STATES_H5AD, backed="r")
        need = ["gsm", "patient", "progression", "neftel_state", "cluster_call", "malignant",
                "leiden", "total_counts"]
        miss = [c for c in need if c not in S.obs]
        sc_cols = [c for c in S.obs if c.startswith("sc_")]
        rep["checks"]["states_obs_missing_columns"] = miss
        rep["checks"]["states_shape"] = list(S.shape)
        rep["checks"]["states_sc_columns"] = sc_cols
        print(f"  states H5AD: {S.shape[0]:,} nuclei x {S.shape[1]:,} genes; missing obs: {miss or 'none'}")
        S.file.close()
    if files["counts_h5ad"].exists() and files["cnv_calls"].exists():
        import anndata as ad
        C = ad.read_h5ad(COUNTS_H5AD, backed="r")
        ic = pd.read_csv(CNV_CALLS, usecols=["cell"])
        ov = np.isin(ic.cell.values, C.obs_names.values).mean()
        ov2 = np.isin(C.obs_names.values, ic.cell.values).mean()
        rep["checks"]["cnv_calls_matching_counts_obs"] = float(ov)
        rep["checks"]["counts_obs_with_cnv_call"] = float(ov2)
        print(f"  CNV calls whose barcode is in counts H5AD: {100 * ov:.1f}%; "
              f"H5AD nuclei with a CNV call: {100 * ov2:.1f}%")
        if ov2 < 0.99:
            print("  WARNING: CNV calls and H5AD barcodes do not match - wrong data directory?")
        C.file.close()

    if files["per_tumor_expression"].exists():
        log("Reproducing the Build 1 CHST11 statistics from results/targets/per_tumor_expression.csv")
        E = pd.read_csv(files["per_tumor_expression"])
        E["delta"] = E.aneuploid - E.diploid
        coh = pd.read_csv(COHORT_CSV)
        s = E[E.gene == "CHST11"].merge(coh[["gsm", "patient_pair"]], on="gsm", how="left")
        m, p, _, _ = signflip(s.delta.values)
        pp_m, pp_p, _, _ = signflip(s.groupby("patient_pair").delta.mean().values)
        ps, k, n = sign_test(s.delta.values)
        rep["checks"]["CHST11_sample"] = dict(mean=m, p=p, positive=k, n=n)
        rep["checks"]["CHST11_patient"] = dict(mean=pp_m, p=pp_p)
        print(f"  sample level : mean {m:+.4f}  P={p:.2e}  {k}/{n} positive   "
              f"(Build 1 reference +{BUILD1_REFERENCE['sample_mean']}, P={BUILD1_REFERENCE['sample_p']}, "
              f"{BUILD1_REFERENCE['sample_positive']}/{BUILD1_REFERENCE['sample_n']})")
        print(f"  patient level: mean {pp_m:+.4f}  P={pp_p:.2e}             "
              f"(Build 1 reference +{BUILD1_REFERENCE['patient_mean']}, P={BUILD1_REFERENCE['patient_p']})")
        if len(s) == BUILD1_REFERENCE["sample_n"] and not (abs(m - BUILD1_REFERENCE["sample_mean"]) < 5e-4
                                                 and k == BUILD1_REFERENCE["sample_positive"]):
            print("  ERROR: Build 1 statistics are not reproduced")
            ok = False
    write_json(out / "build_concordance.json", rep)
    print(f"\nWROTE {rel_path(out / 'build_concordance.json')}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
