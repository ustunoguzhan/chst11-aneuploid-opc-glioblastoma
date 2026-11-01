import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR
import os, io, sys, json, zipfile, warnings, urllib.request, urllib.error
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
OUT = RESULTS_DIR
CACHE = os.path.join(DATA_DIR, "external_cache")
os.makedirs(CACHE, exist_ok=True)


GENES = ["CHST11", "XYLT1", "DSEL", "GALNT13", "ST6GALNAC3"]
CBIO = "https://www.cbioportal.org/api"
STUDY = "gbm_tcga_pan_can_atlas_2018"
IVY_ZIP = ("https://glioblastoma.alleninstitute.org/api/v2/well_known_file_download/"
           "305873915")
STATUS = {}


def api(path, payload=None):
    url = f"{CBIO}{path}"
    if payload is None:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def tcga():
    print("== TCGA-GBM via cBioPortal ==", flush=True)
    ids = {}
    for g in GENES:
        try:
            ids[g] = api(f"/genes/{g}")["entrezGeneId"]
        except Exception as e:
            print(f"  {g}: gene lookup failed, {str(e)[:50]}", flush=True)
    print(f"  resolved {len(ids)} gene ids", flush=True)


    profs = api(f"/studies/{STUDY}/molecular-profiles")
    prof = next((p["molecularProfileId"] for p in profs
                 if p["molecularAlterationType"] == "MRNA_EXPRESSION"
                 and "zscores" not in p["molecularProfileId"]), None)
    if prof is None:
        prof = next(p["molecularProfileId"] for p in profs
                    if p["molecularAlterationType"] == "MRNA_EXPRESSION")
    lists = api(f"/studies/{STUDY}/sample-lists")
    sl = next((x["sampleListId"] for x in lists if x["sampleListId"].endswith("_rna_seq_v2_mrna")),
              lists[0]["sampleListId"])
    print(f"  profile {prof}, sample list {sl}", flush=True)


    md = api(f"/molecular-profiles/{prof}/molecular-data/fetch",
             {"entrezGeneIds": list(ids.values()), "sampleListId": sl})
    E = pd.DataFrame([{"patientId": r["patientId"], "gene": r["entrezGeneId"],
                       "value": r["value"]} for r in md if r.get("value") is not None])
    rev = {v: k for k, v in ids.items()}
    E["gene"] = E.gene.map(rev)
    X = E.pivot_table(index="patientId", columns="gene", values="value", aggfunc="mean")
    print(f"  expression for {X.shape[0]} patients, {X.shape[1]} genes", flush=True)


    cl = api(f"/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY")
    C = pd.DataFrame(cl).pivot_table(index="patientId", columns="clinicalAttributeId",
                                     values="value", aggfunc="first")
    keep = [c for c in ["OS_MONTHS", "OS_STATUS"] if c in C.columns]
    D = X.join(C[keep], how="inner")
    D["OS_MONTHS"] = pd.to_numeric(D.OS_MONTHS, errors="coerce")
    D["event"] = D.OS_STATUS.astype(str).str.startswith("1").astype(int)
    D = D.dropna(subset=["OS_MONTHS"])
    D = D[D.OS_MONTHS > 0]
    D.to_csv(os.path.join(OUT, "independent_cohorts", "tcga_targets_expression.csv"))
    print(f"  survival set: {len(D)} patients, {int(D.event.sum())} events", flush=True)


    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    rows = []
    for g in [c for c in GENES if c in D.columns]:
        d = D[[g, "OS_MONTHS", "event"]].dropna().copy()
        d[g] = np.log2(d[g].clip(lower=0) + 1)
        d["z"] = (d[g] - d[g].mean()) / d[g].std(ddof=0)
        cph = CoxPHFitter().fit(d[["z", "OS_MONTHS", "event"]],
                                duration_col="OS_MONTHS", event_col="event")
        lo, hi = np.exp(cph.confidence_intervals_.loc["z"].values)
        hg = d[d[g] > d[g].median()]; lg = d[d[g] <= d[g].median()]
        lr = logrank_test(hg.OS_MONTHS, lg.OS_MONTHS, hg.event, lg.event)
        rows.append(dict(gene=g, n=len(d), n_event=int(d.event.sum()),
                         cox_HR_per_SD=round(float(np.exp(cph.params_["z"])), 4),
                         cox_CI_low=round(float(lo), 4), cox_CI_high=round(float(hi), 4),
                         cox_p=float(cph.summary.loc["z", "p"]),
                         median_split_logrank_p=float(lr.p_value)))
    R = pd.DataFrame(rows)
    R["cox_p_holm"] = multipletests(R.cox_p, method="holm")[1].round(5)
    R["cox_p"] = R.cox_p.round(6)
    R["median_split_logrank_p"] = R.median_split_logrank_p.round(6)
    R.to_csv(os.path.join(OUT, "independent_cohorts", "tcga_targets_survival.csv"), index=False)
    print(R.to_string(index=False), flush=True)
    STATUS["tcga"] = dict(study=STUDY, profile=prof, n=len(D),
                          events=int(D.event.sum()), genes=list(R.gene))


def ivygap():
    print("\n== IVY-GAP via the Allen Institute download ==", flush=True)
    zp = os.path.join(CACHE, "ivygap_expression.zip")
    if not os.path.exists(zp):
        print("  downloading expression matrix ...", flush=True)
        urllib.request.urlretrieve(IVY_ZIP, zp)
    print(f"  archive {os.path.getsize(zp) / 1e6:.1f} MB", flush=True)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        fn_rows = next(n for n in names if "rows-genes" in n)
        fn_cols = next(n for n in names if "columns-samples" in n)
        fn_fpkm = next(n for n in names if "fpkm_table" in n)
        rows = pd.read_csv(z.open(fn_rows))
        cols = pd.read_csv(z.open(fn_cols))
        fpkm = pd.read_csv(z.open(fn_fpkm), index_col=0)
    print(f"  genes {rows.shape[0]}, samples {cols.shape[0]}, matrix {fpkm.shape}",
          flush=True)


    sym = "gene_symbol" if "gene_symbol" in rows.columns else rows.columns[1]
    gid = "gene_id" if "gene_id" in rows.columns else rows.columns[0]
    idx = rows.set_index(sym)[gid]
    have = [g for g in GENES if g in idx.index]
    take = {g: idx.loc[g] for g in have}
    take = {g: (v.iloc[0] if hasattr(v, "iloc") else v) for g, v in take.items()}
    fpkm.index = fpkm.index.astype(str)
    M = pd.DataFrame({g: np.log2(fpkm.loc[str(v)].astype(float) + 1)
                      for g, v in take.items() if str(v) in fpkm.index})
    M.index.name = "rna_well_id"
    scol = "structure_name" if "structure_name" in cols.columns else cols.columns[-1]
    wcol = "rna_well_id" if "rna_well_id" in cols.columns else cols.columns[0]
    ann = cols.set_index(cols[wcol].astype(str))[[scol]]
    M = M.join(ann, how="inner")
    n_all = len(M)
    ref = M[scol].astype(str).str.contains("sampled by reference histology",
                                           case=False, na=False)
    M = M[ref].copy()
    M["structure"] = (M[scol].astype(str)
                      .str.replace(" sampled by reference histology", "", regex=False)
                      .str.strip())
    M.to_csv(os.path.join(OUT, "independent_cohorts", "ivygap_targets_wells.csv"))
    print(f"  wells with expression and annotation: {n_all}", flush=True)
    print(f"  reference-histology wells retained: {len(M)}", flush=True)
    print(M.structure.value_counts().to_string(), flush=True)

    present = [c for c in GENES if c in M.columns]
    med = M.groupby("structure")[present].median().round(3)
    med.index.name = "structure"
    med.to_csv(os.path.join(OUT, "independent_cohorts",
                            "ivygap_targets_structure_medians.csv"))

    le = M.structure.str.contains("Leading Edge", case=False, na=False)
    rows2 = []
    for g in present:
        a, b = M.loc[le, g].dropna(), M.loc[~le, g].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        groups = [v[g].dropna().values for _, v in M.groupby("structure")
                  if len(v[g].dropna()) >= 3]
        kw = stats.kruskal(*groups)
        u = stats.mannwhitneyu(a, b)
        rb = 1 - 2 * u.statistic / (len(a) * len(b))
        rows2.append(dict(gene=g, n_leading_edge=len(a), n_rest=len(b),
                          median_leading_edge=round(float(a.median()), 3),
                          median_rest=round(float(b.median()), 3),
                          kruskal_p_across_5=float(kw.pvalue),
                          LE_vs_rest_p=float(u.pvalue),
                          rank_biserial=round(float(-rb), 3)))
    R = pd.DataFrame(rows2)
    if len(R):
        R["LE_p_holm"] = multipletests(R.LE_vs_rest_p, method="holm")[1]
        for c in ["kruskal_p_across_5", "LE_vs_rest_p", "LE_p_holm"]:
            R[c] = R[c].map(lambda v: float(f"{v:.4g}"))
        R.to_csv(os.path.join(OUT, "independent_cohorts",
                              "ivygap_targets_leading_edge.csv"), index=False)
        print("\n  leading edge versus the rest:", flush=True)
        print(R.to_string(index=False), flush=True)
    STATUS["ivygap"] = dict(wells_total=n_all, wells_reference_histology=len(M),
                            genes=present)


if __name__ == "__main__":
    for name, fn in [("tcga", tcga), ("ivygap", ivygap)]:
        try:
            fn()
        except Exception as e:
            STATUS[name] = dict(failed=str(e)[:200])
            print(f"  {name} FAILED: {str(e)[:160]}", flush=True)
    io.open(os.path.join(OUT, "independent_cohorts", "external_fetch_summary.json"), "w", encoding="utf-8").write(
        json.dumps(STATUS, indent=1))
    print(f"\n[OK] summary written to {OUT}/independent_cohorts/external_fetch_summary.json", flush=True)
