import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PACKAGE_DATA
import os, gzip, time, collections
import numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as sp
import anndata as ad

OUT_DIR = os.path.join(DATA_DIR, "GSE174554_RAW")
COHORT  = os.path.join(PACKAGE_DATA, "snrna_cohort.csv")
H5AD    = os.path.join(DATA_DIR, "gbm_snrna_raw.h5ad")
MIN_GENES_PER_CELL = 200

coh = pd.read_csv(COHORT).set_index("gsm")
files = os.listdir(OUT_DIR)
MTX_QC = []
QUARANTINE = {}

runs = collections.defaultdict(dict)
for f in files:
    g = f.split("_")[0]
    if g not in coh.index:
        continue
    stem, kind = f.rsplit("_", 1)
    runs[g].setdefault(stem, {})[kind] = f

def read_mtx_safe(path):
    with gzip.open(path, "rb") as fh:
        blob = fh.read()
    lines = blob.split(b"\n")
    i = 0
    if not lines[0].startswith(b"%%MatrixMarket"):
        raise ValueError(f"no MatrixMarket header: {path}")
    i = 1
    while lines[i].startswith(b"%"):
        i += 1
    nrow, ncol, nnz_declared = (int(x) for x in lines[i].split())
    body = lines[i + 1:]

    bad = [k for k, l in enumerate(body) if b"\x00" in l]
    quarantine = set()
    for k in bad:
        j = k - 1
        while j >= 0 and (b"\x00" in body[j] or not body[j].strip()):
            j -= 1
        j2 = k + 1
        while j2 < len(body) and (b"\x00" in body[j2] or not body[j2].strip()):
            j2 += 1
        c_lo = int(body[j].split()[1]) if j >= 0 else 1
        c_hi = int(body[j2].split()[1]) if j2 < len(body) else ncol
        quarantine.update(range(c_lo, c_hi + 1))

    clean = [l for l in body if l.strip() and b"\x00" not in l]
    import io
    dat = np.loadtxt(io.BytesIO(b"\n".join(clean)), dtype=np.int64, ndmin=2)
    rows, cols, vals = dat[:, 0] - 1, dat[:, 1] - 1, dat[:, 2]
    M = sp.coo_matrix((vals, (rows, cols)), shape=(nrow, ncol)).tocsr()
    return M, dict(nnz_declared=nnz_declared, nnz_actual=int(len(vals)),
                   header_mismatch=bool(len(vals) != nnz_declared),
                   n_nul_lines=len(bad),
                   quarantined_cells=sorted(quarantine))


def read_run(d):
    M, qc = read_mtx_safe(os.path.join(OUT_DIR, d["matrix.mtx.gz"]))
    if qc["header_mismatch"] or qc["n_nul_lines"]:
        print(f"  CORRUPT FILE {d['matrix.mtx.gz']}: declared={qc['nnz_declared']:,} "
              f"actual={qc['nnz_actual']:,} nul_lines={qc['n_nul_lines']} "
              f"quarantined_cells={len(qc['quarantined_cells'])}", flush=True)
    MTX_QC.append(dict(file=d["matrix.mtx.gz"],
                       **{k: v for k, v in qc.items() if k != "quarantined_cells"},
                       n_quarantined=len(qc["quarantined_cells"])))
    M = sp.csr_matrix(M.T)
    if qc["quarantined_cells"]:
        drop = np.zeros(M.shape[0], dtype=bool)
        drop[np.array(qc["quarantined_cells"], dtype=int) - 1] = True
        M = M[~drop]
        QUARANTINE[d["matrix.mtx.gz"]] = (~drop)
    with gzip.open(os.path.join(OUT_DIR, d["features.tsv.gz"]), "rt") as fh:
        genes = [l.split("\t")[0].strip() for l in fh]
    with gzip.open(os.path.join(OUT_DIR, d["barcodes.tsv.gz"]), "rt") as fh:
        bcs = [l.strip() for l in fh]
    mask = QUARANTINE.get(d["matrix.mtx.gz"])
    if mask is not None:
        assert len(bcs) == len(mask), f"barcode/mask length mismatch {len(bcs)} vs {len(mask)}"
        bcs = [b for b, k in zip(bcs, mask) if k]
    assert M.shape == (len(bcs), len(genes)), f"shape mismatch {M.shape} vs {len(bcs)}x{len(genes)}"
    return M, genes, bcs

parts, ref_genes = [], None
t0 = time.time()
for i, (g, rr) in enumerate(sorted(runs.items()), 1):
    for stem, d in sorted(rr.items()):
        if not {"matrix.mtx.gz", "features.tsv.gz", "barcodes.tsv.gz"} <= set(d):
            print(f"  SKIPPED (missing file): {stem} -> {sorted(d)}", flush=True)
            continue
        M, genes, bcs = read_run(d)
        if ref_genes is None:
            ref_genes = genes
        assert genes == ref_genes, f"gene order differs: {stem}"
        keep = np.asarray((M > 0).sum(axis=1)).ravel() >= MIN_GENES_PER_CELL
        M = M[keep]
        bcs = [b for b, k in zip(bcs, keep) if k]
        batch = "batch2" if stem.endswith("batch2") else "batch1"
        row = coh.loc[g]
        obs = pd.DataFrame(
            {"gsm": g, "sample": row["sample"], "patient": row["patient_pair"],
             "progression": row["progression"], "age": row["age"],
             "gender": row["gender"], "run": stem, "batch": batch},
            index=[f"{stem}|{b}" for b in bcs])
        parts.append(ad.AnnData(X=M.astype(np.float32), obs=obs))
    if i % 10 == 0:
        n = sum(p.n_obs for p in parts)
        print(f"  {i}/{len(runs)} GSM  |  {n:,} nuclei  |  {time.time()-t0:.0f}s", flush=True)

A = ad.concat(parts, join="outer", index_unique=None)
A.var_names = pd.Index(ref_genes)
A.var_names_make_unique()
del parts

A.uns["build"] = {
    "source": "GSE174554_RAW.tar (GEO)",
    "min_genes_per_cell": MIN_GENES_PER_CELL,
    "n_runs": int(sum(len(v) for v in runs.values())),
    "n_samples": int(A.obs.gsm.nunique()),
    "n_patients": int(A.obs.patient.nunique()),
}
qc_df = pd.DataFrame(MTX_QC)
qc_df.to_csv(os.path.join(DATA_DIR, "mtx_header_qc.csv"), index=False)
A.uns["mtx_header_mismatch_files"] = qc_df.loc[qc_df.header_mismatch, "file"].tolist()
print(f"\nfiles with a mismatched mtx header: {int(qc_df.header_mismatch.sum())}/{len(qc_df)}")
for _, r in qc_df[qc_df.header_mismatch].iterrows():
    print(f"  {r.file}: {r.nnz_declared:,} -> {r.nnz_actual:,}")

A.write_h5ad(H5AD, compression="gzip")
print(f"\nWROTE {H5AD}")
print(f"  nuclei   : {A.n_obs:,}")
print(f"  genes    : {A.n_vars:,}")
print(f"  samples  : {A.obs.gsm.nunique()}   patients: {A.obs.patient.nunique()}")
print(f"  primary/recurrent nuclei: {dict(A.obs.progression.value_counts())}")
print(f"  nuclei per sample: median {int(A.obs.gsm.value_counts().median())}, "
      f"range {A.obs.gsm.value_counts().min()}-{A.obs.gsm.value_counts().max()}")
print(f"  elapsed: {time.time()-t0:.0f}s")
