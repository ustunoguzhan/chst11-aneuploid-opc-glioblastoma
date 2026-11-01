import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR
import numpy as np, scipy.sparse as sp, anndata as ad, json

CHUNK = 20000
A = ad.read_h5ad(os.path.join(DATA_DIR, "gbm_snrna_states.h5ad"))
tot = A.obs["total_counts"].values.astype(np.float64)
assert np.all(tot > 0), "cells with zero or negative total_counts"

blocks, max_dev = [], 0.0
X = sp.csr_matrix(A.X)
for s in range(0, X.shape[0], CHUNK):
    e = min(s + CHUNK, X.shape[0])
    B = X[s:e].copy()
    scale = tot[s:e] / 1e4
    B.data = np.expm1(B.data.astype(np.float64)) * np.repeat(scale, np.diff(B.indptr))
    max_dev = max(max_dev, float(np.abs(B.data - np.round(B.data)).max()) if B.nnz else 0.0)
    B.data = np.round(B.data)
    blocks.append(B.astype(np.float32))
    print(f"  {e:,}/{X.shape[0]:,}", flush=True)

C = sp.vstack(blocks, format="csr")
del blocks, X

print(f"\nmaximum deviation from an integer (before rounding): {max_dev:.3e}")
assert max_dev < 0.01, f"reconstruction is not integral (deviation {max_dev}) - the inverse transform is invalid"

rs = np.asarray(C.sum(1)).ravel()
diff = np.abs(rs - tot)
n_exact = int((diff < 0.5).sum())
print(f"cells whose row sum matches total_counts: {n_exact:,}/{len(tot):,}")
print(f"maximum row-sum difference: {diff.max():.3f}")
assert n_exact == len(tot), "row sums do not match total_counts"

assert C.data.min() >= 0, "negative counts"
print(f"nnz: {C.nnz:,}  |  min/max count: {C.data.min():.0f}/{C.data.max():.0f}")

R = ad.AnnData(X=C, obs=A.obs.copy(), var=A.var.copy())
R.uns["counts_recovery"] = json.dumps({
    "source": "gbm_snrna_states.h5ad (log1p CP10K) + obs.total_counts",
    "method": "counts = expm1(X) * total_counts / 1e4 (inverse of normalize_total followed by log1p)",
    "validation": {"max_deviation_from_integer": max_dev,
                  "cells_with_matching_row_sum": f"{n_exact}/{len(tot)}",
                  "max_row_sum_difference": float(diff.max())},
})
R.write_h5ad(os.path.join(DATA_DIR, "gbm_snrna_counts.h5ad"), compression="gzip")
print(f"\nWROTE gbm_snrna_counts.h5ad: {R.shape[0]:,} x {R.shape[1]:,}")
