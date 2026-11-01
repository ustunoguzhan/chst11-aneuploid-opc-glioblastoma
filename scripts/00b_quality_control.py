import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, time, json

IN  = os.path.join(DATA_DIR, "gbm_snrna_raw.h5ad")
OUT = os.path.join(DATA_DIR, "gbm_snrna_qc.h5ad")
MT_MAX, MIN_NUCLEI, SENS_NUCLEI = 5.0, 100, 500
log = []

A = ad.read_h5ad(IN)
log.append(("start", A.n_obs, A.obs.gsm.nunique(), A.obs.patient.nunique()))

A.var["mt"] = A.var_names.str.startswith(("MT-", "MTRNR"))
sc.pp.calculate_qc_metrics(A, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)

A = A[A.obs.pct_counts_mt < MT_MAX].copy()
log.append((f"MT<{MT_MAX}%", A.n_obs, A.obs.gsm.nunique(), A.obs.patient.nunique()))

n_per = A.obs.gsm.value_counts()
A = A[A.obs.gsm.isin(n_per[n_per >= MIN_NUCLEI].index)].copy()
log.append((f"samples>={MIN_NUCLEI}", A.n_obs, A.obs.gsm.nunique(), A.obs.patient.nunique()))

piv = A.obs.drop_duplicates("gsm").groupby(["patient","progression"], observed=True).size().unstack(fill_value=0)
for c in ("Primary", "Recurrent"):
    if c not in piv: piv[c] = 0
paired = piv[(piv.Primary > 0) & (piv.Recurrent > 0)].index
dropped = sorted(set(piv.index) - set(paired))
A = A[A.obs.patient.isin(paired)].copy()
log.append(("paired patients", A.n_obs, A.obs.gsm.nunique(), A.obs.patient.nunique()))
print("patients dropped because pairing broke:", dropped, flush=True)

t0 = time.time()
sc.pp.scrublet(A, batch_key="gsm", random_state=0)
print(f"scrublet: {time.time()-t0:.0f}s", flush=True)
print("estimated doublet rate: %.2f%%" % (100 * A.obs.predicted_doublet.mean()), flush=True)
print(A.obs.groupby("progression", observed=True).predicted_doublet.mean().mul(100).round(2).to_string(), flush=True)

A.obs["is_singlet"] = ~A.obs.predicted_doublet
A_s = A[A.obs.is_singlet].copy()
log.append(("singlet", A_s.n_obs, A_s.obs.gsm.nunique(), A_s.obs.patient.nunique()))

n_per2 = A_s.obs.gsm.value_counts()
low = n_per2[n_per2 < MIN_NUCLEI]
if len(low):
    print("samples falling below the threshold after doublet removal:", dict(low), flush=True)
    A_s = A_s[A_s.obs.gsm.isin(n_per2[n_per2 >= MIN_NUCLEI].index)].copy()
    piv2 = A_s.obs.drop_duplicates("gsm").groupby(["patient","progression"], observed=True).size().unstack(fill_value=0)
    for c in ("Primary","Recurrent"):
        if c not in piv2: piv2[c] = 0
    keep2 = piv2[(piv2.Primary > 0) & (piv2.Recurrent > 0)].index
    A_s = A_s[A_s.obs.patient.isin(keep2)].copy()
    log.append(("final", A_s.n_obs, A_s.obs.gsm.nunique(), A_s.obs.patient.nunique()))

A_s.obs["passes_sens_filter"] = A_s.obs.gsm.map(
    A_s.obs.gsm.value_counts() >= SENS_NUCLEI).astype(bool)
A_s.uns["qc"] = dict(mt_max=MT_MAX, min_nuclei=MIN_NUCLEI, sens_nuclei=SENS_NUCLEI,
                     dropped_patients=dropped,
                     doublet_rate=float(A.obs.predicted_doublet.mean()))
A_s.write_h5ad(OUT, compression="gzip")

df = pd.DataFrame(log, columns=["step","nuclei","samples","patients"])
df["nuclei_lost"] = -df.nuclei.diff().fillna(0).astype(int)
df.to_csv(os.path.join(DATA_DIR, "qc_waterfall.csv"), index=False)
print("\n" + df.to_string(index=False))
sens = A_s.obs.drop_duplicates("gsm").passes_sens_filter
print(f"\nsamples passing the sensitivity filter (>={SENS_NUCLEI}): {int(sens.sum())}/{len(sens)}")
print(f"WROTE {OUT}: {A_s.n_obs:,} nuclei, {A_s.obs.gsm.nunique()} samples, {A_s.obs.patient.nunique()} patients")
print("primary/recurrent:", dict(A_s.obs.progression.value_counts()))
