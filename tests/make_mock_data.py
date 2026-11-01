#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import neftel

META = ["MES1", "MES2", "AC", "OPC", "NPC1", "NPC2", "G1S", "G2M"]
HOST = {"Oligodendrocytes": "Oligodendrocytes", "Microglia": "Microglia_CG", "Macrophages": "Macrophages",
        "T_cells": "T_cells", "Endothelial": "Endothelial_CG"}


def gene_universe(rng, n_random):
    gp = pd.read_csv(ROOT / "results" / "cnv" / "gene_positions.csv", index_col=0)
    gp = gp.dropna(subset=["chromosome", "start"])
    gp = gp[gp.chromosome.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])]
    gp = gp[~gp.index.duplicated()]
    gp = gp[~gp.index.astype(str).str.contains(r"\s")]
    m = json.load(open(ROOT / "data" / "metabolic_genesets.json", encoding="utf-8"))
    ct = json.load(open(ROOT / "data" / "celltype_genesets.json", encoding="utf-8"))
    meta = neftel.load_metamodules(str(ROOT / "data" / "neftel_metamodules.csv"))
    want = set()
    for d in (m["kegg_metabolism"],):
        for v in d.values():
            want |= set(v["genes"] if isinstance(v, dict) else v)
    for v in meta.values():
        want |= set(v)
    for grp in ("neftel_primary", "guard_beyond_neftel", "cellguide_crosscheck"):
        for v in ct[grp].values():
            want |= set(v)
    want |= {"CHST11", "XYLT1", "DSEL", "VCAN", "BCAN", "NCAN", "ACAN", "CSPG4", "CSPG5", "PTPRZ1",
             "PDGFRA", "OLIG2", "SOX2", "EGFR", "PTEN", "MALAT1", "MT-CO1"}
    syn = ROOT / "results" / "expression_programs" / "detectability_resource.csv"
    if syn.exists():
        try:
            want |= set(pd.read_csv(syn).iloc[:300, 0].astype(str))
        except Exception:
            pass
    rest = [g for g in gp.index if g not in want]
    extra = list(rng.choice(rest, size=min(n_random, len(rest)), replace=False))
    genes = sorted(want | set(extra))
    chrom = gp.chromosome.reindex(genes)
    return genes, chrom, meta, ct


def simulate(rng, n_cells, genes, chrom, base, mult, depth):
    rows, cols, vals = [], [], []
    for i in range(n_cells):
        lam = base * mult[i]
        lam = lam / lam.sum() * depth[i]
        c = rng.poisson(lam)
        nz = np.nonzero(c)[0]
        rows.append(np.full(len(nz), i))
        cols.append(nz)
        vals.append(c[nz])
    X = sp.csr_matrix((np.concatenate(vals).astype(np.float32),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(n_cells, len(genes)))
    return X


def build_cells(rng, genes, chrom, meta, ct, kinds, aneu, depth_scale):
    g_idx = {g: i for i, g in enumerate(genes)}
    n = len(kinds)
    base = rng.lognormal(-1.0, 1.4, len(genes))
    for g in ("MALAT1", "MT-CO1"):
        if g in g_idx:
            base[g_idx[g]] *= 30
    on7 = (chrom.values == "chr7")
    on10 = (chrom.values == "chr10")
    host_sets = {**ct["neftel_primary"], **ct["guard_beyond_neftel"]}
    cg = ct["cellguide_crosscheck"]
    mult = np.ones((n, len(genes)), dtype=np.float32)
    for i, k in enumerate(kinds):
        if aneu[i]:
            mult[i, on7] *= 1.7
            mult[i, on10] *= 0.5
        sets = []
        if k in META:
            sets = [meta[k]]
        elif k in HOST:
            sets = [host_sets[HOST[k]]]
            if k == "Microglia":
                sets.append(host_sets["Macrophages"][:2])
            if k == "Oligodendrocytes":
                sets.append(cg.get("Oligo_CG", []))
        for s in sets:
            cols = [g_idx[g] for g in s if g in g_idx]
            mult[i, cols] *= 12.0
        if k in ("OPC", "NPC1") and aneu[i]:
            mult[i, g_idx["CHST11"]] *= 2.5
            mult[i, g_idx["XYLT1"]] *= 1.6
            for g in ("VCAN", "BCAN", "CSPG4"):
                mult[i, g_idx[g]] *= 1.5
        if k == "Macrophages":
            mult[i, g_idx["CHST11"]] *= 1.8
    depth = rng.lognormal(np.log(2500), 0.35, n) * np.where(aneu, depth_scale, 1.0)
    return simulate(rng, n, genes, chrom, base, mult, depth.astype(int) + 300)


def lognorm(X):
    tot = np.asarray(X.sum(1)).ravel()
    Xn = sp.diags(1e4 / tot) @ X.astype(np.float64)
    Xn = sp.csr_matrix(Xn)
    Xn.data = np.log1p(Xn.data)
    return Xn.astype(np.float32), tot


def kinds_for(rng, n, aneu, host_frac):
    out = np.empty(n, dtype=object)
    for i in range(n):
        if not aneu[i] and rng.random() < host_frac:
            out[i] = rng.choice(list(HOST), p=[0.45, 0.35, 0.1, 0.05, 0.05])
        else:
            out[i] = rng.choice(["OPC", "NPC1", "NPC2", "AC", "MES1", "MES2"], p=[.4, .15, .05, .2, .1, .1])
    return out


def assign_states(Xn, genes, obs, meta, ct, thr):
    nm = ct["neftel_primary"]
    all_ct = {**nm, **ct["guard_beyond_neftel"], **ct["cellguide_crosscheck"]}
    sc_all = neftel.score_sets(Xn, list(genes), {**meta, **all_ct}, seed=0)
    for k, v in sc_all.items():
        obs[f"sc_{k}"] = v
    call_sets = list(nm) + list(ct["guard_beyond_neftel"])
    calls = {}
    for cl, o in obs.groupby("leiden", observed=True):
        med = {k: float(o[f"sc_{k}"].median()) for k in all_ct}
        med.update({k: float(o[f"sc_{k}"].median()) for k in META})
        hits = []
        for k in call_sets:
            t = thr.get(k, {}).get("q999")
            if t is None:
                t = 3 * np.std([med[x] for x in META])
            if med[k] > t:
                hits.append((med[k], k))
        calls[cl] = "malignant" if not hits else max(hits)[1]
    obs["cluster_call"] = obs.leiden.map(calls).astype(str)
    obs["malignant"] = obs.cluster_call == "malignant"
    pl = neftel.neftel_plane({k: obs.loc[obs.malignant, f"sc_{k}"].values for k in META})
    obs["neftel_state"] = "nonmalignant"
    obs.loc[obs.malignant, "neftel_state"] = pl["state"]
    obs["neftel_x"] = np.nan
    obs["neftel_y"] = np.nan
    obs.loc[obs.malignant, "neftel_x"] = pl["x"]
    obs.loc[obs.malignant, "neftel_y"] = pl["y"]
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--per-group", type=int, default=260, help="nuclei per CNV group per sample")
    ap.add_argument("--n-random-genes", type=int, default=2500)
    args = ap.parse_args()
    import anndata as ad
    rng = np.random.default_rng(7)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    if (out / "results").exists():
        shutil.rmtree(out / "results")
    shutil.copytree(ROOT / "results", out / "results",
                    ignore=shutil.ignore_patterns("robustness", "cache"))

    genes, chrom, meta, ct = gene_universe(rng, args.n_random_genes)
    thr = json.load(open(ROOT / "data" / "nonmalignant_thresholds.json", encoding="utf-8"))
    print(f"genes: {len(genes)}")

    calls = pd.read_csv(ROOT / "results" / "cnv" / "infercnvpy_cell_level_calls.csv")
    im = pd.read_csv(ROOT / "results" / "cnv" / "infercnvpy_sample_metadata.csv")
    coh = pd.read_csv(ROOT / "data" / "snrna_cohort.csv")
    flux = pd.read_csv(ROOT / "data" / "scfea_flux_5986_malignant_nuclei.csv.gz", usecols=[0]).iloc[:, 0]
    cnt = calls.groupby(["gsm", "icnv_pred"]).size().unstack(fill_value=0)
    t1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"])
    ok = cnt[(cnt.aneuploid >= args.per_group) & (cnt.diploid >= args.per_group)].index
    ok = [g for g in coh.sort_values("patient_pair").gsm if g in ok and g in t1]
    pp = coh.set_index("gsm").patient_pair
    pairs = pd.Series(ok).groupby(pp.reindex(ok).values).apply(list)
    chosen = [g for v in pairs if len(v) == 2 for g in v][: args.n_samples]
    chosen += [g for g in ok if g not in chosen][: max(0, args.n_samples - len(chosen))]
    t2 = [g for g in im.loc[im.tier == 2, "gsm"] if g in cnt.index][:2]
    print("samples:", chosen, "+ tier2:", t2)

    picks = []
    for g in chosen + t2:
        sub = calls[calls.gsm == g]
        fl = sub[sub.cell.isin(set(flux))]
        for p in ("aneuploid", "diploid"):
            s = sub[sub.icnv_pred == p]
            k = min(len(s), args.per_group if g in chosen else 60)
            f = fl[fl.icnv_pred == p].head(k // 3)
            rest = s[~s.cell.isin(f.cell)].sample(n=k - len(f), random_state=1)
            picks.append(pd.concat([f, rest]))
    P = pd.concat(picks).drop_duplicates("cell").reset_index(drop=True)
    aneu = (P.icnv_pred == "aneuploid").values
    kinds = kinds_for(rng, len(P), aneu, host_frac=0.45)
    t2m = P.gsm.isin(t2).values
    kinds[t2m & ~aneu] = "OPC"
    print(f"simulating {len(P)} nuclei ...")
    X = build_cells(rng, genes, chrom, meta, ct, kinds, aneu, depth_scale=1.5)
    Xn, tot = lognorm(X)
    obs = pd.DataFrame(index=pd.Index(P.cell.values, name=None))
    obs["gsm"] = P.gsm.values
    obs = obs.join(coh.set_index("gsm")[["sample", "patient_pair", "progression"]], on="gsm")
    obs["patient"] = obs.patient_pair.astype(str)
    obs["total_counts"] = tot
    obs["n_genes_by_counts"] = np.asarray((X > 0).sum(1)).ravel()
    lab = {k: i for i, k in enumerate(["OPC", "NPC1", "NPC2", "AC", "MES1", "MES2"] + list(HOST))}
    obs["leiden"] = [str(lab[k] * 2 + int(rng.random() < 0.5)) for k in kinds]
    obs["mock_kind"] = kinds
    obs = assign_states(Xn, genes, obs, meta, ct, thr)
    print(obs.cluster_call.value_counts().to_string())
    print(obs.groupby(["gsm", "neftel_state"]).size().unstack(fill_value=0).to_string())
    var = pd.DataFrame(index=pd.Index(genes))
    for c in ("leiden", "gsm", "patient", "neftel_state", "cluster_call", "progression", "sample", "patient_pair",
              "mock_kind"):
        obs[c] = obs[c].astype(str)
    S = ad.AnnData(X=Xn, obs=obs, var=var)
    S.write_h5ad(out / "data" / "gbm_snrna_states.h5ad")
    C = ad.AnnData(X=X, obs=obs.copy(), var=var)
    C.write_h5ad(out / "data" / "gbm_snrna_counts.h5ad")
    print("wrote", out / "data")


if __name__ == "__main__":
    main()
