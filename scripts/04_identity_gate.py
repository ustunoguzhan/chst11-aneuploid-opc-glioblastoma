import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR, PACKAGE_DATA
import os, io, sys, json, gc, itertools, warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


warnings.filterwarnings("ignore")
sys.path.insert(0, PACKAGE_DATA)
import neftel


BASE = DATA_DIR
OUT = RESULTS_DIR
META = os.path.join(PACKAGE_DATA, "neftel_metamodules.csv")
MIN = 30
STATES = ["OPC", "NPC", "AC", "MES"]


def perm_p(d):
    d = np.asarray(d, float)
    obs = d.mean()
    n = len(d)
    a = np.array([(d * np.array(s)).mean() for s in itertools.product([1, -1], repeat=n)])
    return obs, float((np.abs(a) >= abs(obs)).mean()), 2 / len(a)


def main():
    C = ad.read_h5ad(f"{BASE}/gbm_snrna_counts.h5ad")
    S = ad.read_h5ad(f"{BASE}/gbm_snrna_states.h5ad", backed="r")
    for c in ["gsm", "neftel_state"]:
        C.obs[c] = S.obs[c].astype(str).values
    ic = pd.read_csv(os.path.join(OUT, "cnv", "infercnvpy_cell_level_calls.csv")).set_index("cell")
    im = pd.read_csv(os.path.join(OUT, "cnv", "infercnvpy_sample_metadata.csv"))
    tier1 = set(im.loc[(im.tier == 1) & (im.state == "ok"), "gsm"])
    C.obs["pred"] = ic.icnv_pred.reindex(C.obs_names).values


    o = C.obs[C.obs.gsm.isin(tier1) & (C.obs.neftel_state == "OPC")
              & C.obs.pred.isin(["aneuploid", "diploid"])]
    nn = o.groupby(["gsm", "pred"], observed=True).size()
    keep = {(g, p) for (g, p), v in nn.items() if v >= MIN}
    gsms = sorted({g for g, _ in keep
                   if (g, "aneuploid") in keep and (g, "diploid") in keep})
    print(f"endpoint samples: {len(gsms)}", flush=True)


    A = C[(C.obs.gsm.isin(gsms) & C.obs.neftel_state.isin(
        ["OPC", "NPC", "AC", "MES"])).values].copy()
    del C
    gc.collect()
    print(f"nuclei across all four states: {A.n_obs}", flush=True)
    print("state counts:", A.obs.neftel_state.value_counts().to_dict(), flush=True)


    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)


    mm = neftel.load_metamodules(META)
    mods = {"OPC": mm["OPC"], "AC": mm["AC"],
            "NPC": sorted(set(mm["NPC1"]) | set(mm["NPC2"])),
            "MES": sorted(set(mm["MES1"]) | set(mm["MES2"]))}
    mods = {k: sorted(set(v) & set(A.var_names)) for k, v in mods.items()}
    print("module sizes present in the matrix:",
          {k: len(v) for k, v in mods.items()}, flush=True)


    E = {}
    for k, genes in mods.items():
        idx = [A.var_names.get_loc(g) for g in genes]
        X = A[:, idx].X
        E[k] = np.asarray(X.mean(1)).ravel() if hasattr(X, "mean") else X.mean(1)
    D = pd.DataFrame(E)
    D["gsm"] = A.obs.gsm.values
    D["state"] = A.obs.neftel_state.values


    per = D.groupby(["gsm", "state"])[STATES].mean()
    per.to_csv(os.path.join(OUT, "expression_programs", "identity_gate_marker_based.csv"))


    print("\nOPC module expression: nuclei labeled OPC minus nuclei labeled X")
    rows, all_pass = [], True
    for other in ["NPC", "AC", "MES"]:
        a = per.xs("OPC", level="state")["OPC"]
        b = per.xs(other, level="state")["OPC"]
        d = (a - b).dropna()
        if len(d) < 4:
            rows.append(dict(contrast=f"OPC_vs_{other}", n=len(d), note="not run (n<4)"))
            continue
        obs, p, mn = perm_p(d.values)
        ok = (p < 0.05) and (obs > 0)
        all_pass &= ok
        rows.append(dict(contrast=f"OPC_vs_{other}", n=int(len(d)),
                         delta=round(float(obs), 5), p_perm=round(float(p), 6),
                         min_p=float(mn), positive=f"{int((d>0).sum())}/{len(d)}",
                         passes=bool(ok)))
        print(f"  vs {other:<4} delta={obs:+.5f}  p={p:.6f}  "
              f"{int((d>0).sum())}/{len(d)}  {'PASS' if ok else 'FAIL'}")


    print("\nspecificity: do OPC-labeled nuclei also lead on the other modules?")
    spec = {}
    for mod in ["NPC", "AC", "MES"]:
        a = per.xs("OPC", level="state")[mod]
        b = per.xs(mod, level="state")[mod]
        d = (a - b).dropna()
        obs, p, _ = perm_p(d.values)
        spec[mod] = dict(delta=round(float(obs), 5), p_perm=round(float(p), 6),
                         opc_leads=bool(obs > 0 and p < 0.05))
        print(f"  {mod} module: OPC-labeled minus {mod}-labeled = {obs:+.5f} "
              f"(p={p:.6f}) -> {'OPC leads, BAD' if obs>0 and p<0.05 else 'ok'}")


    res = dict(n_samples=len(gsms), n_nuclei=int(A.n_obs),
               module_sizes={k: len(v) for k, v in mods.items()},
               contrasts=rows, specificity=spec, gate_pass=bool(all_pass))
    io.open(os.path.join(OUT, "expression_programs", "identity_gate_marker_based.json"), "w", encoding="utf-8").write(
        json.dumps(res, indent=1))
    print(f"\nIDENTITY GATE: {'PASS' if all_pass else 'FAIL'}")
    print(f"[OK] two files written to {OUT}/")


if __name__ == "__main__":
    main()
