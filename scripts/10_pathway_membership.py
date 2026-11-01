import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR
import os, io, sys, json, warnings
import pandas as pd
import gseapy


warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
OUT = RESULTS_DIR


WANTED = {
    "Chondroitin / dermatan sulfate": [
        "Glycosaminoglycan biosynthesis - chondroitin sulfate / dermatan sulfate",
        "Glycosaminoglycan biosynthesis-chondroitin sulfate / dermatan sulfate",
        "Chondroitin sulfate biosynthesis",
        "Chondroitin sulfate/dermatan sulfate metabolism",
        "Dermatan sulfate biosynthesis",
    ],
    "Heparan sulfate / heparin": [
        "Glycosaminoglycan biosynthesis - heparan sulfate / heparin",
        "Glycosaminoglycan biosynthesis-heparan sulfate / heparin",
        "Heparan sulfate/heparin (HS-GAG) metabolism",
        "HS-GAG biosynthesis",
    ],
    "Glycosaminoglycan degradation": [
        "Glycosaminoglycan degradation",
        "Glycosaminoglycan metabolism",
    ],
    "Mucin-type O-glycan biosynthesis": [
        "Mucin type O-glycan biosynthesis",
        "Mucin type O-Glycan biosynthesis",
        "O-linked glycosylation of mucins",
        "Termination of O-glycan biosynthesis",
    ],
    "Other O-linked glycosylation": [
        "Other types of O-glycan biosynthesis",
        "O-linked glycosylation",
    ],
    "Sialylation and glycosphingolipid": [
        "Glycosphingolipid biosynthesis - ganglio series",
        "Sialic acid metabolism",
        "Glycosphingolipid biosynthesis-ganglio series",
    ],
}
LIBS = ["KEGG_2021_Human", "Reactome_2022", "Reactome_Pathways_2024",
        "KEGG_2019_Human", "WikiPathway_2023_Human"]


def main():
    enz = pd.read_csv(os.path.join(OUT, "targets", "enzyme_compartment_contrast.csv"))
    genes = list(enz.gene)
    print(f"enzymes to assign: {len(genes)}", flush=True)


    libs, used = {}, {}
    for L in LIBS:
        try:
            libs[L] = gseapy.get_library(L)
            print(f"  retrieved {L}: {len(libs[L])} sets", flush=True)
        except Exception as e:
            print(f"  {L} unavailable: {str(e)[:60]}", flush=True)


    assign = {}
    for branch, names in WANTED.items():
        for L, lib in libs.items():
            for nm in names:
                hit = next((k for k in lib if k.lower().startswith(nm.lower())), None)
                if hit is None:
                    continue
                members = {g.split(",")[0].strip().upper() for g in lib[hit]}
                got = [g for g in genes if g.upper() in members]
                if not got:
                    continue
                used.setdefault(branch, []).append(dict(library=L, set_name=hit,
                                                        genes_matched=len(got)))
                for g in got:
                    assign.setdefault(g, []).append(branch)


    rows = []
    for _, r in enz.iterrows():
        b = sorted(set(assign.get(r.gene, [])))
        rows.append(dict(gene=r.gene,
                         branch="; ".join(b) if b else "unassigned",
                         n_branches=len(b),
                         delta_log=r.delta_log, p_perm=r.p_perm, p_holm=r.p_holm,
                         detection_pct=r.detection_pct, interpretable=r.interpretable))
    A = pd.DataFrame(rows).sort_values(["branch", "delta_log"], ascending=[True, False])
    A.to_csv(os.path.join(OUT, "targets", "pathway_membership.csv"), index=False)
    print("\nassignment:", flush=True)
    print(A[["gene", "branch", "delta_log", "p_holm", "detection_pct"]]
          .to_string(index=False), flush=True)
    n_un = int((A.branch == "unassigned").sum())
    print(f"\nunassigned: {n_un} of {len(A)}", flush=True)


    io.open(os.path.join(OUT, "targets", "pathway_membership_sources.json"), "w", encoding="utf-8").write(
        json.dumps(dict(sources=used, unassigned=n_un, n_genes=len(A)), indent=1))
    print(f"[OK] two files written to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
