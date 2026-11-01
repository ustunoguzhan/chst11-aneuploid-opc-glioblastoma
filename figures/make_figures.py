import os, sys, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, "results")
DATA = os.path.join(REPO, "data")
FIGD = os.path.join(HERE, "output")
SRCD = os.path.join(HERE, "source_data")
os.makedirs(FIGD, exist_ok=True); os.makedirs(SRCD, exist_ok=True)

PFDIR = os.environ.get("PUBFIG_DIR", os.path.join(REPO, "tools", "pubfig"))
if PFDIR not in sys.path:
    sys.path.insert(0, PFDIR)
import pubfig as pf

pf.use_pub_style("nature")

R = lambda *a: pd.read_csv(os.path.join(RES, *a))
J = lambda *a: json.load(open(os.path.join(RES, *a), encoding="utf-8"))

C = dict(CHST11="#0072B2", XYLT1="#D55E00", DSEL="#009E73", GALNT13="#CC79A7",
         neuronal="#999999", enzyme="#0072B2", null="#BBBBBB",
         aneuploid="#0072B2", diploid="#999999", grey="#777777", accent="#E69F00")
GENE3 = ["CHST11", "XYLT1", "DSEL"]


def _nolegendframe(ax, **kw):
    lg = ax.legend(frameon=False, handletextpad=0.4, borderaxespad=0.2, **kw)
    return lg


def _dump(name, df):
    df.to_csv(os.path.join(SRCD, name), index=False)


def figure_1():
    pc = R("expression_programs", "positive_control_chr7_chr10.csv")
    idg = R("expression_programs", "identity_gate_marker_based.csv")
    syn = J("expression_programs", "synaptic_program_contrast.json")
    s1k = J("expression_programs", "synaptic_detection_matched_null_1000.json")
    icnv = R("cnv", "infercnvpy_sample_metadata.csv")

    fig, ax = pf.figure_mosaic("""
        AAB
        CCD
        """, width_mm=183, height_mm=108)

    a = ax["A"];
    steps = [("Cohort\n(GSE174554)", 56, 163481),
             ("Within-sample\nCNV reference", int((icnv.tier == 1).sum()), None),
             ("\u226530 OPC-like nuclei\nper compartment", 20, 13196)]
    xs = np.arange(len(steps))
    bars = a.bar(xs, [s[1] for s in steps], width=0.52, color=[C["grey"], C["grey"], C["aneuploid"]],
                 edgecolor="none")
    for x, (lab, n, nuc) in zip(xs, steps):
        txt = f"{n}" + (f"\n{nuc:,} nuclei" if nuc else "")
        a.text(x, n + 1.5, txt, ha="center", va="bottom", fontsize=5.6)
    a.set_xticks(xs); a.set_xticklabels([s[0] for s in steps], fontsize=5.6)
    a.set_ylabel("Samples"); a.set_ylim(0, 72)
    a.set_yticks([0, 20, 40, 60])
    _dump("Fig1a_analysis_set_flow.csv",
          pd.DataFrame(steps, columns=["stage", "n_samples", "n_nuclei"]))

    b = ax["B"];
    for i, row in pc.iterrows():
        b.plot([0, 1], [row.diploid, row.aneuploid], color="#CCCCCC", lw=0.5, zorder=1)
    b.scatter(np.zeros(len(pc)), pc.diploid, s=7, color=C["diploid"], zorder=3, lw=0)
    b.scatter(np.ones(len(pc)), pc.aneuploid, s=7, color=C["aneuploid"], zorder=3, lw=0)
    for x, v in [(0, pc.diploid.median()), (1, pc.aneuploid.median())]:
        b.plot([x - .22, x + .22], [v, v], color="black", lw=1.1, zorder=4)
    b.set_xticks([0, 1]); b.set_xticklabels(["Diploid", "Aneuploid"])
    b.set_xlim(-0.45, 1.45); b.set_ylabel("chr7 gain \u2212 chr10 loss")
    _dump("Fig1b_chr7_chr10_positive_control.csv", pc)

    c = ax["C"];
    obs = s1k["observed"]; prim = s1k["primary"]; sens = s1k["sensitivity"]
    labels = ["Synaptic\nGO:0007268", "Trans-synaptic\nGO:0098916", "Synaptic mod.\nGO:0050804",
              "Translation\n(neg. control)"]
    vals = [syn["delta"], syn["secondary"]["trans_synaptic_GO0098916"]["delta"],
            syn["secondary"]["synaptic_modulation_GO0050804"]["delta"],
            syn["negative_control_translation"]["delta"]]
    cols = [C["aneuploid"]] * 3 + [C["grey"]]
    c.bar(np.arange(4), vals, width=0.55, color=cols, edgecolor="none")
    c.axhline(0, color="black", lw=0.5)
    c.set_xticks(np.arange(4)); c.set_xticklabels(labels, fontsize=5.6)
    c.set_ylabel("\u0394 AUCell (aneuploid \u2212 diploid)")
    _dump("Fig1c_set_level_deltas.csv", pd.DataFrame({"set": labels, "delta": vals}))

    d = ax["D"];
    for i, (nm, nd) in enumerate([("1,000 sets\nsize-preserving", prim),
                                  ("1,000 sets\nduplicates dropped", sens)]):
        d.barh(i, nd["null_97_5"], height=0.34, color=C["null"], edgecolor="none")
        d.plot([nd["null_median"]] * 2, [i - .17, i + .17], color="black", lw=0.9)
    d.axvline(obs, color=C["accent"], lw=1.2)
    d.set_yticks([0, 1]); d.set_yticklabels(["1,000 sets\nsize-preserving",
                                             "1,000 sets\ndup. dropped"], fontsize=5.6)
    d.set_xlabel("\u0394 AUCell"); d.set_ylim(-0.6, 2.2)
    d.invert_yaxis()
    hs = [Rectangle((0, 0), 1, 1, color=C["null"]), Line2D([0], [0], color="black", lw=0.9),
          Line2D([0], [0], color=C["accent"], lw=1.2)]
    d.legend(hs, ["null 0\u201397.5th pct", "null median", "observed"], frameon=False,
             fontsize=5.4, loc="lower right", handlelength=1.2)
    _dump("Fig1d_synaptic_matched_null.csv", pd.DataFrame([
        dict(construction="size-preserving", **{k: prim[k] for k in
             ["n_sets", "null_median", "null_95", "null_97_5", "p_add_one"]}, observed=obs),
        dict(construction="duplicates dropped", **{k: sens[k] for k in
             ["n_sets", "null_median", "null_95", "null_97_5", "p_add_one"]}, observed=obs)]))

    pf.panel_labels(fig, ax, case="lower")
    return fig, dict(panels={
        "a": "results/cnv/infercnvpy_sample_metadata.csv (tier), cohort counts of the analysis-set definition",
        "b": "results/expression_programs/positive_control_chr7_chr10.csv",
        "c": "results/expression_programs/synaptic_program_contrast.json",
        "d": "results/expression_programs/synaptic_detection_matched_null_1000.json"})


def figure_s1():
    enz = R("targets", "enzyme_compartment_contrast.csv")
    te = R("targets", "target_enrichment.csv")
    det = R("expression_programs", "detectability_resource.csv")
    tract = pd.read_csv(os.path.join(DATA, "target_tractability.csv"))

    fig, ax = pf.figure_mosaic("""
        AAB
        AAC
        """, width_mm=183, height_mm=92)

    a = ax["A"];
    ez = enz[["gene", "detection_pct", "delta_log"]].copy(); ez["set"] = "Carbohydrate enzymes"
    nu = te[~te.gene.isin(enz.gene)][["gene", "detection_pct", "delta_log"]].copy()
    nu["set"] = "Neuronal candidates"
    pts = pd.concat([ez, nu], ignore_index=True)
    for nm, col, mk in [("Neuronal candidates", C["neuronal"], "s"),
                        ("Carbohydrate enzymes", C["enzyme"], "o")]:
        s = pts[pts.set == nm]
        a.scatter(s.detection_pct, s.delta_log, s=14, marker=mk, facecolor=col,
                  edgecolor="white", lw=0.3, label=nm, zorder=3)
    a.axvline(5, color="black", ls="--", lw=0.6, zorder=1)
    a.axhline(0, color="black", lw=0.5, zorder=1)
    a.set_xscale("log")
    a.set_xticks([2, 5, 10, 20, 50])
    a.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_xlabel("Detection across OPC-like nuclei (%)")
    a.set_ylabel("\u0394 log expression (aneuploid \u2212 diploid)")
    for g, ox, oy in [("CHST11", 6, 0), ("XYLT1", 6, 0), ("DSEL", 6, 0), ("GALNT13", 6, 4),
                      ("GRM5", 4, 10), ("CACNA1B", -6, -14), ("GABRB2", 8, -22)]:
        r = pts[pts.gene == g]
        if len(r):
            a.annotate(g, (r.detection_pct.iloc[0], r.delta_log.iloc[0]), xytext=(ox, oy),
                       textcoords="offset points", fontsize=5.6, va="center",
                       ha="right" if ox < 0 else "left",
                       arrowprops=dict(arrowstyle="-", lw=0.4, color="#777777",
                                       shrinkA=0, shrinkB=2.5) if oy else None)
    _nolegendframe(a, loc="lower right", fontsize=6)
    _dump("FigS1a_detection_vs_effect.csv", pts)

    b = ax["B"];
    g = tract.groupby("line").agg(n_drugs=("n_drugs", "sum"),
                                  n_genes=("gene", "count"),
                                  n_with_drug=("n_drugs", lambda s: int((s > 0).sum())))
    order = [2, 1]
    lab = {2: "Neuronal\ncandidates", 1: "Carbohydrate\nenzymes"}
    xs = np.arange(2)
    b.bar(xs - 0.19, [g.loc[i, "n_drugs"] for i in order], width=0.36,
          color=C["neuronal"], edgecolor="none", label="Compounds")
    b.bar(xs + 0.19, [g.loc[i, "n_with_drug"] for i in order], width=0.36,
          color=C["accent"], edgecolor="none", label="Genes with \u22651")
    for x, i in zip(xs, order):
        b.text(x - 0.19, g.loc[i, "n_drugs"] + 4, str(int(g.loc[i, "n_drugs"])),
               ha="center", fontsize=5.6)
        b.text(x + 0.19, g.loc[i, "n_with_drug"] + 4, str(int(g.loc[i, "n_with_drug"])),
               ha="center", fontsize=5.6)
    b.set_xticks(xs); b.set_xticklabels([lab[i] for i in order], fontsize=5.8)
    b.set_ylabel("Count"); b.set_ylim(0, 380)
    _nolegendframe(b, loc="upper right", fontsize=5.6)
    _dump("FigS1b_compound_counts.csv", g.reset_index())

    c = ax["C"];
    nn = det[det.gene_set == "reported_neuronal_10"].sort_values("detection_pct")
    y = np.arange(len(nn))
    c.barh(y, nn.detection_aneuploid_pct, height=0.68, color=C["aneuploid"],
           edgecolor="none", label="Aneuploid")
    c.barh(y, -nn.detection_diploid_pct, height=0.68, color=C["diploid"],
           edgecolor="none", label="Diploid")
    c.axvline(0, color="black", lw=0.5)
    c.set_yticks(y); c.set_yticklabels(nn.gene, fontsize=5.4)
    c.set_xlabel("Nuclei with \u22651 count (%)")
    c.set_xticks([-10, 0, 10, 20]); c.set_xticklabels(["10", "0", "10", "20"])
    _nolegendframe(c, loc="lower right", fontsize=5.4)
    _dump("FigS1c_neuronal_detection.csv", nn)

    pf.panel_labels(fig, ax, case="lower")
    return fig, dict(panels={
        "a": "results/targets/enzyme_compartment_contrast.csv + results/targets/target_enrichment.csv",
        "b": "data/target_tractability.csv",
        "c": "results/expression_programs/detectability_resource.csv (reported_neuronal_10)"})


GAG_STEPS = [("M_129", "M_129\nlinker\ninitiation"), ("M_130", "M_130\nelongation"),
             ("M_131", "M_131\nepimerization"), ("M_110", "M_110\nhyaluronan")]
MOD_LAB = {"M_129": "M_129 GAG linker initiation", "M_130": "M_130 chondroitin elongation",
           "M_131": "M_131 dermatan epimerization", "M_110": "M_110 hyaluronan",
           "M_49": "M_49 glutamate → GABA", "M_85": "M_85 aspartate uptake"}
MOD_COL = {"M_129": C["XYLT1"], "M_130": C["grey"], "M_131": C["grey"], "M_110": C["accent"],
           "M_49": "black", "M_85": "black"}
MOD_OFF = {"M_129": (5, -7), "M_49": (5, 6), "M_85": (6, -7), "M_110": (-6, 6),
           "M_130": (8, 9), "M_131": (8, -8)}


def figure_2():
    mod = R("metabolic_flux", "scfea_module_axis.csv")
    per = R("metabolic_flux", "scfea_module_axis_per_sample.csv")
    enz = R("metabolic_flux", "enzyme_axis_screen.csv")

    fig, ax = pf.figure_mosaic("""
        AB
        CC
        """, width_mm=183, height_mm=122)

    a = ax["A"]
    m = mod.copy(); m["mlq"] = -np.log10(m.q)
    sig = m.q < 0.05
    a.scatter(m.median_rho[~sig], m.mlq[~sig], s=7, facecolor="white", edgecolor=C["null"],
              lw=0.5, zorder=2)
    a.scatter(m.median_rho[sig], m.mlq[sig], s=8, color=C["grey"], lw=0, zorder=3)
    for mid, lab in MOD_LAB.items():
        r = m[m.module == mid].iloc[0]
        a.scatter(r.median_rho, r.mlq, s=20, color=MOD_COL[mid], lw=0, zorder=4, marker="D")
        ox, oy = MOD_OFF[mid]
        a.annotate(lab, (r.median_rho, r.mlq), xytext=(ox, oy), textcoords="offset points",
                   fontsize=5.2, ha="right" if ox < 0 else "left", va="center", color="#222222",
                   arrowprops=dict(arrowstyle="-", lw=0.4, color="#777777",
                                   shrinkA=0, shrinkB=2.5))
    a.axhline(-np.log10(0.05), color="black", ls="--", lw=0.5, zorder=1)
    a.axvline(0, color="black", lw=0.5, zorder=1)
    a.set_xlabel("Median within-sample ρ with state axis\n(AC/MES-like ←  → OPC/NPC-like)")
    a.set_ylabel("−log$_{10}$ q")
    a.set_xlim(m.median_rho.min() - 0.03, m.median_rho.max() + 0.17)
    a.set_ylim(-0.7, m.mlq.max() * 1.16)
    _dump("Fig2a_scfea_module_axis.csv", m.drop(columns="mlq"))

    b = ax["B"]
    rng = np.random.default_rng(3)
    ymax = per[per.module.isin([s for s, _ in GAG_STEPS])].rho.max()
    for i, (mid, _) in enumerate(GAG_STEPS):
        v = per.loc[per.module == mid, "rho"].values
        b.scatter(i + (rng.random(len(v)) - 0.5) * 0.36, v, s=5, color=MOD_COL[mid],
                  alpha=0.75, lw=0, zorder=2)
        b.plot([i - .26, i + .26], [np.median(v)] * 2, color="black", lw=1.1, zorder=3)
        q = mod.loc[mod.module == mid, "q"].iloc[0]
        qs = f"q = {q:.1e}".replace("e-0", "e−") if q < 1e-3 else f"q = {q:.2g}"
        b.text(i, ymax + 0.05, qs, ha="center", va="bottom", fontsize=5.0)
    b.axhline(0, color="black", lw=0.5, zorder=1)
    b.set_xticks(range(len(GAG_STEPS))); b.set_xticklabels([l for _, l in GAG_STEPS], fontsize=5.4)
    b.set_ylabel("Within-sample ρ (flux vs state axis)")
    b.set_ylim(per.loc[per.module.isin([s for s, _ in GAG_STEPS]), "rho"].min() - 0.05, ymax + 0.16)
    _dump("Fig2b_gag_steps_per_sample.csv", per[per.module.isin([s for s, _ in GAG_STEPS])])

    c = ax["C"]
    e = enz.copy()
    cf = e.carried_forward.astype(bool)
    c.scatter(e.detection_pct[~cf], e.median_rho[~cf], s=10, facecolor="white",
              edgecolor=C["grey"], lw=0.5, zorder=2, label="Not carried forward")
    c.scatter(e.detection_pct[cf], e.median_rho[cf], s=13, color=C["enzyme"], lw=0, zorder=3,
              label="Carried forward (28)")
    for g in ["CHST11", "XYLT1", "DSEL", "GALNT13"]:
        r = e[e.gene == g].iloc[0]
        c.scatter(r.detection_pct, r.median_rho, s=22, color=C[g], lw=0, zorder=4, marker="D")
        c.annotate(g, (r.detection_pct, r.median_rho), xytext=(4, 2), textcoords="offset points",
                   fontsize=5.6)
    c.axvline(5, color="black", ls="--", lw=0.6, zorder=1)
    c.axhline(0, color="black", lw=0.5, zorder=1)
    c.set_xscale("log")
    c.set_xticks([1, 2, 5, 10, 20, 40])
    c.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    c.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    c.set_xlabel("Detection in malignant nuclei (%)")
    c.set_ylabel("Median within-sample ρ\nwith state axis")
    _nolegendframe(c, loc="lower right", fontsize=5.6)
    _dump("Fig2c_enzyme_axis_screen.csv", e)

    pf.panel_labels(fig, ax, case="lower")
    return fig, dict(panels={
        "a": "results/metabolic_flux/scfea_module_axis.csv",
        "b": "results/metabolic_flux/scfea_module_axis_per_sample.csv (+ q from scfea_module_axis.csv)",
        "c": "results/metabolic_flux/enzyme_axis_screen.csv"})


def figure_3():
    enz = R("targets", "enzyme_compartment_contrast.csv")
    per = R("targets", "per_tumor_expression.csv")
    dmn = R("targets", "detection_matched_single_gene.csv")
    dep = R("targets", "depth_sensitivity.csv")
    depj = J("targets", "depth_sensitivity.json")
    st = R("targets", "cell_state_scope.csv")

    fig, ax = pf.figure_mosaic("""
        AAB
        AAB
        CDE
        FFF
        """, width_mm=183, height_mm=176)

    a = ax["A"];
    e = enz.sort_values("delta_log").reset_index(drop=True)
    y = np.arange(len(e))
    for i, r in e.iterrows():
        sig = (r.p_holm < 0.05) and bool(r.interpretable)
        col = C.get(r.gene, C["enzyme"]) if sig else C["grey"]
        a.plot([0, r.delta_log], [i, i], color=col, lw=0.9, zorder=2,
               alpha=1.0 if r.interpretable else 0.45)
        a.scatter(r.delta_log, i, s=18 if sig else 11,
                  facecolor=col if r.interpretable else "white",
                  edgecolor=col, lw=0.6, zorder=3,
                  marker="D" if sig else "o")
    a.axvline(0, color="black", lw=0.5)
    a.set_yticks(y); a.set_yticklabels(e.gene, fontsize=5.0)
    a.set_xlabel("\u0394 log expression (aneuploid \u2212 diploid)")
    a.set_ylim(-1, len(e))
    hs = [Line2D([0], [0], marker="D", color="none", markerfacecolor=C["enzyme"],
                 markeredgecolor=C["enzyme"], markersize=3.4),
          Line2D([0], [0], marker="o", color="none", markerfacecolor=C["grey"],
                 markeredgecolor=C["grey"], markersize=2.8),
          Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                 markeredgecolor=C["grey"], markersize=2.8)]
    a.legend(hs, ["Holm P < 0.05", "not significant", "<5% detection"], frameon=False,
             fontsize=5.4, loc="lower right", handlelength=1.0)
    _dump("Fig3a_enzyme_contrast.csv", e)

    b = ax["B"];
    ch = per[per.gene == "CHST11"]
    for _, r in ch.iterrows():
        up = r.aneuploid > r.diploid
        b.plot([0, 1], [r.diploid, r.aneuploid], lw=0.6,
               color=C["CHST11"] if up else C["grey"], alpha=0.85, zorder=2)
    b.scatter(np.zeros(len(ch)), ch.diploid, s=7, color=C["diploid"], zorder=3, lw=0)
    b.scatter(np.ones(len(ch)), ch.aneuploid, s=7, color=C["CHST11"], zorder=3, lw=0)
    b.set_xticks([0, 1]); b.set_xticklabels(["Diploid", "Aneuploid"])
    b.set_xlim(-0.4, 1.4); b.set_ylabel("CHST11, mean log expression")
    _dump("Fig3b_per_tumor_CHST11.csv", ch)

    c = ax["C"];
    dd = dmn.set_index("gene").loc[["CHST11", "XYLT1", "DSEL", "GALNT13"]].reset_index()
    y = np.arange(len(dd))[::-1]
    for i, r in zip(y, dd.itertuples()):
        c.barh(i, r.null_97_5, height=0.42, color=C["null"], edgecolor="none", zorder=1)
        c.plot([r.null_median] * 2, [i - .21, i + .21], color="black", lw=0.8, zorder=2)
        c.scatter(r.delta_log, i, s=16, color=C.get(r.gene, C["enzyme"]), zorder=3,
                  marker="D", lw=0)
    c.set_yticks(y); c.set_yticklabels(dd.gene, fontsize=6)
    c.set_xlabel("\u0394 log expression")
    _dump("Fig3c_detection_matched_null.csv", dd)

    d = ax["D"];
    dm = pd.DataFrame(depj["depth_median_counts"]).T.reset_index().rename(
        columns={"index": "gsm"})
    lim = [dm[["aneuploid", "diploid"]].min().min() * 0.8,
           dm[["aneuploid", "diploid"]].max().max() * 1.25]
    d.plot(lim, lim, color="black", lw=0.5, ls="--", zorder=1)
    d.scatter(dm.diploid, dm.aneuploid, s=9, color=C["aneuploid"], lw=0, zorder=3)
    d.set_xscale("log"); d.set_yscale("log"); d.set_xlim(*lim); d.set_ylim(*lim)
    d.set_xlabel("Diploid, median counts"); d.set_ylabel("Aneuploid, median counts")
    for axx in (d.xaxis, d.yaxis):
        axx.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{int(v):,}"))
        axx.set_minor_formatter(mpl.ticker.NullFormatter())
    d.set_xticks([500, 1000, 2000, 4000]); d.set_yticks([500, 1000, 2000, 4000])
    _dump("Fig3d_depth_per_sample.csv", dm)

    e2 = ax["E"];
    dp = dep.set_index("gene").loc[GENE3].reset_index()
    xs = np.arange(len(dp))
    e2.bar(xs - 0.19, dp.delta_raw, width=0.36, color=C["grey"], edgecolor="none",
           label="Unadjusted")
    e2.bar(xs + 0.19, dp.coef_depth_adjusted, width=0.36,
           color=[C[g] for g in dp.gene], edgecolor="none", label="Depth-adjusted")
    e2.axhline(0, color="black", lw=0.5)
    e2.set_xticks(xs); e2.set_xticklabels(dp.gene, fontsize=6)
    e2.set_ylabel("\u0394 log expression")
    e2.set_ylim(0, max(dp.delta_raw.max(), dp.coef_depth_adjusted.max()) * 1.18)
    _nolegendframe(e2, loc="upper right", fontsize=5.4)
    _dump("Fig3e_depth_adjusted.csv", dp)

    f = ax["F"]
    states = ["OPC", "NPC", "AC", "MES"]
    w = 0.26
    for gi, g in enumerate(GENE3):
        s = st[st.gene == g].set_index("state").loc[states]
        xs = np.arange(len(states)) + (gi - 1) * w
        f.bar(xs, s.delta_log, width=w, color=C[g], edgecolor="none", label=g)
        for x, (_, r) in zip(xs, s.iterrows()):
            if r.p_perm < 0.05:
                f.text(x, r.delta_log + 0.012, "*", ha="center", va="bottom", fontsize=7)
    f.axhline(0, color="black", lw=0.5)
    f.set_xticks(np.arange(len(states)))
    f.set_xticklabels([f"{s}-like (n={int(st[st.state==s].n_samples.iloc[0])})" for s in states],
                      fontsize=5.8)
    f.set_ylabel("Δ log expression")
    f.set_ylim(min(0, st.delta_log.min() * 1.2), st[st.gene.isin(GENE3)].delta_log.max() * 1.2)
    _nolegendframe(f, loc="upper right", fontsize=5.6, ncol=3, columnspacing=0.8)
    _dump("Fig3f_cell_state_scope.csv", st)

    pf.panel_labels(fig, ax, case="lower")
    return fig, dict(panels={
        "a": "results/targets/enzyme_compartment_contrast.csv",
        "b": "results/targets/per_tumor_expression.csv (CHST11)",
        "c": "results/targets/detection_matched_single_gene.csv",
        "d": "results/targets/depth_sensitivity.json (depth_median_counts)",
        "e": "results/targets/depth_sensitivity.csv",
        "f": "results/targets/cell_state_scope.csv"})


def figure_s3():
    sp = R("targets", "enzyme_set_per_sample.csv")
    ss = J("targets", "enzyme_set_summary.json")
    sn = J("targets", "enzyme_set_null_size_preserving.json")
    pm = R("targets", "pathway_membership.csv")

    fig, ax = pf.figure_mosaic("""
        AB
        CC
        """, width_mm=183, height_mm=104)

    a = ax["A"];
    cols = ["GAG_biosynthesis", "NEG_CONTROL_translation"]
    rng = np.random.default_rng(0)
    for i, col in enumerate(cols):
        x = i + (rng.random(len(sp)) - 0.5) * 0.28
        a.scatter(x, sp[col], s=9, color=[C["enzyme"], C["grey"]][i], lw=0, zorder=3, alpha=0.85)
        a.plot([i - .26, i + .26], [sp[col].mean()] * 2, color="black", lw=1.1, zorder=4)
    a.axhline(0, color="black", lw=0.5, zorder=1)
    a.set_xticks([0, 1]); a.set_xticklabels(["28 enzymes\n(program)", "Translation\n(neg. control)"],
                                            fontsize=5.8)
    a.set_xlim(-0.5, 1.5); a.set_ylabel("\u0394 AUCell (aneuploid \u2212 diploid)")
    _dump("FigS3a_enzyme_set_per_sample.csv", sp)

    b = ax["B"];
    obs = ss["GAG_biosynthesis"]["delta"]
    rows = [("500 sets\ndetection-matched", ss["detection_matched_null"]["null_median"],
             ss["detection_matched_null"]["null_97_5"]),
            ("1,000 sets\nsize-preserving", sn["primary"]["null_median"],
             sn["primary"]["null_97_5"])]
    for i, (nm, med, hi) in enumerate(rows):
        b.barh(i, hi, height=0.34, color=C["null"], edgecolor="none", zorder=1)
        b.plot([med] * 2, [i - .17, i + .17], color="black", lw=0.9, zorder=2)
    b.axvline(obs, color=C["accent"], lw=1.2, zorder=3)
    b.set_yticks([0, 1]); b.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    b.set_ylim(-0.6, 2.2); b.invert_yaxis()
    b.set_xlabel("\u0394 AUCell")
    hs = [Rectangle((0, 0), 1, 1, color=C["null"]), Line2D([0], [0], color="black", lw=0.9),
          Line2D([0], [0], color=C["accent"], lw=1.2)]
    b.legend(hs, ["null 0\u201397.5th pct", "null median", "observed"], frameon=False,
             fontsize=5.4, loc="lower right", handlelength=1.2)
    _dump("FigS3b_enzyme_set_null.csv", pd.DataFrame(
        [dict(construction=r[0].replace("\n", " "), null_median=r[1], null_97_5=r[2],
              observed=obs) for r in rows]))

    c = ax["C"];
    pm = pm.copy()
    pm["branch"] = pm.branch.fillna("unassigned")
    SHORT = {"Chondroitin / dermatan sulfate": "CS / DS",
             "Heparan sulfate / heparin": "HS",
             "Glycosaminoglycan degradation": "GAG degradation",
             "Mucin-type O-glycan biosynthesis": "Mucin O-glycan",
             "Other O-linked glycosylation": "Other O-linked",
             "Sialylation and glycosphingolipid": "Sialyl. / GSL",
             "unassigned": "Unassigned"}
    pm["primary_branch"] = pm.branch.str.split(";").str[0].str.strip().map(
        lambda b: SHORT.get(b, b))
    order = (pm.groupby("primary_branch").delta_log.max()
             .sort_values(ascending=False).index.tolist())
    xoff, ticks, labs, spans = 0, [], [], []
    for bi, br in enumerate(order):
        g = pm[pm.primary_branch == br].sort_values("delta_log", ascending=False)
        xs = np.arange(len(g)) + xoff
        if bi % 2 == 0:
            c.add_patch(Rectangle((xoff - 0.5, 0), len(g), 1, color="#F2F2F2",
                                  zorder=0, transform=c.get_xaxis_transform()))
        for x, r in zip(xs, g.itertuples()):
            sig = (r.p_holm < 0.05) and bool(r.interpretable)
            col = C.get(r.gene, C["enzyme"]) if sig else C["grey"]
            c.plot([x, x], [0, r.delta_log], color=col, lw=0.9, zorder=2,
                   alpha=1.0 if r.interpretable else 0.45)
            c.scatter(x, r.delta_log, s=17 if sig else 10,
                      facecolor=col if r.interpretable else "white",
                      edgecolor=col, lw=0.6, marker="D" if sig else "o", zorder=3)
        ticks += list(xs); labs += list(g.gene); spans.append((br, xs.mean()))
        xoff += len(g) + 1
    c.axhline(0, color="black", lw=0.5)
    c.set_xticks(ticks); c.set_xticklabels(labs, rotation=90, fontsize=5.0)
    c.set_xlim(-1, xoff - 1)
    c.set_ylim(pm.delta_log.min() * 1.18, pm.delta_log.max() * 1.60)
    for br, xm in spans:
        c.text(xm, 0.99, br, transform=c.get_xaxis_transform(), ha="center", va="top",
               fontsize=5.2, color="#333333")
    c.set_ylabel("\u0394 log expression")
    _dump("FigS3c_pathway_branches.csv", pm)

    pf.panel_labels(fig, ax, case="lower")
    return fig, dict(panels={
        "a": "results/targets/enzyme_set_per_sample.csv",
        "b": "results/targets/enzyme_set_summary.json + enzyme_set_null_size_preserving.json",
        "c": "results/targets/pathway_membership.csv"})


STRUCT_ORDER = ["Leading Edge", "Infiltrating Tumor", "Cellular Tumor",
                "Microvascular proliferation", "Pseudopalisading cells around necrosis"]
STRUCT_LAB = {"Leading Edge": "Leading\nedge", "Infiltrating Tumor": "Infiltrating\ntumor",
              "Cellular Tumor": "Cellular\ntumor",
              "Microvascular proliferation": "Microvascular\nprolif.",
              "Pseudopalisading cells around necrosis": "Pseudopalisading\nnecrosis"}


def figure_s4():
    wells = R("independent_cohorts", "ivygap_targets_wells.csv")
    tcga = R("independent_cohorts", "tcga_targets_survival.csv")
    cgga = R("independent_cohorts", "cgga_targets_survival.csv")
    cov = R("independent_cohorts", "survival_covariate_adjusted.csv")
    pr = R("independent_cohorts", "cgga_targets_primary_recurrent.csv")

    fig, ax = pf.figure_mosaic("""
        AAB
        AAC
        """, width_mm=183, height_mm=100)

    a = ax["A"];
    present = [s for s in STRUCT_ORDER if s in set(wells.structure)]
    rng = np.random.default_rng(1)
    for gi, g in enumerate(GENE3):
        xs, med = [], []
        for si, s in enumerate(present):
            v = wells.loc[wells.structure == s, g].dropna().values
            x = si + (gi - 1) * 0.25
            a.scatter(x + (rng.random(len(v)) - 0.5) * 0.14, v, s=3.5, color=C[g],
                      alpha=0.45, lw=0, zorder=2)
            a.plot([x - .10, x + .10], [np.median(v)] * 2, color=C[g], lw=1.3, zorder=4)
            xs.append(x); med.append(np.median(v))
        a.plot(xs, med, color=C[g], lw=0.9, ls="-", zorder=3, label=g, alpha=0.9)
    a.set_xticks(np.arange(len(present)))
    a.set_xticklabels([STRUCT_LAB[s] for s in present], fontsize=5.6)
    a.set_ylabel("Expression (IVY-GAP wells, log2)")
    _nolegendframe(a, loc="upper left", fontsize=6, ncol=3, columnspacing=0.8)
    _dump("FigS4a_ivygap_wells.csv", wells)

    b = ax["B"];
    uni = pd.concat([tcga.assign(cohort="TCGA-GBM"), cgga.assign(cohort="CGGA")])
    uni = uni[uni.gene.isin(GENE3)]
    adj = cov[(cov.gene.isin(GENE3)) & (cov.model.str.contains("adjust|covariate", case=False,
                                                               na=False))]
    rows = []
    for coh in ["TCGA-GBM", "CGGA"]:
        for g in GENE3:
            u = uni[(uni.cohort == coh) & (uni.gene == g)]
            if len(u):
                rows.append((coh, g, "Univariate", u.cox_HR_per_SD.iloc[0],
                             u.cox_CI_low.iloc[0], u.cox_CI_high.iloc[0]))
            am = adj[(adj.cohort.str.upper().str.contains(coh.split("-")[0])) & (adj.gene == g)]
            if len(am):
                am = am.iloc[0]
                rows.append((coh, g, "Adjusted", am.HR_per_SD, am.CI_low, am.CI_high))
    F = pd.DataFrame(rows, columns=["cohort", "gene", "model", "HR", "lo", "hi"])
    ylab, yv = [], []
    for i, r in enumerate(F.itertuples()):
        y = len(F) - i
        mk = "o" if r.model == "Univariate" else "s"
        b.plot([r.lo, r.hi], [y, y], color=C[r.gene], lw=0.8, zorder=2)
        b.scatter(r.HR, y, s=11, color=C[r.gene], marker=mk, lw=0, zorder=3)
        tag = "univ." if r.model == "Univariate" else "adj."
        ylab.append(f"{r.gene} \u00b7 {r.cohort.split('-')[0]} \u00b7 {tag}"); yv.append(y)
    b.axvline(1, color="black", lw=0.5, ls="--")
    b.set_xscale("log"); b.set_yticks(yv); b.set_yticklabels(ylab, fontsize=4.8)
    b.set_xlabel("Hazard ratio per s.d.")
    b.set_xticks([0.5, 1, 2]); b.set_xticklabels(["0.5", "1", "2"])
    b.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    b.set_xlim(0.45, 2.2)
    hs = [Line2D([0], [0], marker="o", color="none", markerfacecolor=C["grey"], markersize=2.6),
          Line2D([0], [0], marker="s", color="none", markerfacecolor=C["grey"], markersize=2.6)]
    b.legend(hs, ["Univariate", "Adjusted"], frameon=False, fontsize=5.0,
             loc="lower right", handlelength=0.8)
    _dump("FigS4b_survival_forest.csv", F)

    c = ax["C"];
    p = pr[pr.gene.isin(GENE3)].set_index("gene").loc[GENE3].reset_index()
    xs = np.arange(len(p))
    c.bar(xs, p.log2FC_rec_vs_prim, width=0.5, color=[C[g] for g in p.gene], edgecolor="none")
    c.axhline(0, color="black", lw=0.5)
    c.set_xticks(xs); c.set_xticklabels(p.gene, fontsize=5.8)
    c.set_ylabel("log2 FC\n(recurrent \u2212 primary, CGGA)")
    _dump("FigS4c_cgga_primary_recurrent.csv", pr)

    pf.panel_labels(fig, ax, case="lower")
    return fig, dict(panels={
        "a": "results/independent_cohorts/ivygap_targets_wells.csv",
        "b": "results/independent_cohorts/{tcga,cgga}_targets_survival.csv + survival_covariate_adjusted.csv",
        "c": "results/independent_cohorts/cgga_targets_primary_recurrent.csv"})


def figure_8():
    cspg = R("targets", "cspg_pathway_consistency.csv")
    cspgj = J("targets", "cspg_pathway_consistency.json")
    dm = pd.read_csv(os.path.join(DATA, "depmap_chst11_dependency_by_cellline.csv"))

    fig, ax = pf.figure_mosaic("""
        AB
        CD
        """, width_mm=183, height_mm=104)

    a = ax["A"];
    sp = cspgj["per_sample_delta_vs_CHST11_spearman"]
    cc = cspg.copy()
    cc["rho"] = cc.gene.map(lambda g: sp.get(g, {}).get("spearman_r", np.nan))
    cc.loc[cc.gene == "CHST11", "rho"] = 1.0
    for _, r in cc.iterrows():
        col = C["CHST11"] if r.gene == "CHST11" else C["enzyme"]
        a.scatter(r.delta_log, r.rho, s=8 + r.detection_pct * 1.5,
                  facecolor=col if r.p_perm < 0.05 else "white", edgecolor=col, lw=0.7, zorder=3)
        a.annotate(r.gene, (r.delta_log, r.rho), textcoords="offset points",
                   xytext=(4.5, 2.5), fontsize=5.6)
    a.axhline(0, color="black", lw=0.5); a.axvline(0, color="black", lw=0.5)
    a.set_xlabel("\u0394 log expression (aneuploid \u2212 diploid)")
    a.set_ylabel("Spearman \u03c1 with CHST11\nper-sample \u0394 (n = 20)")
    a.set_xlim(-0.06, 0.62); a.set_ylim(-0.12, 1.14)
    _dump("Fig8a_cspg_consistency.csv", cc)

    b = ax["B"];
    cd = cspg.sort_values("detection_pct")
    y = np.arange(len(cd))
    b.barh(y + 0.19, cd.detection_aneuploid_pct, height=0.36, color=C["aneuploid"],
           edgecolor="none", label="Aneuploid")
    b.barh(y - 0.19, cd.detection_diploid_pct, height=0.36, color=C["diploid"],
           edgecolor="none", label="Diploid")
    b.set_yticks(y); b.set_yticklabels(cd.gene, fontsize=5.8)
    b.set_xlabel("Nuclei with \u22651 count (%)")
    _nolegendframe(b, loc="lower right", fontsize=5.6)
    _dump("Fig8b_cspg_detection.csv", cd)

    c = ax["C"];
    dm = dm.dropna(subset=["CHST11_Chronos_GeneEffect_24Q4"])
    grp = [("GBM", dm[dm.is_GBM]),
           ("Other diffuse\nglioma", dm[(dm.is_Glioma_any) & (~dm.is_GBM)]),
           ("All other\nlineages", dm[~dm.is_Glioma_any])]
    rng = np.random.default_rng(2)
    for i, (nm, g) in enumerate(grp):
        v = g.CHST11_Chronos_GeneEffect_24Q4.values
        c.scatter(i + (rng.random(len(v)) - 0.5) * 0.4, v, s=2.2,
                  color=[C["CHST11"], C["accent"], C["grey"]][i], alpha=0.45, lw=0, zorder=2)
        c.plot([i - .28, i + .28], [np.median(v)] * 2, color="black", lw=1.2, zorder=4)
    c.axhline(0, color="black", lw=0.5, zorder=1)
    c.axhline(-0.5, color=C["XYLT1"], lw=0.7, ls="--", zorder=1)
    c.set_xticks(range(3))
    c.set_xticklabels([f"{nm}\n(n={len(g)})" for nm, g in grp], fontsize=5.6)
    c.set_ylabel("CHST11 CRISPR gene effect\n(Chronos, DepMap 24Q4)")
    _dump("Fig8c_depmap_dependency.csv",
          pd.concat([g.assign(group=nm.replace("\n", " ")) for nm, g in grp]))

    d = ax["D"];
    gl = dm[dm.is_Glioma_any].dropna(subset=["CHST11_Expression_TPM_Logp1_24Q4"])
    gb = gl[gl.is_GBM]; og = gl[~gl.is_GBM]
    d.scatter(og.CHST11_Expression_TPM_Logp1_24Q4, og.CHST11_Chronos_GeneEffect_24Q4,
              s=7, facecolor="none", edgecolor=C["accent"], lw=0.6, zorder=3,
              label=f"Other glioma (n={len(og)})")
    d.scatter(gb.CHST11_Expression_TPM_Logp1_24Q4, gb.CHST11_Chronos_GeneEffect_24Q4,
              s=8, color=C["CHST11"], lw=0, zorder=4, label=f"GBM (n={len(gb)})")
    z = np.polyfit(gl.CHST11_Expression_TPM_Logp1_24Q4, gl.CHST11_Chronos_GeneEffect_24Q4, 1)
    xx = np.linspace(gl.CHST11_Expression_TPM_Logp1_24Q4.min(),
                     gl.CHST11_Expression_TPM_Logp1_24Q4.max(), 50)
    d.plot(xx, np.polyval(z, xx), color="black", lw=0.7, ls="--", zorder=2)
    d.axhline(0, color="black", lw=0.5, zorder=1)
    d.set_xlabel(r"CHST11 expression (log$_2$ TPM+1)")
    d.set_ylabel("CHST11 CRISPR gene effect")
    _nolegendframe(d, loc="lower right", fontsize=5.6)
    _dump("Fig8d_depmap_expression.csv", gl)

    pf.panel_labels(fig, ax, case="lower")
    return fig, dict(panels={
        "a": "results/targets/cspg_pathway_consistency.{csv,json}",
        "b": "results/targets/cspg_pathway_consistency.csv",
        "c": "data/depmap_chst11_dependency_by_cellline.csv",
        "d": "data/depmap_chst11_dependency_by_cellline.csv"})


BUILDERS = [("Figure1", figure_1), ("Figure2", figure_2), ("Figure3", figure_3), ("Figure8", figure_8),
            ("FigureS1", figure_s1), ("FigureS3", figure_s3), ("FigureS4", figure_s4)]

if __name__ == "__main__":
    pf.check_fonts()
    only = sys.argv[1:] or [name for name, _ in BUILDERS]
    for name, fn in BUILDERS:
        if name not in only:
            continue
        fig, prov = fn()
        prov = {**prov, "script": "figures/make_figures.py"}
        res = pf.save_figure(fig, os.path.join(FIGD, name), width_mm=183,
                             provenance=prov, verbose=False)
        aud = res.get("audit", {})
        n_ov = len(aud.get("overlaps", []) or []) if isinstance(aud, dict) else 0
        print(f"{name}: written, {n_ov} overlap flags")
        plt.close(fig)
