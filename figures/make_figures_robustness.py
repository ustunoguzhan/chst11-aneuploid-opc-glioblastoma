import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

R = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 5.8, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 600,
})
C = dict(CHST11="#0072B2", XYLT1="#D55E00", DSEL="#009E73", chr="#E69F00", grey="#8C8C8C",
         ref="#555555", prim="#CC79A7", rec="#0072B2", light="#BBBBBB")


def fp(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}×10$^{{{int(e)}}}$"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def jitter(n, w=0.12, seed=0):
    return np.random.default_rng(seed).uniform(-w, w, n)


RB = "results/robustness/"
INPUTS = {
    "Figure4": [RB + "cnv_classification/cnv_robustness_per_sample.csv",
                RB + "cnv_classification/cnv_robustness_summary.csv",
                RB + "cnv_classification/threshold_free_per_sample.csv"],
    "Figure5": [RB + "sulfotransferase_family/chst_family_evidence_matrix.csv",
                RB + "sulfotransferase_family/per_sample_delta_family.csv"],
    "Figure6": [RB + "cs_pathway_coherence/CHST11_vs_set_delta.csv",
                RB + "cs_pathway_coherence/gene_level_leading_edge.csv",
                RB + "cs_pathway_coherence/null_<set>.csv",
                RB + "cs_pathway_coherence/set_level_summary.csv",
                RB + "cs_pathway_coherence/single_nucleus_coexpression_per_sample.csv"],
    "Figure7": [RB + "primary_recurrent/paired_longitudinal_summary.csv",
                RB + "primary_recurrent/per_sample_effects_with_progression.csv",
                RB + "primary_recurrent/primary_recurrent_summary.csv"],
    "FigureS2": [RB + "patient_level/leave_one_patient_out.csv",
                 RB + "patient_level/leave_one_sample_out.csv",
                 RB + "patient_level/summary.csv"],
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    prov = {"inputs": INPUTS[name], "script": "figures/make_figures_robustness.py"}
    (OUT / f"{name}.provenance.json").write_text(json.dumps(prov, indent=2))


S = pd.read_csv(R / "cnv_classification/cnv_robustness_summary.csv")
P = pd.read_csv(R / "cnv_classification/cnv_robustness_per_sample.csv")
TF = pd.read_csv(R / "cnv_classification/threshold_free_per_sample.csv")
order = ["primary", "p95_rederived", "p97_5", "p99", "gap_p90_p99", "hallmark_7_10", "concordant",
         "ref_primary", "ref_allnm", "ref_microglia", "ref_oligo"]
nice = {"primary": "Primary calls (P95)", "p95_rederived": "P95 re-derived", "p97_5": "Stricter, P97.5",
        "p99": "Stricter, P99", "gap_p90_p99": "Gap: ≤P90 vs >P99", "hallmark_7_10": "chr7+/chr10− hallmark",
        "concordant": "Score AND hallmark", "ref_primary": "Re-run, primary reference",
        "ref_allnm": "Re-run, all non-malignant", "ref_microglia": "Re-run, microglia only",
        "ref_oligo": "Re-run, oligodendrocytes only"}
fig = plt.figure(figsize=(183 * MM, 118 * MM))
gs = fig.add_gridspec(1, 4, width_ratios=[1.45, 0.8, 0.8, 0.8], wspace=0.12, left=0.225, right=0.985,
                      top=0.95, bottom=0.6)
gsb = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.3, left=0.075, right=0.985, top=0.44,
                       bottom=0.1)
ax = [fig.add_subplot(gs[0, i]) for i in range(4)]
y = np.arange(len(order))[::-1]
for gi, (gene, off) in enumerate([("CHST11", 0.18), ("XYLT1", 0.0), ("DSEL", -0.18)]):
    sub = S[S.gene == gene].set_index("scheme").reindex(order)
    for yy, (sc, r) in zip(y, sub.iterrows()):
        d = P[(P.scheme == sc)][gene].dropna().values
        bs = np.random.default_rng(1).choice(d, (4000, len(d))).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax[0].plot([lo, hi], [yy + off] * 2, color=C[gene], lw=0.8)
        ax[0].plot(r.sample_mean, yy + off, "o", ms=3.2, color=C[gene],
                   mfc=C[gene] if r.sample_p_signflip < 0.05 else "white", mew=0.7)
ax[0].set_yticks(y)
ax[0].set_yticklabels([f"{nice[s]} (n={int(S[(S.scheme == s) & (S.gene == 'CHST11')].n_samples.iloc[0])})"
                       for s in order])
ax[0].axvline(0, color="k", lw=0.5, ls="--")
ax[0].axhline(len(order) - 7 - 0.5, color=C["light"], lw=0.5)
ax[0].set_xlabel("Aneuploid − diploid, log1p(CP10k)\n(sample mean, 95% bootstrap CI)")
ax[0].set_title("CHST11, XYLT1, DSEL: sample level", loc="left")
for g in ["CHST11", "XYLT1", "DSEL"]:
    ax[0].plot([], [], "o", color=C[g], ms=3, label=g)
ax[0].legend(loc="upper right", frameon=False, handletextpad=0.1, borderaxespad=0.1)
label(ax[0], "a", x=-0.78)

for k, (col_m, col_p, ttl) in enumerate([("patient_mean", "patient_p_signflip", "Patient level"),
                                         ("depth_adj_mean", "depth_adj_p", "Depth-adjusted")], start=1):
    sub = S[S.gene == "CHST11"].set_index("scheme").reindex(order)
    for yy, (sc, r) in zip(y, sub.iterrows()):
        sig = r[col_p] < 0.05
        ax[k].plot(r[col_m], yy, "o", ms=3.4, color=C["CHST11"], mfc=C["CHST11"] if sig else "white", mew=0.7)
        ax[k].text(2.15, yy, fp(r[col_p]), va="center", ha="right", fontsize=5.2)
    ax[k].set_yticks(y)
    ax[k].set_yticklabels([])
    ax[k].axvline(0, color="k", lw=0.5, ls="--")
    ax[k].set_xlim(-0.1, 2.2)
    ax[k].set_xticks([0, 0.5, 1, 1.5])
    ax[k].axhline(len(order) - 7 - 0.5, color=C["light"], lw=0.5)
    ax[k].set_title(f"CHST11, {ttl.lower()}", loc="left")
    ax[k].set_xlabel("Δ (patient mean)" if k == 1 else "Depth-adjusted coefficient")
    label(ax[k], "bc"[k - 1], x=-0.1)
sub = S[S.gene == "CHST11"].set_index("scheme").reindex(order)
ax[3].barh(y, sub.sample_jaccard_vs_primary, color=C["light"], height=0.7, label="Sample-set Jaccard")
ax[3].plot(sub.opc_nucleus_agreement_with_primary, y, "D", ms=2.8, color="k", label="Nucleus-call agreement")
ax[3].set_yticks(y)
ax[3].set_yticklabels([])
ax[3].set_xlim(0, 1.05)
ax[3].set_xlabel("Agreement with primary calls")
ax[3].set_title("Overlap with primary calls", loc="left")
ax[3].legend(loc="upper left", frameon=False, bbox_to_anchor=(-0.05, -0.18), handletextpad=0.3, ncol=1)
label(ax[3], "d", x=-0.1)

ax2 = fig.add_subplot(gsb[0])
sel = ["primary", "p99", "hallmark_7_10", "concordant", "ref_primary", "ref_microglia", "ref_oligo"]
for i, sc in enumerate(sel):
    d = P[P.scheme == sc]
    ax2.scatter(i + jitter(len(d), seed=i), d.CHST11, s=5, color=C["CHST11"] if i < 4 else C["ref"], lw=0)
    ax2.plot([i - 0.25, i + 0.25], [d.CHST11.median()] * 2, color="k", lw=1)
    pos = (d.CHST11 > 0).sum()
    ax2.text(i, 2.5, f"{pos}/{len(d)}", ha="center", fontsize=5.5)
ax2.axhline(0, color="k", lw=0.5, ls="--")
ax2.set_xticks(range(len(sel)))
short = {"primary": "Primary\n(P95)", "p99": "P99", "hallmark_7_10": "chr7/10\nhallmark",
         "concordant": "Score AND\nhallmark", "ref_primary": "Re-run\nprimary ref.",
         "ref_microglia": "Re-run\nmicroglia", "ref_oligo": "Re-run\noligodendro."}
ax2.set_xticklabels([short[s] for s in sel], fontsize=5.5)
ax2.set_ylabel("Per-sample CHST11 Δ")
ax2.set_ylim(-1.0, 2.75)
ax2.set_title("Per-sample CHST11 differences (numbers: samples with Δ > 0)", loc="left")
label(ax2, "e", x=-0.1)

ax3 = fig.add_subplot(gsb[1])
for j, (gene, xo) in enumerate([("CHST11", 0), ("XYLT1", 1), ("DSEL", 2)]):
    a, b = TF[f"{gene}_rho"].values, TF[f"{gene}_partial_rho_depth"].values
    for u, v in zip(a, b):
        ax3.plot([xo - 0.17, xo + 0.17], [u, v], color=C["light"], lw=0.35)
    ax3.scatter(np.full(len(a), xo - 0.17), a, s=5, color=C[gene], lw=0)
    ax3.scatter(np.full(len(b), xo + 0.17), b, s=5, color=C[gene], lw=0, alpha=0.55)
    sr = stats.wilcoxon(b)
    ax3.text(xo, 0.92, f"median {np.median(a):+.2f} → {np.median(b):+.2f}", ha="center", fontsize=5.0)
ax3.axhline(0, color="k", lw=0.5, ls="--")
ax3.set_xticks([-0.17, 0.17, 0.83, 1.17, 1.83, 2.17])
ax3.set_xticklabels(["raw", "depth", "raw", "depth", "raw", "depth"], fontsize=5.5)
for xo, g in enumerate(["CHST11", "XYLT1", "DSEL"]):
    ax3.text(xo, 1.22, g, ha="center", fontsize=6, color=C[g], fontweight="bold")
ax3.set_ylim(-0.55, 1.35)
ax3.set_ylabel("Within-sample Spearman ρ\n(CNV score vs expression)")
ax3.set_title(f"Threshold-free dose-response ({len(TF)} samples)", loc="left")
label(ax3, "f", x=-0.16)
save(fig, "Figure4")

E = pd.read_csv(R / "sulfotransferase_family/chst_family_evidence_matrix.csv")
PS = pd.read_csv(R / "sulfotransferase_family/per_sample_delta_family.csv")
core = ["CHST11", "UST", "CHST12", "CHST3", "CHST15", "CHST7", "CHST14", "CHST13"]
Ec = E.set_index("gene").loc[core]
cols = [("detection_pct", "Detection\n(%)", None), ("delta", "Δ sample", "p_signflip"),
        ("patient_delta", "Δ patient", "p_patient"), ("percentile_among_detection_matched",
                                                       "Percentile vs\nmatched genes", "p_matched_null"),
        ("depth_adj", "Depth-adj.\ncoef.", "p_depth_adj"),
        ("depth_adj", "Depth-adj. vs\nmatched null", "p_matched_null_depth_adj"),
        ("primary_delta", "Δ primary\n(n=9)", "p_primary"), ("recurrent_delta", "Δ recurrent\n(n=11)", "p_recurrent"),
        ("delta_NPC", "Δ NPC-like", None), ("axis_neftel_median_rho", "State-axis\nmedian ρ", "axis_neftel_p")]
fig = plt.figure(figsize=(183 * MM, 112 * MM))
gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1], width_ratios=[1.1, 1], hspace=0.38, wspace=0.28,
                      left=0.08, right=0.985, top=0.93, bottom=0.08)
axm = fig.add_subplot(gs[0, :])
cmap = plt.get_cmap("RdBu_r")
for j, (c, ttl, pc) in enumerate(cols):
    v = Ec[c].values.astype(float)
    if c == "detection_pct":
        norm = lambda x: 0.5 + 0.5 * np.clip(x / 30, 0, 1)
    elif c.startswith("percentile"):
        norm = lambda x: np.clip(x / 100, 0, 1)
    elif c.startswith("axis"):
        norm = lambda x: 0.5 + 0.5 * np.clip(x / 0.1, -1, 1)
    else:
        norm = lambda x: 0.5 + 0.5 * np.clip(x / 0.6, -1, 1)
    for i, g in enumerate(core):
        val = v[i]
        col = cmap(norm(val)) if np.isfinite(val) else "white"
        axm.add_patch(plt.Rectangle((j, len(core) - 1 - i), 0.96, 0.92, color=col, lw=0))
        txt = (f"{val:.1f}" if c == "detection_pct" else f"{val:.0f}" if c.startswith("percentile")
               else f"{val:+.3f}" if np.isfinite(val) else "–")
        p = Ec[pc].iloc[i] if pc else np.nan
        bold = pc is not None and np.isfinite(p) and p < 0.05
        dark = abs(norm(val) - 0.5) > 0.32 if np.isfinite(val) else False
        axm.text(j + 0.48, len(core) - 1 - i + 0.56, txt, ha="center", va="center", fontsize=5.6,
                 fontweight="bold" if bold else "normal", color="white" if dark else "k")
        if pc:
            axm.text(j + 0.48, len(core) - 1 - i + 0.22, f"P={fp(p)}" if np.isfinite(p) else "",
                     ha="center", va="center", fontsize=4.4, color="white" if dark else "#333")
axm.set_xlim(0, len(cols))
axm.set_ylim(0, len(core))
axm.set_xticks(np.arange(len(cols)) + 0.48)
axm.set_xticklabels([c[1] for c in cols], fontsize=5.6)
axm.xaxis.tick_top()
axm.set_yticks(np.arange(len(core)) + 0.46)
axm.set_yticklabels(core[::-1], fontsize=6.5)
for s in axm.spines.values():
    s.set_visible(False)
axm.tick_params(length=0)
label(axm, "a", x=-0.065, y=1.13)

axb = fig.add_subplot(gs[1, 0])
show = ["CHST11", "UST", "CHST12", "CHST3", "CHST9", "HS3ST5"]
for i, g in enumerate(show):
    d = PS[g].values
    colr = C["CHST11"] if g == "CHST11" else C["grey"]
    axb.scatter(i + jitter(len(d), seed=i), d, s=6, color=colr, lw=0)
    axb.plot([i - 0.25, i + 0.25], [d.mean()] * 2, color="k", lw=1)
    axb.text(i, 1.9, f"{(d > 0).sum()}/{len(d)}", ha="center", fontsize=5.5)
axb.axhline(0, color="k", ls="--", lw=0.5)
axb.set_xticks(range(len(show)))
axb.set_xticklabels(show, rotation=0)
axb.set_ylim(-0.9, 2.1)
axb.set_ylabel("Per-sample Δ (aneuploid − diploid)")
axb.set_title("Per-sample differences (bars, means)", loc="left")
label(axb, "b", x=-0.14)

axc = fig.add_subplot(gs[1, 1])
yv = -np.log10(E.p_matched_null_depth_adj.clip(lower=1e-4))
xv = E.depth_adj
axc.scatter(xv, yv, s=8 + 0.8 * E.detection_pct, c=[C["CHST11"] if g == "CHST11" else C["grey"] for g in E.gene],
            lw=0.3, edgecolor="white")
axc.axhline(-np.log10(0.05), color="k", ls="--", lw=0.5)
axc.axvline(0, color="k", lw=0.4)
offs = {"CHST11": (-0.02, 0.12, "right"), "CHST9": (0.02, 0.05, "left"), "UST": (0.02, -0.15, "left"),
        "HS6ST2": (0.02, 0.07, "left"), "HS3ST5": (0.0, 0.12, "center"), "CHST3": (0.0, 0.12, "center"),
        "NDST4": (-0.01, 0.0, "right")}
for g, (dx, dy, ha) in offs.items():
    r = E[E.gene == g].iloc[0]
    axc.text(r.depth_adj + dx, -np.log10(max(r.p_matched_null_depth_adj, 1e-4)) + dy, g, fontsize=5.6, ha=ha,
             color=C["CHST11"] if g == "CHST11" else "k")
axc.set_xlabel("Depth-adjusted coefficient")
axc.set_ylabel("−log$_{10}$ P, depth-adjusted\ndetection-matched null")
axc.set_title(f"All {len(E)} carbohydrate sulfotransferases", loc="left")
label(axc, "c", x=-0.16)
save(fig, "Figure5")

SL = pd.read_csv(R / "cs_pathway_coherence/set_level_summary.csv").set_index("set")
G = pd.read_csv(R / "cs_pathway_coherence/gene_level_leading_edge.csv")
X = pd.read_csv(R / "cs_pathway_coherence/CHST11_vs_set_delta.csv")
SN = pd.read_csv(R / "cs_pathway_coherence/single_nucleus_coexpression_per_sample.csv")
fig = plt.figure(figsize=(183 * MM, 120 * MM))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1, 0.9], hspace=0.6, wspace=0.45,
                      left=0.16, right=0.985, top=0.93, bottom=0.12)
axa = fig.add_subplot(gs[0, 0])
sets = [("CS_DS_biosynthesis", "CS/DS\n(21 genes)"),
        ("CS_DS_biosynthesis_without_CHST11", "CS/DS − CHST11\n(20)"),
        ("CSPG_core", "CSPG core\n(7)"), ("combined_without_CHST11", "Combined −\nCHST11 (27)")]
for i, (s, nm) in enumerate(sets):
    nl = pd.read_csv(R / f"cs_pathway_coherence/null_{s}.csv").iloc[:, 0].values
    yy = len(sets) - 1 - i
    parts = axa.violinplot(nl, positions=[yy], vert=False, widths=0.75, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(C["light"]); b.set_alpha(0.8); b.set_edgecolor("none")
    axa.plot(SL.loc[s, "mean"], yy, "D", color=C["CHST11"], ms=3.5)
    axa.plot([SL.loc[s, "ci_low"], SL.loc[s, "ci_high"]], [yy] * 2, color=C["CHST11"], lw=0.8)
    pn = SL.loc[s, 'p_matched_null']
    axa.text(0.125, yy + 0.22, "null P = 0.001 (floor)" if pn < 0.0015 else f"null P = {pn:.3f}", fontsize=5.2, ha="right")
axa.set_yticks(range(len(sets)))
axa.set_yticklabels([n for _, n in sets][::-1], fontsize=5.6)
axa.axvline(0, color="k", ls="--", lw=0.5)
axa.set_xlim(-0.02, 0.13)
axa.set_xlabel("Δ AUCell (aneuploid − diploid)")
axa.set_title("Sets vs 1,000 matched random sets", loc="left")
label(axa, "a", x=-0.5)

axb = fig.add_subplot(gs[0, 1:])
Gs = G.sort_values("delta", ascending=False)
colr = [C["CHST11"] if g == "CHST11" else ("#56B4E9" if "CSPG" in str(m) else C["grey"])
        for g, m in zip(Gs.gene, Gs.member)]
axb.bar(range(len(Gs)), Gs.delta, color=colr, width=0.75)
for i, (q, dl) in enumerate(zip(Gs.q_bh, Gs.delta)):
    if q < 0.05:
        axb.text(i, dl + 0.02 if dl >= 0 else dl - 0.08, "*", ha="center", fontsize=7)
axb.set_xticks(range(len(Gs)))
axb.set_xticklabels(Gs.gene, rotation=90, fontsize=5.3)
axb.axhline(0, color="k", lw=0.5)
axb.set_ylabel("Δ log1p(CP10k)")
axb.set_title("Member genes (* BH q < 0.05); blue: CHST11; light blue: CSPG core; grey: KEGG hsa00532",
              loc="left", fontsize=6)
label(axb, "b", x=-0.08)

axc = fig.add_subplot(gs[1, 0])
axc.scatter(X.CHST11_delta, X.set_delta, s=9, color=C["CHST11"], lw=0)
rho, p = stats.spearmanr(X.CHST11_delta, X.set_delta)
cf = np.polyfit(X.CHST11_delta, X.set_delta, 1)
xx = np.linspace(X.CHST11_delta.min(), X.CHST11_delta.max(), 10)
axc.plot(xx, np.polyval(cf, xx), color="k", lw=0.6)
axc.axhline(0, color="k", ls="--", lw=0.4)
axc.axvline(0, color="k", ls="--", lw=0.4)
axc.text(0.03, 0.92, f"Spearman ρ = {rho:.2f}\nP = {fp(p)}, n = 20", transform=axc.transAxes, fontsize=5.6, va="top")
axc.set_xlabel("CHST11 Δ (per sample)")
axc.set_ylabel("CHST11-free combined set Δ")
axc.set_title("Per-sample co-variation", loc="left")
label(axc, "c", x=-0.4)

for k, s in enumerate(["CSPG_core", "combined_without_CHST11"]):
    axd = fig.add_subplot(gs[1, 1 + k])
    for i, comp in enumerate(["aneuploid", "diploid"]):
        d = SN[(SN.set == s) & (SN.compartment == comp)]
        axd.scatter(i + jitter(len(d), 0.1, i), d.partial_rho_depth, s=6, lw=0,
                    color=C["CHST11"] if comp == "aneuploid" else C["grey"])
        axd.plot([i - 0.22, i + 0.22], [d.partial_rho_depth.median()] * 2, color="k", lw=1)
        axd.text(i, 1.0, f"median {d.partial_rho_depth.median():.2f}\n{(d.partial_rho_depth > 0).sum()}/20 > 0",
                 ha="center", fontsize=5.2, va="bottom")
    axd.axhline(0, color="k", ls="--", lw=0.4)
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["aneuploid", "diploid"])
    axd.set_ylim(-0.6, 1.25)
    axd.set_ylabel("Partial ρ, CHST11 vs set score\n(single nuclei, depth-adjusted)" if k == 0 else "")
    axd.set_title(["CSPG core", "Combined, no CHST11"][k], loc="left")
    label(axd, "de"[k], x=-0.3 if k == 0 else -0.15)
save(fig, "Figure6")

PR = pd.read_csv(R / "primary_recurrent/per_sample_effects_with_progression.csv")
SM = pd.read_csv(R / "primary_recurrent/primary_recurrent_summary.csv")
PA = pd.read_csv(R / "primary_recurrent/paired_longitudinal_summary.csv")
fig = plt.figure(figsize=(183 * MM, 66 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 0.9], wspace=0.38, left=0.06, right=0.985, top=0.86,
                      bottom=0.17)
axa = fig.add_subplot(gs[0])
genes = ["CHST11", "XYLT1", "DSEL"]
for gi, g in enumerate(genes):
    for k, pr in enumerate(["Primary", "Recurrent"]):
        d = PR[(PR.gene == g) & (PR.progression == pr)].delta.values
        xpos = gi * 2.8 + k * 1.2
        axa.scatter(xpos + jitter(len(d), 0.13, gi * 2 + k), d, s=6, lw=0, color=C["prim"] if k == 0 else C["rec"])
        axa.plot([xpos - 0.28, xpos + 0.28], [d.mean()] * 2, color="k", lw=1)
        r = SM[(SM.gene == g) & (SM.group == pr)].iloc[0]
        axa.text(xpos, 2.62, f"{int(r.positive)}/{int(r.n)}\nP={fp(r.p_signflip)}", ha="center", fontsize=4.6)
    rr = SM[(SM.gene == g) & (SM.group == "Recurrent_minus_Primary")].iloc[0]
    axa.text(gi * 2.8 + 0.6, -1.38, f"diff {rr['mean']:+.2f}\nperm P={fp(rr.p_label_permutation)}",
             ha="center", fontsize=5, color="#333")
axa.axhline(0, color="k", ls="--", lw=0.5)
axa.set_xticks([gi * 2.8 + 0.6 for gi in range(3)])
axa.set_xticklabels(genes)
for t, g in zip(axa.get_xticklabels(), genes):
    t.set_color(C[g]); t.set_fontweight("bold")
axa.set_ylim(-1.55, 3.05)
axa.set_ylabel("Per-sample Δ (aneuploid − diploid)")
axa.scatter([], [], color=C["prim"], s=6, label="Primary (n=9)")
axa.scatter([], [], color=C["rec"], s=6, label="Recurrent (n=11)")
axa.legend(loc="upper left", bbox_to_anchor=(0.0, 1.17), ncol=2, frameon=False)
label(axa, "a", x=-0.13, y=1.08)
pts = ["#12", "#25", "#32", "#37", "#40"]
for k, g in enumerate(["CHST11", "XYLT1"]):
    axp = fig.add_subplot(gs[1 + k])
    sub = PR[(PR.gene == g) & PR.patient_pair.isin(pts)]
    for pt, d in sub.groupby("patient_pair"):
        a = d[d.progression == "Primary"].delta.mean()
        b = d[d.progression == "Recurrent"].delta.mean()
        axp.plot([0, 1], [a, b], "-o", ms=3, lw=0.8, color=C[g])
    r = PA[PA.gene == g].iloc[0]
    axp.axhline(0, color="k", ls="--", lw=0.4)
    axp.set_xticks([0, 1])
    axp.set_xticklabels(["Primary", "Recurrent"])
    axp.set_xlim(-0.3, 1.3)
    axp.set_title(f"{g}: 5 paired patients\nmean change {r.mean_change:+.2f}, P={fp(r.p_signflip)} "
                  f"(min {r.min_attainable_p:.4f})", loc="left", fontsize=6)
    axp.set_ylabel("Δ (aneuploid − diploid)" if k == 0 else "")
    label(axp, "bc"[k], x=-0.22, y=1.08)
save(fig, "Figure7")

SU = pd.read_csv(R / "patient_level/summary.csv")
LP = pd.read_csv(R / "patient_level/leave_one_patient_out.csv")
LS = pd.read_csv(R / "patient_level/leave_one_sample_out.csv")
fig = plt.figure(figsize=(183 * MM, 62 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.9, 0.9], wspace=0.32, left=0.115, right=0.985, top=0.85, bottom=0.18)
axa = fig.add_subplot(gs[0])
yy = 0
ticks = []
for g in ["CHST11", "XYLT1", "DSEL"]:
    for u in ["sample", "patient"]:
        r = SU[(SU.gene == g) & (SU.unit == u)].iloc[0]
        axa.plot([r.ci_low, r.ci_high], [yy] * 2, color=C[g], lw=0.9)
        axa.plot(r["mean"], yy, "o" if u == "sample" else "s", color=C[g], ms=3.5)
        axa.text(0.72, yy, f"{int(r.positive)}/{int(r.n_units)} > 0; sign-flip P {fp(r.p_signflip)};\nsign test P {fp(r.p_sign_test)}",
                 fontsize=4.8, va="center", ha="left")
        ticks.append(f"{g}, {u}s")
        yy -= 1
    yy -= 0.4
axa.set_yticks([0, -1, -2.4, -3.4, -4.8, -5.8])
axa.set_yticklabels(ticks)
axa.axvline(0, color="k", ls="--", lw=0.5)
axa.set_xlim(-0.05, 1.25)
axa.set_xticks([0, 0.2, 0.4, 0.6])
axa.set_xlabel("Mean Δ (95% bootstrap CI)")
axa.set_title("Sample vs patient unit", loc="left")
label(axa, "a", x=-0.25, y=1.06)
for k, (D, ttl) in enumerate([(LP, "Leave one patient out (15)"), (LS, "Leave one sample out (20)")]):
    ax = fig.add_subplot(gs[1 + k])
    for gi, g in enumerate(["CHST11", "XYLT1", "DSEL"]):
        d = D[D.gene == g]
        ax.scatter(gi + jitter(len(d), 0.15, gi), d["mean"], s=6, color=C[g], lw=0)
        full = SU[(SU.gene == g) & (SU.unit == ("patient" if k == 0 else "sample"))]["mean"].iloc[0]
        ax.plot([gi - 0.3, gi + 0.3], [full] * 2, color="k", lw=0.8, ls=":")
        ax.text(gi, d["mean"].max() + 0.04, f"max P\n{fp(d.p_signflip.max())}", ha="center", fontsize=5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["CHST11", "XYLT1", "DSEL"])
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("Mean Δ after omission" if k == 0 else "")
    ax.yaxis.labelpad = 1
    ax.set_title(ttl, loc="left")
    label(ax, "bc"[k], x=-0.3, y=1.06)
save(fig, "FigureS2")
print("done")
