from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

MM = 1 / 25.4


def use_pub_style(name: str = "nature"):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 5.8, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2,
        "ytick.major.size": 2.2, "lines.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "figure.dpi": 150,
    })


def figure_mosaic(mosaic: str, width_mm: float = 183, height_mm: float = 120, **kw):
    lines = [ln.strip() for ln in textwrap.dedent(mosaic).strip().splitlines() if ln.strip()]
    layout = [list(ln.replace(" ", "")) for ln in lines]
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), layout="constrained")
    ax = fig.subplot_mosaic(layout, empty_sentinel=".", **kw)
    return fig, ax


def panel_labels(fig, ax, case: str = "lower", size: float = 8, pad_pt: float = 14):
    from matplotlib.transforms import ScaledTranslation
    keys = sorted(ax) if isinstance(ax, dict) else range(len(ax))
    for i, k in enumerate(keys):
        a = ax[k]
        lab = str(k) if isinstance(k, str) and len(k) == 1 else chr(ord("A") + i)
        lab = lab.lower() if case == "lower" else lab.upper()
        off = ScaledTranslation(-pad_pt / 72, 4 / 72, fig.dpi_scale_trans)
        a.text(0, 1, lab, transform=a.transAxes + off, fontsize=size, fontweight="bold",
               ha="right", va="bottom", clip_on=False)


def check_fonts():
    names = {f.name for f in font_manager.fontManager.ttflist}
    ok = [f for f in ("Arial", "Helvetica", "Liberation Sans") if f in names]
    if not ok:
        print("[pubfig shim] Arial/Helvetica not installed - falling back to DejaVu Sans "
              "(sudo apt install ttf-mscorefonts-installer fonts-liberation)")
    return ok


def save_figure(fig, path, width_mm=183, provenance=None, verbose=False, formats=("pdf", "tif")):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    if abs(w - width_mm * MM) > 1e-3:
        fig.set_size_inches(width_mm * MM, h * width_mm * MM / w)
    for ext in formats:
        kw = {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(p.with_name(f"{p.name}.{ext}"), **kw)
    if provenance is not None:
        p.with_name(p.name + ".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    if verbose:
        print(f"[pubfig shim] wrote {p}.{{{','.join(formats)}}}")
    return {"path": str(p), "audit": {"overlaps": []}}
