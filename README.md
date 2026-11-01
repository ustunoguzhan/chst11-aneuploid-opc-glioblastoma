# CHST11 and a chondroitin sulfate proteoglycan program in aneuploid OPC-like glioblastoma cells

Code, reference inputs and derived results for the study

> **Copy-number-resolved single-nucleus analysis attributes elevated CHST11 and a coherent
> chondroitin sulfate proteoglycan program to aneuploid oligodendrocyte-progenitor-like
> glioblastoma cells**

Using public single-nucleus RNA-seq data from paired primary and recurrent glioblastomas
(GEO GSE174554), OPC-like nuclei of each tumor are split by inferred copy number into
aneuploid and diploid nuclei, and candidate targets are compared between the two compartments
of the same sample under detection-matched and sequencing-depth controls. The CHST11 result is
then tested under alternative copy-number classifications, at patient level, within the
sulfotransferase family, at pathway level without CHST11, and in primary versus recurrent
tumors.

## Main results

| Result | Value |
| :-- | :-- |
| Analysis set | 20 samples from 15 patients, 13,196 OPC-like nuclei |
| Positive control, chr7 gain minus chr10 loss | +0.0183, *P* = 2.5 × 10⁻⁵, 18 of 20 samples |
| Named neuronal candidates | CACNA1B and GABRB2 below 5% detection; GRM5 not different (*P* = 0.42) |
| CHST11, aneuploid minus diploid | +0.459, *P* = 6.0 × 10⁻⁴, 18 of 20 samples |
| CHST11, patient level | +0.475, *P* = 0.0016, 13 of 15 patients |
| CHST11, detection-matched null | exceeds all 218 matched genes |
| CHST11, depth-adjusted | +0.357, *P* = 4.6 × 10⁻⁵ |
| CHST11, eleven alternative copy-number classifications | positive in all (*P* ≤ 0.021) |
| CHST11, threshold-free dose–response | median ρ = 0.21, 27 of 32 samples, *P* = 5 × 10⁻⁶ |
| Chondroitin sulfate pathway without CHST11 | exceeds detection-matched random sets (*P* = 0.003) |
| Recurrence | CHST11 higher in all 11 recurrent samples |
| DepMap, survival | no fitness dependency in 51 glioblastoma lines; no survival association |

## Repository layout

```
scripts/          analysis pipeline, 00a-29, with the shared library common.py
results/          derived results written by the scripts; results/robustness/ for 18-28
data/             reference tables, provided inputs and the two build annotations in data/builds/ (see data/README.md)
figures/          make_figures.py (Figures 1-3, 8, S1, S3, S4) and make_figures_robustness.py (Figures 4-7, S2);
                  source_data/ holds the values plotted in each panel, figure files are written to figures/output/
tests/            synthetic-data smoke test for scripts 18-28
tools/pubfig/     plotting helper used by the figure scripts
run_all.sh        runs the pipeline and the figure scripts
setup_env.sh      creates a virtual environment with the pinned packages
requirements.txt  pinned Python environment
config.py         filesystem paths
TOOLS.md          software versions, external services and method citations
```

## Reproducing the analysis

```bash
bash setup_env.sh                 # .venv with the pinned packages in requirements.txt
bash tests/run_smoke_test.sh      # optional, a few minutes on synthetic data
bash run_all.sh                   # scripts 01-28 and both figure scripts
bash run_all.sh --from-raw        # additionally rebuilds the matrices (00a-00e) from GEO files
bash run_all.sh --quick           # fewer permutations, no inferCNV re-runs in script 18
bash run_all.sh --only "18 19"    # selected steps (prefix match on the step id; "figures" for the figure scripts)
bash run_all.sh --jobs 8          # parallel workers for the inferCNV re-runs of script 18 (default 4)
```

The primary analyses (01–17) use Build 1 of the discovery cohort and the robustness analyses
that read the processed matrices (18, 21–24) use Build 2. Both builds come from the same
preprocessing scripts run in two software environments and differ only in their cluster-level
malignant calls; their per-nucleus annotations are deposited in `data/builds/` and applied by
`00e`, so `--from-raw` regenerates both exactly (see [`data/README.md`](data/README.md)).

Raw data are public and are not redistributed; download steps are in
[`data/README.md`](data/README.md). Set `GBM_DATA` (single-nucleus matrices and caches),
`GBM_CGGA` (CGGA files) and `GBM_EXT` (clinical covariate files), or edit `config.py`. Each step writes a log to `logs/<step>.log`; a failing step does not stop the run, and the final
summary lists completed, skipped (optional input missing) and failed steps.

| Scripts | Content |
| :-- | :-- |
| 00a–00d | per-sample matrices, quality control, malignant call and Neftel state assignment, count matrix |
| 00e | applies the deposited Build 1 and Build 2 annotations to the 00c matrix |
| 01–02 | copy-number inference, analysis set and chr7/chr10 positive control |
| 03–05 | detectability resource, OPC-like identity gate, scFEA module and enzyme screens along the state axis |
| 06–08 | 28-enzyme contrast, named neuronal candidates, detection-matched single-gene nulls |
| 09–14 | per-tumor values and cell-state scope, pathway membership, patient-level contrast, depth adjustment, synaptic endpoint, set-level matched nulls |
| 15–17 | CGGA, TCGA-GBM and IVY-GAP context, covariate-adjusted Cox models |
| 29 | CHST11 protein in the CPTAC glioblastoma proteome (PDC000204) |
| 28 | barcode concordance between the two preprocessing builds |
| 23 | calibration of the non-malignant reference threshold (its calls are read by 18) |
| 18–22 | alternative copy-number classifications and threshold-free analysis, patient-level and leave-one-out robustness, primary versus recurrent, sulfotransferase family, CHST11-excluded pathway coherence |
| 24–27 | state-scoring harmonization, DepMap dependency, multiplicity corrections, cohort flow |

`run_all.sh` runs them in the order of this table. Network access is needed by scripts 03,
06, 10, 13 and 14, which download gene-set libraries from Enrichr through gseapy (script 22
does the same for its negative-control set and continues without it), and by script 16, which
downloads TCGA-GBM data from the cBioPortal API and the IVY-GAP expression matrix from the
Allen Institute, and by script 29, which downloads the CPTAC glioblastoma proteome from the
Proteomic Data Commons API. The CGGA files read by script 15 and the clinical files read by script 17 are
downloaded manually (`data/README.md`). KEGG gene lists are stored in
`data/metabolic_genesets.json` and are not fetched at run time.

## Statistical conventions

The unit of analysis is the sample or the patient, never the nucleus. Paired contrasts use
two-sided sign-flip permutation, enumerated exactly for *n* ≤ 22 and with 200,000 random
assignments and the add-one estimator otherwise. Candidates are placed against nulls of
genes matched on detection frequency, and single-gene contrasts are repeated with
per-nucleus depth as a covariate. Families of tests are corrected by the Holm procedure;
the axis screens use the Benjamini–Hochberg procedure.

## Authors and citation

Oguzhan Ustun, Sadettin Yavuz Ugurlu and Beste Calimlioglu (corresponding author,
beste.calimlioglu@marmara.edu.tr). Citation details will be added on publication.
