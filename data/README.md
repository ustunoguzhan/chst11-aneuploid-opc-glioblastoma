# Data

No raw sequencing data are stored in this repository. Every dataset is a public release;
download each to the directory `config.py` resolves for it (`GBM_DATA`, `GBM_CGGA`,
`GBM_EXT`).

## Reference tables

| File | Content |
| :-- | :-- |
| `neftel_metamodules.csv` | the eight Neftel metamodule gene lists |
| `celltype_genesets.json` | marker sets for non-malignant cell types |
| `nonmalignant_thresholds.json` | permutation-calibrated score thresholds for those markers |
| `neftel.py` | control-gene signature scoring and the Neftel two-dimensional representation |
| `snrna_cohort.csv` | GSM-to-patient, timepoint and batch mapping for GSE174554 |
| `metabolic_genesets.json` | Neftel metamodules, KEGG metabolism pathway gene lists (KEGG REST) and scFEA M168 module genes |
| `scfea_m168/` | scFEA human M168 files as distributed with the scFEA annotation: `Human_M168_information.symbols.csv` (module annotation, read by `05_state_axis_flux_screen.py`) and `module_gene_m168.csv` (module-to-gene table; no script reads it, the module genes used by the pipeline are stored in `metabolic_genesets.json`) |

The metamodule file holds 50 genes each for MES1, MES2, OPC, NPC1 and NPC2, 39 for AC, 45
for G2M and 29 for G1S, after mapping the published lists onto the gene symbols of this
matrix.

## Provided inputs

These files are inputs to the pipeline rather than outputs of it.

| File | Content | Origin |
| :-- | :-- | :-- |
| `target_tractability.csv` | the two candidate sets with Open Targets tractability and ChEMBL compound annotation | Open Targets Platform GraphQL API, accessed 31 August 2026 |
| `scfea_flux_5986_malignant_nuclei.csv.gz` | scFEA flux (168 modules) for 5,986 malignant nuclei, 101–107 per sample | scFEA, see `TOOLS.md` |
| `depmap_chst11_dependency_by_cellline.csv` | CHST11 CRISPR gene effect (Chronos) and expression for 1,178 cell lines with lineage annotation | DepMap 24Q4 Public release |
| `literature_evidence.csv` | literature and CPTAC evidence for CHST11 (Supplementary Table S20) | PubMed-verified references; the two CPTAC rows are computed by `scripts/29_cptac_proteome.py` (`results/independent_cohorts/cptac_chst11_proteome.json`) |

The candidate sets in `target_tractability.csv` are defined as follows. `line 2`: the 39
genes, among the 342 with higher expression at recurrence in a pseudobulk PyDESeq2 analysis
of OPC-like nuclei (design `~ patient + progression`), whose symbols begin CACNA, GABR, GRM,
KCN, SYT, SNAP, STXBP, SYN, RIMS, CNTN or CNTNAP, together with RYR2, CAMK2A, SHANK2 and
ANK3. `line 1`: the 28 enzymes flagged `carried_forward` in
`results/metabolic_flux/enzyme_axis_screen.csv`.

## Discovery cohort, GSE174554

Download `GSE174554_RAW.tar` from GEO and unpack it into `$GBM_DATA/GSE174554_RAW/`. The
preprocessing steps write:

| Step | Writes | Content |
| :-- | :-- | :-- |
| `00a_build_matrix.py` | `gbm_snrna_raw.h5ad` | all per-sample matrices, ≥200 genes per nucleus |
| `00b_quality_control.py` | `gbm_snrna_qc.h5ad` | mitochondrial, per-sample and doublet filters |
| `00c_state_assignment.py` | `gbm_snrna_states.h5ad` | malignant call and Neftel state, normalized matrix |
| `00e_apply_build_annotation.py --build 2` | `gbm_snrna_states_build2.h5ad` | the 00c matrix with the deposited Build 2 annotation |
| `00e_apply_build_annotation.py --build 1` | `gbm_snrna_states.h5ad` | the 00c matrix with the deposited Build 1 annotation |
| `00d_align_counts.py` | `gbm_snrna_counts.h5ad` | counts aligned to that annotation, 163,481 nuclei × 33,694 genes |

### Two preprocessing builds

The study uses two builds of the discovery cohort. Both were produced by `00a`–`00d` with the
same code and parameters (3,000 highly variable genes, 50 principal components, Leiden
resolution 1.0 with the igraph implementation and two iterations, random seed 0), in two
software environments: Build 1 with scanpy 1.11.5, anndata 0.12.19, NumPy 2.4.6 and SciPy
1.17.1 (igraph and leidenalg versions were not recorded); Build 2 with the pinned environment of
`requirements.txt`. Every analysis uses Build 1; Build 2 is used only by the concordance check
(script 28) and the preprocessing sensitivity analysis (script 30).

The builds share the same 163,481 quality-controlled nuclei, the same normalized matrix and the
same meta-module scores. They differ in the Leiden partition and therefore in the cluster-level
malignant call: 121,788 malignant nuclei in 47 clusters in Build 1 and 126,387 in 48 clusters in
Build 2. Of the nuclei in Build 1 oligodendrocyte clusters, 4,654 fell into malignant clusters
in Build 2 (3,113 of them assigned to the OPC-like state), 12 Build 1 microglia did the same, and
67 Build 1 malignant nuclei fell into non-malignant clusters. Nuclei called malignant in both
builds received the same Neftel state. The OPC-like analysis set contains 13,196 nuclei in Build 1
and 14,911 in Build 2, from the same 20 samples. Of the 1,734 nuclei found only in the Build 2
set, 1,733 belong to Build 1 oligodendrocyte clusters and 95% are diploid; they express mature
oligodendrocyte markers and almost no CHST11, so they dilute the Build 2 diploid compartment and
inflate its compartment differences (script 30).

Because graph-based clustering depends on the software environment, re-running `00c` does not by
itself reproduce either build. Both annotations are therefore deposited in `builds/`, and
`00e_apply_build_annotation.py` applies them to the matrix written by `00c` (expression values are
not modified):

| File | Content |
| :-- | :-- |
| `builds/build1_annotation.csv.gz` | Build 1, all 163,481 nuclei: Leiden cluster, cluster call, malignant flag, Neftel state and coordinates, cycling flag, meta-module scores |
| `builds/build2_annotation.csv.gz` | Build 2, the 126,387 malignant nuclei: Neftel state and the scanpy `score_genes` state scores |
| `builds/build1_cluster_summary.csv`, `builds/build2_cluster_summary.csv` | per-cluster marker scores, patient mix and malignant call of each build |

The Build 2 Leiden labels and the cell-type calls of its non-malignant clusters were not
retained. `00e` carries the Build 1 cell-type call over to Build 2 non-malignant nuclei and labels
the 67 nuclei that are malignant only in Build 1 `nonmalignant_unresolved`.

One file, `GSM5319529_SF8963`, has a MatrixMarket header that declares 4,855,174 entries
where the file holds 4,648,192, and a block of NUL bytes that erases 181 nuclei.
`00a_build_matrix.py` reads it with its own MatrixMarket reader, which ignores the declared
count, discards lines containing NUL and excludes the nuclei adjacent to the damage.

The published non-malignant threshold (a score above 4) was calibrated on SmartSeq2 data in
log2(TPM/10 + 1) and is not applicable to log1p(CP10k) 10x data; the thresholds in
`nonmalignant_thresholds.json` are the 99.9th percentile of a permutation null of
expression-bin-matched random gene sets.

`00d_align_counts.py` reconstructs the count matrix from the normalized matrix by inverting
the CP10k and log1p transforms with each nucleus's `total_counts`, and verifies that every
value is an integer and every row sum matches.

The same steps write four small quality-control tables into this folder:

| File | Written by | Content |
| :-- | :-- | :-- |
| `mtx_header_qc.csv` | `00a_build_matrix.py` | declared versus actual MatrixMarket entries per file, NUL lines, excluded nuclei |
| `qc_waterfall.csv` | `00b_quality_control.py` | nuclei, samples and patients remaining after each filter |
| `geneset_coverage.csv` | `00c_state_assignment.py` | fraction of each signature gene set present in the matrix |
| `cluster_summary.csv` | `00c_state_assignment.py` | per-cluster marker scores, patient mix and malignant call of the re-run (the deposited summaries of the two builds are in `builds/`) |

## Bulk cohorts

**CGGA mRNAseq_693**, release 20200506, from http://www.cgga.org.cn. Place in `$GBM_CGGA`:

```
CGGA.mRNAseq_693.RSEM-genes.20200506.txt
CGGA.mRNAseq_693_clinical.20200506.txt
```

Samples with `Histology` `GBM` or `rGBM` and `PRS_type` `Primary` (n = 140) or `Recurrent`
(n = 109) are used.

**TCGA-GBM** through the cBioPortal REST API, study `gbm_tcga_pan_can_atlas_2018`; 154
patients have expression and survival, with 122 events. **IVY Glioblastoma Atlas Project**
from `glioblastoma.alleninstitute.org`; the 122 wells sampled by reference histology are
used. Both are fetched by `16_external_cohorts.py` and cached under
`$GBM_DATA/external_cache/`.

**CPTAC glioblastoma discovery proteome**, PDC study PDC000204 (Wang et al. 2021), through the
Proteomic Data Commons GraphQL API: the biospecimen table and the gene-level log2 ratios of the
PDC Common Data Analysis Pipeline (110 aliquots, 100 primary tumor and 10 solid-tissue normal,
one aliquot per case). Fetched by `29_cptac_proteome.py` and cached under
`$GBM_DATA/external_cache/`.

For `17_survival_covariates.py`, `$GBM_EXT` (default `ext/` next to `results/`) holds `cgga693_clinical.zip` (the CGGA
mRNAseq_693 clinical table) and `tcga_patient_clinical.json` (cBioPortal patient clinical
data for `gbm_tcga_pan_can_atlas_2018`).

`results/cnv/gene_positions.csv` holds gene genomic coordinates retrieved from Ensembl
BioMart through pybiomart and is read by `01_copy_number_inference.py`.
