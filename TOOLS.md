# Software, versions and method citations

## Execution environment

| Component | Version |
| :-- | :-- |
| Operating systems | Windows 11, Ubuntu Linux |
| Python | 3.12–3.13 |

## Python packages

| Package | Version | Use |
| :-- | :-- | :-- |
| scanpy | 1.12.1 | preprocessing, clustering, normalization, `score_genes` for the state coordinate |
| anndata | 0.12.13 | data structures |
| infercnvpy | 0.6.1 | copy-number inference |
| decoupler | 2.2.0 | AUCell scoring against the full transcriptome |
| gseapy | 1.2.1 | gene sets and pathway membership through Enrichr |
| statsmodels | 0.14.6 | Holm and Benjamini–Hochberg correction, mixed-effects models |
| scipy | 1.17.1 | Spearman correlation, Wilcoxon, Mann–Whitney and Kruskal–Wallis tests |
| lifelines | 0.30.3 | Cox regression and Schoenfeld residual tests |
| numpy / pandas | 2.3.3 / 2.3.2 | numerical work |
| matplotlib | 3.10.6 | figures |
| igraph / leidenalg | 1.0.0 / 0.12.0 | Leiden clustering (`scripts/00c_state_assignment.py`) |
| PyDESeq2 | 0.5.4 | pseudobulk differential expression that defined the neuronal candidate list |

These versions define the pinned environment, in which Build 2 of the discovery cohort was
preprocessed. Scripts 18, 21, 22 and 30 were run on Build 1 with the same package versions
under Python 3.12.10 (recorded in each provenance file). Build 1 was preprocessed by the same scripts with scanpy 1.11.5, anndata 0.12.19,
NumPy 2.4.6 and SciPy 1.17.1; the two builds and their deposited annotations are described in
`data/README.md`.

PyDESeq2 produced the neuronal candidate list provided in `data/target_tractability.csv` and
is not imported by the pipeline. Figures are drawn with matplotlib through the plotting helper
in `tools/pubfig/`.

## External tools

| Tool | Version | Use |
| :-- | :-- | :-- |
| scFEA | changwn/scFEA (GitHub), human M168 map | per-nucleus metabolic flux for 5,986 malignant nuclei |

scFEA was run in a separate environment (Python 3.11, PyTorch CPU build, pandas 1.5.3, numpy
1.26.4, magic-impute 3.0.0) with default settings. Its output is provided as
`data/scfea_flux_5986_malignant_nuclei.csv.gz`; the state coordinate and both axis screens
are computed from it by `scripts/05_state_axis_flux_screen.py`.

## Locally implemented routines

1. The detection-matched null constructions, in `scripts/06_enzyme_compartment_contrast.py`,
   `scripts/08_detection_matched_single_gene.py`, `scripts/12_depth_sensitivity.py`,
   and `scripts/14_set_level_matched_nulls.py`.
   No packaged implementation exists; the construction is described in the manuscript Methods.
2. The two-sided sign-flip permutation test with exact 2ⁿ enumeration. `scripts/11_patient_level_endpoint.py`
   also computes each test with `scipy.stats.permutation_test` and checks that the two agree.
3. Control-gene signature scoring and the Neftel two-dimensional representation, in
   `data/neftel.py` (30 expression bins, 100 control genes per gene).
4. A MatrixMarket reader in `scripts/00a_build_matrix.py`, because the deposited header of one
   sample declares a different entry count from the one the file contains.

## External data services

| Service | Endpoint | Use |
| :-- | :-- | :-- |
| GEO | GSE174554 | discovery single-nucleus cohort |
| cBioPortal | `https://www.cbioportal.org/api`, study `gbm_tcga_pan_can_atlas_2018` | TCGA-GBM expression and survival |
| Proteomic Data Commons | `https://pdc.cancer.gov/graphql`, study PDC000204 (CDAP gene-level log2 ratios) | CPTAC glioblastoma proteome, script 29 |
| Allen Institute | `glioblastoma.alleninstitute.org` | IVY Glioblastoma Atlas Project wells |
| CGGA | `http://www.cgga.org.cn`, release mRNAseq_693 (20200506) | bulk primary and recurrent glioblastoma |
| DepMap | 24Q4 Public release | CHST11 CRISPR gene effect and expression |
| Open Targets Platform | GraphQL API, accessed 31 August 2026 | tractability and ChEMBL compound annotation |
| KEGG | REST interface | pathway gene lists |
| Enrichr, through gseapy | GO Biological Process 2023, KEGG 2021 Human, Reactome, WikiPathways | gene sets and pathway membership |
| Ensembl BioMart | pybiomart | gene genomic coordinates |

## Method citations

- **scanpy** — Wolf FA, Angerer P, Theis FJ (2018) *Genome Biology* 19:15. https://doi.org/10.1186/s13059-017-1382-0
- **Control-gene signature scoring** — Tirosh I, et al. (2016) *Science* 352:189–196. https://doi.org/10.1126/science.aad0501
- **Cell-state model** — Neftel C, et al. (2019) *Cell* 178:835–849. https://doi.org/10.1016/j.cell.2019.06.024
- **inferCNV approach** — Patel AP, et al. (2014) *Science* 344:1396–1401. https://doi.org/10.1126/science.1254257 (infercnvpy has no primary publication and is identified by name and version)
- **scFEA** — Alghamdi N, et al. (2021) *Genome Research* 31:1867–1884. https://doi.org/10.1101/gr.271205.120
- **KEGG** — Kanehisa M, Goto S (2000) *Nucleic Acids Research* 28:27–30. https://doi.org/10.1093/nar/28.1.27
- **AUCell** — Aibar S, et al. (2017) *Nature Methods* 14:1083–1086. https://doi.org/10.1038/nmeth.4463
- **decoupleR** — Badia-i-Mompel P, et al. (2022) *Bioinformatics Advances* 2:vbac016. https://doi.org/10.1093/bioadv/vbac016
- **gseapy** — Fang Z, Liu X, Peltz G (2023) *Bioinformatics* 39:btac757. https://doi.org/10.1093/bioinformatics/btac757
- **PyDESeq2** — Muzellec B, et al. (2023) *Bioinformatics* 39:btad547. https://doi.org/10.1093/bioinformatics/btad547
- **SciPy** — Virtanen P, et al. (2020) *Nature Methods* 17:261–272. https://doi.org/10.1038/s41592-019-0686-2
- **statsmodels** — Seabold S, Perktold J (2010) *Proceedings of the 9th Python in Science Conference*, 92–96.
- **Add-one permutation estimator** — Phipson B, Smyth GK (2010) *Statistical Applications in Genetics and Molecular Biology* 9:Article 39. https://doi.org/10.2202/1544-6115.1585
- **Scrublet** — Wolock SL, Lopez R, Klein AM (2019) *Cell Systems* 8:281–291.e9. https://doi.org/10.1016/j.cels.2018.11.005
- **Chronos** — Dempster JM, et al. (2021) *Genome Biology* 22:343. https://doi.org/10.1186/s13059-021-02540-7
- **Open Targets Platform** — Ochoa D, et al. (2023) *Nucleic Acids Research* 51:D1353–D1359. https://doi.org/10.1093/nar/gkac1046
- **ChEMBL** — Zdrazil B, et al. (2024) *Nucleic Acids Research* 52:D1180–D1192. https://doi.org/10.1093/nar/gkad1004

## Data-resource citations

- **GSE174554** — Wang L, et al. (2022) *Nature Cancer* 3:1534–1552. https://doi.org/10.1038/s43018-022-00475-x
- **CGGA** — Zhao Z, et al. (2021) *Genomics, Proteomics and Bioinformatics* 19:1–12. https://doi.org/10.1016/j.gpb.2020.10.005
- **cBioPortal** — Cerami E, et al. (2012) *Cancer Discovery* 2:401–404. https://doi.org/10.1158/2159-8290.CD-12-0095; Gao J, et al. (2013) *Science Signaling* 6:pl1. https://doi.org/10.1126/scisignal.2004088
- **TCGA PanCancer Atlas** — Hoadley KA, et al. (2018) *Cell* 173:291–304.e6. https://doi.org/10.1016/j.cell.2018.03.022
- **IVY Glioblastoma Atlas Project** — Puchalski RB, et al. (2018) *Science* 360:660–663. https://doi.org/10.1126/science.aaf2666
