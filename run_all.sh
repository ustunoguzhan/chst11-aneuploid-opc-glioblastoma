#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

QUICK=0; FROM_RAW=0; JOBS=4; ONLY=""; PY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --from-raw) FROM_RAW=1 ;;
    --jobs) JOBS="$2"; shift ;;
    --only) ONLY="$2"; shift ;;
    --python) PY="$2"; shift ;;
    -h|--help) echo "usage: bash run_all.sh [--quick] [--from-raw] [--only \"STEPS\"] [--jobs N] [--python P]"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
[[ -f "$ROOT/config.py" ]] || { echo "run_all.sh must sit in the repository root (next to config.py)"; exit 2; }

export TQDM_DISABLE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
RES="${GBM_RESULTS:-$ROOT/results}"
LOG="${GBM_LOGS:-$ROOT/logs}"
mkdir -p "$LOG"
STATUS="$LOG/status.tsv"
printf "step\tstatus\tminutes\tlog\n" > "$STATUS"
T_ALL=$(date +%s)

want() {
  [[ -z "$ONLY" ]] && return 0
  for s in $ONLY; do [[ "$1" == "$s"* ]] && return 0; done
  return 1
}

run_step() {
  local id="$1" name="$2"; shift 2
  want "$id" || return 0
  local lf="$LOG/$name.log" t0 rc st
  t0=$(date +%s)
  printf "\n\033[1m==> [%s] %s\033[0m\n" "$(date +%H:%M:%S)" "$name"
  printf "    %s\n" "$*"
  "$@" > "$lf" 2>&1
  rc=$?
  case $rc in 0) st=OK ;; 3) st=SKIPPED ;; *) st="FAILED(rc=$rc)" ;; esac
  local mins; mins=$(awk "BEGIN{printf \"%.1f\", ($(date +%s)-$t0)/60}")
  printf "%s\t%s\t%s\t%s\n" "$name" "$st" "$mins" "logs/$name.log" >> "$STATUS"
  if [[ $st == OK ]]; then
    printf "    \033[32mOK\033[0m (%s min)\n" "$mins"
  elif [[ $st == SKIPPED ]]; then
    printf "    \033[33mSKIPPED\033[0m: %s\n" "$(grep -m1 SKIPPED "$lf" | cut -c1-200)"
  else
    printf "    \033[31m%s\033[0m - last lines of %s:\n" "$st" "$lf"; tail -n 8 "$lf" | sed 's/^/      /'
  fi
}

echo "Repository : $ROOT"
echo "Python     : $PY ($("$PY" --version 2>&1))"
echo "Data dir   : ${GBM_DATA:-$ROOT/data}"
echo "Results dir: $RES"
echo "Mode       : $([[ $QUICK == 1 ]] && echo quick || echo full)"

if [[ $FROM_RAW == 1 ]]; then
  for s in 00a_build_matrix 00b_quality_control 00c_state_assignment; do
    run_step "${s%%_*}" "$s" "$PY" "scripts/$s.py"
  done
  run_step 00e 00e_apply_build2_annotation "$PY" scripts/00e_apply_build_annotation.py --build 2
  run_step 00e 00e_apply_build1_annotation "$PY" scripts/00e_apply_build_annotation.py --build 1
  run_step 00d 00d_align_counts "$PY" scripts/00d_align_counts.py
fi
B2="${GBM_STATES_BUILD2:-${GBM_DATA:-$ROOT/data}/gbm_snrna_states_build2.h5ad}"
echo "Build 2   : $B2"

for s in 01_copy_number_inference 02_compartment_positive_control 03_detectability_resource \
         04_identity_gate 05_state_axis_flux_screen 06_enzyme_compartment_contrast \
         07_neuronal_candidate_contrast 08_detection_matched_single_gene 09_target_characterization \
         10_pathway_membership 11_patient_level_endpoint 12_depth_sensitivity \
         13_synaptic_program_contrast 14_set_level_matched_nulls 15_cgga_bulk_targets \
         16_external_cohorts 17_survival_covariates 29_cptac_proteome; do
  run_step "${s%%_*}" "$s" "$PY" "scripts/$s.py"
done

if [[ $QUICK == 1 ]]; then
  Q18=(--references none); Q19=(--n-boot 20000); Q22=(--n-null 200)
else
  Q18=(); Q19=(); Q22=()
fi
run_step 28 28_build_concordance_check       env GBM_STATES_H5AD="$B2" "$PY" scripts/28_build_concordance_check.py
run_step 18 18_cnv_classification_robustness "$PY" scripts/18_cnv_classification_robustness.py --n-jobs "$JOBS" "${Q18[@]}"
run_step 19 19_patient_level_robustness      "$PY" scripts/19_patient_level_robustness.py "${Q19[@]}"
run_step 20 20_primary_recurrent             "$PY" scripts/20_primary_recurrent.py
run_step 21 21_sulfotransferase_family       "$PY" scripts/21_sulfotransferase_family.py
run_step 22 22_cs_pathway_coherence          "$PY" scripts/22_cs_pathway_coherence.py "${Q22[@]}"
run_step 25 25_depmap_dependency             "$PY" scripts/25_depmap_dependency.py
run_step 26 26_multiplicity_corrections      "$PY" scripts/26_multiplicity_corrections.py
run_step 27 27_cohort_flow                   "$PY" scripts/27_cohort_flow.py
run_step 30 30_build2_sensitivity            "$PY" scripts/30_build2_sensitivity.py

run_step figures make_figures env PUBFIG_DIR="${PUBFIG_DIR:-$ROOT/tools/pubfig}" \
  "$PY" figures/make_figures.py
run_step figures make_figures_robustness env PUBFIG_DIR="${PUBFIG_DIR:-$ROOT/tools/pubfig}" \
  "$PY" figures/make_figures_robustness.py "$RES/robustness" figures/output

echo
echo "================================ RUN STATUS ================================"
column -t -s $'\t' "$STATUS" 2>/dev/null || cat "$STATUS"
echo "total: $(awk "BEGIN{printf \"%.1f\", ($(date +%s)-$T_ALL)/60}") min"
grep -q FAILED "$STATUS" && exit 1 || exit 0
