#!/bin/bash

# Invoke with:
#   bash merge_rerun_results.sh rerun_all
#   bash merge_rerun_results.sh rerun_australian
#   bash merge_rerun_results.sh rerun_neg2_fix

set -euo pipefail

MODE="${1:-rerun_all}"
WORK_ROOT="/data/horse/ws/<HPC_USER>-marrmot_recal"
RUN_ROOT_BASE="${WORK_ROOT}/output_seeded"
MANIFEST_DIR="${WORK_ROOT}/manifests"
SEEDS=(1 2 3 4 5)

resolve_mode() {
  case "${MODE}" in
    rerun_all)
      MANIFEST_FILE="${MANIFEST_DIR}/compile_rerun_all.tsv"
      RUN_ROOT="${RUN_ROOT_BASE}/all"
      ;;
    rerun_australian)
      MANIFEST_FILE="${MANIFEST_DIR}/compile_rerun_australian.tsv"
      RUN_ROOT="${RUN_ROOT_BASE}/australian"
      ;;
    rerun_neg2_fix)
      MANIFEST_FILE="${MANIFEST_DIR}/compile_rerun_neg2_fix.tsv"
      RUN_ROOT="${RUN_ROOT_BASE}/neg2_fix"
      ;;
    *)
      echo "Unknown mode: ${MODE}. Use 'rerun_all', 'rerun_australian', or 'rerun_neg2_fix'."
      exit 1
      ;;
  esac
}

merge_combo_csv() {
  local combo_dir="$1"
  local merged_csv="${combo_dir}/all_results.csv"
  local first=1

  : > "${merged_csv}"

  for seed in "${SEEDS[@]}"; do
    local seed_csv
    seed_csv=$(printf '%s/seed_%04d/summary.csv' "${combo_dir}" "${seed}")
    if [[ ! -f "${seed_csv}" ]]; then
      continue
    fi

    if [[ "${first}" -eq 1 ]]; then
      cat "${seed_csv}" >> "${merged_csv}"
      first=0
    else
      tail -n +2 "${seed_csv}" >> "${merged_csv}"
    fi
  done
}

resolve_mode

GLOBAL_CSV="${RUN_ROOT}/all_results.csv"
: > "${GLOBAL_CSV}"

while IFS=$'\t' read -r catchment objective model; do
  combo_dir="${RUN_ROOT}/${catchment}/${objective}/${model}"
  combo_csv="${combo_dir}/all_results.csv"

  merge_combo_csv "${combo_dir}"

  if [[ -s "${combo_csv}" ]]; then
    if [[ ! -s "${GLOBAL_CSV}" ]]; then
      cat "${combo_csv}" > "${GLOBAL_CSV}"
    else
      tail -n +2 "${combo_csv}" >> "${GLOBAL_CSV}"
    fi
  fi
done < "${MANIFEST_FILE}"
