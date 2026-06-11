#!/bin/bash
# Fixed version of run_all.sh — reads compiled binaries from compiled/*_fixed/
# and writes results to output_seeded/*_fixed/
#
# Usage (from login node):
#   bash run_all_fixed.sh all          <- CAMELS + other non-Australian catchments
#   bash run_all_fixed.sh australian   <- Australian catchments
#   bash run_all_fixed.sh neg2_fix     <- neg2 objective only

#SBATCH --job-name=marrmot_run_fixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=50:00:00
#SBATCH --mem=4G
#SBATCH --array=1-1%15
#SBATCH --output=logs/run_all_fixed_%A_%a.out
#SBATCH --error=logs/run_all_fixed_%A_%a.err
#SBATCH -A p_extruso

set -euo pipefail

MODE="${1:-all}"
WORK_ROOT="/data/horse/ws/<HPC_USER>-marrmot_recal"
COMPILE_ROOT_BASE="${WORK_ROOT}/compiled"
RUN_ROOT_BASE="${WORK_ROOT}/output_seeded"
MANIFEST_DIR="${WORK_ROOT}/manifests"
SEEDS=(1 2 3 4 5)

resolve_mode() {
  case "${MODE}" in
    all)
      COMPILE_ROOT="${COMPILE_ROOT_BASE}/all_fixed"
      MANIFEST_FILE="${MANIFEST_DIR}/compile_all_fixed.tsv"
      RUN_ROOT="${RUN_ROOT_BASE}/all_fixed"
      ;;
    australian)
      COMPILE_ROOT="${COMPILE_ROOT_BASE}/australian_fixed"
      MANIFEST_FILE="${MANIFEST_DIR}/compile_australian_fixed.tsv"
      RUN_ROOT="${RUN_ROOT_BASE}/australian_fixed"
      ;;
    neg2_fix)
      COMPILE_ROOT="${COMPILE_ROOT_BASE}/neg2_fix_fixed"
      MANIFEST_FILE="${MANIFEST_DIR}/compile_neg2_fix_fixed.tsv"
      RUN_ROOT="${RUN_ROOT_BASE}/neg2_fix_fixed"
      ;;
    *)
      echo "Unknown mode: ${MODE}. Use 'all', 'australian', or 'neg2_fix'."
      exit 1
      ;;
  esac
}

submit_array() {
  local total
  total=$(wc -l < "${MANIFEST_FILE}")
  if [[ "${total}" -eq 0 ]]; then
    echo "Manifest is empty: ${MANIFEST_FILE}"
    echo "Run the matching compile script first."
    exit 1
  fi
  sbatch --array="1-${total}" "$0" "${MODE}"
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

append_global_csv() {
  local combo_dir="$1"
  local combo_csv="${combo_dir}/all_results.csv"
  local global_csv="${RUN_ROOT}/all_results.csv"
  local lock_file="${RUN_ROOT}/all_results.lock"

  [[ -f "${combo_csv}" ]] || return 0

  mkdir -p "${RUN_ROOT}"
  (
    flock 200
    if [[ ! -f "${global_csv}" ]]; then
      cat "${combo_csv}" > "${global_csv}"
    else
      tail -n +2 "${combo_csv}" >> "${global_csv}"
    fi
  ) 200>"${lock_file}"
}

run_one() {
  local line catchment objective model compile_dir combo_dir
  line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST_FILE}")
  IFS=$'\t' read -r catchment objective model <<< "${line}"

  compile_dir="${COMPILE_ROOT}/${catchment}/${objective}/${model}"
  combo_dir="${RUN_ROOT}/${catchment}/${objective}/${model}"

  if [[ ! -x "${compile_dir}/run_runMARRMoT.sh" ]]; then
    echo "Compiled runner missing: ${compile_dir}/run_runMARRMoT.sh"
    echo "Run the matching compile script first."
    exit 1
  fi

  mkdir -p "${combo_dir}"
  module load release/25.06
  module load MATLAB/2025a
  cd "${combo_dir}"

  local mcr_cache_root
  mcr_cache_root="${combo_dir}/mcr_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  mkdir -p "${mcr_cache_root}"
  export MCR_CACHE_ROOT="${mcr_cache_root}"

  for seed in "${SEEDS[@]}"; do
    "${compile_dir}/run_runMARRMoT.sh" "${EBROOTMATLAB}" "${seed}" "${combo_dir}"
  done

  merge_combo_csv "${combo_dir}"
  append_global_csv "${combo_dir}"

  rm -rf "${mcr_cache_root}"
}

resolve_mode

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  submit_array
  exit 0
fi

run_one
