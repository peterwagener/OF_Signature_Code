#!/bin/bash
# Fixed version of build_final_combined_results.sh
# Reads from export_for_matlab_fixed/ and writes to final_results_combined_fixed/
# Produces the same combined_seed_index.tsv format as before.
#
# Run after export_authoritative_results_fixed.sh completes.
# Usage: bash build_final_combined_results_fixed.sh

set -euo pipefail

WORK_ROOT="/data/horse/ws/<HPC_USER>-marrmot_recal"
SRC_ROOT="${WORK_ROOT}/export_for_matlab_fixed/data"
OUT_ROOT="${WORK_ROOT}/final_results_combined_fixed"

RESULTS_ROOT="${OUT_ROOT}/results"
INDEX_ROOT="${OUT_ROOT}/index"

mkdir -p "${RESULTS_ROOT}" "${INDEX_ROOT}"

SEED_INDEX="${INDEX_ROOT}/combined_seed_index.tsv"
COMBO_INDEX="${INDEX_ROOT}/combined_combo_index.tsv"
INCOMPLETE_INDEX="${INDEX_ROOT}/incomplete_combos.tsv"

printf "catchment\tobjective\tmodel\tseed\tsource_namespace\tsummary_csv\tsummary_mat\tsignature_log\tstatus\n" > "${SEED_INDEX}"
printf "catchment\tobjective\tmodel\tsource_namespace\tn_seeds\tstatus\tcombined_path\n" > "${COMBO_INDEX}"
printf "catchment\tobjective\tmodel\tsource_namespace\tn_seeds\tcombined_path\n" > "${INCOMPLETE_INDEX}"

copy_combo() {
  local src_ns="$1"
  local catchment="$2"
  local objective="$3"
  local model="$4"

  local src_combo="${SRC_ROOT}/${src_ns}/${catchment}/${objective}/${model}"
  local dst_combo="${RESULTS_ROOT}/${catchment}/${objective}/${model}"

  if [[ ! -d "${src_combo}" ]]; then
    return 0
  fi

  mkdir -p "${dst_combo}"

  find "${src_combo}" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | sort | while read -r seed_dir; do
    seed=$(basename "${seed_dir}")
    dst_seed="${dst_combo}/${seed}"
    mkdir -p "${dst_seed}"

    summary_csv=""
    summary_mat=""
    signature_log=""
    status="complete"

    if [[ -f "${seed_dir}/summary.csv" ]]; then
      cp -p "${seed_dir}/summary.csv" "${dst_seed}/summary.csv"
      summary_csv="${dst_seed}/summary.csv"
    else
      status="missing_summary_csv"
    fi

    if [[ -f "${seed_dir}/summary.mat" ]]; then
      cp -p "${seed_dir}/summary.mat" "${dst_seed}/summary.mat"
      summary_mat="${dst_seed}/summary.mat"
    fi

    if [[ -f "${seed_dir}/signature_log.csv" ]]; then
      cp -p "${seed_dir}/signature_log.csv" "${dst_seed}/signature_log.csv"
      signature_log="${dst_seed}/signature_log.csv"
    elif [[ -f "${seed_dir}/signature_log.mat" ]]; then
      cp -p "${seed_dir}/signature_log.mat" "${dst_seed}/signature_log.mat"
      signature_log="${dst_seed}/signature_log.mat"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${catchment}" "${objective}" "${model}" "${seed}" "${src_ns}" \
      "${summary_csv}" "${summary_mat}" "${signature_log}" "${status}" >> "${SEED_INDEX}"
  done

  n_seeds=$(find "${dst_combo}" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | wc -l | tr -d ' ')
  combo_status="complete"
  if [[ "${n_seeds}" -lt 5 ]]; then
    combo_status="incomplete"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${catchment}" "${objective}" "${model}" "${src_ns}" "${n_seeds}" "${dst_combo}" >> "${INCOMPLETE_INDEX}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${catchment}" "${objective}" "${model}" "${src_ns}" "${n_seeds}" "${combo_status}" "${dst_combo}" >> "${COMBO_INDEX}"
}

echo "=== Building combined fixed tree ==="

for ns in all australian; do
  root="${SRC_ROOT}/${ns}"
  [[ -d "${root}" ]] || continue

  find "${root}" -mindepth 3 -maxdepth 3 -type d | sort | while read -r combo; do
    rel="${combo#${root}/}"
    catchment=$(echo "${rel}" | cut -d/ -f1)
    objective=$(echo "${rel}" | cut -d/ -f2)
    model=$(echo "${rel}" | cut -d/ -f3)
    copy_combo "${ns}" "${catchment}" "${objective}" "${model}"
  done
done

echo "=== Done ==="
echo "Combined root: ${OUT_ROOT}"
echo "Seed index:    ${SEED_INDEX}"
echo "Combo index:   ${COMBO_INDEX}"
echo "Incomplete:    ${INCOMPLETE_INDEX}"
echo ""
echo "Quick stats:"
find "${RESULTS_ROOT}" -name summary.csv | wc -l | awk '{print "  summary.csv files: " $1}'
find "${RESULTS_ROOT}" -name signature_log.csv | wc -l | awk '{print "  signature_log.csv files: " $1}'
awk -F'\t' 'NR>1 && $6=="complete"{c++} END{print "  complete combos: " c+0}' "${COMBO_INDEX}"
awk -F'\t' 'NR>1 && $6=="incomplete"{c++} END{print "  incomplete combos: " c+0}' "${COMBO_INDEX}"
