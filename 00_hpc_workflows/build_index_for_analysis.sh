#!/bin/bash
# build_index_for_analysis.sh
# Generates combined_seed_index.tsv directly from output_seeded/all_fixed
# and output_seeded/australian_fixed.
#
# Paths in the TSV point to the cluster output_seeded directories so
# the localize() function in MATLAB can remap them to the local all_fixed folder.
#
# Usage: bash build_index_for_analysis.sh

set -euo pipefail

WORK_ROOT="/data/horse/ws/<HPC_USER>-marrmot_recal"
OUT_DIR="${WORK_ROOT}/final_results_combined_fixed/index"
mkdir -p "${OUT_DIR}"

SEED_INDEX="${OUT_DIR}/combined_seed_index.tsv"

printf "catchment\tobjective\tmodel\tseed\tsource_namespace\tsummary_csv\tsummary_mat\tsignature_log\tstatus\n" \
  > "${SEED_INDEX}"

for ns in all_fixed australian_fixed; do
  root="${WORK_ROOT}/output_seeded/${ns}"
  [[ -d "${root}" ]] || continue

  find "${root}" -mindepth 4 -maxdepth 4 -type d -name 'seed_*' | sort | while read -r seed_dir; do
    rel="${seed_dir#${root}/}"
    catchment=$(echo "${rel}" | cut -d/ -f1)
    objective=$(echo "${rel}" | cut -d/ -f2)
    model=$(echo "${rel}" | cut -d/ -f3)
    seed=$(echo "${rel}" | cut -d/ -f4)

    summary_csv=""
    summary_mat=""
    signature_log=""
    status="complete"

    if [[ -f "${seed_dir}/summary.csv" ]]; then
      summary_csv="${seed_dir}/summary.csv"
    else
      status="missing_summary_csv"
    fi

    if [[ -f "${seed_dir}/summary.mat" ]]; then
      summary_mat="${seed_dir}/summary.mat"
    fi

    if [[ -f "${seed_dir}/signature_log.csv" ]]; then
      signature_log="${seed_dir}/signature_log.csv"
    elif [[ -f "${seed_dir}/signature_log.mat" ]]; then
      signature_log="${seed_dir}/signature_log.mat"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${catchment}" "${objective}" "${model}" "${seed}" "${ns}" \
      "${summary_csv}" "${summary_mat}" "${signature_log}" "${status}" \
      >> "${SEED_INDEX}"
  done
done

echo "=== Done ==="
echo "Index: ${SEED_INDEX}"
echo ""
echo "Quick stats:"
awk -F'\t' 'NR>1 && $9=="complete"{c++} END{print "  complete seeds: " c+0}' "${SEED_INDEX}"
awk -F'\t' 'NR>1 && $9!="complete"{c++} END{print "  incomplete seeds: " c+0}' "${SEED_INDEX}"
awk -F'\t' 'NR>1 && $8!=""{c++} END{print "  seeds with signature_log: " c+0}' "${SEED_INDEX}"
