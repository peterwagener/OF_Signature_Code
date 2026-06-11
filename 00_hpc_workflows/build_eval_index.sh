#!/bin/bash
# build_eval_index.sh
# Builds eval_seed_index.tsv from output_seeded/eval_fixed/.
# One row per model x catchment x objective (no seed — best seed already selected).
#
# Usage: bash build_eval_index.sh

set -euo pipefail

WORK_ROOT="/data/horse/ws/<HPC_USER>-marrmot_recal"
EVAL_ROOT="${WORK_ROOT}/output_seeded/eval_fixed"
OUT_DIR="${WORK_ROOT}/final_results_combined_fixed/index"
mkdir -p "${OUT_DIR}"

EVAL_INDEX="${OUT_DIR}/eval_index.tsv"

printf "catchment\tobjective\tmodel\teval_summary_csv\teval_signatures_csv\tstatus\n" \
  > "${EVAL_INDEX}"

find "${EVAL_ROOT}" -mindepth 3 -maxdepth 3 -type d | sort | while read -r combo_dir; do
  rel="${combo_dir#${EVAL_ROOT}/}"
  catchment=$(echo "${rel}" | cut -d/ -f1)
  objective=$(echo "${rel}" | cut -d/ -f2)
  model=$(echo "${rel}" | cut -d/ -f3)

  eval_summary=""
  eval_signatures=""
  status="complete"

  if [[ -f "${combo_dir}/eval_summary.csv" ]]; then
    eval_summary="${combo_dir}/eval_summary.csv"
  else
    status="missing_eval_summary"
  fi

  if [[ -f "${combo_dir}/eval_signatures.csv" ]]; then
    eval_signatures="${combo_dir}/eval_signatures.csv"
  elif [[ "${status}" == "complete" ]]; then
    status="missing_eval_signatures"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${catchment}" "${objective}" "${model}" \
    "${eval_summary}" "${eval_signatures}" "${status}" \
    >> "${EVAL_INDEX}"
done

echo "=== Eval index built ==="
echo "Index: ${EVAL_INDEX}"
echo ""
echo "Quick stats:"
awk -F'\t' 'NR>1 && $6=="complete"{c++}      END{print "  complete:           " c+0}' "${EVAL_INDEX}"
awk -F'\t' 'NR>1 && $6!="complete"{c++}      END{print "  incomplete:         " c+0}' "${EVAL_INDEX}"
awk -F'\t' 'NR>1{c++}                         END{print "  total combos:       " c+0}' "${EVAL_INDEX}"
