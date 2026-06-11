#!/bin/bash
# Builds index/combined_seed_index.tsv from the output_seeded/all_fixed and
# output_seeded/australian_fixed directories.
#
# The TSV is read by analysis scripts (e.g. seed_uncertainty_analysis.m) with:
#   T = readtable(...'combined_seed_index.tsv', 'FileType','text','Delimiter','\t');
#
# Columns: status, catchment, objective, model, seed, summary_csv, signature_log
#
# Usage:
#   bash build_seed_index.sh
# Output:
#   /data/horse/ws/<HPC_USER>-marrmot_recal/output_seeded/all_fixed/index/combined_seed_index.tsv

set -euo pipefail

WORK_ROOT="/data/horse/ws/<HPC_USER>-marrmot_recal"
OUT_DIR="${WORK_ROOT}/output_seeded/all_fixed/index"
TSV="${OUT_DIR}/combined_seed_index.tsv"

mkdir -p "${OUT_DIR}"

# Header
printf 'status\tcatchment\tobjective\tmodel\tseed\tsummary_csv\tsignature_log\n' > "${TSV}"

scan_dir() {
  local base="$1"

  # Walk: base/catchment/objective/model/seed_XXXX/
  for summary in "${base}"/*/*/*/*/summary.csv; do
    [[ -f "${summary}" ]] || continue

    seed_dir=$(dirname "${summary}")
    model_dir=$(dirname "${seed_dir}")
    obj_dir=$(dirname "${model_dir}")
    catch_dir=$(dirname "${obj_dir}")

    catchment=$(basename "${catch_dir}")
    objective=$(basename "${obj_dir}")
    model=$(basename "${model_dir}")
    seed_folder=$(basename "${seed_dir}")
    seed="${seed_folder#seed_}"     # strip "seed_" prefix, keep zero-padded number
    seed="${seed#0}"                # strip leading zeros so MATLAB reads as number
    seed="${seed:-0}"               # handle seed_0000 edge case

    # signature_log path — prefer .csv over .mat
    sig_csv="${seed_dir}/signature_log.csv"
    sig_mat="${seed_dir}/signature_log.mat"
    if [[ -f "${sig_csv}" ]]; then
      sig_log="${sig_csv}"
    elif [[ -f "${sig_mat}" ]]; then
      sig_log="${sig_mat}"
    else
      sig_log=""
    fi

    printf 'complete\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${catchment}" "${objective}" "${model}" "${seed}" \
      "${summary}" "${sig_log}" >> "${TSV}"
  done
}

echo "Scanning all_fixed..."
scan_dir "${WORK_ROOT}/output_seeded/all_fixed"

echo "Scanning australian_fixed..."
scan_dir "${WORK_ROOT}/output_seeded/australian_fixed"

total=$(tail -n +2 "${TSV}" | wc -l)
echo "Done. ${total} complete runs indexed."
echo "Index written to: ${TSV}"
