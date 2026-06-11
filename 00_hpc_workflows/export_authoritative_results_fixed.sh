#!/bin/bash
# Fixed version of export_authoritative_results.sh
# Sources from all_fixed and australian_fixed, which already contain all
# objectives correctly (no separate kge02_fix / neg2_fix needed).
#
# Run via SLURM:  sbatch export_results_fixed.sh
# Or directly:   bash export_authoritative_results_fixed.sh

set -euo pipefail

WORK_ROOT="/data/horse/ws/<HPC_USER>-marrmot_recal"
SRC_BASE="${WORK_ROOT}/output_seeded"
EXPORT_ROOT="${WORK_ROOT}/export_for_matlab_fixed"

mkdir -p "${EXPORT_ROOT}/data"
mkdir -p "${EXPORT_ROOT}/manifests"

copy_root() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  rsync -a --prune-empty-dirs \
    --exclude='mcr_cache*' \
    --exclude='cmaes_*.dat' \
    --exclude='cmaesvars.mat' \
    --exclude='summary.mat' \
    "$src"/ "$dst"/
}

echo "=== Copying authoritative roots ==="

# all_fixed contains all objectives — no exclusions needed
copy_root "${SRC_BASE}/all_fixed"        "${EXPORT_ROOT}/data/all"
copy_root "${SRC_BASE}/australian_fixed" "${EXPORT_ROOT}/data/australian"

echo "=== Building authoritative index ==="

INDEX="${EXPORT_ROOT}/manifests/authoritative_index.tsv"
INCOMPLETE="${EXPORT_ROOT}/manifests/incomplete_authoritative.tsv"

printf "namespace\tcatchment\tobjective\tmodel\tn_seeds\tstatus\tpath\n" > "$INDEX"
: > "$INCOMPLETE"

for ns in all australian; do
  root="${EXPORT_ROOT}/data/${ns}"
  [[ -d "$root" ]] || continue

  find "$root" -mindepth 3 -maxdepth 3 -type d | sort | while read -r combo; do
    rel="${combo#$root/}"
    catchment=$(echo "$rel" | cut -d/ -f1)
    objective=$(echo "$rel" | cut -d/ -f2)
    model=$(echo "$rel" | cut -d/ -f3)
    n_seeds=$(find "$combo" -path '*/seed_*/summary.csv' | wc -l)

    status="complete"
    if [[ "$n_seeds" -lt 5 ]]; then
      status="incomplete"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ns" "$catchment" "$objective" "$model" "$n_seeds" "$status" "$combo" >> "$INCOMPLETE"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$ns" "$catchment" "$objective" "$model" "$n_seeds" "$status" "$combo" >> "$INDEX"
  done
done

echo "=== Done ==="
echo "Export root: ${EXPORT_ROOT}"
echo "Index:       ${INDEX}"
echo "Incomplete:  ${INCOMPLETE}"
echo ""
awk -F'\t' 'NR>1 && $6=="complete"{c++} END{print "Complete combos: " c+0}' "$INDEX"
awk -F'\t' 'NR>1 && $6=="incomplete"{c++} END{print "Incomplete combos: " c+0}' "$INDEX"
