#!/bin/bash
# run_eval_fixed.sh
# Submits evaluation-period forward runs for all model x catchment x objective combos.
# Picks the best calibration seed automatically and writes to output_seeded/eval_fixed/.
#
# Usage: bash run_eval_fixed.sh   (from login node, self-submitting)

#SBATCH --job-name=marrmot_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --array=1-1%1
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH -A p_extruso

set -euo pipefail

WORK_ROOT="/data/horse/ws/<HPC_USER>-marrmot_recal"
CATCHMENTS_DIR="${WORK_ROOT}/caravan_subset"
COMPILED_DIR="${WORK_ROOT}/compiled/eval_fixed"
MANIFEST="${WORK_ROOT}/manifests/run_eval_fixed.tsv"
EVAL_ROOT="${WORK_ROOT}/output_seeded/eval_fixed"

OBJECTIVES=(
  of_KGE of_KGE_02_transf of_KGE_neg2_transf of_KGE_non_parametric
  of_KGE_split of_log_NSE of_NSE of_SHE of_diagnostic_efficiency
)

MODELS=(
  m_01_collie1_1p_1s m_02_wetland_4p_1s m_03_collie2_4p_1s m_04_newzealand1_6p_1s m_05_ihacres_7p_1s
  m_06_alpine1_4p_2s m_07_gr4j_4p_2s m_08_us1_5p_2s m_09_susannah1_6p_2s m_10_susannah2_6p_2s
  m_11_collie3_6p_2s m_12_alpine2_6p_2s m_13_hillslope_7p_2s m_14_topmodel_7p_2s m_15_plateau_8p_2s
  m_16_newzealand2_8p_2s m_17_penman_4p_3s m_18_simhyd_7p_3s m_19_australia_8p_3s m_20_gsfb_8p_3s
  m_21_flexb_9p_3s m_22_vic_10p_3s m_23_lascam_24p_3s m_24_mopex1_5p_4s m_25_tcm_6p_4s
  m_26_flexi_10p_4s m_27_tank_12p_4s m_28_xinanjiang_12p_4s m_29_hymod_5p_5s m_30_mopex2_7p_5s
  m_31_mopex3_8p_5s m_32_mopex4_10p_5s m_33_sacramento_11p_5s m_34_flexis_12p_5s m_35_mopex5_12p_5s
  m_36_modhydrolog_15p_5s m_37_hbv_15p_5s m_38_tank2_16p_5s m_39_mcrm_16p_5s m_40_smar_8p_6s
  m_41_nam_10p_6s m_42_hycymodel_12p_6s m_43_gsmsocont_12p_6s m_44_echo_16p_6s m_45_prms_18p_7s
  m_46_classic_12p_8s m_47_IHM19_16p_4s
)

# Catchments → calibration namespace mapping
declare -A NS_MAP
for c in camels_02017500 camels_03460000 camels_12381400 \
          camelsaus_143110A camelsbr_60615000 \
          camelsgb_27035 camelsgb_39037 \
          hysets_01AF007 lamah_200048; do
  NS_MAP[$c]="all_fixed"
done
NS_MAP[camelsaus_607155]="australian_fixed"

ALL_CATCHMENTS=( "${!NS_MAP[@]}" )

build_manifest() {
  mkdir -p "$(dirname "${MANIFEST}")" "${EVAL_ROOT}" logs

  # Build manifest: only include combos where at least one calibration seed finished
  # and eval result does NOT yet exist (idempotent re-runs).
  : > "${MANIFEST}"
  local skipped=0 queued=0

  for catchment in "${ALL_CATCHMENTS[@]}"; do
    local ns="${NS_MAP[$catchment]}"
    local cali_base="${WORK_ROOT}/output_seeded/${ns}/${catchment}"

    for objective in "${OBJECTIVES[@]}"; do
      for model in "${MODELS[@]}"; do
        # Skip if eval already done
        eval_csv="${EVAL_ROOT}/${catchment}/${objective}/${model}/eval_summary.csv"
        if [[ -f "${eval_csv}" ]]; then
          skipped=$((skipped+1))
          continue
        fi

        # Skip if no calibration seed finished
        has_cali=false
        for seed in 1 2 3 4 5; do
          if [[ -f "${cali_base}/${objective}/${model}/$(printf 'seed_%04d' $seed)/summary.csv" ]]; then
            has_cali=true
            break
          fi
        done
        if ! $has_cali; then
          skipped=$((skipped+1))
          continue
        fi

        printf '%s\t%s\t%s\t%s\n' \
          "${catchment}" "${objective}" "${model}" "${ns}" >> "${MANIFEST}"
        queued=$((queued+1))
      done
    done
  done

  echo "Eval manifest: ${queued} jobs to run, ${skipped} skipped (done or no cali)"
}

submit_array() {
  local total
  total=$(wc -l < "${MANIFEST}")
  if [[ "${total}" -eq 0 ]]; then
    echo "Nothing to submit."
    exit 0
  fi
  sbatch --array="1-${total}" "$0"
  echo "Submitted ${total} eval jobs"
}

run_one() {
  local line catchment objective model ns cali_root binary

  line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")
  IFS=$'\t' read -r catchment objective model ns <<< "${line}"

  cali_root="${WORK_ROOT}/output_seeded/${ns}"
  runner="${COMPILED_DIR}/${model}/run_runMARRMoT_eval.sh"

  if [[ ! -x "${runner}" ]]; then
    echo "ERROR: runner not found: ${runner}" >&2
    exit 1
  fi

  module load release/25.06
  module load MATLAB/2025a

  if [[ -z "${EBROOTMATLAB:-}" ]]; then
    echo "ERROR: EBROOTMATLAB not set after module load" >&2
    exit 1
  fi

  export MCR_CACHE_ROOT="${TMPDIR:-/tmp}/mcr_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  mkdir -p "${MCR_CACHE_ROOT}"

  echo "[$(date)] Running eval: ${catchment} / ${objective} / ${model}"

  "${runner}" "${EBROOTMATLAB}" \
    "${catchment}" \
    "${objective}" \
    "${CATCHMENTS_DIR}" \
    "${cali_root}" \
    "${EVAL_ROOT}"

  rm -rf "${MCR_CACHE_ROOT}"
  echo "[$(date)] Done: ${catchment} / ${objective} / ${model}"
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  build_manifest
  submit_array
  exit 0
fi

run_one
