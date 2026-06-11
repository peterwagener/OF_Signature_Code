#!/bin/bash
# compile_eval_fixed.sh
# Compiles ONE binary per model (47 total) — catchment/objective are
# runtime args so we don't need 4,230 separate binaries.
#
# Usage: bash compile_eval_fixed.sh   (from login node, self-submitting)

#SBATCH --job-name=marrmot_compile_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --array=1-47%5
#SBATCH --output=logs/compile_eval_%A_%a.out
#SBATCH --error=logs/compile_eval_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH -A p_extruso

set -euo pipefail

WORK_ROOT="/data/horse/ws/<HPC_USER>-marrmot_recal"
MARRMOT_ROOT="${WORK_ROOT}/MARRMoT_GR4J_fix"
CATCHMENTS_DIR="${WORK_ROOT}/caravan_subset"
TOSSH_ROOT="${WORK_ROOT}/TOSSH-master"
TEMPLATE="${WORK_ROOT}/updated_scripts/workflow_hpc_eval_template.m"
COMPILED_DIR="${WORK_ROOT}/compiled/eval_fixed"
MANIFEST="${WORK_ROOT}/manifests/compile_eval_fixed.tsv"

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

build_manifest() {
  mkdir -p "$(dirname "${MANIFEST}")" "${COMPILED_DIR}" logs
  : > "${MANIFEST}"
  for model in "${MODELS[@]}"; do
    echo "${model}" >> "${MANIFEST}"
  done
  echo "Manifest: $(wc -l < "${MANIFEST}") models"
}

submit_array() {
  local total
  total=$(wc -l < "${MANIFEST}")
  sbatch --array="1-${total}%15" "$0"
}

compile_one() {
  local model
  model=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")

  local results_dir matlab_file
  results_dir="${COMPILED_DIR}/${model}"
  matlab_file="${results_dir}/workflow_hpc_eval.m"

  mkdir -p "${results_dir}"

  # Skip if already compiled successfully
  if [[ -f "${results_dir}/runMARRMoT_eval" && -f "${results_dir}/runMARRMoT_eval.ctf" ]]; then
    echo "Already compiled: ${model} — skipping"
    exit 0
  fi

  # Only __MODEL_NAME__ is a compile-time placeholder
  sed "s#__MODEL_NAME__#${model}#g" "${TEMPLATE}" > "${matlab_file}"

  module load release/25.06
  module load MATLAB/2025a
  cd "${results_dir}"

  mcc \
    -a "${MARRMOT_ROOT}/MARRMoT/Functions" \
    -a "${MARRMOT_ROOT}/MARRMoT/Models" \
    -a "${CATCHMENTS_DIR}" \
    -a "${TOSSH_ROOT}" \
    -m workflow_hpc_eval.m \
    -o runMARRMoT_eval \
    -R -nodisplay \
    -R -nosplash

  rm -f includedSupportPackages.txt mccExcludedFiles.log readme.txt \
        requiredMCRProducts.txt unresolvedSymbols.txt

  echo "Compiled: ${model}"
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  build_manifest
  submit_array
  exit 0
fi

compile_one
