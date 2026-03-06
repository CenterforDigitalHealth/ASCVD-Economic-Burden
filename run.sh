#!/usr/bin/env bash
set -euo pipefail

# Optional argument: disease selector (all, IHD, IS, PAD, comma-separated list)
DISEASE="${1:-all}"

# Fixed run settings per request.
SCENARIOS=("val" "lower" "upper")
DISCOUNT="0.02"
INFORMAL="0"
TC="1"
MB="1"

normalize_one_disease_tag() {
  local token
  token="$(echo "${1}" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' | tr '[:upper:]' '[:lower:]')"
  case "${token}" in
    "" )
      echo ""
      ;;
    "all" )
      echo "ALL"
      ;;
    "ihd"|"ischemic heart disease" )
      echo "IHD"
      ;;
    "is"|"ischemic stroke" )
      echo "IS"
      ;;
    "pad"|"lower extremity peripheral arterial disease" )
      echo "PAD"
      ;;
    * )
      echo "${token}" | tr '[:lower:]' '[:upper:]' | sed -E 's/[^A-Z0-9]+/_/g; s/^_+//; s/_+$//; s/__+/_/g'
      ;;
  esac
}

build_disease_tag() {
  local input="${1:-all}"
  local -a raw_tokens=()
  local -a tags=()
  local token mapped joined existing found tag_count
  tag_count=0
  IFS=',' read -r -a raw_tokens <<< "${input}"
  for token in "${raw_tokens[@]}"; do
    mapped="$(normalize_one_disease_tag "${token}")"
    if [[ -z "${mapped}" ]]; then
      continue
    fi
    if [[ "${mapped}" == "ALL" ]]; then
      echo "ALL"
      return
    fi
    found=0
    for existing in "${tags[@]-}"; do
      if [[ "${existing}" == "${mapped}" ]]; then
        found=1
        break
      fi
    done
    if [[ ${found} -eq 0 ]]; then
      tags+=("${mapped}")
      tag_count=$((tag_count + 1))
    fi
  done
  if [[ ${tag_count} -eq 0 ]]; then
    echo "ALL"
    return
  fi
  joined="${tags[*]-}"
  echo "${joined// /_}"
}

DISEASE_TAG="$(build_disease_tag "${DISEASE}")"
AGG_FILE="tmpresults/aggregate_results_${DISEASE_TAG}.csv"
IMPUTED_FILE="results/aggregate_results_imputed_${DISEASE_TAG}.csv"

if [[ -f "model.py" ]]; then
  MODEL_SCRIPT="model.py"
else
  MODEL_SCRIPT="main.py"
fi

mkdir -p logs tmpresults results tables

echo "[1/4] Run model: ${MODEL_SCRIPT}"
for SCENARIO in "${SCENARIOS[@]}"; do
  echo "  - scenario: ${SCENARIO}"
  python "${MODEL_SCRIPT}" \
    -t "${TC}" \
    -m "${MB}" \
    -i "${INFORMAL}" \
    -d "${DISCOUNT}" \
    -s "${SCENARIO}" \
    --disease "${DISEASE}" \
    --file-tag "${DISEASE_TAG}" \
    2>&1 | tee "logs/log_model_${DISEASE_TAG}_${SCENARIO}_d${DISCOUNT}_i${INFORMAL}.txt"
done

echo "[2/4] Combine outputs"
python combine.py \
  --disease "${DISEASE}" \
  --file-tag "${DISEASE_TAG}" \
  --discount "${DISCOUNT}" \
  --informal "${INFORMAL}" \
  2>&1 | tee "logs/log_combine_${DISEASE_TAG}_allscen_d${DISCOUNT}_i${INFORMAL}.txt"

echo "[3/4] Imputation"
python imputation.py \
  -i "${AGG_FILE}" \
  -o "${IMPUTED_FILE}" \
  --disease "${DISEASE}" \
  --tc "${TC}" \
  --mb "${MB}" \
  --discount "${DISCOUNT}" \
  --informal "${INFORMAL}" \
  --output-tag "${DISEASE_TAG}" \
  2>&1 | tee "logs/log_imputation_${DISEASE_TAG}_allscen_d${DISCOUNT}_i${INFORMAL}.txt"

echo "[4/4] Generate table 1"
python generate_tables.py \
  -f "${IMPUTED_FILE}" \
  -d "${DISCOUNT}" \
  -i "${INFORMAL}" \
  --disease "${DISEASE}" \
  --output-tag "${DISEASE_TAG}" \
  --only-table1 \
  2>&1 | tee "logs/log_tables_${DISEASE_TAG}_allscen_d${DISCOUNT}_i${INFORMAL}.txt"

echo "Done"
