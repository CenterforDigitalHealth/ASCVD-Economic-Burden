#!/usr/bin/env bash
set -euo pipefail

# Optional argument: disease selector (all, IHD, IS, PAD, comma-separated list)
DISEASE="${1:-all}"

# Fixed run settings per request.
SCENARIO="val"
DISCOUNT="0.02"
INFORMAL="0"

if [[ -f "model.py" ]]; then
  MODEL_SCRIPT="model.py"
else
  MODEL_SCRIPT="main.py"
fi

mkdir -p logs tmpresults results tables

echo "[1/4] Run model: ${MODEL_SCRIPT}"
python "${MODEL_SCRIPT}" \
  -t 1 \
  -m 1 \
  -i "${INFORMAL}" \
  -d "${DISCOUNT}" \
  -s "${SCENARIO}" \
  --disease "${DISEASE}" \
  2>&1 | tee "logs/log_model_${SCENARIO}_d${DISCOUNT}_i${INFORMAL}.txt"

echo "[2/4] Combine outputs"
python combine.py \
  --disease "${DISEASE}" \
  --scenario "${SCENARIO}" \
  --discount "${DISCOUNT}" \
  --informal "${INFORMAL}" \
  2>&1 | tee "logs/log_combine_${SCENARIO}_d${DISCOUNT}_i${INFORMAL}.txt"

echo "[3/4] Imputation"
python imputation.py \
  -i tmpresults/aggregate_results.csv \
  -o results/aggregate_results_imputed.csv \
  --disease "${DISEASE}" \
  2>&1 | tee "logs/log_imputation_${SCENARIO}_d${DISCOUNT}_i${INFORMAL}.txt"

echo "[4/4] Generate tables"
python generate_tables.py \
  -f results/aggregate_results_imputed.csv \
  -d "${DISCOUNT}" \
  -i "${INFORMAL}" \
  --disease "${DISEASE}" \
  2>&1 | tee "logs/log_tables_${SCENARIO}_d${DISCOUNT}_i${INFORMAL}.txt"

echo "Done"
