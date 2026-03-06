# ASCVD Economic Burden

This repository computes the macroeconomic burden of ASCVD using country-level data, then combines outputs, imputes missing countries, and generates summary tables.

## What `run.sh` does

`run.sh` is the main entrypoint. It runs 4 steps in order:

1. Model run (`main.py` or `model.py` if present)
2. File combination (`combine.py`)
3. Imputation (`imputation.py`)
4. Table generation (`generate_tables.py`)

Current `run.sh` uses fixed settings:

- `SCENARIOS=(val, lower, upper)`
- `DISCOUNT=0.02`
- `INFORMAL=0`
- `TC=1`
- `MB=1`

Imputation is restricted to these fixed settings and runs only the 3 scenarios above.

## Quick Start

```bash
bash run.sh all
```

You can also run a specific disease:

```bash
bash run.sh IHD
bash run.sh IS
bash run.sh PAD
```

Or a list:

```bash
bash run.sh IHD,IS
```

## Disease selector and output tag

`run.sh` converts the disease selector into a file tag:

- `all` -> `ALL`
- `IHD` / `Ischemic heart disease` -> `IHD`
- `IS` / `Ischemic stroke` -> `IS`
- `PAD` / `Lower extremity peripheral arterial disease` -> `PAD`
- Comma lists are joined (example: `IHD,IS` -> `IHD_IS`)

This tag is appended to output filenames.

## Output files

After `bash run.sh all`, key outputs are:

### `tmpresults/`

- `annual_results_TC1_MB1_informal0.0_discount0.02_val_ALL.csv`
- `annual_results_TC1_MB1_informal0.0_discount0.02_lower_ALL.csv`
- `annual_results_TC1_MB1_informal0.0_discount0.02_upper_ALL.csv`
- `aggregate_results_TC1_MB1_informal0.0_discount0.02_val_ALL.csv`
- `aggregate_results_TC1_MB1_informal0.0_discount0.02_lower_ALL.csv`
- `aggregate_results_TC1_MB1_informal0.0_discount0.02_upper_ALL.csv`
- `annual_results_ALL.csv`
- `aggregate_results_ALL.csv`
- `est_ALL.csv`

### `results/`

- `annual_results_ALL.csv`
- `aggregate_results_ALL.csv`
- `aggregate_results_imputed_ALL.csv`

### `tables/`

- `Table1_ALL.csv`

Notes:

- `run.sh` generates only **Table1** (Table2/Table3 are intentionally skipped).
- Legacy files from older runs may still exist in these folders.

### `logs/`

- `log_model_ALL_val_d0.02_i0.txt`
- `log_model_ALL_lower_d0.02_i0.txt`
- `log_model_ALL_upper_d0.02_i0.txt`
- `log_combine_ALL_allscen_d0.02_i0.txt`
- `log_imputation_ALL_allscen_d0.02_i0.txt`
- `log_tables_ALL_allscen_d0.02_i0.txt`

## Table1 interval behavior (`totalloss`, `pc_loss`, `tax %`)

Table1 uses:

- `val` as the central estimate
- `lower` and `upper` as interval bounds

If `lower`/`upper` rows are not available for a country, the table shows only the central value (it no longer forces `lower=mean=upper`).

## Running components manually (advanced)

### 1) Model

```bash
python main.py \
  -t 1 -m 1 -i 0 -d 0.02 -s val \
  --disease all \
  --file-tag ALL
```

### 2) Combine

```bash
python combine.py \
  --disease all \
  --file-tag ALL \
  --discount 0.02 \
  --informal 0
```

### 3) Imputation (fixed TC/MB/informal/discount, 3 scenarios)

```bash
python imputation.py \
  -i tmpresults/aggregate_results_ALL.csv \
  -o results/aggregate_results_imputed_ALL.csv \
  --disease all \
  --tc 1 --mb 1 \
  --discount 0.02 \
  --informal 0 \
  --output-tag ALL
```

### 4) Table1 only

```bash
python generate_tables.py \
  -f results/aggregate_results_imputed_ALL.csv \
  -d 0.02 -i 0 \
  --disease all \
  --output-tag ALL \
  --only-table1
```
