# ASCVD Economic Burden

This repository computes the macroeconomic burden of ASCVD using country-level data, then combines outputs, imputes missing countries, and generates summary tables.

## How to Run

`run.sh` is the main entrypoint. It runs 4 steps in order:

1. Model run (`main.py`)
2. File combination (`combine.py`)
3. Imputation (`imputation.py`)
4. Table generation (`generate_tables.py`)

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

## Output files

After `bash run.sh all`, key outputs are:

### `tmpresults/`

- `annual_results_val_ALL.csv`
- `annual_results_lower_ALL.csv`
- `annual_results_upper_ALL.csv`
- `aggregate_results_val_ALL.csv`
- `aggregate_results_lower_ALL.csv`
- `aggregate_results_upper_ALL.csv`
- `annual_results_ALL.csv`
- `aggregate_results_ALL.csv`
- `est_ALL.csv`

### `results/`

- `annual_results_ALL.csv`
- `aggregate_results_ALL.csv`
- `aggregate_results_imputed_ALL.csv`

### `tables/`

- `Table_ALL.csv`

## Table1 interval behavior (`totalloss`, `pc_loss`, `tax %`)

Table1 uses:

- `val` as the central estimate
- `lower` and `upper` as interval bounds
