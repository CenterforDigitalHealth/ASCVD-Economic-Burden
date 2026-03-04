#!/bin/bash

## generate tables
python generate_tables.py -f 'results/aggregate_results_imputed.csv' -i 0.11 -d 0
python generate_tables.py -f 'results/aggregate_results_imputed.csv' -i 0.11 -d 0.02
python generate_tables.py -f 'results/aggregate_results_imputed.csv' -i 0.11 -d 0.03

python generate_tables.py -f 'results/aggregate_results_imputed.csv' -i 0 -d 0.02
python generate_tables.py -f 'results/aggregate_results_imputed.csv' -i 0.05 -d 0.02
python generate_tables.py -f 'results/aggregate_results_imputed.csv' -i 0.23 -d 0.02


## 
runipy generate_figures-Choropleth.ipynb
runipy generate_figures-TC_proportion.ipynb
runipy generate_figures-informal_proportion.ipynb