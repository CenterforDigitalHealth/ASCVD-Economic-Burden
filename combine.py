#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
import numpy as np

DISEASE_ALIASES = {
    'IHD': 'Ischemic heart disease',
    'IS': 'Ischemic stroke',
    'PAD': 'Lower extremity peripheral arterial disease',
}


def find_result_files(folder, prefix):
    files = []
    for filename in sorted(os.listdir(folder)):
        if filename.startswith(prefix) and filename.endswith('.csv'):
            files.append(os.path.join(folder, filename))
    return files


def combine_csv(files):
    pieces_file = []
    for filename in files:
        df = pd.read_csv(filename)
        pieces_file.append(df)
    return pd.concat(pieces_file, ignore_index=True)


def parse_disease_filter(disease_arg, available_diseases):
    if disease_arg is None or str(disease_arg).strip() == '' or str(disease_arg).strip().lower() == 'all':
        return None

    available = list(pd.Series(available_diseases).dropna().astype(str).unique())
    available_lower = {name.lower(): name for name in available}
    selected = []
    for token in str(disease_arg).split(','):
        token = token.strip()
        if token == '':
            continue
        alias = DISEASE_ALIASES.get(token.upper(), token)
        match = available_lower.get(alias.lower(), alias)
        selected.append(match)

    selected = sorted(set(selected))
    return selected


def filter_by_disease(df, disease_arg):
    if 'disease' not in df.columns:
        return df

    selected = parse_disease_filter(disease_arg, df['disease'].unique())
    if selected is None:
        return df

    return df[df['disease'].isin(selected)].copy()


def filter_by_state(df, scenario=None, discount=None, informal=None):
    out = df
    if scenario is not None and 'scenario' in out.columns:
        out = out[out['scenario'].astype(str) == str(scenario)]

    if discount is not None and 'discount' in out.columns:
        discount_series = pd.to_numeric(out['discount'], errors='coerce')
        out = out[np.isclose(discount_series, float(discount), equal_nan=False)]

    if informal is not None and 'informal' in out.columns:
        informal_series = pd.to_numeric(out['informal'], errors='coerce')
        out = out[np.isclose(informal_series, float(informal), equal_nan=False)]

    return out.copy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine result csv files')
    parser.add_argument('--folder', type=str, default='tmpresults')
    parser.add_argument('--output-folder', type=str, default='results')
    parser.add_argument('--disease', type=str, default='all',
                        help='Disease selector: all, IHD, IS, PAD, full disease name, or comma-separated list')
    parser.add_argument('--scenario', type=str, default=None, help='Optional scenario filter (e.g., val)')
    parser.add_argument('--discount', type=float, default=None, help='Optional discount filter (e.g., 0.02)')
    parser.add_argument('--informal', type=float, default=None, help='Optional informal rate filter (e.g., 0)')
    args = parser.parse_args()

    folder = args.folder
    os.makedirs(folder, exist_ok=True)

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    print('aggregate data processing')
    aggregate_files = find_result_files(folder, 'aggregate_results_TC')
    if len(aggregate_files) > 0:
        df_aggregate_results = combine_csv(aggregate_files)
        df_aggregate_results = filter_by_disease(df_aggregate_results, args.disease)
        df_aggregate_results = filter_by_state(
            df_aggregate_results,
            scenario=args.scenario,
            discount=args.discount,
            informal=args.informal,
        )
        df_aggregate_results.to_csv(os.path.join(output_folder, 'aggregate_results.csv'), index=False)
        df_aggregate_results.to_csv(os.path.join(folder, 'aggregate_results.csv'), index=False)
        print(f'combined {len(aggregate_files)} aggregate files')
    else:
        print('no aggregate result files found')

    print('annual data processing')
    annual_files = find_result_files(folder, 'annual_results_TC')
    if len(annual_files) > 0:
        df_annual_results = combine_csv(annual_files)
        df_annual_results = filter_by_disease(df_annual_results, args.disease)
        df_annual_results = filter_by_state(
            df_annual_results,
            scenario=args.scenario,
            discount=args.discount,
            informal=args.informal,
        )
        df_annual_results.to_csv(os.path.join(output_folder, 'annual_results.csv'), index=False)
        df_annual_results.to_csv(os.path.join(folder, 'annual_results.csv'), index=False)
        print(f'combined {len(annual_files)} annual files')
    else:
        print('no annual result files found')

    print("Done!")
