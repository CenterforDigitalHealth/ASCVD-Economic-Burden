#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd


def combine_csv(filelist, startswith):
    pieces_file = []
    for file in filelist:
        filename = os.path.join(folder, file)
        if file.startswith(startswith):
            df = pd.read_csv(filename)
            pieces_file.append(df)
    df = pd.concat(pieces_file)
    return df


if __name__ == '__main__':

    folder = 'tmpresults'
    filelist = sorted(os.listdir(folder))

    if not os.path.exists('results'):
        os.mkdir('results')

    print('aggregate data processing')
    if not os.path.exists('results/aggregate_results.csv'):
        df_aggregate_results = combine_csv(filelist, 'aggregate')
        df_aggregate_results.to_csv('results/aggregate_results.csv', index=False)
    else:
        print ('aggregate_results already exists!')

    print('annual data processing')
    if not os.path.exists('results/annual_results.csv'):
        df_annual_results = combine_csv(filelist, 'annual')
        df_annual_results.to_csv('results/annual_results.csv', index=False)
    else:
        print ('annual_results already exists!')

    print("Done!")

