#!/usr/bin/env python
# coding: utf-8
"""Compare original Table 1 results with new model results"""

import pandas as pd
import re

# Read files
df_orig = pd.read_csv('tables/Table1_detailed_countries_d2i0.csv')
df_new = pd.read_csv('tmpresults/aggregate_results_TC1_MB1_informal0.0_discount0.02_val.csv')

def parse_value(val):
    if pd.isna(val): 
        return None
    val = str(val).replace(',', '')
    match = re.match(r'([0-9.-]+)', val)
    return float(match.group(1)) if match else None

# Key countries to compare
countries = [
    ('USA', 'United States'), 
    ('CHN', 'China'), 
    ('IND', 'India'), 
    ('DEU', 'Germany'), 
    ('JPN', 'Japan'), 
    ('GBR', 'United Kingdom'), 
    ('BRA', 'Brazil'), 
    ('FRA', 'France'),
    ('MEX', 'Mexico'),
    ('KOR', 'Korea')
]

print('='*85)
print('COMPARISON: Original Table 1 vs New Model Results')
print('Parameters: TC=1, MB=1, informal=0, discount=0.02, scenario=val')
print('='*85)
print(f"{'Country':<22} {'Original(B)':<15} {'New(B)':<15} {'Diff%':<10}")
print('-'*85)

for code, name in countries:
    orig_row = df_orig[df_orig['WBCountry'].str.contains(name, case=False, na=False)]
    new_row = df_new[df_new['Country Code'] == code]
    
    if len(orig_row) > 0 and len(new_row) > 0:
        orig_val = orig_row.iloc[0]['Economic cost in millions of 2017 INT$']
        orig_gdp = parse_value(orig_val)
        if orig_gdp:
            orig_gdp = orig_gdp / 1000  # Convert millions to billions
        new_gdp = abs(new_row.iloc[0]['GDPloss'])
        
        if orig_gdp and new_gdp:
            diff = ((new_gdp - orig_gdp) / orig_gdp) * 100
            print(f'{name:<22} {orig_gdp:>12.1f} B {new_gdp:>12.1f} B {diff:>8.1f}%')
        else:
            print(f'{name:<22} Parse error')
    else:
        print(f'{name:<22} NOT FOUND (orig:{len(orig_row)}, new:{len(new_row)})')

print()
print(f'New results: {len(df_new)} countries')
print(f'Original table: {len(df_orig)} rows')

# Calculate totals
print()
print('='*85)
print('SUMMARY')
print('='*85)

# Find common countries
orig_countries_set = set()
for _, row in df_orig.iterrows():
    orig_countries_set.add(row['WBCountry'])

new_countries = set(df_new['Country Code'].unique())
print(f'Countries in new model run: {len(new_countries)}')
print(f'Countries in original table: {len(orig_countries_set)}')
