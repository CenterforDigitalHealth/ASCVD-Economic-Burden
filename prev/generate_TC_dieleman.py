
import pandas as pd
import numpy as np
import os

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
prep_dir = os.path.join(base_dir, 'dataPreparation', 'TreatmentCost', 'adjust_prevalence')

# Constants from Dieleman et al / Notebook analysis
TC_USA_TOTAL_BILLIONS = 101.4 
HE_USA_TOTAL_BILLIONS_CONST = 2100.1
# Note: HE_USA_TOTAL_BILLIONS_CONST (2100.1) is the value used in the notebook. 
# It likely corresponds to Total Health Expenditure in USA in the base year (2013 or 2016).

# Load Prevalence to get USA rate
prev_df = pd.read_csv(os.path.join(prep_dir, 'prevalence.csv'))
usa_prev_row = prev_df[prev_df['Country Code'] == 'USA']
if usa_prev_row.empty:
    raise ValueError("USA not found in prevalence.csv")
PREV_USA_RATE = usa_prev_row['Diabetes mellitus'].values[0]

print(f"USA Prevalence Rate (per 100k?): {PREV_USA_RATE}")

# Derived Constant
# Formula: Cost_Country = HE_Country_2021 * (Share_Diabetes_in_HE_USA * (Prev_Country / Prev_USA))
# Share_Diabetes_in_HE_USA = TC_USA / HE_USA_Total
SHARE_USA = TC_USA_TOTAL_BILLIONS / HE_USA_TOTAL_BILLIONS_CONST
CONSTANT_FACTOR = SHARE_USA / PREV_USA_RATE

print(f"Diabetes Share of HE in USA: {SHARE_USA}")
print(f"Constant Factor (Share / Prev): {CONSTANT_FACTOR}")

# Load HE Data
he_df = pd.read_csv(os.path.join(data_dir, 'hepc_ppp.csv'))

# Prepare HE Data (2021)
# Ensure we have 2021 column
if '2021' not in he_df.columns:
    raise ValueError("Column '2021' not found in hepc_ppp.csv")

he_2021 = he_df[['Country Code', '2021']].rename(columns={'2021': 'HE_2021'})

# Prepare Prevalence Data
prev_data = prev_df[['Country Code', 'Diabetes mellitus']].rename(columns={'Diabetes mellitus': 'Prevalence'})

# Merge
df = pd.merge(he_2021, prev_data, on='Country Code', how='inner')

# Calculate Dieleman TC
df['Diabetes mellitus'] = df['HE_2021'] * CONSTANT_FACTOR * df['Prevalence']

# Format output
output_df = df[['Country Code', 'Diabetes mellitus']]

# Verify USA value
usa_val = output_df[output_df['Country Code'] == 'USA']['Diabetes mellitus'].values[0]
print(f"Calculated USA Dieleman TC Per Capita (2021): {usa_val}")

# Save
output_path = os.path.join(data_dir, 'TC_dieleman.csv')
output_df.to_csv(output_path, index=False)
print(f"Saved {output_path}")
