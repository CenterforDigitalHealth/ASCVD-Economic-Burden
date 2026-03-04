
import pandas as pd
import re
import os

def parse_val(s):
    if pd.isna(s): return None
    # Match the number before the parenthesis or the whole number if no parenthesis
    # Remove commas
    s_clean = str(s).replace(',', '')
    match = re.match(r'([0-9\.]+)', s_clean)
    if match: 
        return float(match.group(1))
    return None

def main():
    # File paths
    orig_path = 'tables/Table1_detailed_countries_d2i0.csv'
    # new_path = 'tmpresults/aggregate_results_TC1_MB1_informal0_discount0.02_val.csv'
    new_path = 'results/aggregate_results_imputed.csv'
    output_path = 'comparison_result.csv'

    print(f"Reading original table: {orig_path}")
    if not os.path.exists(orig_path):
        print(f"Error: File not found: {orig_path}")
        return

    print(f"Reading new table: {new_path}")
    if not os.path.exists(new_path):
        print(f"Error: File not found: {new_path}")
        return

    df_orig = pd.read_csv(orig_path)
    df_new = pd.read_csv(new_path)

    # Load Country Mapping
    mapping_path = 'data/dl1_countrycodeorg_country_name.csv'
    if os.path.exists(mapping_path):
        df_map = pd.read_csv(mapping_path, encoding='latin1')
        if 'Country Code' in df_map.columns and 'WBCountry' in df_map.columns:
            df_new = pd.merge(df_new, df_map[['Country Code', 'WBCountry']], on='Country Code', how='left')
        else:
            print("Warning: Mapping file columns mismatch")
    else:
        print(f"Warning: Mapping file not found: {mapping_path}")

    # Prepare Original Data
    # Columns likely: Region, WBCountry, Economic cost..., Percentage..., Per capita...
    # We need 'WBCountry' and 'Economic cost in millions of 2017 INT$'
    print("Original Columns:", df_orig.columns.tolist())
    val_col_orig = 'Economic cost in millions of 2017 INT$'
    if val_col_orig not in df_orig.columns:
        print(f"Warning: Column '{val_col_orig}' not found in original table.")
        # Try to find a column that looks like cost
        possible_cols = [c for c in df_orig.columns if 'cost' in c.lower()]
        if possible_cols:
            val_col_orig = possible_cols[0]
            print(f"Using '{val_col_orig}' instead.")
    
    df_orig['Original_Value'] = df_orig[val_col_orig].apply(parse_val)
    
    # Clean WBCountry in Original Data (remove *)
    if 'WBCountry' in df_orig.columns:
        df_orig['WBCountry'] = df_orig['WBCountry'].astype(str).str.replace('*', '', regex=False).str.strip()
        
    df_orig_clean = df_orig[['WBCountry', 'Original_Value']].copy()

    # Prepare New Data
    # Columns likely: Region, country, WBCountry, totalloss, pc_loss, tax %
    print("New Columns:", df_new.columns.tolist())
    # val_col_new = 'totalloss'
    val_col_new = 'GDPloss'
    
    # GDPloss is in Billions (from code /1e9), Original is in Millions
    # Convert Billions to Millions -> * 1000
    df_new['New_Value'] = df_new[val_col_new].apply(parse_val) * 1000
    df_new_clean = df_new[['WBCountry', 'New_Value']].copy()

    # Merge
    # Merge
    merged = pd.merge(df_orig_clean, df_new_clean, on='WBCountry', how='right')

    # Calculate Ratio and Difference
    merged['Ratio'] = merged['New_Value'] / merged['Original_Value']
    merged['Difference'] = merged['New_Value'] - merged['Original_Value']

    # Sort by Original Value descending (to see major economies first)
    merged = merged.sort_values('Original_Value', ascending=False)

    # Save
    merged.to_csv(output_path, index=False)
    print(f"\nSaved comparison result to {output_path}")

    # Display Top 10 Discrepancies (by absolute difference)
    print("\nTop 10 Countries by Value (Millions of 2017 INT$):")
    print(f"{'Country':<30} {'Original':<15} {'New':<15} {'Ratio':<10}")
    print("-" * 75)
    for _, row in merged.head(10).iterrows():
        print(f"{row['WBCountry']:<30} {row['Original_Value']:<15,.0f} {row['New_Value']:<15,.0f} {row['Ratio']:<10.2f}")

    print("\n(Note: 2,505,656 Million = 2.5 Trillion)")

if __name__ == "__main__":
    main()
