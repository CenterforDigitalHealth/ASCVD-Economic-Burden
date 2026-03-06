#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.pca import PCA
import os
import numpy as np
import argparse
PPP = True
endyear = 2051
projectStartYear = 2020
DoPCA = False
IHME_PROXY_YEAR = '2019'
ASCVD_BASE_DIR = 'data/ASCVD'
ASCVD_GROUPS = ['IHD', 'IS', 'PAD']
ASCVD_MEASURE_FILES = {
    'Prevalence': 'Prevalence',
    'Incidence': 'Incidence',
    'Deaths': 'Deaths',
    'DALYs (Disability-Adjusted Life Years)': 'DALYs',
    'YLDs (Years Lived with Disability)': 'YLDs',
    'YLLs (Years of Life Lost)': 'YLLs',
}
SCENARIOS = ['val', 'lower', 'upper']
DISEASE_ALIASES = {
    'IHD': 'Ischemic heart disease',
    'IS': 'Ischemic stroke',
    'PAD': 'Lower extremity peripheral arterial disease',
}
if DoPCA:
    summaryfile = 'tmpresults/models_summary_prevalence_dalys_pca.txt'
else:
    summaryfile = 'tmpresults/models_summary_prevalence_dalys_nopca.txt'
# ## Part 0. Get population / GDP / per GDP data
coefficient_file = 'tmpresults/models_coefficient.txt'
# In[ ]:


def normalize_file_tag(file_tag):
    if file_tag is None:
        return None
    tag = str(file_tag).strip()
    if tag == '':
        return None
    if tag.lower() == 'all':
        return 'ALL'
    safe = []
    for ch in tag:
        if ch.isalnum() or ch in ['_', '-']:
            safe.append(ch.upper())
        elif ch in [',', ' ', '/']:
            safe.append('_')
    safe_tag = ''.join(safe).strip('_')
    while '__' in safe_tag:
        safe_tag = safe_tag.replace('__', '_')
    return safe_tag if safe_tag != '' else None


def append_file_tag(path, file_tag):
    tag = normalize_file_tag(file_tag)
    if tag is None:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{tag}{ext}"


def read_csv_safe(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs['engine'] = 'python'
        return pd.read_csv(path, **fallback_kwargs)


def build_population_weights(year=IHME_PROXY_YEAR):
    pop = read_csv_safe('data/population_un.csv')
    if year not in pop.columns:
        raise KeyError(f"population_un.csv does not contain year {year}")

    pop_year = pop[['Country Code', 'sex', 'age', year]].copy()
    pop_year = pop_year.rename(columns={year: 'pop'})
    pop_year['pop'] = pd.to_numeric(pop_year['pop'], errors='coerce').fillna(0)
    pop_year = pop_year[pop_year['pop'] > 0]

    both = pop_year.groupby(['Country Code', 'age'], as_index=False)['pop'].sum()
    both['sex'] = 'B'

    return pd.concat([pop_year[['Country Code', 'sex', 'age', 'pop']], both], ignore_index=True)


def aggregate_country_metric(df_metric, metric_name, year, population_weights):
    if year not in df_metric.columns:
        return pd.DataFrame(columns=['Country Code', 'cause', 'measure', 'value'])

    df = df_metric.rename(columns={
        'ISO3': 'Country Code',
        'cause_name': 'cause',
        'sex_name': 'sex',
        'age_name': 'age',
        'measure_name': 'measure',
    }).copy()

    if 'measure' not in df.columns:
        df['measure'] = metric_name
    else:
        df['measure'] = metric_name

    df['sex'] = df['sex'].replace({
        'Female': 'F',
        'Male': 'M',
        'Both': 'B',
        'female': 'F',
        'male': 'M',
        'both': 'B',
    })
    df = df[df['sex'].isin(['B', 'F', 'M'])]
    df[year] = pd.to_numeric(df[year], errors='coerce')
    df = df.dropna(subset=['Country Code', 'sex', 'age', 'cause', year])

    df = df.merge(population_weights, on=['Country Code', 'sex', 'age'], how='left')
    df['pop'] = pd.to_numeric(df['pop'], errors='coerce').fillna(0)
    df['weighted'] = df[year] * df['pop']

    agg = df.groupby(['Country Code', 'cause', 'measure'], as_index=False).agg(
        weighted=('weighted', 'sum'),
        pop=('pop', 'sum'),
        mean_val=(year, 'mean'),
    )
    agg['value'] = np.where(agg['pop'] > 0, agg['weighted'] / agg['pop'], agg['mean_val'])
    agg['value'] = pd.to_numeric(agg['value'], errors='coerce')
    agg = agg.dropna(subset=['value'])
    return agg[['Country Code', 'cause', 'measure', 'value']]


def load_ihme_proxy_from_ascvd(base_dir=ASCVD_BASE_DIR, year=IHME_PROXY_YEAR):
    population_weights = build_population_weights(year=year)
    all_rows = []

    for group in ASCVD_GROUPS:
        for measure_name, file_prefix in ASCVD_MEASURE_FILES.items():
            by_scenario = []
            for scenario in SCENARIOS:
                path = os.path.join(base_dir, group, f'{file_prefix}_{scenario}.csv')
                if not os.path.exists(path):
                    continue

                df_metric = read_csv_safe(path, encoding='utf-8-sig')
                agg = aggregate_country_metric(df_metric, measure_name, year, population_weights)
                if len(agg) == 0:
                    continue
                agg = agg.rename(columns={'value': scenario})
                by_scenario.append(agg)

            if len(by_scenario) == 0:
                continue

            merged = by_scenario[0]
            for extra in by_scenario[1:]:
                merged = merged.merge(extra, on=['Country Code', 'cause', 'measure'], how='outer')
            all_rows.append(merged)

    if len(all_rows) == 0:
        return pd.DataFrame(columns=['Country Code', 'cause', 'measure', 'val', 'lower', 'upper'])

    df_ihme = pd.concat(all_rows, ignore_index=True)
    for scenario in SCENARIOS:
        if scenario not in df_ihme.columns:
            df_ihme[scenario] = np.nan
        df_ihme[scenario] = pd.to_numeric(df_ihme[scenario], errors='coerce')
    df_ihme = df_ihme.dropna(subset=['Country Code', 'cause', 'measure'])
    return df_ihme


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
    return sorted(set(selected))


def filter_df_by_disease(df, disease_arg):
    if 'disease' not in df.columns:
        return df
    selected = parse_disease_filter(disease_arg, df['disease'].unique())
    if selected is None:
        return df
    return df[df['disease'].isin(selected)].copy()


## MAIN FUNCTION - input
## OUTPUT FILE
save_filename_gdp_total1 = 'tmpresults/GDP_TOTAL_discount0.csv'
save_filename_gdp_total2 = 'tmpresults/GDP_TOTAL_discount0.02.csv'
save_filename_gdp_total3 = 'tmpresults/GDP_TOTAL_discount0.03.csv'
save_filename_pop_total = 'tmpresults/POP_TOTAL.csv'
save_filename_gdp_psy = 'tmpresults/GDP_PSY.csv'
save_filename_pop_psy = 'tmpresults/POP_PSY.csv'




if os.path.exists(save_filename_gdp_total1) and \
    os.path.exists(save_filename_gdp_total2) and \
    os.path.exists(save_filename_gdp_total3) and \
    os.path.exists(save_filename_pop_total) and \
    os.path.exists(save_filename_gdp_psy) and \
    os.path.exists(save_filename_pop_psy) :
    pass
else:
    PPP = True
    if PPP:
        gdp_data = pd.read_csv('data/GDP_ppp.csv')
        gdp_cia = pd.read_csv('data/GDP_ppp_cia.csv')
    else:
        gdp_data = pd.read_csv('data/GDP.csv')

    pop_data_un = pd.read_csv('data/population_un.csv')
    pop_data_total = pd.read_csv('data/population_total.csv')

    startyear = 2019
    projectStartYear = 2020
    endyear = 2051
    # percap = pd.read_csv('../data/GDP_per_cia.csv')
    grow_rate = 0.03
    pop_un = pop_data_un.groupby('Country Code').sum()
    pop_total = pop_un[[str(i) for i in range(projectStartYear,endyear,1)]].sum(axis=1).to_frame('totalPOP')
    pop_fill = pop_data_total.set_index('Country Code')
    pop_filled = pop_fill[[str(i) for i in range(projectStartYear,endyear,1)]].sum(axis=1).to_frame('totalPOP')
    allPOP = pd.concat([pop_total, pop_filled]).reset_index()
    allPOP = allPOP.drop_duplicates(subset=['Country Code'], keep='first')
    allPOP.to_csv('tmpresults/POP_TOTAL.csv', index=False)

    for discount in [0, 0.02, 0.03]:
        DiscountRate = []
        rate = 1 / (1 - discount) ** (projectStartYear - startyear)
        DiscountRate.append(rate)
        for i in range(1, endyear-startyear, 1):
            rate = rate * (1 - discount)
            DiscountRate.append(rate)

        years = [str(i) for i in range(startyear, endyear, 1)]
        gdp = gdp_data[years].values
        index = gdp_data['Country Code']
        gdp_total = (gdp * DiscountRate)[:, projectStartYear-startyear:].sum(axis=1)
        gdp_total = pd.DataFrame(index=index, columns=['totalGDP'] ,data=gdp_total)

        gdp_fill = gdp_cia.set_index('Country Code')
        gdp_fill = gdp_fill.sort_index()
        years = (endyear - projectStartYear)

        r =  (1 + grow_rate) * (1 - discount)
        if r == 0:
            gdp_filled = years * gdp_fill['value']
        else:
            gdp_filled = (r ** years - 1) / (r - 1) * gdp_fill['value']
        gdp_filled = gdp_filled.to_frame('totalGDP')
        allGDP = pd.concat([gdp_total, gdp_filled]).reset_index()
        allGDP = allGDP.drop_duplicates(subset=['Country Code'], keep='first')
        allGDP.to_csv('tmpresults/GDP_TOTAL_discount%s.csv'%(discount), index=False)

    pop_psy1 = pop_un[[str(projectStartYear)]].rename(columns={str(projectStartYear):'pop_psy'})
    pop_psy2 = pop_fill[[str(projectStartYear)]].rename(columns={str(projectStartYear):'pop_psy'})
    pop_psy = pd.concat([pop_psy1, pop_psy2]).reset_index().drop_duplicates(subset=['Country Code'], keep='first')

    gdp_psy1 = gdp_data.set_index('Country Code')[str(projectStartYear)].to_frame('gdp_psy')
    gdp_psy2 = gdp_fill['value'].to_frame('gdp_psy')
    gdp_psy = pd.concat([gdp_psy1, gdp_psy2], axis=0).reset_index().drop_duplicates(subset=['Country Code'], keep='first')

    pop_psy.to_csv('tmpresults/POP_PSY.csv', index=False)
    gdp_psy.to_csv('tmpresults/GDP_PSY.csv', index=False)


# ## Part 1. raw estimate for each disease here


# In[ ]:


def get_df(df_result, ConsiderTC, ConsiderMB, informal, discount, scenario):
    df = df_result[(df_result['discount']==discount)&
                   (df_result['ConsiderTC']==ConsiderTC)&
                   (df_result['ConsiderMB']==ConsiderMB)&
                   (df_result['informal']==informal)&
                   (df_result['scenario']==scenario)]
    df = df[df['tax'] > 0]
    return df

def get_IHME_data(df_IHME, disease, scenario):
    df_IHME_disease = df_IHME[df_IHME['cause'] == disease]
    if len(df_IHME_disease) == 0:
        return pd.DataFrame(columns=['Country Code', 'comp_0'])

    if scenario not in df_IHME_disease.columns:
        if 'val' in df_IHME_disease.columns:
            scenario = 'val'
        else:
            return pd.DataFrame(columns=['Country Code', 'comp_0'])

    data = pd.pivot_table(df_IHME_disease, index='Country Code', columns='measure', values=scenario)
    if DoPCA:
        required_cols = ['DALYs (Disability-Adjusted Life Years)', 'Deaths', 'Prevalence', 'Incidence',
                         'YLDs (Years Lived with Disability)', 'YLLs (Years of Life Lost)']
        for col in required_cols:
            if col not in data.columns:
                data[col] = np.nan
        x = data[required_cols].dropna()
        if len(x) == 0:
            return pd.DataFrame(columns=['Country Code', 'comp_0', 'comp_1'])
        pca_model = PCA(x, standardize=False, demean=True)
        x_input = pca_model.factors.iloc[:, :2] 
    else:
        if 'Prevalence' not in data.columns:
            return pd.DataFrame(columns=['Country Code', 'comp_0'])
        x = data[['Prevalence']].dropna()
        if len(x) == 0:
            return pd.DataFrame(columns=['Country Code', 'comp_0'])
        x.columns = ['comp_0']
        x_input = x

    x_input = x_input.replace([np.inf, -np.inf], np.nan).dropna()
    x_input = x_input.reset_index()
    return x_input

def get_Indicator_data():
    countries_info = pd.read_csv('data/dl1_countrycodeorg_country_name.csv', encoding='ISO-8859-1')
    Income = countries_info[['Country Code', 'Income group']]
    # print(Income['Income group'].unique())
    col1 = Income['Country Code']
    col2 = (Income['Income group'] == 'High income').astype(int)
    col3 = (Income['Income group'] == 'Upper middle income').astype(int)
    col4 = (Income['Income group'] == 'Lower middle income').astype(int)
    col5 = (Income['Income group'] == 'Low income').astype(int)
    col6 = col2
    df_income = pd.concat([col1, col6], axis=1)
    # df_income.columns = ['Country Code', 'High income', 'Upper middle income', 'Lower middle income', 'Low income']
    df_income.columns = ['Country Code', 'Upper income']
    return df_income

def get_aggregate_data(df_agg, disease, IHME_data):
    df = df_agg[df_agg['disease'] == disease]
    data = df.merge(IHME_data, on='Country Code',how='inner')
    # for IHME_data in IHMElist:
    #     data = data.merge(IHME_data, on='Country Code',how='inner')
    # country_touse = pd.read_csv('../country_touse.csv')
    # data = data.merge(country_touse, on=['Country Code'],how='inner')
    data = data.dropna()
    return data

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def get_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] <= fence_low) | (df_in[col_name] >= fence_high)]
    return df_out

# def fit_ols_model(data):
#     X = sm.add_constant(data[['Prevalence', 'DALYs (Disability-Adjusted Life Years)']])#, 'Incidence','DALYs (Disability-Adjusted Life Years)']])
#     ols_model = sm.OLS(data['tax'], X)
#     # ols_model = sm.RLM(data['tax'], X, M=sm.robust.norms.HuberT())
#     ols_results = ols_model.fit()
#     return ols_results

def fit_ols_model_pca(data):
    # x_input = data[['comp_0', 'comp_1']]
    if DoPCA:
        x_input = data[['comp_0', 'comp_1', 'Upper income']]
    else:
        x_input = data[['comp_0']]
    # x0, x1 = data[['comp_0']].values, data[['comp_1']].values
    # x00 = x0 * x0
    # x11 = x1 * x1
    # x01 = x0 * x1
    # x_input = np.concatenate([x0, x1, x00, x01, x11], axis=1)
    x_input = np.log(x_input.clip(lower=1e-12))
    X = sm.add_constant(x_input)#, 'Incidence','DALYs (Disability-Adjusted Life Years)']])
    ols_model = sm.OLS(np.log(data['tax'].clip(lower=1e-12)), X)
    # ols_model = sm.RLM(data['tax'], X, M=sm.robust.norms.HuberT())
    ols_results = ols_model.fit()
    return ols_results

# for estimation
def get_estimation_prepare(df_agg, disease, IHME_data):
    df = df_agg[df_agg['disease'] == disease]
    # for IHME_data in IHMElist:
    #     est = est.merge(IHME_data, on='Country Code',how='inner')    
    est = IHME_data.merge(df[['Country Code','tax']],on='Country Code',how='outer')
    est = est[(est['tax'].isnull())]
    # print (len(est))
    return est

def get_estimation_result(STATISTICS_DATA, est, ols_results):
    if len(est) == 0:
        return pd.DataFrame(columns=['Country Code', 'pc_loss', 'GDPloss', 'tax'])

    estdata = est.merge(STATISTICS_DATA, on='Country Code').set_index('Country Code')
    if len(estdata) == 0:
        return pd.DataFrame(columns=['Country Code', 'pc_loss', 'GDPloss', 'tax'])

    # save_folder = os.path.join(result_folder, disease)
    if DoPCA:
        est['tax'] = ols_results.params[0]+ols_results.params[1]*est['comp_0']+ols_results.params[2]*est['comp_1'] +ols_results.params[3]*est['Upper income'] ##+ols_results.params[3]*est['Incidence']+ols_results.params[3]*est['DALYs (Disability-Adjusted Life Years)']
    else:
        est['tax'] = ols_results.params[0] + ols_results.params[1] * np.log(est['comp_0'].clip(lower=1e-12))
    est['tax'] = np.exp(est['tax'])
    est = est.set_index('Country Code')
    est['GDPloss'] = est['tax']*estdata['totalGDP']
    est['pc_loss'] = est['GDPloss']/(estdata['totalPOP']/(endyear-projectStartYear))
    # convert to billions
    est['GDPloss'] = est['GDPloss']/1e9
    # print(est)
    est = est.reset_index()
    # print(est.columns)
    est = est[['Country Code', 'pc_loss','GDPloss','tax']]
    
    # if not os.path.exists(save_folder):
        # os.makedirs(save_folder)
    # print ('saving estimation results in ', save_folder)
    # est.to_csv(os.path.join(save_folder, 'est_discount_%s.csv'%(discount)))
    return est


# In[ ]:


def Process(df_result, df_IHME, diseases, STATISTICS_DATA, ConsiderTC, ConsiderMB, informal, discount, scenario):
    pieces = []
    df_agg = get_df(df_result, ConsiderTC, ConsiderMB, informal, discount, scenario)
    if len(df_agg) == 0:
        return df_agg
    Indicator = get_Indicator_data()

    with open(summaryfile, 'a+') as f:
        with open(coefficient_file, 'a+') as f2:   
            for i, disease in enumerate(diseases):
                disease = diseases[i]
                IHMEdata = get_IHME_data(df_IHME, disease, scenario)
                if len(IHMEdata) == 0:
                    continue
                IHMEdata = IHMEdata.merge(Indicator, on='Country Code')
                data = get_aggregate_data(df_agg, disease, IHMEdata)
                if len(data) < 5:
                    continue
                # data = remove_outlier(data, 'tax')
                try:
                    ols_results = fit_ols_model_pca(data)
                except Exception:
                    continue
                print("****************************", file=f)
                print("****************************", file=f)
                print("****************************", file=f)
                print("ConsiderTC, ConsiderMB, informal, discount, scenario", file=f)
                print(ConsiderTC, ConsiderMB, informal, discount, scenario, file=f)
                print(i, disease, file=f)
                print(ols_results.summary(), file=f)
                est_prepare = get_estimation_prepare(df_agg, disease, IHMEdata)
                est = get_estimation_result(STATISTICS_DATA, est_prepare, ols_results)
                if len(est) == 0:
                    continue
                est['disease'] = disease
                est['scenario'] = scenario
                est['ConsiderTC'] = ConsiderTC
                est['ConsiderMB'] = ConsiderMB
                est['informal'] = informal
                est['discount'] = discount     
                pieces.append(est)  
                print("ConsiderTC, informal, discount, scenario, ols_results.params[0](ols_results.pvalues[0]), ols_results.params[1](ols_results.pvalues[1]), R-squared", file=f2)
                print(ConsiderTC, informal, discount, scenario, "%s (%.3e)"%(ols_results.params[0],ols_results.pvalues[0]), "%s (%.3e)"%(ols_results.params[1],ols_results.pvalues[1]), ols_results.rsquared, file=f2)


    if len(pieces) == 0:
        return pd.DataFrame(columns=['Country Code', 'pc_loss', 'GDPloss', 'tax', 'disease',
                                     'scenario', 'ConsiderTC', 'ConsiderMB', 'informal', 'discount'])

    df = pd.concat(pieces, ignore_index=True)
    
    return df
        
if __name__ == "__main__":

    if not os.path.exists('tmpresults/'):
        os.makedirs('tmpresults')

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--input', type=str, default='tmpresults/aggregate_results.csv') # or 0
    parser.add_argument('-o', '--output', type=str, default='tmpresults/aggregate_results_imputed.csv') # or 0
    parser.add_argument('-g', '--gdpfile', '--gpdfile', dest='gdpfile', type=str, default='tmpresults/GDP_TOTAL.csv') # or 'lower', 'upper'
    parser.add_argument('-p', '--popfile', type=str, default='tmpresults/POP_TOTAL.csv') # or 0.02, 0.03
    parser.add_argument('--disease', type=str, default='all',
                        help='Disease selector: all, IHD, IS, PAD, full disease name, or comma-separated list')
    parser.add_argument('--tc', type=int, default=None, help='Optional ConsiderTC filter/value for imputation')
    parser.add_argument('--mb', type=int, default=None, help='Optional ConsiderMB filter/value for imputation')
    parser.add_argument('--scenario', type=str, default=None, help='Optional scenario filter/value for imputation')
    parser.add_argument('--informal', type=float, default=None, help='Optional informal filter/value for imputation')
    parser.add_argument('--discount', type=float, default=None, help='Optional discount filter/value for imputation')
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Optional output file suffix tag, e.g. ALL, IHD, IS, PAD')
    args = parser.parse_args()
    output_tag = normalize_file_tag(args.output_tag)
    if output_tag is not None:
        summaryfile = append_file_tag(summaryfile, output_tag)
        coefficient_file = append_file_tag(coefficient_file, output_tag)
    ## ## MAIN FUNCTION - input
    

    ## INPUT FILE
    df_result = read_csv_safe(args.input)
    df_result = filter_df_by_disease(df_result, args.disease)
    if len(df_result) == 0:
        raise ValueError(f"No rows left after disease filter: {args.disease}")
    diseases = sorted(df_result['disease'].unique())
    legacy_ihme_path = "bigdata/data_diabetes/IHME.csv"
    if os.path.exists(legacy_ihme_path):
        countries_info = read_csv_safe('data/dl1_countrycodeorg_country_name.csv', encoding='ISO-8859-1')
        code_map = dict(zip(countries_info.country, countries_info['Country Code']))
        df_IHME = read_csv_safe(legacy_ihme_path)

        def get_code(x):
            try:
                return code_map[x]
            except KeyError:
                if "Cote d'Ivoire" in x or "Côte d'Ivoire" in x:
                    return 'CIV'
                return None

        df_IHME['Country Code'] = df_IHME['location'].apply(get_code)
        df_IHME = df_IHME.dropna(subset=['Country Code'])
        df_IHME = df_IHME[(df_IHME['year'] == int(IHME_PROXY_YEAR)) & (df_IHME['metric'] == 'Rate')]
    else:
        df_IHME = load_ihme_proxy_from_ascvd()
    if 'cause' in df_IHME.columns:
        df_IHME = df_IHME[df_IHME['cause'].isin(diseases)]
    print(df_result.columns)


    # In[ ]:


    import warnings
    warnings.filterwarnings("ignore")
    print('imputing.........')
    tc_values = [int(args.tc)] if args.tc is not None else [1, 0]
    mb_values = [int(args.mb)] if args.mb is not None else [1]
    scenario_values = [str(args.scenario)] if args.scenario is not None else ['val', 'lower', 'upper']
    informal_values = [float(args.informal)] if args.informal is not None else [0, 0.05, 0.11, 0.23]
    discount_values = [float(args.discount)] if args.discount is not None else [0, 0.02, 0.03]
    est_pieces = []
    for ConsiderTC in tc_values:
        for ConsiderMB in mb_values:
            for scenario in scenario_values:
                for informal in informal_values:
                    for discount in discount_values:
                        gdp_base = os.path.splitext(args.gdpfile)[0]
                        gdp_file_name = gdp_base + '_discount%s.csv' % (discount)
                        GDP_filled = read_csv_safe(gdp_file_name).set_index('Country Code')
                        POP_filled = read_csv_safe(args.popfile).set_index('Country Code')
                        STATISTICS_DATA = GDP_filled.merge(POP_filled, on='Country Code')
                        print(ConsiderTC, ConsiderMB, informal, discount, scenario)
                        est_df = Process(df_result, df_IHME, diseases, STATISTICS_DATA, ConsiderTC, ConsiderMB, informal, discount, scenario)
                        est_pieces.append(est_df)
    if len(est_pieces) > 0:
        df = pd.concat(est_pieces, ignore_index=True)
    else:
        df = pd.DataFrame(columns=['Country Code', 'pc_loss', 'GDPloss', 'tax', 'disease',
                                   'scenario', 'ConsiderTC', 'ConsiderMB', 'informal', 'discount'])
    est_output_path = append_file_tag('tmpresults/est.csv', output_tag)
    df.to_csv(est_output_path, index=False)


    # In[ ]:
    df_imputed = pd.concat([df_result, df])
    df_imputed = df_imputed.drop_duplicates(subset=['ConsiderTC','ConsiderMB','informal','discount','scenario','disease','Country Code'], keep='last')
    df_imputed.sort_values(['ConsiderTC', 'ConsiderMB','informal','discount','scenario','disease','Country Code'], inplace=True)
    df_imputed.to_csv(args.output, index=False)

    # In[ ]:
    print('primary data:', len(df_result))
    print('imputed data:', len(df_imputed))

    # In[ ]:
