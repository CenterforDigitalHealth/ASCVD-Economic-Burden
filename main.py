#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import math
import sys
import os
import argparse
import pdb

# Correct age group ordering (numeric, not alphabetic)
CORRECT_AGE_ORDER = ['d0', 'd5', 'd10', 'd15', 'd20', 'd25', 'd30', 'd35', 'd40', 'd45', 'd50', 'd55', 'd60', 'd65']
CORRECT_INDEX = pd.MultiIndex.from_product([['F', 'M'], CORRECT_AGE_ORDER], names=['sex', 'age'])

# Working age order (d15-d65 only, for labor and education data)
WORKING_AGE_ORDER = ['d15', 'd20', 'd25', 'd30', 'd35', 'd40', 'd45', 'd50', 'd55', 'd60', 'd65']
WORKING_AGE_INDEX = pd.MultiIndex.from_product([['F', 'M'], WORKING_AGE_ORDER], names=['sex', 'age'])

# ASCVD data root and file naming conventions.
ASCVD_BASE_DIR = 'data/ASCVD'
ASCVD_METRIC_FILE_PREFIX = {
    'mortality': 'Deaths',
    'prevalence': 'Prevalence',
    'morbidity': 'morbidity',
}
RATE_PER_100K_METRICS = {'mortality', 'prevalence'}
REQUIRED_COUNTRY_DATASETS = [
    'data/physical_ppp.csv',
    'data/savings.csv',
    'data/GDP_ppp.csv',
    'data/population_un.csv',
    'data/laborparticipation_final.csv',
    'data/education_filled.csv',
]
ASCVD_METRIC_CACHE = {}
HE_GROWTH_CACHE = None
TC_TABLE_CACHE = None
SUPPORTED_COUNTRIES_CACHE = None


def read_csv_safe(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs['engine'] = 'python'
        return pd.read_csv(path, **fallback_kwargs)


def get_country_codes(path):
    df = read_csv_safe(path)
    if 'Country Code' not in df.columns:
        return set()
    return set(df['Country Code'].dropna().astype(str).unique())


def get_supported_countries():
    global SUPPORTED_COUNTRIES_CACHE
    if SUPPORTED_COUNTRIES_CACHE is not None:
        return SUPPORTED_COUNTRIES_CACHE

    supported = None
    for path in REQUIRED_COUNTRY_DATASETS:
        country_codes = get_country_codes(path)
        supported = country_codes if supported is None else supported & country_codes

    SUPPORTED_COUNTRIES_CACHE = supported if supported is not None else set()
    return SUPPORTED_COUNTRIES_CACHE

# # Helper functions to retrieve data

# In[5]:


# get 2015's data as default
def get_params(country, year=2019):
    econ_params = pd.read_csv('data/alpha.csv').set_index('Country Code')
    if country in econ_params.index:
        alpha = econ_params.loc[country, 'alpha']
    else:
        alpha = 0.33
    
    delta = 1-0.05
    # get physical capital
    df = pd.read_csv('data/physical_ppp.csv').set_index('Country Code')
    
    # key is to get the initial capital stocks, most recent date is 2019
    InitialCapitalStock = df.loc[country, str(year)]*1000000
    df = pd.read_csv('data/savings.csv').set_index('Country Code')
    s = df.loc[country, '2050']/100
    return alpha, delta, InitialCapitalStock, s

def aggregate_age_groups(df, method='sum'):
    """
    Aggregates age groups >= d65 (e.g., d70, d75) into d65.
    If method is 'sum', values are summed (for Population).
    If method is 'keep_first', only d65 is kept (for Rates).
    Assume index includes 'age' level or column.
    """
    # Check if 'age' is in index
    is_multi_index = False
    if 'age' in df.index.names:
        is_multi_index = True
        df = df.reset_index()
    
    # Identify d65+ groups
    # We look for d65, d70, d75, ...
    # Simple logic: extract number from 'dXX', if >= 65, it's target
    
    # First, find valid age columns
    # Actually, the dataframe structure at this point (before pivoting/setting index in some cases) 
    # usually has 'age' column.
    
    if 'age' not in df.columns:
        # If no age column, return as is (should not happen based on usage)
        if is_multi_index:
             df = df.set_index(['sex', 'age'])
        return df

    # Filter rows with age >= d65
    # We assume format 'dXX'
    def get_age_num(x):
        try:
            return int(x.replace('d', ''))
        except:
            return -1

    df['age_num'] = df['age'].apply(get_age_num)
    
    # Split into under 65 and over 65
    df_under = df[df['age_num'] < 65].copy()
    df_over = df[df['age_num'] >= 65].copy()
    
    if df_over.empty:
        df = df.drop(columns=['age_num'])
        if is_multi_index:
             df = df.set_index(['sex', 'age'])
        return df

    # Aggregate df_over
    if method == 'sum':
        # Sum by sex (and Country Code if present)
        # We need to preserve non-numeric columns like Country Code, sex
        # Numeric columns (years) should be summed
        group_cols = ['sex']
        if 'Country Code' in df.columns:
            group_cols.append('Country Code')
        
        # d65 row for metadata
        # We want the result to be labelled 'd65'
        # We group by sex (and Country) and sum numeric columns
        df_agg = df_over.groupby(group_cols).sum(numeric_only=True).reset_index()
        df_agg['age'] = 'd65'
        
        # We might lose some metadata columns if they are not in group_cols
        # But getPop structure is simple: Country Code, sex, age, years...
        
    else: # keep_first (essentially just keep d65 rows)
        df_agg = df_over[df_over['age'] == 'd65'].copy()
        
    # Combine
    df_final = pd.concat([df_under, df_agg], ignore_index=True)
    df_final = df_final.drop(columns=['age_num'])
    
    if is_multi_index:
         df_final = df_final.set_index(['sex', 'age'])
         
    return df_final


def get_available_ascvd_groups(base_dir=ASCVD_BASE_DIR):
    if not os.path.isdir(base_dir):
        return []
    groups = []
    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if os.path.exists(os.path.join(full_path, 'Deaths_val.csv')):
            groups.append(entry)
    return groups


def get_disease_name_from_group(group, scen='val', base_dir=ASCVD_BASE_DIR):
    path = os.path.join(base_dir, group, f'Deaths_{scen}.csv')
    if not os.path.exists(path):
        return group

    df = read_csv_safe(path, encoding='utf-8-sig')
    for col in ['cause_name', 'disease', 'cause']:
        if col in df.columns and len(df) > 0:
            return df[col].iloc[0]
    return group


def load_ascvd_metric(metric, scen='val', disease_group=None, base_dir=ASCVD_BASE_DIR):
    metric = metric.lower()
    if metric not in ASCVD_METRIC_FILE_PREFIX:
        raise ValueError(f"Unsupported metric '{metric}'")

    cache_key = (metric, scen, disease_group, base_dir)
    if cache_key in ASCVD_METRIC_CACHE:
        return ASCVD_METRIC_CACHE[cache_key].copy()

    legacy_path = os.path.join(base_dir, f'{metric}_{scen}.csv')
    target_path = None

    if disease_group:
        prefix = ASCVD_METRIC_FILE_PREFIX[metric]
        target_path = os.path.join(base_dir, disease_group, f'{prefix}_{scen}.csv')
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Cannot find file for metric='{metric}', scenario='{scen}', group='{disease_group}': {target_path}"
            )
    elif os.path.exists(legacy_path):
        target_path = legacy_path
    else:
        raise FileNotFoundError(
            f"Cannot find legacy file for metric='{metric}', scenario='{scen}': {legacy_path}"
        )

    df = read_csv_safe(target_path, encoding='utf-8-sig')
    rename_map = {
        'ISO3': 'Country Code',
        'sex_name': 'sex',
        'age_name': 'age',
        'cause_name': 'disease',
        'measure_name': 'measure',
        'cause': 'disease',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if 'sex' in df.columns:
        df['sex'] = df['sex'].replace({
            'Female': 'F',
            'Male': 'M',
            'Both': 'B',
            'female': 'F',
            'male': 'M',
            'both': 'B',
        })

    year_cols = [col for col in df.columns if str(col).isdigit()]
    if len(year_cols) > 0:
        df[year_cols] = df[year_cols].apply(pd.to_numeric, errors='coerce')
        df[year_cols] = df[year_cols].replace([np.inf, -np.inf], np.nan)
        if metric in RATE_PER_100K_METRICS:
            df[year_cols] = (df[year_cols] / 100000.0).clip(lower=0, upper=1)
        else:
            df[year_cols] = df[year_cols].clip(lower=0)
        df[year_cols] = df[year_cols].fillna(0)

    ASCVD_METRIC_CACHE[cache_key] = df
    return df.copy()


# In[6]:


# get GDP
def getGDP(country, startyear, endyear):
    df = pd.read_csv('data/GDP_ppp.csv')
    df = df[df['Country Code'] == country]
    df = df.drop('Country Code',axis=1)
    years = [str(i) for i in range(startyear, endyear, 1)]
    df = df[years] 
    # df = 1.124115573 * df[years]  ## 112.4115573 constant2010->constant2017
    gdp = df.values.tolist()[0]
    return gdp


# In[7]:


# get population
def getPop(country, startyear, endyear):
    df = pd.read_csv('data/population_un.csv')
    df = df[df['Country Code']==country]
    # Drop exact duplicates first
    df = df.drop_duplicates(subset=['sex', 'age'])
    df = df.set_index(['sex','age'])
    years = [str(i) for i in range(startyear, endyear, 1)]
    df = df[years]
    # Reindex to ensure correct age ordering
    df = df.reindex(CORRECT_INDEX)
    return df


# In[8]:


# get labor
def getLaborRate(country, startyear, endyear):
    df = pd.read_csv('data/laborparticipation_final.csv')
    df = df[df['Country Code']==country]
    df = df.drop_duplicates(subset=['sex', 'age'])
    df = df.set_index(['sex','age'])
    years = [str(i) for i in range(startyear, endyear, 1)]
    df = df[years]
    # Reindex to ensure correct age ordering (working age only), fill with 0 for non-working ages
    df = df.reindex(CORRECT_INDEX).fillna(0)
    return df


# In[10]:


# get mortality
def getMortalityDiseaseRate(disease, country, startyear, projectStartYear, endyear, scen='val', disease_group=None):
    df = load_ascvd_metric('mortality', scen=scen, disease_group=disease_group)
    if 'disease' in df.columns:
        df = df[df['disease'] == disease]
    df = df[df['Country Code']==country]
    years = [str(i) for i in range(startyear, endyear, 1)]
    df = df.drop_duplicates(subset=['sex', 'age'])
    df = df.set_index(['sex', 'age'])
    df = df[years]
    # Reindex to ensure correct age ordering
    df = df.reindex(CORRECT_INDEX).fillna(0)
    # before the project start time, assume the mortality is zero
    df[[str(i) for i in range(startyear, projectStartYear, 1)]] = 0
    return df


# In[11]:


def getMorbidityDisease(disease, country, startyear, projectStartYear, endyear, scen='val', disease_group=None):
    df = load_ascvd_metric('morbidity', scen=scen, disease_group=disease_group)
    df = df[df['Country Code']==country]
    if 'disease' in df.columns:
        df = df[df['disease'] == disease]
    df = df.drop_duplicates(subset=['sex', 'age'])
    df = df.set_index(['sex', 'age'])
    years = [str(i) for i in range(startyear, endyear, 1)]
    df = df[years]
    df = df.fillna(0)
    # Reindex to ensure correct age ordering
    df = df.reindex(CORRECT_INDEX).fillna(0)
    # before the project start time, assume the morbidity is zero
    df[[str(i) for i in range(startyear, projectStartYear, 1)]] = 0
    return df


def get_prevalence(disease, country, startyear, projectStartYear, endyear, scen='val', disease_group=None):
    df = load_ascvd_metric('prevalence', scen=scen, disease_group=disease_group)
    if 'disease' in df.columns:
        df = df[df['disease'] == disease]
    df = df[df['Country Code']==country]
    df = df.drop_duplicates(subset=['sex', 'age'])
    df = df.set_index(['sex', 'age'])
    years = [str(i) for i in range(startyear, endyear, 1)]
    df = df[years]
    # Reindex to ensure correct age ordering
    df = df.reindex(CORRECT_INDEX).fillna(0)
    # before the project start time, assume the mortality is zero
    df[[str(i) for i in range(startyear, projectStartYear, 1)]] = 0
    return df

# In[12]:


def getHumanCapital(country, startyear, endyear):
    years = [str(i) for i in range(startyear, endyear, 1)]
    df1 = pd.read_csv('data/education_filled.csv')
    df1 = df1[df1['Country Code']==country]
    df1 = df1.drop_duplicates(subset=['sex', 'age'])
    df1 = df1.set_index(['sex','age'])
    df1 = df1[years]
    # Reindex to ensure correct age ordering (working age only), fill with 0 for non-working ages
    df1 = df1.reindex(CORRECT_INDEX).fillna(0)

    ys = df1 

    agedf = pd.read_csv('data/gd.csv')
    agedf = agedf.set_index(['sex','age'])
    # Reindex gd.csv to correct age ordering
    agedf = agedf.reindex(CORRECT_INDEX).fillna(0)

    ageaf = ys.copy()
    for key in df1.keys():
        ageaf[key] = agedf

    h = 0.091 * ys + 0.1301 * (ageaf - ys - 5) -0.0023 * (ageaf - ys - 5) * (ageaf - ys - 5)
    hh = np.exp(h)
    # hh.to_csv('temp.csv', index=True)

    # hh = pd.read_csv('temp.csv')
    # hh = hh.set_index(['sex', 'age'])
    return hh


# In[13]:


def age_convert(aa):
    if aa<5:
        a=0
    elif aa<10:
        a=1
    elif aa < 15:
        a = 2
    elif aa < 20:
        a =3
    elif aa < 25:
        a = 4
    elif aa < 30:
        a = 5
    elif aa<35:
        a=6
    elif aa < 40:
        a = 7
    elif aa<45:
        a=8
    elif aa < 50:
        a = 9
    elif aa < 55:
        a = 10
    elif aa<60:
        a=11
    elif aa<65:
        a=12
    else:
        a=13
    return a


# In[14]:


def getSigma2(a,t,sigma, Morbidity):
    """
    now configure to include time t
    """
    midages={0:2, 1:7, 2:12, 3:17,
    4:22,
    5:27,
    6:32,
    7:37,
    8:42,
    9:47,
    10:52,
    11:57,
    12:62,
    13:70}
    aa = midages[a]
    temp = 1
    if aa < t:
        for i in range(0, aa+1):
            temp = temp * (1-sigma[age_convert(aa-1-i)][t-1-i])*(1-Morbidity[age_convert(aa-1-i)][t-1-i]*sigma[age_convert(aa-1-i)][t-1-i])
            # temp = temp + sigma[age_convert(aa-1-i)][t-1-i]*(1+Morbidity[age_convert(aa-1-i)][t-1-i])
    else:
        for i in range(0, t+1):
            temp = temp * (1-sigma[age_convert(aa-1-i)][t-1-i])*(1-Morbidity[age_convert(aa-1-i)][t-1-i]*sigma[age_convert(aa-1-i)][t-1-i])
            # temp = temp + sigma[age_convert(aa-1-i)][t-1-i]*(1+Morbidity[age_convert(aa-1-i)][t-1-i])

    result = (1.0/temp)-1
    return result


# In[15]:


def project(disease, country, startyear, projectStartYear, endyear, ConsiderMB, Reduced, TC, scen='val',
            informal=0.0, discount=0.02, disease_group=None):
    # pdb.set_trace()
    alpha, delta,InitialCapitalStock,s = get_params(country)
    GDP_SQ = getGDP(country, startyear, endyear)
    population = getPop(country, startyear, endyear) 
    TotalPopulation = population.sum(axis = 0).values.tolist()
    LaborRate = getLaborRate(country, startyear, endyear)
    MortalityRateDisease = getMortalityDiseaseRate(disease, country, startyear, projectStartYear, endyear, scen,
                                                   disease_group=disease_group)
    HumanCapital = getHumanCapital(country, startyear, endyear)
    Morbidity = np.asarray(getMorbidityDisease(disease, country, startyear, projectStartYear, endyear, scen,
                                               disease_group=disease_group))
    prevalence = get_prevalence(disease, country, startyear, projectStartYear, endyear, scen,
                                disease_group=disease_group)
   ###############################################################################################
    ####### status quo scenario
    ###############################################################################################
    labor_SQ = population * LaborRate * HumanCapital
    FTE_SQ = labor_SQ.sum(axis = 0).values.tolist()# labor supply in status quo scenario per year
    
    # project informal care worker's FTE
    total_labor = (population * LaborRate).sum(axis = 0).values
    informal_care_labor = informal * (population * prevalence).sum(axis = 0).values
    with np.errstate(divide='ignore', invalid='ignore'):
        informal_care_labor_ratio = np.divide(
            informal_care_labor,
            total_labor,
            out=np.zeros_like(informal_care_labor, dtype=float),
            where=total_labor > 0
        )
    informal_care_labor_loss = informal_care_labor_ratio * labor_SQ.sum(axis = 0).values
#     print(informal_care_labor_ratio)
    ###############################################################################################
    #####capital accumulation
    ###############################################################################################
    K_SQ = []
    K_SQ.append(InitialCapitalStock)

    for i in range(1, endyear-startyear, 1):
        temp = GDP_SQ[i-1]*s+delta*K_SQ[i-1]
        K_SQ.append(temp)
    Y = np.multiply(np.power(K_SQ, alpha), np.power(FTE_SQ, 1-alpha))
    with np.errstate(divide='ignore', invalid='ignore'):
        Scalings = np.divide(
            GDP_SQ,
            Y,
            out=np.zeros(len(GDP_SQ), dtype=float),
            where=np.asarray(Y) > 0
        )
    Scalings = np.nan_to_num(Scalings, nan=0.0, posinf=0.0, neginf=0.0)
    ###############################################################################################
    ###############################################################################################
    sigma = np.asarray(MortalityRateDisease)
    sigma_f = sigma[0:14][:]
    sigma_m = sigma[14:28][:]
    N = np.asarray((population * LaborRate * HumanCapital).fillna(0))###number of labor a*t
    N_f = N[0:14][:]
    N_m = N[14:28][:]
    ###############################################################################################
    ###############################################################################################
    PercentageLoss = []
    dN_m = np.zeros([14,endyear-startyear])
    dN_f = np.zeros([14,endyear-startyear])
    for a in range(0,14):
        for t in range(0,endyear-startyear):
            dN_m[a][t] = N_m[a][t] * getSigma2(a,t,sigma_m* Reduced, Morbidity[14:,]*ConsiderMB)#considermorbidity is used
            dN_f[a][t] = N_f[a][t] * getSigma2(a,t,sigma_f* Reduced, Morbidity[:14,]*ConsiderMB)

    NN_m = N_m + dN_m
    NN_f = N_f + dN_f
    NN = np.append(NN_m,NN_f,axis=0)
    FTE_CF = np.sum(NN,axis=0) + informal_care_labor_loss

     ###################################
    K_CF = []
    K_CF.append(InitialCapitalStock)
    GDP_CF = []
    
     # discount rate
    DiscountRate = []
    rate = 1 / (1 - discount) ** (projectStartYear - startyear)
    DiscountRate.append(rate)
    
    
    GDP_CF.append(GDP_SQ[0])
    for i in range(1,endyear-startyear,1):
        temp = GDP_CF[i-1]*s + delta*K_CF[i-1]+ TC*TotalPopulation[i-1]*s*Reduced*get_he(country, startyear+i-1, projectStartYear)  #treatment
        K_CF.append(temp)
        capital_base = max(float(K_CF[i]), 0.0)
        labor_base = max(float(FTE_CF[i]), 0.0)
        temp2 = Scalings[i]*math.pow(capital_base, alpha) * math.pow(labor_base, (1-alpha))
        GDP_CF.append(temp2)
        
        rate = rate * (1 - discount)
        DiscountRate.append(rate)
        
    GDP_CF = np.multiply(GDP_CF, DiscountRate)
    GDP_SQ = np.multiply(GDP_SQ, DiscountRate)
    GDPloss = np.sum(np.subtract(GDP_CF,GDP_SQ))/1000000000
    
    # tax rate loss
    tax = np.sum(np.subtract(GDP_CF,GDP_SQ))/sum(GDP_SQ[projectStartYear-startyear:])
    # per capita loss
    # pdb.set_trace()
    pc_loss = np.sum(np.subtract(GDP_CF,GDP_SQ))/(population.sum(axis = 0)[projectStartYear-startyear:].mean())
    df = pd.DataFrame()
    df['GDP_loss_percapita'] = np.subtract(GDP_CF,GDP_SQ)/ (population.sum(axis = 0))
    df = df.reset_index()
    df = df.rename(columns={'index':'year'})
    df['GDP_loss'] = np.subtract(GDP_CF,GDP_SQ)
    df['GDP_loss_percentage'] = np.subtract(GDP_CF,GDP_SQ)/GDP_SQ
    df['EffectiveLabor_loss_percentage'] = np.subtract(FTE_CF,FTE_SQ)/FTE_SQ
    df = df.set_index('year')
    return df.iloc[projectStartYear-startyear:], GDPloss, tax, pc_loss


# In[16]:


# get health expenditure growth rate
def get_he(country, year, projectStartYear):
    global HE_GROWTH_CACHE
    if HE_GROWTH_CACHE is None:
        he = read_csv_safe('data/hepc_ppp.csv').set_index('Country Code')
        he = he.apply(pd.to_numeric, errors='coerce')
        if '2021' in he.columns:
            baseline = he['2021'].replace(0, np.nan)
            he = he.div(baseline, axis='index').T
        else:
            he = he.T
        HE_GROWTH_CACHE = he.replace([np.inf, -np.inf], np.nan).fillna(0)

    if year < projectStartYear:
        return 0

    year_col = str(year)
    if country not in HE_GROWTH_CACHE.columns:
        return 0
    if year_col not in HE_GROWTH_CACHE.index:
        return 0
    return float(HE_GROWTH_CACHE.at[year_col, country])


# In[17]:


# get TC
def get_TC(country, disease, disease_group):
    global TC_TABLE_CACHE
    # Toggle between IDF (TC_ppp.csv) and Dieleman (TC_dieleman.csv)
    # USE_DIELEMAN_DATA = True 
    filename = f'data/TC_{disease_group}.csv' # Switch to GBD data
    # filename = 'data/TC_ppp.csv' # Original IDF data

    if TC_TABLE_CACHE is None:
        TC_TABLE_CACHE = read_csv_safe(filename).set_index('ISO3')
    df_tc = TC_TABLE_CACHE
    if country not in df_tc.index:
        return 0
    if disease not in df_tc.columns:
        return 0

    TC = df_tc.loc[country, disease]
    return TC


def build_disease_runs(requested_disease='all', scen='val'):
    """
    Return list of (disease_name, disease_group).
    disease_group is folder name under data/ASCVD (e.g., IHD/IS/PAD) or None for legacy format.
    """
    requested = (requested_disease or 'all').strip()
    if ',' in requested and requested.lower() != 'all':
        merged = []
        for token in requested.split(','):
            token = token.strip()
            if token == '':
                continue
            merged.extend(build_disease_runs(token, scen=scen))
        dedup = []
        seen = set()
        for disease_name, disease_group in merged:
            key = (str(disease_name), disease_group)
            if key in seen:
                continue
            seen.add(key)
            dedup.append((disease_name, disease_group))
        return dedup

    groups = get_available_ascvd_groups()
    if len(groups) == 0:
        return [(requested, None)]

    group_lookup = {g.upper(): g for g in groups}

    if requested.lower() == 'all':
        return [(get_disease_name_from_group(g, scen=scen), g) for g in groups]

    if requested.upper() in group_lookup:
        g = group_lookup[requested.upper()]
        return [(get_disease_name_from_group(g, scen=scen), g)]

    for g in groups:
        disease_name = get_disease_name_from_group(g, scen=scen)
        if str(disease_name).lower() == requested.lower():
            return [(disease_name, g)]

    # Fallback for legacy datasets where disease is a column in one combined CSV.
    return [(requested, None)]


def get_available_countries(scen='val', disease_group=None):
    df = load_ascvd_metric('mortality', scen=scen, disease_group=disease_group)
    return sorted(df['Country Code'].dropna().unique())


# In[18]:


# notice that here InitialCapitalStock need to be updated, convert from 2011 international to 2010
# download GDP data from penn world data, then compare to that of world bank

"""
projection parameters
Note that mortality and morbidity file should only have data for years
"""

if __name__ == "__main__":
    startyear = 2019 # because of most recent data for physical capital
    projectStartYear = 2020
    endyear = 2051
    Reduced = 1

    if not os.path.exists('tmpresults/'):
        os.makedirs('tmpresults')

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-t', '--tc', type=int, default=1) # or 0
    parser.add_argument('-m', '--mb', type=int, default=1) # or 0
    parser.add_argument('-s', '--scenario', type=str, default='val') # or 'lower', 'upper'
    parser.add_argument('-d', '--discount', type=float, default=0) # or 0.02, 0.03
    parser.add_argument('-i', '--informal', type=float, default=0) # range 0-1
    parser.add_argument('--disease', type=str, default='all',
                        help='Disease selector: IHD, IS, PAD, exact disease name, all, or comma-separated list')
    parser.add_argument('--list-diseases', action='store_true',
                        help='List selectable diseases and exit')
    parser.add_argument('-r', '--ran', type=bool, default=False)
    args = parser.parse_args()
    # In[19]:

    scenario = args.scenario
    ConsiderTC = args.tc
    ConsiderMB = args.mb
    discount = args.discount
    informal = args.informal

    if args.list_diseases:
        print("Available diseases")
        for disease_name, disease_group in build_disease_runs('all', scen=scenario):
            print(f"- {disease_group}: {disease_name}")
        sys.exit(0)

    disease_runs = build_disease_runs(args.disease, scen=scenario)
    disease_runs = sorted(disease_runs, key=lambda x: x[0])
    supported_countries = get_supported_countries()

    if args.ran:
        print("Run quick test mode")
        disease_runs = disease_runs[:1]

    countries_by_group = {}
    for _, disease_group in disease_runs:
        if disease_group in countries_by_group:
            continue
        try:
            countries = get_available_countries(scen=scenario, disease_group=disease_group)
        except Exception:
            countries = []
        countries = [country for country in countries if country in supported_countries]
        if args.ran:
            if 'USA' in countries:
                countries = ['USA']
            elif len(countries) > 0:
                countries = [countries[0]]
        countries_by_group[disease_group] = countries

    total_country_count = sum(len(countries_by_group[group]) for _, group in disease_runs)
    if total_country_count == 0:
        print("No matching countries found for the selected disease/scenario.")
        print("Try --list-diseases to check available disease options.")
        sys.exit(1)

    pieces_df = []
    pieces_result = []
    print("Country runs", total_country_count, "Diseases", len(disease_runs))
    for disease, disease_group in disease_runs:
        print(f"{disease} ({disease_group}) ----------------------")
        countries = countries_by_group.get(disease_group, [])
        for country in countries:
            print(country)
            try:
                TC = get_TC(country, disease, disease_group)
                if not ConsiderTC:
                    TC = 0
                df, GDPloss, tax, pc_loss = project(disease, country, startyear, projectStartYear, endyear, 
                                                    ConsiderMB, Reduced, TC, scen=scenario, informal=informal,
                                                    discount=discount, disease_group=disease_group)
                # print(GDPloss)
                result = pd.DataFrame()
                result['disease'] = [disease]
                result['Country Code'] = country
                result['scenario'] = scenario
                result['ConsiderTC'] = ConsiderTC
                result['ConsiderMB'] = ConsiderMB
                result['informal'] = informal
                result['discount'] = discount
                result['GDPloss'] = GDPloss
                result['tax'] = tax
                result['pc_loss'] = pc_loss
                
                df['disease'] = disease
                df['Country Code'] = country
                df['scenario'] = scenario
                df['ConsiderTC'] = ConsiderTC
                df['ConsiderMB'] = ConsiderMB
                df['informal'] = informal
                df['discount'] = discount

                pieces_df.append(df)
                pieces_result.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print('failed %s [%s]: scenario:%s, TC:%s, MB%s'
                      %(country, disease, scenario, ConsiderTC, ConsiderMB))
    save_annfilename = 'tmpresults/annual_results_TC%s_MB%s_informal%s_discount%s_%s.csv'%(ConsiderTC,ConsiderMB,informal,discount,scenario)
    save_aggfilename = 'tmpresults/aggregate_results_TC%s_MB%s_informal%s_discount%s_%s.csv'%(ConsiderTC,ConsiderMB,informal,discount,scenario)
    if args.ran:
        save_annfilename = 'tmpresults/runexampleann.csv'
        save_aggfilename = 'tmpresults/runexampleagg.csv'

    if len(pieces_df) > 0:
        df = pd.concat(pieces_df).reset_index()
        df.to_csv(save_annfilename, index=False)
        print(f"Annual results saved to {save_annfilename}")
    else:
        print("Warning: No annual results to save")

    if len(pieces_result) > 0:
        df = pd.concat(pieces_result)
        df.to_csv(save_aggfilename, index=False)
        print(f"Aggregate results saved to {save_aggfilename}")
    else:
        print("Warning: No aggregate results to save")
