#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import math
import sys
import os
import argparse

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
    
    # key is to get the initial capital stocks, most recent date is 2017
    InitialCapitalStock = df.loc[country, str(year)]*1000000
    df = pd.read_csv('data/savings.csv').set_index('Country Code')
    s = df.loc[country, '2050']/100
    return alpha, delta, InitialCapitalStock, s


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
    df = pd.read_csv('data/population_final.csv')
    df = df[df['Country Code']==country]
    df = df.set_index(['sex','age'])
    years = [str(i) for i in range(startyear, endyear, 1)]
    df = df[years]
    return df


# In[8]:


# get labor
def getLaborRate(country, startyear, endyear):
    df = pd.read_csv('data/laborparticipation_final.csv')
    df = df[df['Country Code']==country]
    df = df.set_index(['sex','age'])
    years = [str(i) for i in range(startyear, endyear, 1)]
    df = df[years]
    return df


# In[10]:


# get mortality
def getMortalityDiseaseRate(disease, country, startyear, projectStartYear, endyear, scen='val'):
    df = pd.read_csv('bigdata/mortality_%s.csv'%scen)
    df = df[df['disease'] == disease]
    df = df[df['Country Code']==country]
    years = [str(i) for i in range(startyear, endyear, 1)]
# This leads to scaling factors of:
# 1. for all 0.136/0.06298
# 2. for men 0.19/0.057257
# 3. for women 0.081/0.06873
    # if country == 'CHN':
    #     # scale China accordingly
    #     df.loc[df['sex']=='F',years] *= 0.081/0.06873
    #     df.loc[df['sex']=='M',years] *= 0.19/0.057257
    df = df.set_index(['sex', 'age'])
    df = df[years]
    # before the project start time, assume the mortality is zero
    df[[str(i) for i in range(startyear, projectStartYear, 1)]] = 0
    return df


# In[11]:


def getMorbidityDisease(disease, country, startyear, projectStartYear, endyear, scen='val'):
    df = pd.read_csv('bigdata/morbidity_%s.csv'%scen)
    df = df[df['Country Code']==country]
    df = df[df['disease'] == disease]
    df = df.set_index(['sex', 'age'])
    years = [str(i) for i in range(startyear, endyear, 1)]
    df = df[years]
    df = df.fillna(0)
    # before the project start time, assume the morbidity is zero
    df[[str(i) for i in range(startyear, projectStartYear, 1)]] = 0
    return df


# In[12]:


def getHumanCapital(country, startyear, endyear):
    years = [str(i) for i in range(startyear, endyear, 1)]
    df1 = pd.read_csv('data/education_filled.csv').set_index(['sex','age'])
    df1 = df1[df1['Country Code']==country]
    df1 = df1[years]

    ys = df1 

    agedf = pd.read_csv('data/gd.csv')
    agedf = agedf.set_index(['sex','age'])

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


def project(disease, country, startyear, projectStartYear, endyear, ConsiderMB, Reduced, TC, scen='', discount = 0.0):
    alpha, delta,InitialCapitalStock,s = get_params(country)
    GDP_SQ = getGDP(country, startyear, endyear)
    population = getPop(country, startyear, endyear) 
    TotalPopulation = population.sum(axis = 0).values.tolist()
    LaborRate = getLaborRate(country, startyear, endyear)
    MortalityRateDisease = getMortalityDiseaseRate(disease, country, startyear, projectStartYear, endyear, scen)
    HumanCapital = getHumanCapital(country, startyear, endyear)
    Morbidity = np.asarray(getMorbidityDisease(disease, country, startyear, projectStartYear, endyear, scen))


    ###############################################################################################
    ####### status quo scenario
    ###############################################################################################
    labor_SQ = population * LaborRate * HumanCapital
    FTE_SQ = labor_SQ.sum(axis = 0).values.tolist()# labor supply in status quo scenario
    ###############################################################################################
    #####capital accumulation
    ###############################################################################################
    K_SQ = []
    K_SQ.append(InitialCapitalStock)

    for i in range(1,endyear-startyear,1):
        temp = GDP_SQ[i-1]*s+delta*K_SQ[i-1]
        K_SQ.append(temp)
    Y = np.multiply(np.power(K_SQ, alpha), np.power(FTE_SQ, 1-alpha))
    Scalings = np.divide(GDP_SQ, Y)
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
    FTE_CF = np.sum(NN,axis=0)

     ###################################
    K_CF = []
    K_CF.append(InitialCapitalStock)
    GDP_CF = []
    
    # discount rate
    DiscountRate = []
    rate = 1 / (1 - discount) ** (projectStartYear - startyear)
    DiscountRate.append(rate)
    
    
    GDP_CF.append(GDP_SQ[0])
    for i in range(1, endyear-startyear,1):
        temp = GDP_CF[i-1]*s + delta*K_CF[i-1]+ TC*TotalPopulation[i-1]*s*Reduced*get_he(country, startyear+i-1, projectStartYear)  #treatment
        K_CF.append(temp)
        temp2 = Scalings[i]*math.pow(K_CF[i],alpha) * math.pow(FTE_CF[i],(1-alpha))
        GDP_CF.append(temp2)
        
        rate = rate * (1 - discount)
        DiscountRate.append(rate)
     
    GDP_CF = np.multiply(GDP_CF, DiscountRate)
    GDP_SQ = np.multiply(GDP_SQ, DiscountRate)
    
    GDPloss = np.sum(np.subtract(GDP_CF,GDP_SQ))/1000000000

    # tax rate loss
    tax = np.sum(np.subtract(GDP_CF,GDP_SQ))/sum(GDP_SQ[projectStartYear-startyear:])
    
    # per capita loss
    pc_loss = np.sum(np.subtract(GDP_CF,GDP_SQ))/population.sum(axis = 0).mean()
    df = pd.DataFrame()
    df['GDP_loss_percapita'] = np.subtract(GDP_CF,GDP_SQ)/ population.sum(axis = 0)
    df = df.reset_index()
    df = df.rename(columns={'index':'year'})
    df['GDP_loss'] = np.subtract(GDP_CF,GDP_SQ)
    df['GDP_loss_percentage'] = np.subtract(GDP_CF,GDP_SQ)/GDP_SQ
    df['EffectiveLabor_loss_percentage'] = np.subtract(FTE_CF,FTE_SQ)/FTE_SQ
    df = df.set_index('year')
    return df.iloc[projectStartYear-startyear:], GDPloss, tax, pc_loss


# # Now run the results

# In[113]:


"""
This block generate treatment cost data used in the projection, refer to the write-up for 
detailed assumptions
"""

# we only know the costs from USA from deleman in 2013, numbers were reported in 2015 USA and then convert to 2010 USD (0.92 inflation)
# TC_USA = 8191.208111 * 53.8 / 2100.1 # 53.8 billion cost, out of 2100 billion total cost


# he = pd.read_csv('data/hepc_ppp.csv')
# he = he[['2013','Country Code']]
# he = he.rename(columns = {'2013':'he'})
# he = he.set_index('Country Code')
# TC_USA = he.loc['USA', 'he'] * 53.8 / 2100.1 # 53.8 billion cost, out of 2100 billion total cost

# # prevanlence in 2013
# prev = pd.read_csv('data/Prevalence.csv')
# prev = prev[['Country Code', 'prevalence']]

# data = he.merge(prev, on='Country Code')
# data = data.set_index('Country Code')
# const = TC_USA / data.loc['USA','prevalence'] / data.loc['USA','he']
# data[disease] = data['he']*data['prevalence']*const

# # treat China differently
# data.loc['CHN','Chronic obstructive pulmonary disease'] *= 0.136/0.06298
# data.to_csv('data/TC_ppp.csv')


# In[16]:


# get health expenditure growth rate
def get_he(country, year, projectStartYear):
    he = pd.read_csv('data/hepc_ppp.csv').set_index('Country Code')
    he = he.div(he['2013'], axis='index').T
    if year>=projectStartYear:
        return he[country].loc[str(year)]
    else:
        return 0


# In[17]:


# get TC
def get_TC(country, disease):
    df_tc = pd.read_csv('data/TC_ppp.csv').set_index('Country Code')
    TC = df_tc.loc[country, disease]
    return TC


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
    disease = "Liver cancer"
    country = "USA"

    if not os.path.exists('tmpresults/'):
        os.makedirs('tmpresults')

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-t', '--tc', type=int, default=1) # or 0
    parser.add_argument('-m', '--mb', type=int, default=1) # or 0
    parser.add_argument('-s', '--scenario', type=str, default='val') # or 'lower', 'upper'
    parser.add_argument('-d', '--discount', type=float, default=0) # or 0.02, 0.03
    parser.add_argument('-r', '--ran', type=bool, default=False)
    args = parser.parse_args()
    # In[19]:

    scenario = args.scenario
    ConsiderTC = args.tc
    ConsiderMB = args.mb
    discount = args.discount

    
    # countries = pd.read_csv('country_touse_ppp.csv')['Country Code'].unique()
    diseases = np.array(["Lip and oral cavity cancer", "Nasopharynx cancer", "Other pharynx cancer", "Esophageal cancer", "Stomach cancer", 
    "Colon and rectum cancer", "Liver cancer", "Gallbladder and biliary tract cancer", "Pancreatic cancer", "Larynx cancer", 
    "Tracheal, bronchus, and lung cancer", "Malignant skin melanoma", "Non-melanoma skin cancer", "Breast cancer", "Cervical cancer", 
    "Uterine cancer", "Ovarian cancer", "Prostate cancer", "Testicular cancer", "Kidney cancer", 
    "Bladder cancer", "Brain and central nervous system cancer", "Thyroid cancer", "Mesothelioma", "Hodgkin lymphoma", 
    "Non-Hodgkin lymphoma", "Multiple myeloma", "Leukemia", "Other neoplasms"]) # "Other malignant neoplasms"
    diseases = sorted(diseases)

    countries = pd.read_csv('bigdata/mortality_val.csv')['Country Code'].unique()
    # diseases = pd.read_csv('bigdata/mortality_val.csv')['disease'].unique()
    countries = sorted(countries)
    if args.ran: # Run on only five countries and two diseases for test
        print("random choose diseases and countries for test")
        diseases = np.random.choice(diseases, 2)
        countries = np.random.choice(countries, 5)
    diseases = sorted(diseases)
    pieces_df = []
    pieces_result = []
    print("Countries", len(countries), "Diseases", len(diseases))
    for disease in diseases:
        print(disease, '----------------------')
        for country in countries:
            print(country)
            try:
                TC = get_TC(country, disease)
                if not ConsiderTC:
                    TC = 0
                df, GDPloss, tax, pc_loss = project(disease, country, startyear, projectStartYear, endyear, 
                                                    ConsiderMB, Reduced, TC, scen=scenario,discount = discount)
                # print(GDPloss)
                result = pd.DataFrame()
                result['disease'] = [disease]
                result['Country Code'] = country
                result['scenario'] = scenario
                result['ConsiderTC'] = ConsiderTC
                result['ConsiderMB'] = ConsiderMB
                result['GDPloss'] = GDPloss
                result['tax'] = tax
                result['pc_loss'] = pc_loss
                result['discount'] = discount
                
                df['disease'] = disease
                df['Country Code'] = country
                df['scenario'] = scenario
                df['ConsiderTC'] = ConsiderTC
                df['ConsiderMB'] = ConsiderMB
                df['discount'] = discount

                pieces_df.append(df)
                pieces_result.append(result)
            except:
                print("failed %s: scenario:%s, TC:%s, MB%s"%(country, scenario, ConsiderTC, ConsiderMB))

    df = pd.concat(pieces_df).reset_index()
    df.to_csv('tmpresults/annual_results_TC%s_MB%s_%s_discount%s.csv'%(ConsiderTC,ConsiderMB,scenario,discount), index=False)

    df = pd.concat(pieces_result)
    df.to_csv('tmpresults/aggregate_results_TC%s_MB%s_%s_discount%s.csv'%(ConsiderTC,ConsiderMB,scenario,discount), index=False)



