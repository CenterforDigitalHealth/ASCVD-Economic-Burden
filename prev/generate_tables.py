import pandas as pd
import numpy as np
import math
import sys
import os
import argparse
import pdb


class Tables():
    def __init__(self, discount=0.02, informal=0.11, filename='results/aggregate_results_imputed.csv'):
        self.df_input = pd.read_csv(filename)
        self.countries = self.df_input['Country Code'].unique()
        if discount == 0.0 :
            discount = 0
        self.default_discount = discount
        self.default_informal = informal
        self.set_state()
        self.set_params()
        self.df_state = self.get_data()
        

    def set_params(self):
        countries_info = pd.read_csv('data/dl1_countrycodeorg_country_name.csv', encoding='ISO-8859-1')
        self.countries_info = countries_info[['Country Code', 'Region', 'Income group', 'WBCountry', 'country']]
        # self.codemap = countries_info.dropna()
        self.endyear = 2051
        self.projectStartYear = 2020
        gdp_total_df = pd.read_csv('tmpresults/GDP_TOTAL_discount%s.csv'%(self.default_discount)).set_index('Country Code')
        pop_total_df = pd.read_csv('tmpresults/POP_TOTAL.csv').set_index('Country Code')
        gdp_psy_df = pd.read_csv('tmpresults/GDP_PSY.csv').set_index('Country Code')
        pop_psy_df = pd.read_csv('tmpresults/POP_PSY.csv').set_index('Country Code')
        self.INFODATA = gdp_total_df.merge(pop_total_df, on='Country Code').merge(gdp_psy_df, on='Country Code').merge(pop_psy_df, on='Country Code')
        self.INFODATA = self.INFODATA.merge(countries_info, on='Country Code')
        self.INFODATA = self.INFODATA[self.INFODATA['Country Code'].isin(self.countries)]
        print(len(self.INFODATA))
               
    def set_state(self, state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'val'}):
        state['discount'] = self.default_discount
        state['informal'] = self.default_informal
        self.state = state

    def get_data(self):
        df = self.df_input[(self.df_input['discount']==self.state['discount'])&
                       (self.df_input['ConsiderTC']==self.state['ConsiderTC'])&
                       (self.df_input['ConsiderMB']==self.state['ConsiderMB'])&
                       (self.df_input['informal']==self.state['informal'])&
                       (self.df_input['scenario']==self.state['scenario'])]
        self.imputed = df.merge(self.INFODATA, on='Country Code')
        return self.imputed

    def get_group_data(self, identify=['Country Code', 'disease']): #each Country, each disease
        imputed = self.imputed
        group = pd.DataFrame()
        assert 'disease' in identify
        group['GDPloss'] = imputed.groupby(identify).sum()['GDPloss']
        group['GDPlossRatio'] = imputed.groupby(identify).sum()['GDPloss']/imputed['GDPloss'].sum()
        group['tax']=  imputed.groupby(identify).sum()['GDPloss'] * 1000000000 / imputed.groupby(identify)['totalGDP'].sum()#
        group['pc_loss']=  imputed.groupby(identify).sum()['GDPloss'] * 1000000000 /(imputed.groupby(identify)['totalPOP'].sum() / (self.endyear - self.projectStartYear) ) #

        group = group.reset_index()
        group.sort_values([identify[0],'GDPlossRatio'], ascending = [True,False], inplace=True)

        # group['tax'] = group['tax']*1000 # rate - > 1‰
        group['tax'] = group['tax']*100 # rate - > 1%
        return group

    def generate_table1(self):
        identify =['Country Code', 'disease']
        self.set_state(state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'lower'})
        self.get_data()
        group1 = self.get_group_data(identify)[['Country Code', 'disease', 'GDPloss', 'tax', 'pc_loss']]

        self.set_state(state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'upper'})
        self.get_data()
        group2 = self.get_group_data(identify)[['Country Code', 'disease', 'GDPloss', 'tax', 'pc_loss']]

        self.set_state(state={'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'val'})
        self.get_data()
        group3 = self.get_group_data(identify)[['Country Code', 'disease', 'GDPloss', 'tax', 'pc_loss']]

        lower = group1.groupby('Country Code').sum()
        upper = group2.groupby('Country Code').sum()
        base = group3.groupby('Country Code').sum()
        
        df = base.merge(upper,on='Country Code',suffixes=('', '_upper'))
        df= df.merge(lower,on='Country Code',suffixes=('', '_lower'))
        df = df.merge(self.countries_info, on='Country Code')
        # pdb.set_trace()
        data = df.copy()
        data['totalloss'] = data.apply(lambda row: str(round(1000*row['GDPloss']))+'('+ str(round(1000*row['GDPloss_lower']))+'-'+str(round(1000*row['GDPloss_upper']))+')', axis=1)
        # data['tax ‰'] = data.apply(lambda row: str(round(row['tax'],3))+'('+ str(round(row['tax_lower'],3))+'-'+str(round(row['tax_upper'],3))+')', axis=1)
        data['tax %'] = data.apply(lambda row: str(round(row['tax'],2))+'('+ str(round(row['tax_lower'],2))+'-'+str(round(row['tax_upper'],2))+')', axis=1)
        data['pc_loss'] = data.apply(lambda row: str(round(row['pc_loss']))+'('+ str(round(row['pc_loss_lower']))+'-'+str(round(row['pc_loss_upper']))+')', axis=1)
        data = data.sort_values(['Region','country'])[['Region','country', 'WBCountry', 'totalloss','pc_loss','tax %']]
        # df.reset_index().to_csv('tables/tmp_Table1_informal%s_discount%s.csv'%(self.state['informal'], self.state['discount']), index=False, float_format='%.3f')
        data.to_csv('tables/Table1_informal%s_discount%s.csv'%(self.state['informal'], self.state['discount']), index=False)

    def generate_table2(self):
        identify =['Region', 'disease']
        group1 = self.get_group_data(identify)
        data1 = group1.groupby('Region').head(5).reset_index()
        data1['location'] = data1['Region']
        data1.sort_values(['Region','GDPlossRatio'], ascending = [True,False], inplace=True)


        identify =['Income group', 'disease']
        group2 = self.get_group_data(identify)
        data2 = group2.groupby('Income group').head(5).reset_index()
        data2['location'] = data2['Income group']
        data2.sort_values(['Income group','GDPlossRatio'], ascending = [True,False], inplace=True)

        identify =['disease']
        group3 = self.get_group_data(identify)
        data4 = group3.head(5).reset_index()
        data4['location'] = 'global'

        data = pd.concat([data1, data2, data4])   
        data['burden'] = data.apply(lambda row: str(round(row['GDPloss']))+' ('+   "{0:.1%}".format(row['GDPlossRatio'])   +')', axis=1)
        data['tax %'] = data.apply(lambda row: str(round(row['tax'],2)), axis=1)
        data['pc_loss'] = data.apply(lambda row: str(round(row['pc_loss'])), axis=1) 
        data[['location','disease','burden','tax %', 'pc_loss']].to_csv('tables/Table2_informal%s_discount%s.csv'%(self.state['informal'], self.state['discount']), index=False) 

    def generate_table3(self):
        data1 = self.INFODATA.groupby('Region').sum().reset_index()
        data1['location'] = data1['Region']
        data1['gdp_psy_Ratio'] = data1['gdp_psy']/data1['gdp_psy'].sum()   
        data1['pop_psy_Ratio'] = data1['pop_psy']/data1['pop_psy'].sum()  
        data1['gdp_psy'] = data1['gdp_psy'] / 1000000000
        data1['pop_psy'] = data1['pop_psy'] / 1000000
        data1['totalGDP_Ratio'] = data1['totalGDP']/data1['totalGDP'].sum()   
        data1['totalPOP_Ratio'] = data1['totalPOP']/data1['totalPOP'].sum()  
        data1['averageGDP'] = data1['totalGDP'] / 1000000000 / (self.endyear - self.projectStartYear)
        data1['averagePOP'] = data1['totalPOP'] / 1000000 / (self.endyear - self.projectStartYear)
        data2 = self.INFODATA.groupby('Income group').sum().reset_index()
        data2['location'] = data2['Income group']
        data2['gdp_psy_Ratio'] = data2['gdp_psy']/data2['gdp_psy'].sum()   
        data2['pop_psy_Ratio'] = data2['pop_psy']/data2['pop_psy'].sum()   
        data2['gdp_psy'] = data2['gdp_psy'] / 1000000000
        data2['pop_psy'] = data2['pop_psy'] / 1000000
        data2['totalGDP_Ratio'] = data2['totalGDP']/data2['totalGDP'].sum()   
        data2['totalPOP_Ratio'] = data2['totalPOP']/data2['totalPOP'].sum()  
        data2['averageGDP'] = data2['totalGDP'] / 1000000000 / (self.endyear - self.projectStartYear)
        data2['averagePOP'] = data2['totalPOP'] / 1000000 / (self.endyear - self.projectStartYear)
        # print(self.INFODATA.columns)
        # pdb.set_trace()
        data3 = self.INFODATA.sum(numeric_only=True).to_frame().T
        # print(data3.columns)
        data3['location'] = 'global'
        data3['gdp_psy_Ratio'] = data3['gdp_psy']/data3['gdp_psy'].sum()   
        data3['pop_psy_Ratio'] = data3['pop_psy']/data3['pop_psy'].sum()   
        data3['gdp_psy'] = data3['gdp_psy'] / 1000000000
        data3['pop_psy'] = data3['pop_psy'] / 1000000
        data3['totalGDP_Ratio'] = data3['totalGDP']/data3['totalGDP'].sum()   
        data3['totalPOP_Ratio'] = data3['totalPOP']/data3['totalPOP'].sum()  
        data3['averageGDP'] = data3['totalGDP'] / 1000000000 / (self.endyear - self.projectStartYear)
        data3['averagePOP'] = data3['totalPOP'] / 1000000 / (self.endyear - self.projectStartYear)        

        yll = pd.read_csv('./bigdata/data_diabetes/YLL_val.csv').set_index(['Country Code', 'sex', 'age']).drop(columns=['disease'])
        yld = pd.read_csv('./bigdata/data_diabetes/YLD_val.csv').set_index(['Country Code', 'sex', 'age']).drop(columns=['disease'])
        pop = pd.read_csv('./data/population_un.csv').set_index(['Country Code', 'sex', 'age'])
        DALYs = (yll + yld) * pop 
        DALY = pd.concat([DALYs['2020'].dropna(), DALYs['2050'].dropna()], axis=1)
        DALY = DALY.groupby('Country Code').sum().reset_index()
        DALY = DALY.merge(self.INFODATA[['Country Code', 'Region', 'Income group']], on='Country Code')
        data4 = DALY.groupby('Region').sum().reset_index()
        data4['location'] = data4['Region']
        data4['daly2020'] = data4['2020'] / 1000000
        data4['daly2020_Ratio'] = data4['daly2020'] / data4['daly2020'].sum()
        data4['daly2050'] = data4['2050'] / 1000000
        data4['daly2050_Ratio'] = data4['daly2050'] / data4['daly2050'].sum()
        data5 = DALY.groupby('Income group').sum().reset_index()
        data5['location'] = data5['Income group']
        data5['daly2020'] = data5['2020'] / 1000000
        data5['daly2020_Ratio'] = data5['daly2020'] / data5['daly2020'].sum()
        data5['daly2050'] = data5['2050'] / 1000000
        data5['daly2050_Ratio'] = data5['daly2050'] / data5['daly2050'].sum()
        # pdb.set_trace()
        data6 = DALY.sum(numeric_only=True).to_frame().T
        data6['location'] = 'global'
        data6['daly2020'] = data6['2020'] / 1000000
        data6['daly2020_Ratio'] = data6['daly2020'] / data6['daly2020'].sum()
        data6['daly2050'] = data6['2050'] / 1000000
        data6['daly2050_Ratio'] = data6['daly2050'] / data6['daly2050'].sum()

        prev = pd.read_csv('./bigdata/data_diabetes/prevalence_val.csv').set_index(['Country Code', 'sex', 'age']).drop(columns=['disease'])
        pop = pd.read_csv('./data/population_un.csv').set_index(['Country Code', 'sex', 'age'])
        PREVs = prev * pop 
        PREV = pd.concat([PREVs['2020'].dropna(), PREVs['2050'].dropna()], axis=1)
        PREV = PREV.groupby('Country Code').sum().reset_index()
        PREV = PREV.merge(self.INFODATA[['Country Code', 'Region', 'Income group']], on='Country Code')
        data7 = PREV.groupby('Region').sum().reset_index()
        data7['location'] = data7['Region']
        data7['prev2020'] = data7['2020'] / 1000000
        data7['prev2020_Ratio'] = data7['prev2020'] / data7['prev2020'].sum()
        data7['prev2050'] = data7['2050'] / 1000000
        data7['prev2050_Ratio'] = data7['prev2050'] / data7['prev2050'].sum()
        data8 = PREV.groupby('Income group').sum().reset_index()
        data8['location'] = data8['Income group']
        data8['prev2020'] = data8['2020'] / 1000000
        data8['prev2020_Ratio'] = data8['prev2020'] / data8['prev2020'].sum()
        data8['prev2050'] = data8['2050'] / 1000000
        data8['prev2050_Ratio'] = data8['prev2050'] / data8['prev2050'].sum()
        data9 = PREV.sum(numeric_only=True).to_frame().T
        data9['location'] = 'global'
        data9['prev2020'] = data9['2020'] / 1000000
        data9['prev2020_Ratio'] = data9['prev2020'] / data9['prev2020'].sum()
        data9['prev2050'] = data9['2050'] / 1000000
        data9['prev2050_Ratio'] = data9['prev2050'] / data9['prev2050'].sum()     
        
        data_info = pd.concat([data1, data2, data3])
        data_daly = pd.concat([data4, data5, data6])
        data_prev = pd.concat([data7, data8, data9])
        data = data_info.merge(data_daly, on='location').merge(data_prev, on='location')

        data['GDP 2020'] = data.apply(lambda row: str(round(row['gdp_psy']))+' ('+   "{0:.1%}".format(row['gdp_psy_Ratio'])   +')', axis=1)
        data['POP 2020'] = data.apply(lambda row: str(round(row['pop_psy']))+' ('+   "{0:.1%}".format(row['pop_psy_Ratio'])   +')', axis=1)
        data['DALY 2020'] = data.apply(lambda row: str(round(row['daly2020']))+' ('+   "{0:.1%}".format(row['daly2020_Ratio'])   +')', axis=1)
        data['DALY 2050'] = data.apply(lambda row: str(round(row['daly2050']))+' ('+   "{0:.1%}".format(row['daly2050_Ratio'])   +')', axis=1)
        data['PREV 2020'] = data.apply(lambda row: str(round(row['prev2020']))+' ('+   "{0:.1%}".format(row['prev2020_Ratio'])   +')', axis=1)
        data['PREV 2050'] = data.apply(lambda row: str(round(row['prev2050']))+' ('+   "{0:.1%}".format(row['prev2050_Ratio'])   +')', axis=1)
        data['averageGDP'] = data.apply(lambda row: str(round(row['averageGDP']))+' ('+   "{0:.1%}".format(row['totalGDP_Ratio'])   +')', axis=1)
        data['averagePOP'] = data.apply(lambda row: str(round(row['averagePOP']))+' ('+   "{0:.1%}".format(row['totalPOP_Ratio'])   +')', axis=1)
        data[['location', 'GDP 2020', 'averageGDP', 'averagePOP', 'POP 2020', 'DALY 2020', 'DALY 2050', 'PREV 2020', 'PREV 2050']].to_csv('tables/Table3_discount%s.csv'%(self.state['discount']), index=False)
        self.INFODATA.to_csv('tmpresults/infodata.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--filename', type=str, default='results/aggregate_results_imputed.csv') 
    parser.add_argument('-d', '--discount', type=float, default=0.02) # or 0, 0.02, 0.03
    parser.add_argument('-i', '--informal', type=float, default=0.11) # or 0, 0.05, 0.11, 0.23
    args = parser.parse_args()
    # In[19]:
    mytable = Tables(discount=args.discount, informal=args.informal, filename=args.filename)
    mytable.generate_table1()
    mytable.generate_table2()
    mytable.generate_table3()


