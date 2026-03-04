# Data Preparation For diseases

Workspace：

```
local for download data, processing small data **puppy**:../HMM_diabetes/dataPreparation
Computing Platform MacBook Air (M1, 2020)
Updated to 26 April 2022.
```

## Country Name Cross-Data

INPUT: Artificial Download

OUTPUT: dl1_countrycodeorg_country_name.csv

Details:

1. Get country code from [https://countrycode.org/](https://countrycode.org/)
2. Get WBCountry from [https://data.worldbank.org/country](https://data.worldbank.org/country)
3. Get WBCountry Code、Region、Income group from [http://databank.worldbank.org/data/download/site-content/CLASS.xlsx](http://databank.worldbank.org/data/download/site-content/CLASS.xlsx)
4. Get country name from [http://ghdx.healthdata.org/gbd-results-tool](http://ghdx.healthdata.org/gbd-results-tool)

Artificial processing to make sure 1 Country Code 2 WBCountry 3 country are matching.

1. Channel Islands has no corresponding country code (GB, the same as the United Kingdom), so we take the code in CHI from CLASS.xlsx of the world bank.

[Country Name Summary](https://www.notion.so/af016a5fc9414826ad6dd7a209c6d2b0)

## Education

Data source Barro Lee education database (from 1990-2040):

[http://www.barrolee.com](http://www.barrolee.com/)

```python
## 输入与输出

## INPUT FILE
df_input_female = pd.read_csv('BL2013_F_v2.2.csv')
df_input_male = pd.read_csv('BL2013_M_v2.2.csv')
df_project_female = pd.read_csv('OUP_proj_F1564_v1.csv')
df_project_male = pd.read_csv('OUP_proj_M1564_v1.csv')

country_names = pd.read_csv('../dl1_countrycodeorg_country_name.csv')
codemap = dict(zip(country_names['Country Code'], country_names.country))
WBcodes = df_input_female['WBcode'].unique() # 146

## OUTPUT FILE
save_file_name = "education.csv"
save_filled_file_name = "education_filled.csv"
save_check_name = 'check_country_name.csv'
```

Details:

First, Prepare input files. 

Download these education data to PATH="./education"

```bash
# Download 1950-2010 BarroLeeDataset
# wget https://barrolee.github.io/BarroLeeDataSet/BLData/BL2013_MF_v2.2.csv
wget https://barrolee.github.io/BarroLeeDataSet/BLData/BL2013_F_v2.2.csv
wget https://barrolee.github.io/BarroLeeDataSet/BLData/BL2013_M_v2.2.csv

# Download 2015-2040 Pojections
# wget https://barrolee.github.io/BarroLeeDataSet/OUP/OUP_proj_MF1564_v1.csv
wget https://barrolee.github.io/BarroLeeDataSet/OUP/OUP_proj_F1564_v1.csv
wget https://barrolee.github.io/BarroLeeDataSet/OUP/OUP_proj_M1564_v1.csv
```

Then, run all codes in "./education/process.ipynb"

Finally, output 146 countries education data:  "./education/education.csv"

output 216 countries education data:"./education/education_filled.csv"

```
216 countries = 216 WB countries 
+ TWN(Taiwan, China, has economic data, East Asia & Pacific, High income) 
+ REU(Reunion, has primary education data) 
- PRK(North Korea, the only country as East Asia & Pacific Low income) 
- SYC(Seychelles, the only country as Sub-Saharan Africa High income) 

204 GBD 
- {'PRK', 'SYC', 'PSE', 'NIU', 'COK', 'TKL'} 6
+ {'CUW', 'GIB', 'REU', 'CYM', 'PYF', 'ABW', 'MAF', 'SXM', 'IMN', 'XKX', 'MAC', 'VGB', 'TCA', 'CHI', 'FRO', 'HKG', 'LIE', 'NCL'} 18
```

Total Run time: 10s.

### Labour data

Data source ILOSTAT International Labour Organization Statistics (from 2010-2030)

 [https://ilostat.ilo.org/data/](https://ilostat.ilo.org/data/)

Details:

First, Prepare input files. 

Download these labor data to PATH="./labor"


```bash
# Population by sex and age - - UN estimates and projections, Nov. 2020 (thousands) - Annual }
https://www.ilo.org/shinyapps/bulkexplorer25/?lang=en&segment=indicator&id=HOW_2LSS_NOC_RT_A
# Download ILO modelled estimates labour force participation rate by sex and age (annual) 
save as "population_p.csv"
wget https://www.ilo.org/ilostat-files/WEB_bulk_download/indicator/EAP_2WAP_SEX_AGE_RT_A.csv.gz
# \text { Labour force participation rate by sex and age - - ILO modelled estimates, Nov. } 2020 \text { (\%) - Annual }
save as "laborparticipation_p.csv"
```

Then, run all codes in "./labor/labor.ipynb"

Total Run time: 3s.

### Population Data

Population Data Source: Department of Economic and Social Affairs Population Dynamics

[United Nations Population Website](https://population.un.org/wpp/DataQuery/)

Downloads: 

1. Total Population from 1950 to 2100. 
https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_TotalPopulationBySex.csv

2. Population by Age and Sex from 1950 to 2100.
https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_PopulationByAgeSex_Medium.csv

[https://population.un.org/wpp/Download/Files/4_Metadata/WPP2019_F01_LOCATIONS.XLSX](https://population.un.org/wpp/Download/Files/4_Metadata/WPP2019_F01_LOCATIONS.XLSX)


Then RUN 'population_labor/UNpopulation.ipynb'

Total Run time: 7s.

### GDP: World Bank Statistics updated to Apr 8, 2022; growth projections: IMF Data, updated to Apr 2022, Accessed Apr, 2022

**Data Source: The world bank**

GDP, PPP (constant 2017 international $) save as GDP_raw_ppp.csv

[https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.PP.KD?downloadformat=csv](https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.PP.KD?downloadformat=csv)
https://data.worldbank.org/indicator/NY.GDP.MKTP.PP.KD

*GDP Projection data: the growth rate from 2021-2027 save as GDP_projection.csv*

https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/2022/WEOApr2022all.ashx

then we use the average of growing rate at 2015-2019 as the growth for years 2028-2050.

Due to the high growth rate at 2015-2019 in China, we set the average growth rate as 3.5% in 2028-2050.

Given the covid crisis, we will project the GDP beyond 2021 without using growth data for 2020 and 2021.

**Code for Processing Raw Data: ./**GDP/GDP_process.ipynb

**Output:** GDP.csv or GDP_ppp.csv

### Savings, Updated to Apr 8, 2022, Accessed Apr, 2022

**Data Source: The world bank**, assume savings rate going forward is the average between 2010-2019

Gross Savings of GDP

[https://data.worldbank.org/indicator/NY.GNS.ICTR.ZS](https://data.worldbank.org/indicator/NY.GNS.ICTR.ZS)

save as *savings_raw.csv*

assuming that the saving from 2021 to 2050 is the average from 2010-2019.

**Code for Processing Raw Data:**

Savings/savings_process.ipynb

**Output:**

savings.csv

### Health expenditure, Updated to Apr 8, 2022, Accessed Apr, 2022

**Data Source: The world bank**, 

Current health expenditure (% of GDP)

[https://api.worldbank.org/v2/en/indicator/SH.XPD.CHEX.GD.ZS?downloadformat=csv](https://api.worldbank.org/v2/en/indicator/SH.XPD.CHEX.GD.ZS?downloadformat=csv)
https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS

Current health expenditure per capita, PPP (current international $)

[https://api.worldbank.org/v2/en/indicator/SH.XPD.CHEX.PP.CD?downloadformat=csv](https://api.worldbank.org/v2/en/indicator/SH.XPD.CHEX.PP.CD?downloadformat=csv)


Keep data starting from 2010.

Projection: the difficulty is some countries grow too fast or decrease too fast, pure linear projection may not make sense.

- If the % is increasing between 2000-2019.
    - If the % is larger than OECD average 12.44%, stop and keep constant
    - If the % is smaller than OECD average, keep increasing
- If the % is decreasing between 2000-2019.
    - If the % is smaller than 3% (1/3 of world average), stop decreasing

**Code for Processing/projecting Data:**

health_expenditure/data_processing.ipynb

**Output:**

*hepc.csv* or hepc_ppp.csv if using GDP_ppp.csv

### Mortality and Morbidity, Updated Sep, 2021, Accessed Apr, 2022

Mortality and morbidity data source: Global Health Data Exchange

[IHME dataset website](http://ghdx.healthdata.org/gbd-results-tool)

Go to the IHME dataset website, press F12 button and fill the javascript console, then press Enter to run the script.
Please refer to the [offical codebook](https://ghdx.healthdata.org/sites/default/files/ihme_query_tool/IHME_GBD_2019_CODEBOOK.zip) to find the specic meaning of the code for ages(all 5 year groups), locations(only countries and territories), measures(Deaths, DALYs, YLDs, YLLs, Prevalence, Incidence), cause(Diabetes mellitus) and so on.
<details>
    <summary>Click to see details of javascript codes.</summary>
    <pre><code>
const years = [2019,2018,2017,2016,2015,2014,2013,2012,2011,2010]
const cause = [587];


const tasks = [];

years.map(year => {
  cause.forEach(item => {
    tasks.push({
      year,
      cause: item
    })
  });
});

function sleep() {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve();
    }, 1200);
  })
}

(async () => {
  for (const task of tasks) {
    await sleep();
    doTask(task);
  }
})()

function doTask(task) {
  var data = {
    version: 7266,
    "year[0]": task.year,
    "age[0]": 1,
    "age[1]": 6,
    "age[2]": 7,
    "age[3]": 8,
    "age[4]": 9,
    "age[5]": 10,
    "age[6]": 11,
    "age[7]": 12,
    "age[8]": 13,
    "age[9]": 14,
    "age[10]": 15,
    "age[11]": 16,
    "age[12]": 17,
    "age[13]": 18,
    "age[14]": 19,
    "age[15]": 234,
    "cause[0]": task.cause,
    "location[0]": 33,
    "location[1]": 34,
    "location[2]": 35,
    "location[3]": 36,
    "location[4]": 37,
    "location[5]": 38,
    "location[6]": 39,
    "location[7]": 40,
    "location[8]": 41,
    "location[9]": 43,
    "location[10]": 44,
    "location[11]": 45,
    "location[12]": 46,
    "location[13]": 47,
    "location[14]": 48,
    "location[15]": 50,
    "location[16]": 49,
    "location[17]": 51,
    "location[18]": 52,
    "location[19]": 53,
    "location[20]": 54,
    "location[21]": 55,
    "location[22]": 57,
    "location[23]": 58,
    "location[24]": 59,
    "location[25]": 60,
    "location[26]": 61,
    "location[27]": 62,
    "location[28]": 63,
    "location[29]": 71,
    "location[30]": 72,
    "location[31]": 66,
    "location[32]": 67,
    "location[33]": 68,
    "location[34]": 69,
    "location[35]": 101,
    "location[36]": 349,
    "location[37]": 102,
    "location[38]": 97,
    "location[39]": 98,
    "location[40]": 99,
    "location[41]": 74,
    "location[42]": 75,
    "location[43]": 76,
    "location[44]": 77,
    "location[45]": 78,
    "location[46]": 79,
    "location[47]": 80,
    "location[48]": 81,
    "location[49]": 82,
    "location[50]": 83,
    "location[51]": 84,
    "location[52]": 85,
    "location[53]": 86,
    "location[54]": 87,
    "location[55]": 88,
    "location[56]": 367,
    "location[57]": 89,
    "location[58]": 90,
    "location[59]": 91,
    "location[60]": 396,
    "location[61]": 92,
    "location[62]": 93,
    "location[63]": 94,
    "location[64]": 95,
    "location[65]": 121,
    "location[66]": 122,
    "location[67]": 123,
    "location[68]": 105,
    "location[69]": 106,
    "location[70]": 107,
    "location[71]": 108,
    "location[72]": 305,
    "location[73]": 109,
    "location[74]": 110,
    "location[75]": 111,
    "location[76]": 112,
    "location[77]": 113,
    "location[78]": 114,
    "location[79]": 115,
    "location[80]": 385,
    "location[81]": 393,
    "location[82]": 116,
    "location[83]": 117,
    "location[84]": 118,
    "location[85]": 119,
    "location[86]": 422,
    "location[87]": 125,
    "location[88]": 126,
    "location[89]": 127,
    "location[90]": 128,
    "location[91]": 129,
    "location[92]": 130,
    "location[93]": 131,
    "location[94]": 132,
    "location[95]": 133,
    "location[96]": 135,
    "location[97]": 136,
    "location[98]": 160,
    "location[99]": 139,
    "location[100]": 140,
    "location[101]": 141,
    "location[102]": 142,
    "location[103]": 143,
    "location[104]": 144,
    "location[105]": 145,
    "location[106]": 146,
    "location[107]": 147,
    "location[108]": 148,
    "location[109]": 150,
    "location[110]": 149,
    "location[111]": 151,
    "location[112]": 152,
    "location[113]": 522,
    "location[114]": 153,
    "location[115]": 154,
    "location[116]": 155,
    "location[117]": 156,
    "location[118]": 157,
    "location[119]": 161,
    "location[120]": 162,
    "location[121]": 163,
    "location[122]": 164,
    "location[123]": 165,
    "location[124]": 6,
    "location[125]": 7,
    "location[126]": 8,
    "location[127]": 298,
    "location[128]": 320,
    "location[129]": 22,
    "location[130]": 351,
    "location[131]": 23,
    "location[132]": 24,
    "location[133]": 25,
    "location[134]": 369,
    "location[135]": 374,
    "location[136]": 376,
    "location[137]": 380,
    "location[138]": 26,
    "location[139]": 27,
    "location[140]": 28,
    "location[141]": 413,
    "location[142]": 29,
    "location[143]": 416,
    "location[144]": 30,
    "location[145]": 10,
    "location[146]": 11,
    "location[147]": 12,
    "location[148]": 13,
    "location[149]": 14,
    "location[150]": 183,
    "location[151]": 15,
    "location[152]": 16,
    "location[153]": 186,
    "location[154]": 17,
    "location[155]": 18,
    "location[156]": 19,
    "location[157]": 20,
    "location[158]": 168,
    "location[159]": 169,
    "location[160]": 170,
    "location[161]": 171,
    "location[162]": 172,
    "location[163]": 173,
    "location[164]": 175,
    "location[165]": 176,
    "location[166]": 177,
    "location[167]": 178,
    "location[168]": 179,
    "location[169]": 180,
    "location[170]": 181,
    "location[171]": 182,
    "location[172]": 184,
    "location[173]": 185,
    "location[174]": 187,
    "location[175]": 435,
    "location[176]": 190,
    "location[177]": 189,
    "location[178]": 191,
    "location[179]": 193,
    "location[180]": 197,
    "location[181]": 194,
    "location[182]": 195,
    "location[183]": 196,
    "location[184]": 198,
    "location[185]": 200,
    "location[186]": 201,
    "location[187]": 203,
    "location[188]": 202,
    "location[189]": 204,
    "location[190]": 205,
    "location[191]": 206,
    "location[192]": 207,
    "location[193]": 208,
    "location[194]": 209,
    "location[195]": 210,
    "location[196]": 211,
    "location[197]": 212,
    "location[198]": 213,
    "location[199]": 214,
    "location[200]": 215,
    "location[201]": 216,
    "location[202]": 217,
    "location[203]": 218,
    "measure[0]": 1,
    "measure[1]": 2,
    "measure[2]": 3,
    "measure[3]": 4,
    "measure[4]": 5,
    "measure[5]": 6,
    "sex[0]": 1,
    "sex[1]": 2,
    "metric[0]": 1,
    "metric[1]": 2,
    "metric[2]": 3,
    base: "single",
    context: "cause",
    audience: "public",
    gbdRound: 2019,
    toolID: 1,
    email: "tech@zhongcao.fit",
    idsOrNames: "names",
    numParams: 18,
    rows: 500000,
    singleOrMult: "single",
  };

  fetch('https://ghdx.healthdata.org/sites/all/modules/custom/ihme_query_tool/gbd-search/php/download.php', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: Object.keys(data).map(key => {
      return `${key}=${data[key]}`
    }).join('&')
  })
}
    </code></pre>
</details>


Then, run ./Mortality_morbidity/IHME/IHME.ipynb for examples of generating mortality/morbidity data for the CANCER project.

Finally, output 204 countries mortality and morbidity data:  "./Mortality_morbidity/IHME/data_cancer/mortality_val.csv, morbidity_val.csv"


### Physical Capital, updated to Nov 8, 2021. Accessed Apr, 2022

**Data Source:** Penn World Table (from 1990-2019)

[Physical PPP 2017 Download Link](https://fred.stlouisfed.org/categories/33404?t=capital&ob=pv&od=desc)

[Physical constant 2017 Download Link](https://fred.stlouisfed.org/categories/33405?cid=33405&et=&ob=pv&od=desc&pageID=1&t=stocks)(failed to download)

save data files into "./physicalcapital/physical_ppp"

**Code for Processing Data:**

Physicalcapital/process_physical.ipynb

Finally, output 167 countries physical capital data:  "./physicalcapital/physical_ppp.csv"

```
181 countries

Subtraction from WB: ['ABW', 'AFG', 'AND', 'ARE', 'ASM', 'CHI', 'CUB', 'CUW', 'CYM', 'DZA', 'ERI', 'FRO', 'FSM', 'GIB', 'GRL', 'GUM', 'GUY', 'HTI', 'IMN', 'KIR', 'LBY', 'LIE', 'MAF', 'MCO', 'MHL', 'MMR', 'MNP', 'NCL', 'NIC', 'NRU', 'PLW', 'PNG', 'PRI', 'PRK', 'PYF', 'SLB', 'SMR', 'SOM', 'SSD', 'SXM', 'SYC', 'TCA', 'TLS', 'TON', 'TUV', 'VGB', 'VIR', 'VUT', 'WSM', 'XKX'] 50
Plus from WB: ['TWN'] 1
Subtraction from GBD : ['AFG', 'AND', 'ARE', 'ASM', 'COK', 'CUB', 'DZA', 'ERI', 'FSM', 'GRL', 
'GUM', 'GUY', 'HTI', 'KIR', 'LBY', 'MCO', 'MHL', 'MMR', 'MNP', 'NIC', 
'NIU', 'NRU', 'PLW', 'PNG', 'PRI', 'PRK', 'PSE', 'SLB', 'SMR', 'SOM', 
'SSD', 'SYC', 'TKL', 'TLS', 'TON', 'TUV', 'VIR', 'VUT', 'WSM'] 39
Plus from GBD : ['HKG', 'MAC'] 2
```

### Treatment Cost, updated to Nov 8 2021. Accessed Apr, 2022

HMM previous papers concentrate on estimating with JAMA 2016[1].
This is the tricky part, and each disease may vary significantly. 

1. Assuming that TC(country, disease) = const * prevalence(country, disease) * house expenditure(country, disease)
2. According to JAMA 2016 [1], we could calculate the radio of each disease treatment cost to all disease treatment costs in the USA. Then TC(USA, Diabetes mellitus) = USA_house_expenditure * 101.4 / 2100.1 

Due to the recent study on direct cost of diabetes mellitus, we adopted [treatment data](https://diabetesatlas.org/data/en/indicators/19/) for in 2011 or 2021 estimated by International Diabetes Federation (IDF)[2] 2011-2021. The data is per patient cost. So we calculate per capita cost, which is per patient cost * prevalence rate.

[1] Dieleman JL, Baral R, Birger M, Bui AL, Bulchis A, Chapin A, Hamavid H, Horst C, Johnson EK, Joseph J, Lavado R, Lomsadze L, Reynolds A, Squires E, Campbell M, DeCenso B, Dicker D, Flaxman AD, Gabert R, Highfill T, Naghavi M, Nightingale N, Templin T, Tobias MI, Vos T, Murray CJ. US Spending on Personal Health Care and Public Health, 1996-2013. JAMA. 2016 Dec 27;316(24):2627-2646. doi: 10.1001/jama.2016.16885. PMID: 28027366; PMCID: PMC5551483.
[2] Williams R, Karuranga S, Malanda B, Saeedi P, Basit A, Besançon S, Bommer C, Esteghamati A, Ogurtsova K, Zhang P, Colagiuri S. Global and regional estimates and projections of diabetes-related health expenditure: Results from the International Diabetes Federation Diabetes Atlas, 9th edition. Diabetes Res Clin Pract. 2020 Apr;162:108072. doi: 10.1016/j.diabres.2020.108072. Epub 2020 Feb 13. PMID: 32061820.