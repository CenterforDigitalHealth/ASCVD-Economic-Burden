import pandas as pd
import numpy as np
import math
import sys
import os
import argparse
import pdb

ASCVD_GROUPS = ['IHD', 'IS', 'PAD']
TABLE_YEARS = ['2020', '2050']
LOCATION_ORDER = [
    'East Asia and Pacific',
    'Europe and Central Asia',
    'Latin America and Caribbean',
    'Middle East and North Africa',
    'North America',
    'South Asia',
    'Sub-Saharan Africa',
    'Low income',
    'Lower middle income',
    'Upper middle income',
    'High income',
    'Global',
]
DISEASE_ALIASES = {
    'IHD': 'Ischemic heart disease',
    'IS': 'Ischemic stroke',
    'PAD': 'Lower extremity peripheral arterial disease',
}
DISEASE_NAME_TO_ALIAS = {v.lower(): k for k, v in DISEASE_ALIASES.items()}


def read_csv_safe(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs['engine'] = 'python'
        return pd.read_csv(path, **fallback_kwargs)


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


def normalize_output_tag(output_tag, disease='all'):
    candidate = output_tag
    if candidate is None or str(candidate).strip() == '':
        candidate = disease

    text = str(candidate).strip()
    if text == '' or text.lower() == 'all':
        return 'ALL'

    tags = []
    for token in text.split(','):
        token = token.strip()
        if token == '':
            continue
        lower_token = token.lower()
        if lower_token == 'all':
            mapped = 'ALL'
        elif token.upper() in DISEASE_ALIASES:
            mapped = token.upper()
        elif lower_token in DISEASE_NAME_TO_ALIAS:
            mapped = DISEASE_NAME_TO_ALIAS[lower_token]
        else:
            safe = []
            for ch in token:
                if ch.isalnum() or ch in ['_', '-']:
                    safe.append(ch.upper())
                elif ch in [' ', '/']:
                    safe.append('_')
            mapped = ''.join(safe).strip('_')
            while '__' in mapped:
                mapped = mapped.replace('__', '_')
        if mapped != '':
            tags.append(mapped)

    if len(tags) == 0:
        return 'ALL'

    dedup = []
    seen = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        dedup.append(tag)
    return '_'.join(dedup)


def parse_ascvd_groups(disease_arg):
    if disease_arg is None:
        return list(ASCVD_GROUPS)

    text = str(disease_arg).strip()
    if text == '':
        return list(ASCVD_GROUPS)

    groups = []
    for token in text.split(','):
        token = token.strip()
        if token == '':
            continue

        upper_token = token.upper()
        lower_token = token.lower()

        if lower_token in ['all', 'ascvd']:
            return list(ASCVD_GROUPS)

        if upper_token in ASCVD_GROUPS:
            mapped = upper_token
        elif lower_token in DISEASE_NAME_TO_ALIAS:
            mapped = DISEASE_NAME_TO_ALIAS[lower_token]
        else:
            mapped = None

        if mapped is not None and mapped not in groups:
            groups.append(mapped)

    if len(groups) == 0:
        return list(ASCVD_GROUPS)
    return groups


def format_value(value, digits):
    if pd.isna(value):
        return ''
    rounded = round(float(value), digits)
    if digits > 0:
        text = f"{rounded:,.{digits}f}"
        if abs(rounded) < 10000:
            text = text.replace(',', '')
        if text in ['-0', '-0.0', '-0.00']:
            return f"{0:.{digits}f}"
        return text

    integer = int(rounded)
    if abs(integer) >= 10000:
        return f"{integer:,}"
    if digits == 0:
        return str(integer)
    return str(rounded)


def format_interval(mean_value, lower_value, upper_value, digits=0, scale=1.0):
    if pd.isna(mean_value):
        return ''
    mean_text = format_value(scale * mean_value, digits)
    if pd.isna(lower_value) or pd.isna(upper_value):
        return mean_text
    lower_bound = min(lower_value, upper_value)
    upper_bound = max(lower_value, upper_value)
    lower_text = format_value(scale * lower_bound, digits)
    upper_text = format_value(scale * upper_bound, digits)
    return f"{mean_text}({lower_text}-{upper_text})"


def format_value_with_ratio_interval(
    value,
    lower_value,
    upper_value,
    ratio,
    lower_ratio,
    upper_ratio,
    value_digits=0,
    value_scale=1.0,
    ratio_digits=1,
    ratio_scale=100.0,
    show_ratio_interval=True,
):
    value_text = format_interval(value, lower_value, upper_value, digits=value_digits, scale=value_scale)
    if value_text == '':
        return ''
    if pd.isna(ratio):
        return value_text
    if show_ratio_interval:
        ratio_text = format_interval(ratio, lower_ratio, upper_ratio, digits=ratio_digits, scale=ratio_scale)
    else:
        ratio_text = format_value(ratio_scale * ratio, ratio_digits)
    if ratio_text == '':
        return value_text
    return f"{value_text} ({ratio_text}%)"


def apply_location_order(data, location_col='location', secondary_sort_cols=None, secondary_ascending=None):
    if location_col not in data.columns:
        return data

    out = data.copy()
    out[location_col] = out[location_col].replace({'global': 'Global'})

    dynamic_categories = list(pd.Series(out[location_col].dropna().astype(str).unique()))
    categories = list(LOCATION_ORDER)
    for name in dynamic_categories:
        if name not in categories:
            categories.append(name)

    out['_location_order'] = pd.Categorical(out[location_col], categories=categories, ordered=True)
    sort_cols = ['_location_order']
    ascending = [True]
    if secondary_sort_cols:
        sort_cols.extend(secondary_sort_cols)
        if secondary_ascending is None:
            ascending.extend([True] * len(secondary_sort_cols))
        else:
            ascending.extend(list(secondary_ascending))
    out = out.sort_values(sort_cols, ascending=ascending).drop(columns=['_location_order'])
    return out


class Tables():
    def __init__(self, discount=0.02, informal=0.11, filename='results/aggregate_results_imputed.csv',
                 disease='all', output_tag=None):
        self.df_input = read_csv_safe(filename)
        self.df_input = filter_df_by_disease(self.df_input, disease)
        if len(self.df_input) == 0:
            raise ValueError(f"No rows left after disease filter: {disease}")
        self.countries = self.df_input['Country Code'].unique()
        if discount == 0.0 :
            discount = 0
        self.default_discount = discount
        self.default_informal = informal
        self.output_tag = normalize_output_tag(output_tag, disease=disease)
        self.ascvd_groups = parse_ascvd_groups(disease)
        os.makedirs('tables', exist_ok=True)
        self.set_state()
        self.set_params()
        self.df_state = self.get_data()
        

    def set_params(self):
        countries_info = read_csv_safe('data/countrycode.csv', encoding='utf-8-sig')
        self.countries_info = countries_info[['Country Code', 'Region', 'Income group', 'WBCountry', 'country']]
        # self.codemap = countries_info.dropna()
        self.endyear = 2051
        self.projectStartYear = 2020
        gdp_total_df = read_csv_safe('tmpresults/GDP_TOTAL_discount%s.csv'%(self.default_discount)).set_index('Country Code')
        pop_total_df = read_csv_safe('tmpresults/POP_TOTAL.csv').set_index('Country Code')
        gdp_psy_df = read_csv_safe('tmpresults/GDP_PSY.csv').set_index('Country Code')
        pop_psy_df = read_csv_safe('tmpresults/POP_PSY.csv').set_index('Country Code')
        self.INFODATA = gdp_total_df.merge(pop_total_df, on='Country Code').merge(gdp_psy_df, on='Country Code').merge(pop_psy_df, on='Country Code')
        self.INFODATA = self.INFODATA.merge(countries_info, on='Country Code')
        self.INFODATA = self.INFODATA[self.INFODATA['Country Code'].isin(self.countries)]
        print(len(self.INFODATA))
               
    def set_state(self, state=None):
        if state is None:
            state = {'ConsiderTC':1, 'ConsiderMB':1, 'scenario':'val'}
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
        assert 'disease' in identify
        if len(imputed) == 0:
            return pd.DataFrame(columns=identify + ['GDPloss', 'GDPlossRatio', 'tax', 'pc_loss'])

        group = pd.DataFrame()
        grouped_loss = imputed.groupby(identify)['GDPloss'].sum()
        group['GDPloss'] = grouped_loss
        total_loss = imputed['GDPloss'].sum()
        if total_loss == 0:
            group['GDPlossRatio'] = 0
        else:
            group['GDPlossRatio'] = grouped_loss / total_loss
        grouped_gdp = imputed.groupby(identify)['totalGDP'].sum().replace(0, np.nan)
        grouped_pop = imputed.groupby(identify)['totalPOP'].sum().replace(0, np.nan)
        group['tax'] = grouped_loss * 1000000000 / grouped_gdp
        group['pc_loss'] = grouped_loss * 1000000000 / (grouped_pop / (self.endyear - self.projectStartYear))
        group = group.fillna(0).reset_index()
        group.sort_values([identify[0],'GDPlossRatio'], ascending = [True,False], inplace=True)

        # group['tax'] = group['tax']*1000 # rate - > 1‰
        group['tax'] = group['tax']*100 # rate - > 1%
        return group

    def _load_ascvd_rate_sum(self, metric_prefix, scenario='val', years=None):
        if years is None:
            years = TABLE_YEARS

        combined = None
        for group in self.ascvd_groups:
            path = os.path.join('data', 'ASCVD', group, f'{metric_prefix}_{scenario}.csv')
            if not os.path.exists(path):
                continue
            df = read_csv_safe(path, encoding='utf-8-sig')
            df = df.rename(columns={'ISO3': 'Country Code', 'sex_name': 'sex', 'age_name': 'age'})
            if not {'Country Code', 'sex', 'age'}.issubset(df.columns):
                continue
            df['sex'] = df['sex'].replace({
                'Female': 'F',
                'Male': 'M',
                'Both': 'B',
                'female': 'F',
                'male': 'M',
                'both': 'B',
            })
            df = df[df['sex'].isin(['F', 'M'])]
            year_cols = [year for year in years if year in df.columns]
            if len(year_cols) == 0:
                continue
            df = df[['Country Code', 'sex', 'age'] + year_cols]
            for col in year_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df = df.groupby(['Country Code', 'sex', 'age'])[year_cols].sum()
            if combined is None:
                combined = df
            else:
                combined = combined.add(df, fill_value=0)

        if combined is None:
            return pd.DataFrame(columns=years)

        # ASCVD files are rates per 100,000.
        return combined / 100000.0

    def _rate_to_country_total(self, rate_df, pop_df, years=None):
        if years is None:
            years = TABLE_YEARS
        if len(rate_df) == 0:
            return pd.DataFrame(columns=['Country Code'] + years)

        common_index = rate_df.index.intersection(pop_df.index)
        if len(common_index) == 0:
            return pd.DataFrame(columns=['Country Code'] + years)

        counts = rate_df.loc[common_index, years] * pop_df.loc[common_index, years]
        country = counts.groupby('Country Code').sum().reset_index()
        return country

    def _get_group_data_by_scenario(self, identify):
        scenario_groups = {}
        original_state = dict(self.state)
        for scenario in ['lower', 'upper', 'val']:
            self.set_state(state={'ConsiderTC': 1, 'ConsiderMB': 1, 'scenario': scenario})
            self.get_data()
            scenario_groups[scenario] = self.get_group_data(identify)
        self.set_state(state=original_state)
        self.get_data()
        return scenario_groups

    def _merge_interval_columns(self, base_df, lower_df, upper_df, keys, value_cols):
        data = base_df.copy()
        if len(data) == 0:
            for key in keys:
                if key not in data.columns:
                    data[key] = pd.Series(dtype='object')
            for col in value_cols:
                if col not in data.columns:
                    data[col] = pd.Series(dtype='float64')
            for col in value_cols:
                data[f'{col}_lower'] = pd.Series(dtype='float64')
                data[f'{col}_upper'] = pd.Series(dtype='float64')
            ordered_cols = keys + value_cols + [f'{col}_lower' for col in value_cols] + [f'{col}_upper' for col in value_cols]
            return data[ordered_cols]

        for col in value_cols:
            if col not in data.columns:
                data[col] = np.nan

        lower_available = [col for col in value_cols if col in lower_df.columns]
        if len(lower_df) > 0 and len(lower_available) > 0:
            lower = lower_df[keys + lower_available].rename(columns={col: f'{col}_lower' for col in lower_available})
            data = data.merge(lower, on=keys, how='left')
        for col in value_cols:
            if f'{col}_lower' not in data.columns:
                data[f'{col}_lower'] = np.nan

        upper_available = [col for col in value_cols if col in upper_df.columns]
        if len(upper_df) > 0 and len(upper_available) > 0:
            upper = upper_df[keys + upper_available].rename(columns={col: f'{col}_upper' for col in upper_available})
            data = data.merge(upper, on=keys, how='left')
        for col in value_cols:
            if f'{col}_upper' not in data.columns:
                data[f'{col}_upper'] = np.nan

        ordered_cols = keys + value_cols + [f'{col}_lower' for col in value_cols] + [f'{col}_upper' for col in value_cols]
        return data[ordered_cols]

    def _select_base_scenario_df(self, scenario_map):
        base = scenario_map.get('val', pd.DataFrame())
        if len(base) == 0:
            base = scenario_map.get('lower', pd.DataFrame())
        if len(base) == 0:
            base = scenario_map.get('upper', pd.DataFrame())
        return base

    def _build_daly_country(self, pop_df, scenario='val', years=None):
        if years is None:
            years = TABLE_YEARS
        daly_rate = self._load_ascvd_rate_sum('DALYs', scenario=scenario, years=years)
        if len(daly_rate) > 0:
            daly = self._rate_to_country_total(daly_rate, pop_df, years=years)
        else:
            # Fallback for compatibility when DALYs files are unavailable.
            yll_rate = self._load_ascvd_rate_sum('YLLs', scenario=scenario, years=years)
            yld_rate = self._load_ascvd_rate_sum('YLDs', scenario=scenario, years=years)
            if len(yll_rate) == 0 and len(yld_rate) == 0:
                daly = pd.DataFrame(columns=['Country Code'] + years)
            elif len(yll_rate) == 0:
                daly = self._rate_to_country_total(yld_rate, pop_df, years=years)
            elif len(yld_rate) == 0:
                daly = self._rate_to_country_total(yll_rate, pop_df, years=years)
            else:
                daly = self._rate_to_country_total(yll_rate.add(yld_rate, fill_value=0), pop_df, years=years)
        if len(daly) == 0:
            return pd.DataFrame(columns=['Country Code', 'Region', 'Income group'] + years)
        return daly.merge(self.INFODATA[['Country Code', 'Region', 'Income group']], on='Country Code', how='inner')

    def _build_prev_country(self, pop_df, scenario='val', years=None):
        if years is None:
            years = TABLE_YEARS
        prev_rate = self._load_ascvd_rate_sum('Prevalence', scenario=scenario, years=years)
        prev = self._rate_to_country_total(prev_rate, pop_df, years=years)
        if len(prev) == 0:
            return pd.DataFrame(columns=['Country Code', 'Region', 'Income group'] + years)
        return prev.merge(self.INFODATA[['Country Code', 'Region', 'Income group']], on='Country Code', how='inner')

    def _summarize_metric_by_location(self, country_df, metric_prefix, years=None, divisor=1000000.0):
        if years is None:
            years = TABLE_YEARS
        value_cols = [f'{metric_prefix}{year}' for year in years]
        ratio_cols = [f'{metric_prefix}{year}_Ratio' for year in years]
        out_cols = ['location'] + value_cols + ratio_cols

        if len(country_df) == 0:
            return pd.DataFrame(columns=out_cols)

        def summarize_one(group_col):
            if group_col is None:
                part = country_df[years].sum(numeric_only=True).to_frame().T
                part['location'] = 'Global'
            else:
                part = country_df.groupby(group_col)[years].sum().reset_index()
                part['location'] = part[group_col]

            for year in years:
                value_col = f'{metric_prefix}{year}'
                ratio_col = f'{metric_prefix}{year}_Ratio'
                part[value_col] = part[year] / divisor
                total = part[value_col].sum()
                if total == 0:
                    part[ratio_col] = 0
                else:
                    part[ratio_col] = part[value_col] / total
            return part[out_cols]

        return pd.concat(
            [
                summarize_one('Region'),
                summarize_one('Income group'),
                summarize_one(None),
            ],
            ignore_index=True,
        )

    def generate_table1(self):
        table1_path = f"tables/Table1_{self.output_tag}.csv"
        identify = ['Country Code', 'disease']
        scenario_groups = {}
        for scenario in ['lower', 'upper', 'val']:
            self.set_state(state={'ConsiderTC': 1, 'ConsiderMB': 1, 'scenario': scenario})
            self.get_data()
            group = self.get_group_data(identify)
            if len(group) == 0:
                scenario_groups[scenario] = pd.DataFrame(columns=['Country Code', 'GDPloss', 'tax', 'pc_loss'])
            else:
                scenario_groups[scenario] = group.groupby('Country Code', as_index=False)[['GDPloss', 'tax', 'pc_loss']].sum()

        base = scenario_groups['val']
        if len(base) == 0 and len(scenario_groups['lower']) > 0:
            base = scenario_groups['lower']
        if len(base) == 0 and len(scenario_groups['upper']) > 0:
            base = scenario_groups['upper']
        if len(base) == 0:
            columns = ['Region', 'country', 'WBCountry', 'totalloss', 'pc_loss', 'tax %']
            pd.DataFrame(columns=columns).to_csv(table1_path, index=False)
            return

        lower = scenario_groups['lower'].rename(columns={
            'GDPloss': 'GDPloss_lower',
            'tax': 'tax_lower',
            'pc_loss': 'pc_loss_lower',
        })
        upper = scenario_groups['upper'].rename(columns={
            'GDPloss': 'GDPloss_upper',
            'tax': 'tax_upper',
            'pc_loss': 'pc_loss_upper',
        })

        df = base.merge(lower, on='Country Code', how='left')
        df = df.merge(upper, on='Country Code', how='left')
        df = df.merge(self.countries_info, on='Country Code')
        data = df.copy()
        data['totalloss'] = data.apply(
            lambda row: format_interval(
                row['GDPloss'],
                row.get('GDPloss_lower', np.nan),
                row.get('GDPloss_upper', np.nan),
                digits=1,
                scale=1000.0,
            ),
            axis=1,
        )
        data['tax %'] = data.apply(
            lambda row: format_interval(
                row['tax'],
                row.get('tax_lower', np.nan),
                row.get('tax_upper', np.nan),
                digits=2,
                scale=1.0,
            ),
            axis=1,
        )
        data['pc_loss'] = data.apply(
            lambda row: format_interval(
                row['pc_loss'],
                row.get('pc_loss_lower', np.nan),
                row.get('pc_loss_upper', np.nan),
                digits=1,
                scale=1.0,
            ),
            axis=1,
        )
        data = data.sort_values(['Region', 'country'])[['Region', 'country', 'WBCountry', 'totalloss', 'pc_loss', 'tax %']]
        data.to_csv(table1_path, index=False)

    def generate_table2(self):
        table2_path = f"tables/Table2_{self.output_tag}.csv"
        value_cols = ['GDPloss', 'GDPlossRatio', 'tax', 'pc_loss']

        region_groups = self._get_group_data_by_scenario(['Region', 'disease'])
        region_base = self._select_base_scenario_df(region_groups)
        if len(region_base) > 0:
            region_base = region_base.sort_values(['Region', 'GDPlossRatio'], ascending=[True, False]).groupby('Region').head(5).copy()
            region_data = self._merge_interval_columns(
                region_base,
                region_groups['lower'],
                region_groups['upper'],
                ['Region', 'disease'],
                value_cols,
            )
            region_data['location'] = region_data['Region']
        else:
            region_data = pd.DataFrame(columns=['location', 'disease'] + value_cols + [f'{col}_lower' for col in value_cols] + [f'{col}_upper' for col in value_cols])

        income_groups = self._get_group_data_by_scenario(['Income group', 'disease'])
        income_base = self._select_base_scenario_df(income_groups)
        if len(income_base) > 0:
            income_base = income_base.sort_values(['Income group', 'GDPlossRatio'], ascending=[True, False]).groupby('Income group').head(5).copy()
            income_data = self._merge_interval_columns(
                income_base,
                income_groups['lower'],
                income_groups['upper'],
                ['Income group', 'disease'],
                value_cols,
            )
            income_data['location'] = income_data['Income group']
        else:
            income_data = pd.DataFrame(columns=['location', 'disease'] + value_cols + [f'{col}_lower' for col in value_cols] + [f'{col}_upper' for col in value_cols])

        global_groups = self._get_group_data_by_scenario(['disease'])
        global_base = self._select_base_scenario_df(global_groups)
        if len(global_base) > 0:
            global_base = global_base.sort_values('GDPlossRatio', ascending=False).head(5).copy()
            global_data = self._merge_interval_columns(
                global_base,
                global_groups['lower'],
                global_groups['upper'],
                ['disease'],
                value_cols,
            )
            global_data['location'] = 'Global'
        else:
            global_data = pd.DataFrame(columns=['location', 'disease'] + value_cols + [f'{col}_lower' for col in value_cols] + [f'{col}_upper' for col in value_cols])

        data = pd.concat([region_data, income_data, global_data], ignore_index=True, sort=False)
        if len(data) == 0:
            pd.DataFrame(columns=['location', 'disease', 'burden', 'pc_loss', 'tax']).to_csv(table2_path, index=False)
            return

        data['burden'] = data.apply(
            lambda row: format_value_with_ratio_interval(
                row['GDPloss'],
                row.get('GDPloss_lower', np.nan),
                row.get('GDPloss_upper', np.nan),
                row['GDPlossRatio'],
                row.get('GDPlossRatio_lower', np.nan),
                row.get('GDPlossRatio_upper', np.nan),
                value_digits=1,
                value_scale=1.0,
                ratio_digits=1,
                ratio_scale=100.0,
                show_ratio_interval=False,
            ),
            axis=1,
        )
        data['tax'] = data.apply(
            lambda row: format_interval(
                row['tax'],
                row.get('tax_lower', np.nan),
                row.get('tax_upper', np.nan),
                digits=2,
                scale=1.0,
            ),
            axis=1,
        )
        data['pc_loss'] = data.apply(
            lambda row: format_interval(
                row['pc_loss'],
                row.get('pc_loss_lower', np.nan),
                row.get('pc_loss_upper', np.nan),
                digits=1,
                scale=1.0,
            ),
            axis=1,
        )
        data = apply_location_order(
            data,
            secondary_sort_cols=['GDPlossRatio', 'disease'],
            secondary_ascending=[False, True],
        )
        data[['location', 'disease', 'burden', 'pc_loss', 'tax']].to_csv(table2_path, index=False)

    def generate_table3(self):
        table3_path = f"tables/Table3_{self.output_tag}.csv"
        data1 = self.INFODATA.groupby('Region').sum(numeric_only=True).reset_index()
        data1['location'] = data1['Region']
        data1['gdp_psy_Ratio'] = data1['gdp_psy']/data1['gdp_psy'].sum()
        data1['pop_psy_Ratio'] = data1['pop_psy']/data1['pop_psy'].sum()
        data1['gdp_psy'] = data1['gdp_psy'] / 1000000000
        data1['pop_psy'] = data1['pop_psy'] / 1000000
        data1['totalGDP_Ratio'] = data1['totalGDP']/data1['totalGDP'].sum()
        data1['totalPOP_Ratio'] = data1['totalPOP']/data1['totalPOP'].sum()
        data1['averageGDP'] = data1['totalGDP'] / 1000000000 / (self.endyear - self.projectStartYear)
        data1['averagePOP'] = data1['totalPOP'] / 1000000 / (self.endyear - self.projectStartYear)
        data2 = self.INFODATA.groupby('Income group').sum(numeric_only=True).reset_index()
        data2['location'] = data2['Income group']
        data2['gdp_psy_Ratio'] = data2['gdp_psy']/data2['gdp_psy'].sum()
        data2['pop_psy_Ratio'] = data2['pop_psy']/data2['pop_psy'].sum()
        data2['gdp_psy'] = data2['gdp_psy'] / 1000000000
        data2['pop_psy'] = data2['pop_psy'] / 1000000
        data2['totalGDP_Ratio'] = data2['totalGDP']/data2['totalGDP'].sum()
        data2['totalPOP_Ratio'] = data2['totalPOP']/data2['totalPOP'].sum()
        data2['averageGDP'] = data2['totalGDP'] / 1000000000 / (self.endyear - self.projectStartYear)
        data2['averagePOP'] = data2['totalPOP'] / 1000000 / (self.endyear - self.projectStartYear)
        data3 = self.INFODATA.sum(numeric_only=True).to_frame().T
        data3['location'] = 'Global'
        data3['gdp_psy_Ratio'] = data3['gdp_psy']/data3['gdp_psy'].sum()
        data3['pop_psy_Ratio'] = data3['pop_psy']/data3['pop_psy'].sum()
        data3['gdp_psy'] = data3['gdp_psy'] / 1000000000
        data3['pop_psy'] = data3['pop_psy'] / 1000000
        data3['totalGDP_Ratio'] = data3['totalGDP']/data3['totalGDP'].sum()
        data3['totalPOP_Ratio'] = data3['totalPOP']/data3['totalPOP'].sum()
        data3['averageGDP'] = data3['totalGDP'] / 1000000000 / (self.endyear - self.projectStartYear)
        data3['averagePOP'] = data3['totalPOP'] / 1000000 / (self.endyear - self.projectStartYear)

        pop = read_csv_safe('./data/population_un.csv').set_index(['Country Code', 'sex', 'age'])
        pop = pop[TABLE_YEARS].apply(pd.to_numeric, errors='coerce').fillna(0)

        daly_by_scenario = {}
        prev_by_scenario = {}
        for scenario in ['lower', 'upper', 'val']:
            daly_country = self._build_daly_country(pop, scenario=scenario, years=TABLE_YEARS)
            daly_by_scenario[scenario] = self._summarize_metric_by_location(daly_country, metric_prefix='daly', years=TABLE_YEARS)
            prev_country = self._build_prev_country(pop, scenario=scenario, years=TABLE_YEARS)
            prev_by_scenario[scenario] = self._summarize_metric_by_location(prev_country, metric_prefix='prev', years=TABLE_YEARS)

        data_info = pd.concat([data1, data2, data3], ignore_index=True, sort=False)

        data_daly_base = self._select_base_scenario_df(daly_by_scenario)
        data_daly = self._merge_interval_columns(
            data_daly_base,
            daly_by_scenario.get('lower', pd.DataFrame()),
            daly_by_scenario.get('upper', pd.DataFrame()),
            ['location'],
            ['daly2020', 'daly2020_Ratio', 'daly2050', 'daly2050_Ratio'],
        )

        data_prev_base = self._select_base_scenario_df(prev_by_scenario)
        data_prev = self._merge_interval_columns(
            data_prev_base,
            prev_by_scenario.get('lower', pd.DataFrame()),
            prev_by_scenario.get('upper', pd.DataFrame()),
            ['location'],
            ['prev2020', 'prev2020_Ratio', 'prev2050', 'prev2050_Ratio'],
        )

        data = data_info.merge(data_daly, on='location', how='left').merge(data_prev, on='location', how='left')

        data['GDP 2020'] = data.apply(
            lambda row: format_value_with_ratio_interval(
                row['gdp_psy'],
                np.nan,
                np.nan,
                row['gdp_psy_Ratio'],
                np.nan,
                np.nan,
                value_digits=1,
                ratio_digits=1,
                show_ratio_interval=False,
            ),
            axis=1,
        )
        data['POP 2020'] = data.apply(
            lambda row: format_value_with_ratio_interval(
                row['pop_psy'],
                np.nan,
                np.nan,
                row['pop_psy_Ratio'],
                np.nan,
                np.nan,
                value_digits=1,
                ratio_digits=1,
                show_ratio_interval=False,
            ),
            axis=1,
        )
        data['DALY 2020'] = data.apply(
            lambda row: format_value_with_ratio_interval(
                row['daly2020'],
                np.nan,
                np.nan,
                row['daly2020_Ratio'],
                np.nan,
                np.nan,
                value_digits=1,
                ratio_digits=1,
                show_ratio_interval=False,
            ),
            axis=1,
        )
        data['DALY 2050'] = data.apply(
            lambda row: format_value_with_ratio_interval(
                row['daly2050'],
                np.nan,
                np.nan,
                row['daly2050_Ratio'],
                np.nan,
                np.nan,
                value_digits=1,
                ratio_digits=1,
                show_ratio_interval=False,
            ),
            axis=1,
        )
        data['PREV 2020'] = data.apply(
            lambda row: format_value_with_ratio_interval(
                row['prev2020'],
                np.nan,
                np.nan,
                row['prev2020_Ratio'],
                np.nan,
                np.nan,
                value_digits=1,
                ratio_digits=1,
                show_ratio_interval=False,
            ),
            axis=1,
        )
        data['PREV 2050'] = data.apply(
            lambda row: format_value_with_ratio_interval(
                row['prev2050'],
                np.nan,
                np.nan,
                row['prev2050_Ratio'],
                np.nan,
                np.nan,
                value_digits=1,
                ratio_digits=1,
                show_ratio_interval=False,
            ),
            axis=1,
        )
        data['averageGDP'] = data.apply(
            lambda row: format_value_with_ratio_interval(
                row['averageGDP'],
                np.nan,
                np.nan,
                row['totalGDP_Ratio'],
                np.nan,
                np.nan,
                value_digits=1,
                ratio_digits=1,
                show_ratio_interval=False,
            ),
            axis=1,
        )
        data['averagePOP'] = data.apply(
            lambda row: format_value_with_ratio_interval(
                row['averagePOP'],
                np.nan,
                np.nan,
                row['totalPOP_Ratio'],
                np.nan,
                np.nan,
                value_digits=1,
                ratio_digits=1,
                show_ratio_interval=False,
            ),
            axis=1,
        )
        data = apply_location_order(data)
        data[['location', 'GDP 2020', 'averageGDP', 'averagePOP', 'POP 2020', 'DALY 2020', 'DALY 2050', 'PREV 2020', 'PREV 2050']].to_csv(table3_path, index=False)
        self.INFODATA.to_csv('tmpresults/infodata.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--filename', type=str, default='results/aggregate_results_imputed.csv') 
    parser.add_argument('-d', '--discount', type=float, default=0.02) # or 0, 0.02, 0.03
    parser.add_argument('-i', '--informal', type=float, default=0.11) # or 0, 0.05, 0.11, 0.23
    parser.add_argument('--disease', type=str, default='all',
                        help='Disease selector: all, IHD, IS, PAD, full disease name, or comma-separated list')
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Optional output tag for table file names (e.g., ALL, IHD, IS, PAD)')
    # parser.add_argument('--only-table1', action='store_true',
    #                     help='Generate only Table1')
    args = parser.parse_args()
    # In[19]:
    mytable = Tables(
        discount=args.discount,
        informal=args.informal,
        filename=args.filename,
        disease=args.disease,
        output_tag=args.output_tag,
    )
    mytable.generate_table1()
    mytable.generate_table2()
    mytable.generate_table3()
