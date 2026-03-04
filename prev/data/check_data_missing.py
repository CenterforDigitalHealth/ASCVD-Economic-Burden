import pandas as pd
from typing import Dict, List, Union

class DataIntegrityChecker:
    """
    数据完整性检查器
    ---------------------------
    功能：
    1. 检查关键数据字段是否存在
    2. 生成详细的缺失报告
    3. 支持多年度数据验证

    使用示例：
    >>> data_sources = {
    ...     'physical_ppp': 'path/to/physical_ppp.csv',
    ...     'gdp': 'path/to/gdp.csv',
    ...     'hepc': 'path/to/hepc.csv',
    ...     'savings': 'path/to/savings.csv'
    ... }
    >>> checker = DataIntegrityChecker('TJK', data_sources)
    >>> result = checker.validate(
    ...     fields_to_check=['InitialCapitalStock', 'GDP', 'HealthExpenditure', 'SavingRate'],
    ...     check_years={'InitialCapitalStock': 2024, 'GDP': 2025}
    ... )
    >>> print(result)
    """

    def __init__(self, country_code: str, data_sources: Dict[str, str]):
        """
        初始化验证器
        :param country_code: 3位国家代码(如'TJK')
        :param data_sources: 数据源路径字典，应包含：
            - physical_ppp: 实物资本存量数据路径
            - gdp: GDP数据路径
            - hepc: 健康支出数据路径
            - savings: 储蓄率数据路径
        """
        self.country_code = country_code
        self.data_sources = data_sources
        self._data_cache = {}  # 数据缓存
        self.validation_report = {
            'missing_fields': [],
            'detailed_logs': []
        }

    def _load_dataset(self, dataset_key: str) -> pd.DataFrame:
        """加载并缓存数据集"""
        if dataset_key not in self._data_cache:
            try:
                df = pd.read_csv(self.data_sources[dataset_key])
                self._data_cache[dataset_key] = df
            except FileNotFoundError:
                raise FileNotFoundError(f"数据文件 {dataset_key} 不存在于 {self.data_sources[dataset_key]}")
        return self._data_cache[dataset_key]

    def _check_country_existence(self, dataset: pd.DataFrame, country_col: str = 'Country Code') -> bool:
        """检查国家代码是否存在"""
        return self.country_code in dataset[country_col].values

    def _check_year_value(self, dataset: pd.DataFrame, year: Union[int, str], country_col: str = 'Country Code') -> bool:
        """检查指定年份数据是否存在"""
        year_col = str(year)
        if year_col not in dataset.columns:
            return False

        country_data = dataset[dataset[country_col] == self.country_code]
        if country_data.empty:
            return False

        value = country_data[year_col].values[0]
        return not pd.isnull(value)

    def validate_field(self, field: str, check_year: int = None) -> Dict:
        """验证单个字段完整性"""
        validation_map = {
            'InitialCapitalStock': {
                'dataset': 'physical_ppp',
                'checks': [
                    {'type': 'country_existence', 'col': 'Country Code'},
                    {'type': 'year_value', 'col': check_year}
                ]
            },
            'GDP': {
                'dataset': 'gdp',
                'checks': [
                    {'type': 'country_existence', 'col': 'Country Code'},
                    {'type': 'year_value', 'col': check_year}
                ]
            },
            'HealthExpenditure': {
                'dataset': 'hepc',
                'checks': [
                    {'type': 'country_existence', 'col': 'Country Code'},
                    {'type': 'year_value', 'col': check_year}
                ]
            },
            'SavingRate': {
                'dataset': 'savings',
                'checks': [
                    {'type': 'country_existence', 'col': 'Country Code'},
                    {'type': 'year_value', 'col': check_year}
                ]
            }
        }

        if field not in validation_map:
            raise ValueError(f"不支持的字段检查: {field}")

        config = validation_map[field]
        dataset = self._load_dataset(config['dataset'])
        logs = []
        is_valid = True

        for check in config['checks']:
            if check['type'] == 'country_existence':
                exists = self._check_country_existence(dataset, check['col'])
                if not exists:
                    logs.append(f"国家代码 {self.country_code} 在 {config['dataset']} 数据中不存在")
                    is_valid = False
                    break

            elif check['type'] == 'year_value' and check['col'] is not None:
                year_valid = self._check_year_value(dataset, check['col'])
                if not year_valid:
                    logs.append(f"在 {config['dataset']} 中缺少 {check['col']} 年数据")
                    is_valid = False

        return {
            'field': field,
            'is_valid': is_valid,
            'logs': logs
        }

    def validate(self, fields_to_check: List[str], check_years: Dict[str, int] = None) -> Dict:
        """
        执行完整性检查
        :param fields_to_check: 需要检查的字段列表
        :param check_years: 需要检查的年份映射，格式为 {字段: 年份}
        :return: 验证结果字典
        """
        self.validation_report = {
            'missing_fields': [], 
            'detailed_logs': [],
            'required_fields': fields_to_check  # 添加这个键
        }

        for field in fields_to_check:
            result = self.validate_field(field, check_years.get(field) if check_years else None)
            if not result['is_valid']:
                self.validation_report['missing_fields'].append(field)
                self.validation_report['detailed_logs'].extend(result['logs'])
        
        # 添加is_complete键到validation_report
        self.validation_report['is_complete'] = len(self.validation_report['missing_fields']) == 0
        self.validation_report['country_code'] = self.country_code

        return {
            'country_code': self.country_code,
            'is_complete': self.validation_report['is_complete'],
            'missing_fields': self.validation_report['missing_fields'],
            'logs': self.validation_report['detailed_logs'],
            'required_fields': fields_to_check
        }

    def generate_report(self) -> str:
        """生成可读性报告"""
        report = [
            f"数据完整性检查报告 - 国家: {self.country_code}",
            "="*50,
            f"检查字段: {', '.join(self.validation_report.get('required_fields', []))}",
            f"完整状态: {'通过' if self.validation_report['is_complete'] else '未通过'}",
            ""
        ]

        if self.validation_report['missing_fields']:
            report.append("缺失字段:")
            for field in self.validation_report['missing_fields']:
                report.append(f"  - {field}")
            
            report.append("\n详细日志:")
            for log in self.validation_report['logs']:
                report.append(f"  • {log}")

        return "\n".join(report)
    
if __name__ == '__main__':
    data_sources = {
        'physical_ppp': './database/data/physical_lyh_update.csv',
        'gdp': "./database/data/GDP_lyh_update.csv",
        'hepc': "./database/data/hepc_lyh_update.csv",
        'savings': "./database/data/savings_lyh_update.csv"
    }
    # data_sources = {
    #     'physical_ppp': 'physical_lyh_update.csv',
    #     'gdp': "GDP_lyh_update.csv",
    #     'hepc': "hepc_lyh_update.csv",
    #     'savings': "savings_lyh_update.csv"
    # }
    # 单个国家检查示例
    # checker = DataIntegrityChecker('TJK', data_sources)
    # result = checker.validate(
    #     fields_to_check=['InitialCapitalStock', 'GDP', 'HealthExpenditure', 'SavingRate'],
    #     check_years={'InitialCapitalStock':None, 
    #                  'GDP': None}
    # )
    # print(result)
    # print(checker.generate_report())

    # 检查所有国家
    all_country_code=['AFG', 'AGO', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ASM', 'ATG', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BFA', 'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'BRN', 'BTN', 'BWA', 'CAF', 'CAN', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', 'COD', 'COG', 'COK', 'COL', 'COM', 'CPV', 'CRI', 'CUB', 'CYP', 'CZE', 'DEU', 'DJI', 'DMA', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FRA', 'FSM', 'GAB', 'GBR', 'GEO', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ', 'GRC', 'GRD', 'GRL', 'GTM', 'GUM', 'GUY', 'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA', 'JAM', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KIR', 'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LKA', 'LSO', 'LTU', 'LUX', 'LVA', 'MAR', 'MCO', 'MDA', 'MDG', 'MDV', 'MEX', 'MHL', 'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MNP', 'MOZ', 'MRT', 'MUS', 'MWI', 'MYS', 'NAM', 'NER', 'NGA', 'NIC', 'NIU', 'NLD', 'NOR', 'NPL', 'NRU', 'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PLW', 'PNG', 'POL', 'PRI', 'PRK', 'PRT', 'PRY', 'PSE', 'QAT', 'ROU', 'RUS', 'RWA', 'SAU', 'SDN', 'SEN', 'SGP', 'SLB', 'SLE', 'SLV', 'SMR', 'SOM', 'SRB', 'SSD', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SYC', 'SYR', 'TCD', 'TGO', 'THA', 'TJK', 'TKL', 'TKM', 'TLS', 'TON', 'TTO', 'TUN', 'TUR', 'TUV', 'TWN', 'TZA', 'UGA', 'UKR', 'URY', 'USA', 'UZB', 'VCT', 'VEN', 'VIR', 'VNM', 'VUT', 'WSM', 'YEM', 'ZAF', 'ZMB', 'ZWE']
    
    # 存储所有国家的检查结果
    all_results = []
    complete_countries = []
    incomplete_countries = []
    
    # 检查每个国家
    for country_code in all_country_code:
        try:
            checker = DataIntegrityChecker(country_code, data_sources)
            result = checker.validate(
                fields_to_check=['InitialCapitalStock', 'GDP', 'HealthExpenditure', 'SavingRate'],
                check_years={'InitialCapitalStock': None, 'GDP': None}
            )
            all_results.append(result)
            
            # 分类完整和不完整的国家
            if result['is_complete']:
                complete_countries.append(country_code)
            else:
                incomplete_countries.append(country_code)
                
            # 打印每个国家的检查进度
            print(f"已检查: {country_code} - {'完整' if result['is_complete'] else '不完整'}")
        except Exception as e:
            print(f"检查 {country_code} 时出错: {str(e)}")
    
    # 生成汇总报告
    print("\n" + "="*50)
    print("数据完整性检查汇总报告")
    print("="*50)
    print(f"检查的国家总数: {len(all_country_code)}")
    print(f"数据完整的国家数: {len(complete_countries)}")
    print(f"数据不完整的国家数: {len(incomplete_countries)}")
    
    # 显示数据完整的国家
    print("\n数据完整的国家:")
    for i, country in enumerate(complete_countries):
        print(f"{country}", end=", " if (i+1) % 10 != 0 and i != len(complete_countries)-1 else "\n")
        if (i+1) % 10 == 0 and i != len(complete_countries)-1:
            print()
    
    # 显示数据不完整的国家及缺失字段
    print("\n数据不完整的国家及缺失字段:")
    for country_code in incomplete_countries:
        result = next((r for r in all_results if r['country_code'] == country_code), None)
        if result:
            missing = ", ".join(result['missing_fields'])
            print(f"{country_code}: 缺失 {missing}")
    
    # 保存结果到CSV文件
    try:
        import csv
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"country_data_integrity_check_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['国家代码', '是否完整', '缺失字段', '详细日志']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in all_results:
                writer.writerow({
                    '国家代码': result['country_code'],
                    '是否完整': '是' if result['is_complete'] else '否',
                    '缺失字段': ', '.join(result['missing_fields']),
                    '详细日志': '; '.join(result['logs'])
                })
        
        print(f"\n详细检查结果已保存到: {filename}")
    except Exception as e:
        print(f"保存CSV文件时出错: {str(e)}")