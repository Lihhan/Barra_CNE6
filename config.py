import configparser
from os.path import expanduser

# get the data source path in the whole models

config = configparser.ConfigParser()
config.read(expanduser("~")+'/myproject/myBarra/config.ini')

industry_index_path = config.get('Industry_Partition','industry_index_path')

style_factor_path_CNE6 = config.get('getfactorreturn2','style_factor_path_CNE6')

src_mkt_equd = config.get('getfactorreturn2','src_mkt_equd')
src_indi_gh = config.get('getfactorreturn2','src_indi_gh')
md_security = config.get('getfactorreturn2','md_security')

calendar_path = config.get('getfactorreturn2','calendar_path')

neg_val_path = config.get('writeFactorreturns', 'neg_val_path')

IDX_path = config.get('writeFactorreturns', 'IDX_path')

returns_df_path = config.get('writeFactorreturns', 'returns_df_path')

equ_shares_df_path = config.get('writeFactorreturns', 'equ_shares_df_path')

PE_df_path = config.get('writeFactorreturns', 'PE_df_path')
