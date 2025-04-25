from utils import pandas as pd
import utils.r_scripts as r_script
from utils import pandas2ri
from utils import robjects as ro
from rpy2.robjects.conversion import localconverter

pandas2ri.activate()
"""
这里主要是调用R语言的包来完成这步工作
"""
need_index = format_list = ('co2_flux_add_strg', 'h2o_flux_add_strg',
                            'le_add_strg', 'h_add_strg', 'rh', 'rg', 'vpd',
                            'record_time', 'tair', 'tsoil', 'ustar')

re_format_list = ('NEE', 'H2O', 'LE', 'H', 'rH', 'Rg', 'VPD', 'DateTime',
                  'Tair', 'Tsoil', 'Ustar')

del_list = ('NEE_orig', 'H2O_orig', 'LE_orig', 'H_orig', 'Tair_orig',
            'Tsoil_orig', 'VPD_orig', 'Rg_orig')

no_use_list = ['co2_flux', 'h2o_flux', 'le', 'h']


def ustar_data(file_name, longitude, latitude, timezone, data, qc_indicators):
    # 先把record_time格式转了
    # 这里数据应该都是float的了
    data['record_time'] = pd.to_datetime(data['record_time'])


    # 把r_data的参数名称变r的默认形式
    data = data.rename(columns={'record_time': 'DateTime'})

    data['NEE'] = data['co2_despiking']
    data['rH'] = data['rh_threshold_limit']
    data['Rg'] = data['rg_1_1_2_threshold_limit']
    data['Tair'] = data['ta_1_2_1_threshold_limit']
    # VPD pa to hpa
    data['VPD'] = data['vpd_threshold_limit'] * 0.01

    # 先把传入的参数给处理掉
    # print(longitude, latitude, timezone)
    r_filename = r_script.my_robjects.StrVector([file_name])
    r_longitude = r_script.my_robjects.FloatVector([longitude])
    r_latitude = r_script.my_robjects.FloatVector([latitude])
    r_timezone = r_script.my_robjects.IntVector([timezone])
    data_r = pandas2ri.py2rpy(data)

    # gapfilling indicators
    gapfill_indicators = []
    for i in qc_indicators:
        if i['is_gapfill'] == 1 and i['code'] not in no_use_list:
            if i['belong_to'] == 'flux' and i['code'] in data.columns:
                gapfill_indicators.append(i['code'] + '_threshold_limit')
    gapfill_indicators += [
        'h2o_despiking', 'le_despiking', 'h_despiking'
    ]
    # 然后根据 flux_type 的类型用不同的r脚本处理数据
    result_data = r_script.my_robjects.r['r_co2_flux'](r_filename, r_longitude,
                                                       r_latitude, r_timezone,
                                                       data_r,
                                                       gapfill_indicators)

    # 这里ns 和ns Shanghai 不可直接合并，所以要转一下
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_data = ro.conversion.rpy2py(result_data)

    result_data = result_data.rename(columns={'DateTime': 'record_time'})
    result_data['record_time'] = result_data['record_time'].dt.tz_localize(
        None)
    result_data.set_index('record_time')
    del_list = ['rH','Rg','Tair','VPD']
    for item in result_data.columns.tolist():
        if item in del_list:
            del result_data[item]
    result_data = delete_format(result_data)

    return result_data


def formatter(data):
    """
    把数据库的indicator name 变成 r那边的default name
    """
    trans_list = dict(zip(data.columns.tolist(), re_format_list))
    # trans_list = dict(zip(format_list, re_format_list))
    for item in data.columns.tolist():
        if item in format_list:
            data = data.rename(columns={item: trans_list[item]})
            if item != 'record_time':
                data[trans_list[item]] = data[trans_list[item]].astype('float')
    return data


def delete_format(data):
    # print(data.columns)
    for item in data.columns.tolist():
        if item.split("_")[-1] == 'orig':
            del data[item]
        else:
            data = data.rename(columns={item: item.lower()})
    # print(data.columns)
    return data


if __name__ == "__main__":

    import datetime

    file_path = r"/Users/luffy/Desktop/BEON/r_test/badaling (2019-2020).txt"

    def parse(yr, doy, hr):
        yr, doy = [int(x) for x in [yr, doy]]
        hr = float(hr)
        dt = datetime.datetime(yr - 1, 12, 31)
        delta = datetime.timedelta(days=doy, hours=hr)
        return dt + delta

    # pd_data = pd.read_csv(file_path, sep='\t')
    pd_data = pd.read_csv(file_path,
                          sep='\t',
                          parse_dates={'record_time': ['Year', 'DoY', 'Hour']},
                          date_parser=parse)
    # print(pd_data)
    # ustar_data('h2o_flux', 'test', 115.94, 40.37, 8, pd_data)
    pd_data.to_csv("test_data.csv")
