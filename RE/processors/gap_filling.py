"""
数据插补模块
"""
import pandas as pd
import numpy as np
from r_scripts import robjects
from rpy2.robjects import pandas2ri, robject
from rpy2.robjects.conversion import localconverter


def gap_fill_par(file_name, longitude, latitude, timezone, data):
    """
    插补Par（光合有效辐射）
    
    Args:
        file_name: 文件名
        longitude: 经度
        latitude: 纬度
        timezone: 时区
        data: 数据DataFrame
        
    Returns:
        插补后的数据
    """
    # 将数据转化成R语言需要的格式
    data['record_time'] = pd.to_datetime(data['record_time'])
    data = data.rename(columns={"record_time": "DateTime"})
    
    # 准备R需要的变量
    data['rH'] = data['rh_threshold_limit'] if 'rh_threshold_limit' in data.columns else np.nan
    data['Rg'] = data['rg_1_1_2_threshold_limit'] if 'rg_1_1_2_threshold_limit' in data.columns else np.nan
    data['Tair'] = data['ta_1_2_1_threshold_limit'] if 'ta_1_2_1_threshold_limit' in data.columns else np.nan
    data['VPD'] = data['vpd_threshold_limit'] * 0.01 if 'vpd_threshold_limit' in data.columns else np.nan
    data['Par'] = data['ppfd_1_1_1_threshold_limit'] if 'ppfd_1_1_1_threshold_limit' in data.columns else np.nan
 
    # 转换为R对象
    r_filename = robject.StrVector([file_name])
    r_longitude = robject.FloatVector([longitude])
    r_latitude = robject.FloatVector([latitude])
    r_timezone = robject.IntVector([timezone])
    
    # 使用localconverter来转换DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        data_r = robjects.conversion.py2rpy(data)
    
    # 调用R函数
    result_data = robjects.r['r_gap_fill_par'](r_filename, r_longitude, r_latitude, r_timezone, data_r)

    # 将R结果转回Python
    with localconverter(robjects.default_converter + pandas2ri.converter):
        result_data = robjects.conversion.rpy2py(result_data)
        
    # 处理结果数据
    result_data = result_data.rename(columns={"DateTime": "record_time"})
    result_data["record_time"] = result_data["record_time"].dt.tz_localize(None)
    result_data = result_data.set_index("record_time")
    
    # 删除不需要的列
    columns_to_drop = ["rH", "Rg", "Tair", "VPD", "Par"]
    result_data = result_data.drop([col for col in columns_to_drop if col in result_data.columns], axis=1)
    
    # 重置索引
    result_data = result_data.reset_index()
    
    return result_data