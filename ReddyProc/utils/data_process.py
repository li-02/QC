from utils import re
from utils import pandas as pd
from utils import numpy as np
from utils import robjects, pandas2ri,ro,localconverter
from judge_day_night import judge_day_night
from add_window_tag import add_window_tag
from calculate_diff import calculate_diff
from calculate_md_mad import md_method,mad_method
from set_data_nan import set_data_nan

def do_add_strg(self, data):
    # 定义需要处理的变量列表
    variables = [
        ('co2_flux', 'co2_flux_filter_label', 'co2_flux_strg', 'co2_flux_add_strg'),
        ('h2o_flux', 'h2o_flux_filter_label', 'h2o_flux_strg', 'h2o_flux_add_strg'),
        ('le', 'le_filter_label', 'le_strg', 'le_add_strg'),
        ('h', 'h_filter_label', 'h_strg', 'h_add_strg')
    ]

    for _, filter_col, strg_col, result_col in variables:
        if filter_col in data.columns:
            data[filter_col] = data[filter_col].astype('float')
            if strg_col in data.columns:
                data[strg_col] = data[strg_col].astype('float')
                data[result_col] = data[filter_col] + data[strg_col]

    return data

def not_add_strg(self, data):
    # 定义需要处理的变量列表（简化版本）
    variables = [
        ('co2_flux_filter_label', 'co2_flux_add_strg'),
        ('h2o_flux_filter_label', 'h2o_flux_add_strg'),
        ('le_filter_label', 'le_add_strg'),
        ('h_filter_label', 'h_add_strg')
    ]

    for filter_col, result_col in variables:
        if filter_col in data.columns:
            data[filter_col] = data[filter_col].astype('float')
            data[result_col] = data[filter_col]

    return data

def copy_flux_columns_without_qc_filter(data: pd.DataFrame):
    """直接复制通量列而不进行QC标记过滤"""
    data['co2_flux_filter_label'] = data['co2_flux']
    data['h2o_flux_filter_label'] = data['h2o_flux']
    data['le_filter_label'] = data['le']
    data['h_filter_label'] = data['h']

    data['qc_co2_flux'] = data['qc_co2_flux'].astype('float')
    data['qc_h2o_flux'] = data['qc_h2o_flux'].astype('float')
    data['qc_le'] = data['qc_le'].astype('float')
    data['qc_h'] = data['qc_h'].astype('float')
    return data

def handle_campbell_special_case(data: pd.DataFrame):
    """处理Campbell站点特殊情况(缺少某些通量数据)"""
    data['co2_flux_filter_label'] = data['co2_flux']

    # 八达岭、奥森没有h2o_flux、qc_co2_flux、qc_h2o_flux
    data['h2o_flux'] = (data['le'].astype(float)) / 2450 / 18 * 1000
    data['h2o_flux_filter_label'] = data['h2o_flux']

    data['le_filter_label'] = data['le']
    data['h_filter_label'] = data['h']

    return data

def filter_flux_by_qc_flags(data, excluded_qc_flags):
    """根据QC标记过滤通量数据"""
    data['co2_flux_filter_label'] = data['co2_flux']
    data['h2o_flux_filter_label'] = data['h2o_flux']
    data['le_filter_label'] = data['le']
    data['h_filter_label'] = data['h']
    try:
        data.loc[data['qc_co2_flux'].isin(excluded_qc_flags), 'co2_flux_filter_label'] = np.NAN
        data.loc[data['qc_h2o_flux'].isin(excluded_qc_flags), 'h2o_flux_filter_label'] = np.NAN
        data.loc[data['qc_le'].isin(excluded_qc_flags), 'le_filter_label'] = np.NAN
        data.loc[data['qc_h'].isin(excluded_qc_flags), 'h_filter_label'] = np.NAN

        data['qc_co2_flux'] = data['qc_co2_flux'].astype('float')
        data['qc_h2o_flux'] = data['qc_h2o_flux'].astype('float')
        data['qc_le'] = data['qc_le'].astype('float')
        data['qc_h'] = data['qc_h'].astype('float')

        return data
    except Exception as e:
        print(e)

def threshold_limit(data, qc_indicators, data_type):
    """ 阈值过滤

    Args:
        data (dataframe): data
        qc_indicators (list): all indicators

    Returns:
        dataframe: result
    """
    if data_type == 'flux':
        try:
            # 特殊变量与其对应的 add_strg 和 threshold_limit 列名的映射
            special_vars = {
                'co2_flux': {'add_strg': 'co2_flux_add_strg', 'threshold': 'co2_flux_threshold_limit'},
                'h2o_flux': {'add_strg': 'h2o_flux_add_strg', 'threshold': 'h2o_flux_threshold_limit'},
                'le': {'add_strg': 'le_add_strg', 'threshold': 'le_threshold_limit'},
                'h': {'add_strg': 'h_add_strg', 'threshold': 'h_threshold_limit'},
            }
             # 创建一个字典来快速查找 qc_indicators 中的值
            qc_dict = {item['code']: {'lower': float(item['qc_lower_limit']), 'upper': float(item['qc_upper_limit'])} for item in qc_indicators}
            for col in data.columns:
                if col in qc_dict:
                    limits=qc_dict[col]
                    if col in special_vars:
                        # 对特殊变量处理
                        add_strg_col=special_vars[col]['add_strg']
                        threshold_col=special_vars[col]['threshold']
                        if add_strg_col in data.columns:
                            condition=(data[add_strg_col]<limits['lower'])|(data[add_strg_col]>limits['upper'])
                            data[threshold_col]=data[add_strg_col]
                            data.loc[condition, threshold_col]=np.NAN
                    else:
                        # 对其他变量处理
                        threshold_col=col + "_threshold_limit"
                        condition=(data[col]<limits['lower'])|(data[col]>limits['upper'])
                        data[threshold_col]=data[col]
                        data.loc[condition, threshold_col]=np.NAN
            return data
        except Exception as e:
            print(e)
    else:
        # 先转成float
        for i in data.columns:
            if i != 'record_time':
                print(data[i])
                data[i] = data[i].astype('float')
        try:
            for i in qc_indicators:
                for j in data.columns:
                    if data_type == 'aqi':
                        if re.sub(r"\W", "_", i["en_name"]).lower() == j:
                            condition = (data[j] < float(i['qc_lower_limit'])) | (data[j] > float(i['qc_upper_limit']))
                            data[j + "_threshold_limit"] = data[j]
                            data.loc[condition, j + "_threshold_limit"] = np.NAN
                    else:
                        if i['code'] == j:
                            condition = (data[j] < float(i['qc_lower_limit'])) | (data[j] > float(i['qc_upper_limit']))
                            data[j + "_threshold_limit"] = data[j]
                            data.loc[condition, j + "_threshold_limit"] = np.NAN
        except Exception as e:
            print(e)
        if data_type == 'sapflow':
            for i in qc_indicators:
                for j in data.columns:
                    # 如果列名以 tc_dtca_ 开头，执行 del_abnormal_data 函数
                    if i['code'] == j and j.startswith('tc_dtca_'):
                        data = del_abnormal_data_sapflow(data, ta_name="ta_1_2_1_threshold_limit", daca_name=j + "_threshold_limit")
            # 茎流速率 用5倍标准差再筛选一遍数据
            data.to_csv('weishanzhuang.csv',index=False)
            print(data.columns)
            print(data.dtypes)
            data = standard_deviation_limit(data)
        elif data_type == 'aqi':
            # 补半点数据将用前后整点数据的均值来插补，若前后至少有一个是NaN那么这个半点的数据就是NaN
            # 将时间设为index
            data = data.set_index(pd.to_datetime(data['record_time'])).drop(
                'record_time', axis=1)
            # 补全时间序列 半点数据置为NaN
            data = data.resample('30min').mean()
            # 将半点的值置为前后整点数据的均值
            for i in range(data.shape[0]):
                if data.iloc[i].name.minute == 30:
                    data.iloc[i] = (data.iloc[i - 1] + data.iloc[i + 1]) / 2
            data = data.reset_index()
        return data
    
def del_abnormal_data_sapflow(raw_data, ta_name="ta_1_2_1_threshold_limit", daca_name="tc_dtca_1__threshold_limit"):
    df = raw_data.copy()
    df[daca_name + "_old"] = df[daca_name]

    if ta_name in df.columns and 'record_time' in df.columns:
        df['record_time'] = pd.to_datetime(df['record_time'])
        df['date'] = df['record_time'].dt.date

        # 计算每日的平均温度
        daily_avg_temp = df.groupby('date')[ta_name].mean()

        daily_df = pd.DataFrame({
            'date': daily_avg_temp.index,
            'day_avg_tair': daily_avg_temp.values
        })

        # 以天为单位对 'ta' 列进行滚动平均，并确保窗口包含3天的数据
        daily_df['ta_three_avg'] = daily_df['day_avg_tair'].rolling(window=3, min_periods=3, center=True).mean()

        # 前 和 后 赋值最近的值
        first_non_nan = daily_df['ta_three_avg'].first_valid_index()
        last_non_nan = daily_df['ta_three_avg'].last_valid_index()
        if last_non_nan is not None:
            last_non_nan_value = daily_df.at[last_non_nan, 'ta_three_avg']
        else:
            last_non_nan_value = pd.NA  # 如果整个列都是 NaN，则这里也是 NaN

        # 用第一个非NaN值填充前面的NaN
        daily_df['ta_three_avg'] = daily_df['ta_three_avg'].fillna(method='bfill',
                                                                   limit=daily_df.index.get_loc(first_non_nan))

        # 用最后一个非NaN值填充后面的NaN
        daily_df['ta_three_avg'] = daily_df['ta_three_avg'].fillna(last_non_nan_value)

        # 添加是否是生长季的列（温度是否小于5）
        daily_df['is_grow_season'] = (daily_df['ta_three_avg'] >= 5).astype(int)

        df = df.merge(daily_df[['date', 'is_grow_season']], on='date', how='left')
        del df['date']

        # 在生长季剔除不在 [3, 12] 范围内的数据
        condition = (df['is_grow_season'] == 1) & ((df[daca_name] < 3) | (df[daca_name] > 12))
        df.loc[condition, daca_name] = pd.NA

        # 删除 'is_grow_season' 列
        del df['is_grow_season']

    return df

def standard_deviation_limit(data):
    """sapflow 5 times Standard deviation limit
        滑动窗口是480 然后滑动范围可以滑一天96
    Args:
        data (df): the pandas dataframe
        完了 一个月后看不懂自己当初写的是啥了😭
        关键代码还是要写注释....
    """
    sapflow_data = pd.DataFrame()
    #sapflow_data['record_time'] = data['record_time']
    for i in data.columns.tolist():
        if i.split("_")[-1] == 'limit':
            sapflow_data[i+'_std'] = data[i]
    index = 0
    while (index < sapflow_data.shape[0]):
        if (index + 479) > (sapflow_data.shape[0]):
            break

            # 打印当前的 index 值
        print(f"Index: {index}")

        # 计算当前窗口内的数据的均值和标准差
        window_mean = sapflow_data.iloc[index:index + 480].mean()
        window_std = sapflow_data.iloc[index:index + 480].std()

        print(f"window_mean-----{window_mean}")
        print(f"window_std------{window_std}")
        # 检查均值是否全是 NaN
        if window_mean.isna().all():
            print(f"Skipping index {index} due to all NaN mean values.")
            index += 96
            continue

        sapflow_data[(sapflow_data.iloc[index:index + 480] >
              (sapflow_data.iloc[index:index + 480].mean() + sapflow_data.iloc[index:index + 480].std())) |
             (sapflow_data.iloc[index:index + 480] < (sapflow_data.iloc[index:index + 480].mean() -
                                      sapflow_data.iloc[index:index + 480].std()))] = np.nan
        index += 96
    sapflow_data['record_time']=data['record_time']
    full_data = pd.merge(data, sapflow_data, how='outer', on='record_time')
    # std 筛选后，sapflow只保留00和30分的数据，15的和45的就不要了，数据产品暂时都是30分钟的
    full_data['record_time'] = pd.to_datetime(full_data['record_time'])
    new_data = full_data.drop(full_data[full_data['record_time'].apply(
        lambda x: x.minute == 15 or x.minute == 45)].index)
    return new_data


def gap_fill_par(file_name,longitude,latitude,timezone,data):
    """
    插补par光合有效辐射
    Args:
        file_name (str): 文件名
        longitude (float): 经度
        latitude (float): 纬度
        timezone (int): 时区
        data (pd.DataFrame): 数据
    """
    # 将数据转化成R语言的类型
    data['record_time'] = pd.to_datetime(data['record_time'])
    data=data.rename(columns={"record_time": "DateTime"})
    data['rH'] = data['rh_threshold_limit'] if 'rh_threshold_limit' in data.columns else None
    data['Rg'] = data['rg_1_1_2_threshold_limit'] if 'rg_1_1_2_threshold_limit' in data.columns else None
    data['Tair'] = data['ta_1_2_1_threshold_limit'] if 'ta_1_2_1_threshold_limit' in data.columns else None
    # vpd pa to hpa
    data['VPD'] = data['vpd_threshold_limit'] * 0.01 if 'vpd_threshold_limit' in data.columns else None
    data['Par'] = data['ppfd_1_1_1_threshold_limit'] if 'ppfd_1_1_1_threshold_limit' in data.columns else None
 
    r_filename=robjects.StrVector([file_name])
    r_longitude=robjects.FloatVector([longitude])
    r_latitude=robjects.FloatVector([latitude])
    r_timezone=robjects.IntVector([timezone])
    data_r=pandas2ri.py2ri(data)
    result_data=robjects.r['r_gap_fill_par'](r_filename,r_longitude,r_latitude,r_timezone,data_r)

    with localconverter(ro.default_converter+pandas2ri.converter):
        result_data=ro.conversion.rpy2py(result_data)
    result_data.rename(columns={"DateTime":"record_time"},inplace=True)
    result_data["record_time"]=result_data["record_time"].dt.tz_localize(None)
    result_data.set_index("record_time")
    result_data.drop(["rH", "Rg", "Tair", "VPD", "Par"], axis=1, inplace=True)
    return result_data



def despiking_data(data, despiking_z):
    """
    Parameters:
    ----------
        data : dataframe
            插补后的par数据
        despiking_z : int 
            z

    Returns:
        data: df
            去峰值后的数据
    """

    # 1.判断白天黑夜，老版代码有两个判断方式，但这边会直接在一个值中直接判断
    # 如果总辐射 global_radiation(rg_1_1_2) > 20 这个值没有，就用 入射光合有效辐射(ppfd_1_1_1 >5)
    # 如果两个都没有，只能用最死板的 daytime(白天1晚上0)

    data['co2_despiking'] = data['co2_flux_threshold_limit']
    data['h2o_despiking'] = data['h2o_flux_threshold_limit']
    data['le_despiking'] = data['le_threshold_limit']
    data['h_despiking'] = data['h_threshold_limit']

    data = judge_day_night(data)

    # 2.加上window标签
    # 这里注意window data是 flux 值不为NaN的条目！，先get到这些条目去设置这个mad和md，
    # 再在原始条目中根据这个来设nan，最后存下来的就是时间齐全而且去掉这些条件的数据啦

    co2_window_data = data[data['co2_despiking'].notnull()].reset_index(drop=True)
    h2o_window_data = data[data['h2o_despiking'].notnull()].reset_index(drop=True)
    le_window_data = data[data['le_despiking'].notnull()].reset_index(drop=True)
    h_window_data = data[data['h_despiking'].notnull()].reset_index(drop=True)

    co2_window_data, co2_window_size, co2_window_nums = add_window_tag(co2_window_data)
    h2o_window_data, h2o_window_size, h2o_window_nums = add_window_tag(h2o_window_data)
    le_window_data, le_window_size, le_window_nums = add_window_tag(le_window_data)
    h_window_data, h_window_size, h_window_nums = add_window_tag(h_window_data)

    # 3.根据每个window的值计算对应的MAD和Md
    variables_config = {
        'co2': {'data': co2_window_data, 'window_nums': co2_window_nums},
        'h2o': {'data': h2o_window_data, 'window_nums': h2o_window_nums},
        'le': {'data': le_window_data, 'window_nums': le_window_nums},
        'h': {'data': h_window_data, 'window_nums': h_window_nums}
    }
    for var_name,window_data in variables_config.items():
        window_nums=window_data['window_nums']
        data = process_variable_despiking(data, window_data['data'], var_name, window_nums, despiking_z)
    return data

def process_variable_despiking(data, window_data, var_name, window_nums, despiking_z):
    """
    对指定变量进行去尖峰处理
    
    参数:
    data: 主数据集
    window_data: 按窗口划分的数据
    var_name: 变量名称（如'co2', 'h2o'等）
    window_nums: 窗口数量
    despiking_z: 去尖峰的z系数
    
    返回:
    处理后的主数据集
    """
    diff_col = f"{var_name}_diff"
    md_col = f"{var_name}_Md"
    mad_col = f"{var_name}_MAD"
    despiking_col = f"{var_name}_despiking"
    
    # 预先创建diff列，避免重复创建
    if diff_col not in window_data.columns:
        window_data[diff_col] = np.nan
    
    for i in range(window_nums):
        # 基于窗口ID和白天/黑夜标志筛选数据
        window_condition = window_data['windowID'] == i
        day_condition = window_condition & (window_data['is_day_night'] == 1)
        night_condition = window_condition & (window_data['is_day_night'] == 0)
        
        # 一次性获取白天和夜晚数据，避免重复筛选
        window_data_D = window_data[day_condition]
        window_data_N = window_data[night_condition]
        
        # 计算白天和夜晚的差分
        if not window_data_D.empty:
            temp_diff = calculate_diff(window_data_D, despiking_col)
            window_data.loc[temp_diff.index, diff_col] = temp_diff
        
        if not window_data_N.empty:
            temp_diff = calculate_diff(window_data_N, despiking_col)
            window_data.loc[temp_diff.index, diff_col] = temp_diff
        
        # 重新获取更新后的白天和夜晚数据
        window_data_D = window_data[day_condition]
        window_data_N = window_data[night_condition]
        
        # 计算MD和MAD
        window_data = md_method(window_data_D, window_data_N, window_data, var_name)
        
        # 重新获取更新后的白天和夜晚数据
        window_data_D = window_data[day_condition]
        window_data_N = window_data[night_condition]
        window_data = mad_method(window_data_D, window_data_N, window_data, var_name)
        
        # 计算并标记峰值
        di_low_range = window_data[md_col] - (despiking_z * window_data[mad_col]) / 0.6745
        di_high_range = window_data[md_col] + (despiking_z * window_data[mad_col]) / 0.6745
        
        # 使用向量化操作检测条件
        condition = (window_data[diff_col] < di_low_range) | (window_data[diff_col] > di_high_range)
        condition = condition & window_condition  # 只考虑当前窗口
        
        if condition.any():
            # 只处理符合条件的记录
            spike_times = window_data.loc[condition, 'record_time'].tolist()
            data_condition = data['record_time'].isin(spike_times)
            data = set_data_nan(data, data_condition, despiking_col)
    
    return data




def del_abnormal_data(raw_data, ta_name="ta_1_2_1_threshold_limit", par_name="ppfd_1_1_1_threshold_limit",
                      nee_name="co2_flux_threshold_limit"):

    df = judge_day_night(data=raw_data, ppfd_1_1_1=par_name)
    # 新的一行
    df[nee_name + "_old"] = df[nee_name]
    if ta_name in df.columns and 'record_time' in df.columns:
        df['record_time'] = pd.to_datetime(df['record_time'])

        df['date'] = df['record_time'].dt.date
        # 计算每日的平均温度
        daily_avg_temp = df.groupby('date')[ta_name].mean()

        daily_df = pd.DataFrame({
            'date': daily_avg_temp.index,
            'day_avg_tair': daily_avg_temp.values
        })

        # 以天为单位对 'ta' 列进行滚动平均，并确保窗口包含3天的数据
        daily_df['ta_three_avg'] = daily_df['day_avg_tair'].rolling(window=3, min_periods=3, center=True).mean()
        # 前 和 后 赋值最近的值
        # 找到第一个非NaN值的位置
        first_non_nan = daily_df['ta_three_avg'].first_valid_index()

        # 找到最后一个非NaN值
        last_non_nan = daily_df['ta_three_avg'].last_valid_index()
        if last_non_nan is not None:
            last_non_nan_value = daily_df.at[last_non_nan, 'ta_three_avg']
        else:
            last_non_nan_value = pd.NA  # 如果整个列都是 NaN，则这里也是 NaN

        # 用第一个非NaN值填充前面的NaN
        daily_df['ta_three_avg'] = daily_df['ta_three_avg'].fillna(method='bfill',
                                                       limit=daily_df.index.get_loc(first_non_nan))

        # 用最后一个非NaN值填充后面的NaN
        daily_df['ta_three_avg'] = daily_df['ta_three_avg'].fillna(last_non_nan_value)

        # 添加是否是生长季的列（温度是否小于5）
        daily_df['is_grow_season'] = 0
        daily_df.loc[daily_df['ta_three_avg'] >= 5, 'is_grow_season'] = 1
        df = df.merge(daily_df[['date', 'is_grow_season']].rename(columns={'is_grow_season': 'is_grow_season'}), on='date',how='left')

        del df['date']

        codition =(((df['is_day_night'] == 1) & (-1 >= df[nee_name]) & (df[nee_name] >= 1) & (df['is_grow_season'] == 0)) | 
                   ((df['is_day_night'] == 0) & (df[nee_name] < -0.2) & (df['is_grow_season'] == 0)))
        df.loc[codition, nee_name] = pd.NA

        # 输出结果，查看 DataFrame
        # print(df)
        del df['is_grow_season']
        return df
    return df
