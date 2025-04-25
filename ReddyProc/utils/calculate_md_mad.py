from utils import numpy as np

def md_method(data_D, data_N, data, value):
    '''
    # calculate MD
    :param data_D:  <dataframe>day data
    :param data_N:  <dataframe>night data
    :param data: <dataframe>data set
    :param value: <string> index need to calculate md
    :return:  <dataframe> data set including {value}_md
    '''
    # day's data_md
    Day = calculate_md(data_D, value)
    try:
        a = Day[value+'_Md'].values[0]
    except:
        a = np.nan
    data.loc[data_D.index.tolist(), value+'_Md'] = a

    # night's data_md
    Night = calculate_md(data_N, value)
    try:
        a = Night[value+'_Md'].values[0]
    except:
        a = np.nan
    data.loc[data_N.index.tolist(), value+'_Md'] = a

    # print(data[value+'_Md'])

    return data


def mad_method(data_D, data_N, data, value):
    '''
    # calculate MD
    :param data_D:  <dataframe>day data
    :param data_N:  <dataframe>night data
    :param data: <dataframe>data set
    :param value: <string> index need to calculate mad
    :return:  <dataframe> data set including {value}_mad
    '''
    try:
        a = calculate_mad(data_D, value)
    except:
        a = np.nan
    data.loc[data_D.index.tolist(), value+'_MAD'] = a
    try:
        a = calculate_mad(data_N, value)
    except:
        a = np.nan
    data.loc[data_N.index.tolist(), value+'_MAD'] = a

    return data

def calculate_md(data, value):
    '''
    calculate Md value
    :param data: <dataframe> data set including co2_Md
    :param value: index need to calculate
    :return: <dataframe> =windowID=,=daytime=,={value}_Md=
    '''
    Md_data = data.groupby(by=['windowID', 'is_day_night'], as_index=False).median()
    data = {
        'windowID': Md_data['windowID'],
        'is_day_night': Md_data['is_day_night'],
        value+'_Md': Md_data[value+'_diff']
    }
    return pd.DataFrame(data)


def calculate_mad(data, value):
    '''
    calculate MAD value
    :param data: <dataframe> data set including {value}_Md
    :param value: <value> index need to calculate MAD
    :return: <float>{value}_MAD
    '''

    MAD = (data[value+'_diff'] - data[value+'_Md']).abs().median()
    return MAD