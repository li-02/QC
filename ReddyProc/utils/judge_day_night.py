from utils import pandas as pd
from utils import numpy as np

def judge_day_night(data, ppfd_1_1_1='Par_f', ppfd_1_1_1_threshold=5):
    """
    新增一列 is_day_night, 1表示白天,0表示夜晚
    """
    data['is_day_night'] = np.NAN
    if ppfd_1_1_1 not in data.columns.tolist():
        print('数据表缺失 ppfd_1_1_1 无法判断白天黑夜')
    else:
        data[ppfd_1_1_1] = data[ppfd_1_1_1].astype('float')
        data.loc[(data[ppfd_1_1_1] > ppfd_1_1_1_threshold) & (pd.isnull(data['is_day_night'])), 'is_day_night'] = 1
        data.loc[(data[ppfd_1_1_1] <= ppfd_1_1_1_threshold) & (pd.isnull(data['is_day_night'])), 'is_day_night'] = 0
    return data
