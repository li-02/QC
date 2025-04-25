from utils import numpy as np

def set_data_nan(data, condition, column_name):
    data.loc[condition, column_name] = np.nan
    return data