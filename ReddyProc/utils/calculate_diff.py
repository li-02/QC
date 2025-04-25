def calculate_diff(data, value):
    """
    计算时间序列的二阶差分
    
    Parameters:
    ------
    data : DataFrame
        包含需要进行差分的数据
    value : str
        需要进行差分的列名
        
    Returns:
    ------
    Series
        计算后的二阶差分序列
    """
    a = data[value]
    b = a.shift(1)
    c = a.shift(-1)
    temp_value = (a-c)-(b-a)
    return temp_value