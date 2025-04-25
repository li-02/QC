def add_window_tag(data, day_size=13):
    """
    添加一列window标签序号,如果最后一个的个数不够window_size的,则算前一个window
    
    Parameters:
    -----------
    data : DataFrame
        所有数据
    day_size : int, default=13
        设定的天数
        
    Returns:
    --------
    data : DataFrame
        增加了windowID的数据
    window_size : int
        一个window大小
    window_nums : int
        windows的个数
    """
    window_size = day_size * 48
    window_nums = data.shape[0] // window_size
    data['windowID'] = data.index // window_size
    if data.shape[0] % window_size != 0:
        data.loc[data['windowID'] == window_nums, 'windowID'] = window_nums - 1
    return data, window_size, window_nums
