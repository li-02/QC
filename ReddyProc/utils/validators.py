from utils import os

def validate_args(args):
    """验证输入参数是否符合要求"""
    valid = True
    error_msgs = []
    
    # 检查文件路径
    if not os.path.exists(args.file_path):
        valid = False
        error_msgs.append(f"错误：文件 {args.file_path} 不存在")
    
    # 检查数据类型
    valid_data_types = ["flux", "aqi", "sapflow", "nai", "micro_meteorology"]
    if args.data_type not in valid_data_types:
        valid = False
        error_msgs.append(f"错误：数据类型 {args.data_type} 不合法，有效选项为: {', '.join(valid_data_types)}")
    
    # 检查经纬度
    if args.longitude < -180 or args.longitude > 180:
        valid = False
        error_msgs.append(f"错误：经度 {args.longitude} 超出有效范围 [-180, 180]")
        
    if args.latitude < -90 or args.latitude > 90:
        valid = False
        error_msgs.append(f"错误：纬度 {args.latitude} 超出有效范围 [-90, 90]")
    
    return valid, error_msgs