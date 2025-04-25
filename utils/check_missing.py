"""
用于检查原始数据除了目标列，其他列是否存在缺失值
如果存在缺失值，则返回缺失值所在行
example:
TIMESTAMP,          RH	NEE_VUT_REF	TS_F_MDS_1	    SWC_F_MDS_1	    VPD_F_MDS	    TA_F_MDS	    NETRAD	    SW_IN_F
2024-01-01 0:00     0.0	NaN	        -0.0001	        -0.0001	0.0	    -0.0001	        -0.0001	        -0.0001     -0.0001

"""

import os
import pandas as pd


def check_missing_values(df, target_col):
    """
    检查 DataFrame 中除目标列外的其他列是否存在缺失值
    缺失值定义：空字符串""、仅含空格的字符串、或值为"nan"(不区分大小写)

    参数:
    df (pandas.DataFrame): 输入的数据框
    target_col (str): 目标列名

    返回:
    tuple: (表头列表, 包含缺失值的行数据的列表)
    """
    # 获取表头
    headers = list(df.columns)

    # 创建一个掩码，检查缺失值条件
    # 排除目标列
    check_columns = [col for col in df.columns if col != target_col]

    # 定义缺失值检查函数
    def is_missing(value):
        if isinstance(value, str):
            return (
                value.strip() == "" or value.lower() == "nan"  # 空字符串或仅含空格
            )  # 值转为小写后等于"nan"
        return False

    # 初始化结果列表
    missing_rows = []

    # 遍历每一行
    for index, row in df.iterrows():
        # 检查除目标列外的其他列
        for col in check_columns:
            if is_missing(str(row[col])):
                # 如果发现缺失值，添加整行数据
                missing_rows.append({"row_index": index, "values": row.to_dict()})
                break  # 一行只要有一个缺失值就记录，不需要继续检查该行

    return headers, missing_rows


def main():
    file_path = os.path.join(
        "D:\\", "Code", "QC", "data", "2024_shisanling_pm2_5_raw_data.csv"
    )
    data = pd.read_csv(file_path)
    headers, missing = check_missing_values(data, "target")
    # 打印结果
    print("表头:", headers)
    print("\n包含缺失值的行:")
    for row in missing:
        print(f"行 {row['row_index']}: {row['values']}")


# 示例用法
if __name__ == "__main__":
    main()
