import pandas as pd


def merge_csv_by_record_time(csv1_path: str, csv2_path: str) -> pd.DataFrame:
    """
    根据 record_time 列合并两个 CSV 文件，并使用较晚的开始时间进行同步。

    参数：
        csv1_path (str): 第一个 CSV 文件路径
        csv2_path (str): 第二个 CSV 文件路径

    返回：
        pd.DataFrame: 合并后的 DataFrame，包含两个 CSV 文件的内容
    """
    # 读取两个 CSV 文件
    df1 = pd.read_csv(csv1_path, parse_dates=["record_time"])
    df2 = pd.read_csv(csv2_path, parse_dates=["record_time"])

    # 获取两个文件中 record_time 的较晚起始时间
    start_date = max(df1["record_time"].min(), df2["record_time"].min())

    # 过滤数据，仅保留从较晚起始时间开始的记录
    df1_filtered = df1[df1["record_time"] >= start_date]
    df2_filtered = df2[df2["record_time"] >= start_date]

    # 基于 record_time 进行外连接合并
    merged_df = pd.merge(df1_filtered, df2_filtered, on="record_time", how="outer")

    # 按照时间排序
    merged_df.sort_values("record_time", inplace=True)

    return merged_df


if __name__ == "__main__":
    # 示例文件路径，请根据实际情况修改
    common_indicators_path = "../data/2023_common_indicators.csv"
    target_columns_path = "../data/2023_shisanling_pm2_5_qc_data.csv"

    # 执行合并操作
    merged_result = merge_csv_by_record_time(
        common_indicators_path, target_columns_path
    )

    # 将结果保存到第二个 CSV 路径中（可根据需要修改）
    merged_result.to_csv(target_columns_path, index=False)
