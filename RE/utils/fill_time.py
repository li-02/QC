import numpy as np
import pandas as pd
import os


def fill_time(raw_data: pd.DataFrame, time_freq: str = "30min") -> pd.DataFrame:

    # 删除 id 字段（如果存在）
    if "id" in raw_data.columns:
        del raw_data["id"]

    # 将 'NaN', 'nan', '' 替换成 np.NAN，再填充 NaN 值
    raw_data = raw_data.replace(["NaN", "nan", ""], np.nan)
    raw_data = raw_data.fillna(np.nan)

    # 确保 record_time 转换为 datetime 类型
    raw_data["record_time"] = pd.to_datetime(raw_data["record_time"], errors="coerce")
    raw_data = raw_data.sort_values("record_time")

    # 生成完整的时间序列
    start_time = raw_data["record_time"].min()
    end_time = raw_data["record_time"].max()
    complete_time_index = pd.date_range(start=start_time, end=end_time, freq=time_freq)

    # 构建包含完整时间序列的 DataFrame
    df_complete = pd.DataFrame({"record_time": complete_time_index})

    # 将完整时间序列与原始数据按 record_time 合并（左连接）
    merged = pd.merge(df_complete, raw_data, on="record_time", how="left")

    # 对于除 record_time 之外的其他字段，如果缺失则填充为空字符串
    other_cols = [col for col in merged.columns if col != "record_time"]
    merged[other_cols] = merged[other_cols].fillna("")

    # 更新 raw_data
    raw_data = merged
    return raw_data


if __name__ == "__main__":
    # 测试代码
    raw_data_path = os.path.join(
        "D:\\", "Code", "QC", "data", "2023_shisanling_pm2_5_raw_data.csv"
    )
    raw_data = pd.read_csv(raw_data_path)
    raw_data["record_time"] = pd.to_datetime(raw_data["record_time"])
    # 使用30分钟间隔填充
    filled_data = fill_time(raw_data, time_freq="1h")
    # 另存为csv
    filled_data.to_csv(raw_data_path, index=False)
