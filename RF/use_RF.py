import os
import pickle
import pandas as pd
import numpy as np
import argparse


def read_data(csv_file):
    """读取CSV数据并解析时间戳"""
    df = pd.read_csv(csv_file, parse_dates=["TIMESTAMP"])
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%d/%m/%Y %H:%M:%S")
    return df


def preprocess_data(df, target_column):
    """预处理数据，与训练时保持一致"""
    # 保存时间戳列，之后恢复用
    timestamps = df["TIMESTAMP"].copy()

    # 在预测前去掉时间戳列（与训练时保持一致）
    df_processed = df.drop("TIMESTAMP", axis=1)

    # 返回处理后的数据和时间戳
    return df_processed, timestamps


def fill_missing_values(df, model_path, output_path, target_column):
    """使用训练好的随机森林模型填充缺失值"""
    # 预处理数据
    df_processed, timestamps = preprocess_data(df, target_column)

    # 找出包含NaN值的目标列的行索引
    missing_idx = df_processed[target_column].isna()

    if not missing_idx.any():
        print(f"没有发现{target_column}列中的缺失值，无需填充。")
        return df

    print(f"发现{target_column}列中{missing_idx.sum()}个缺失值需要填充。")

    # 加载模型
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # 准备特征数据（去掉目标列）
    X_missing = df_processed.loc[missing_idx].drop(columns=[target_column])

    # 确保X_missing没有NaN值，如有必要可以进行简单填充
    if X_missing.isna().any().any():
        print("警告：特征中存在缺失值，使用均值填充")
        X_missing = X_missing.fillna(X_missing.mean())

    # 预测缺失值
    predicted_values = model.predict(X_missing)

    # 创建结果DataFrame
    result_df = df.copy()

    # 添加一列用于标记原始值和填充值
    flag_column = f"{target_column}_FLAG"
    result_df[flag_column] = "original"
    result_df.loc[missing_idx, flag_column] = "filled"

    # 填充缺失值
    result_df.loc[missing_idx, target_column] = predicted_values

    # 恢复时间戳
    result_df["TIMESTAMP"] = timestamps

    # 保存结果
    result_df.to_csv(output_path, index=False)
    print(f"{target_column}列的缺失值填充完成，结果已保存到 {output_path}")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="应用随机森林模型填充缺失值")
    parser.add_argument(
        "--data", type=str, required=True, help="包含缺失值的数据文件路径"
    )
    parser.add_argument("--model", type=str, required=True, help="训练好的模型文件路径")
    parser.add_argument(
        "--output", type=str, required=True, help="填充后的输出文件路径"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="NEE_VUT_REF",
        help="需要填充缺失值的目标列名（默认为NEE_VUT_REF）",
    )

    args = parser.parse_args()

    # 读取数据
    print(f"读取数据文件: {args.data}")
    df = read_data(args.data)

    # 检查目标列是否存在
    if args.target not in df.columns:
        print(f"错误：数据中不存在指定的目标列 '{args.target}'")
        print(f"可用的列名: {', '.join(df.columns)}")
        return

    # 填充缺失值
    print(f"使用模型 {args.model} 填充 {args.target} 列的缺失值")
    fill_missing_values(df, args.model, args.output, args.target)


if __name__ == "__main__":
    main()
