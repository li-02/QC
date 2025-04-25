import os
import pickle
import time
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def SLOPE(true, pred):
    x_mean = np.mean(pred)
    y_mean = np.mean(true)
    numerator = np.sum((pred - x_mean) * (true - y_mean))
    denominator = np.sum((pred - x_mean) ** 2)
    return numerator / denominator if denominator != 0 else 0


def mybias(true, pred):
    n = len(pred)
    return np.sum(pred - true) / n if n > 0 else 0


def set_season_tag(df):
    # 根据月份计算季节并添加到 DataFrame 中
    df["month"] = df["TIMESTAMP"].dt.month
    df["season"] = (df["month"] % 12 + 3) // 3
    # 删除临时创建的 'month' 列
    df.drop("month", axis=1, inplace=True)
    return df, ["season"]


def set_rg_tag(df, rg):
    if rg not in df.columns:
        print(f"警告: 列 '{rg}' 不存在于数据中。跳过辐射标签设置。")
        return df, []

    # 根据辐射值 rg 将数据点分为四个等级
    df["rg_rank"] = np.select(
        condlist=[df[rg] < 10, (df[rg] >= 10) & (df[rg] < 100), df[rg] >= 100],
        choicelist=[1, 2, 3],
        default=0,
    )
    # 删除辐射列
    df.drop(rg, axis=1, inplace=True)
    return df, ["rg_rank"]


def set_doy_year_tag(df):
    df["doy"] = df["TIMESTAMP"].dt.dayofyear  # 添加一年中的第几天标签
    df["hour"] = df["TIMESTAMP"].dt.hour  # 添加小时数标签
    return df, ["doy", "hour"]


def fill_missing_with_rf(df, train_idx, test_idx, model_dir, j, i, target_column):
    try:
        # 检查索引是否在有效范围内
        max_idx = len(df) - 1
        valid_train_idx = [idx for idx in train_idx if 0 <= idx <= max_idx]
        valid_test_idx = [idx for idx in test_idx if 0 <= idx <= max_idx]

        if len(valid_train_idx) != len(train_idx):
            print(
                f"警告: 训练集索引超出范围。原始: {len(train_idx)}, 有效: {len(valid_train_idx)}"
            )

        if len(valid_test_idx) != len(test_idx):
            print(
                f"警告: 测试集索引超出范围。原始: {len(test_idx)}, 有效: {len(valid_test_idx)}"
            )

        if len(valid_train_idx) == 0 or len(valid_test_idx) == 0:
            print(f"错误: 没有有效的训练或测试索引，跳过训练。")
            return 0, 0, 0, 0, 0, 0

        # 确保目标列存在
        if target_column not in df.columns:
            print(
                f"错误: 目标列 '{target_column}' 不存在于数据中。可用列有: {df.columns.tolist()}"
            )
            return 0, 0, 0, 0, 0, 0

        # 分割数据集
        X_train = df.iloc[valid_train_idx].drop(columns=[target_column])
        X_test = df.iloc[valid_test_idx].drop(columns=[target_column])
        y_train = df.iloc[valid_train_idx][target_column]
        y_test = df.iloc[valid_test_idx][target_column]

        # 检查是否有足够的数据
        if len(X_train) < 10 or len(X_test) < 10:
            print(
                f"警告: 训练或测试数据集太小 (训练: {len(X_train)}, 测试: {len(X_test)})。结果可能不可靠。"
            )

        # 实例化随机森林回归器
        rf_regressor = RandomForestRegressor(random_state=42)

        start_time = time.time()

        # 训练模型
        rf_regressor.fit(X_train, y_train)

        # 确保模型保存目录存在
        os.makedirs(model_dir, exist_ok=True)

        model_filename = os.path.join(model_dir, f"RF_model_{j * 48}_mask{i}.pkl")
        with open(model_filename, "wb") as file:
            pickle.dump(rf_regressor, file)

        # 预测缺失值
        y_pred = rf_regressor.predict(X_test)

        inference_time = time.time() - start_time

        # 保存原始值和预测值
        df.loc[valid_test_idx, f"{target_column}_raw"] = y_test
        df.loc[valid_test_idx, f"{target_column}_pred"] = y_pred

        # 计算模型性能指标
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        slope = SLOPE(y_test, y_pred)
        bias = mybias(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mae, rmse, slope, bias, r2, inference_time

    except Exception as e:
        print(f"错误: 在 fill_missing_with_rf 中发生异常: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return 0, 0, 0, 0, 0, 0


def read_data(csv_file, time_column="record_time"):
    try:
        # 尝试读取数据文件
        print(f"正在读取数据文件: {csv_file}")
        df = pd.read_csv(csv_file)

        # 检查是否存在指定的时间列
        if time_column not in df.columns:
            print(
                f"警告: 数据文件中没有 '{time_column}' 列。可用列有: {df.columns.tolist()}"
            )
            return None
        else:
            try:
                df.rename(columns={time_column: "TIMESTAMP"}, inplace=True)
                df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
                print(f"成功将列 '{time_column}' 转换为时间戳")
            except Exception as e:
                print(f"警告: 无法将列 '{time_column}' 转换为时间戳: {str(e)}")

        print(f"成功读取数据, 形状: {df.shape}")
        return df

    except Exception as e:
        print(f"错误: 无法读取数据文件: {str(e)}")
        sys.exit(1)


def adapt_mask_to_data(mask_df, data_df):
    """
    调整掩码大小以匹配数据大小，处理半小时掩码与一小时数据的情况
    """
    # 检查掩码和数据大小
    print(f"原始掩码形状: {mask_df.shape}")
    print(f"数据形状: {data_df.shape}")

    # 检查时间频率
    if "TIMESTAMP" in data_df.columns:
        time_diffs = data_df["TIMESTAMP"].diff().dropna()
        if len(time_diffs) > 0:
            most_common_diff = time_diffs.mode()[0]
            data_hours = most_common_diff.total_seconds() / 3600

            # 假设掩码是半小时一行的标准格式
            mask_hours = 0.5

            # 如果数据是一小时一行，而掩码是半小时一行，需要重采样掩码
            if abs(data_hours - 1.0) < 0.1 and abs(mask_hours - 0.5) < 0.1:
                print("检测到数据是一小时一行，掩码是半小时一行，将重采样掩码")

                # 将半小时掩码转换为一小时掩码（取每隔一行）
                new_mask_df = pd.DataFrame()
                for col in mask_df.columns:
                    # 对于每个掩码列，每隔一行取一次值
                    new_mask_df[col] = mask_df[col].iloc[::2].reset_index(drop=True)

                mask_df = new_mask_df
                print(f"调整后的掩码形状: {mask_df.shape}")

    # 如果掩码行数仍然超过数据行数，截断掩码
    if len(mask_df) > len(data_df):
        print(
            f"掩码行数 ({len(mask_df)}) 仍然多于数据行数 ({len(data_df)})，将截断掩码"
        )
        mask_df = mask_df.iloc[: len(data_df)]

    # 如果掩码行数少于数据行数，填充掩码（全部标记为训练集）
    elif len(mask_df) < len(data_df):
        print(f"掩码行数 ({len(mask_df)}) 少于数据行数 ({len(data_df)})，将填充掩码")
        padding = pd.DataFrame(
            1, index=range(len(data_df) - len(mask_df)), columns=mask_df.columns
        )
        mask_df = pd.concat([mask_df, padding], ignore_index=True)

    return mask_df


def main():
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="随机森林模型训练与评估")

    # 添加命令行参数
    parser.add_argument("--csv_path", type=str, required=True, help="目标CSV文件路径")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="",
        help="结果保存目录，默认为当前目录同级的result/{target_column}目录",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="模型保存目录，默认为当前目录同级的model/{target_column}目录",
    )
    parser.add_argument(
        "--target_column", type=str, required=True, help="目标指标的列名"
    )
    parser.add_argument(
        "--time_column",
        type=str,
        default="record_time",
        help="时间戳列名，默认为'record_time'",
    )
    parser.add_argument("--site_name", type=str, default="shisanling", help="站点名称")

    parser.add_argument(
        "--masks_dir",
        type=str,
        default="",
        help="掩码文件目录，默认为当前目录同级的masks目录",
    )
    parser.add_argument(
        "--adapt_mask",
        action="store_true",
        help="自适应调整掩码以匹配数据频率",
        default=True,
    )
    parser.add_argument("--verbose", action="store_true", help="是否输出详细信息")

    # 解析命令行参数
    args = parser.parse_args()

    # 设置默认的结果和模型保存目录（如果未指定）
    if not args.result_dir:
        args.result_dir = os.path.join(current_dir, "result", args.target_column)

    if not args.model_dir:
        args.model_dir = os.path.join(current_dir, "model", args.target_column)
    # 设置默认的掩码文件目录（如果未指定）
    if not args.masks_dir:
        args.masks_dir = os.path.join(current_dir, "masks")

    # 显示执行参数
    print("=" * 80)
    print("运行参数:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 80)

    # 确保结果目录存在
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "process"), exist_ok=True)
    # 检查掩码目录是否存在
    if not os.path.exists(args.masks_dir):
        print(f"警告: 掩码目录 '{args.masks_dir}' 不存在。请确认掩码文件路径是否正确。")

    # 读取数据
    df = read_data(args.csv_path, args.time_column)

    # 检查目标列是否存在
    if args.target_column not in df.columns:
        print(f"错误: 目标列 '{args.target_column}' 不存在于数据中。")
        print(f"可用列有: {df.columns.tolist()}")
        sys.exit(1)

    # 显示数据的基本信息
    print(f"数据概览:")
    print(f"  行数: {len(df)}")
    print(f"  列数: {len(df.columns)}")
    print(f"  列名: {df.columns.tolist()}")
    print(f"  目标列 '{args.target_column}' 统计信息:")
    print(df[args.target_column].describe())

    # 缺失值填充的时间段长度
    seg_len = [1, 7, 15, 30, 90]

    # 打开结果文件
    result_path = os.path.join(args.result_dir, "result_imputation.txt")
    with open(result_path, "a") as f:
        f.write(f"Target column: {args.target_column}\n")
        f.write(f"Site: {args.site_name}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 备份时间戳列以便之后使用
    timestamp_column = None
    if "TIMESTAMP" in df.columns:
        timestamp_column = df["TIMESTAMP"].copy()
        df.drop("TIMESTAMP", axis=1, inplace=True)

    for j in seg_len:
        setting = f"RF_{args.site_name}_{j * 48}"
        print(f"\n处理设置: {setting}")

        # 构建掩码文件路径
        mask_path = os.path.join(args.masks_dir, f"masks{j}.csv")

        # 检查掩码文件是否存在
        if not os.path.exists(mask_path):
            print(f"警告: 掩码文件 {mask_path} 不存在，跳过此设置。")
            continue

        try:
            mask_df = pd.read_csv(mask_path)
            print(f"掩码文件形状: {mask_df.shape}")

            # 调整掩码以匹配数据
            if args.adapt_mask:
                # 为了进行频率分析，临时将时间戳放回数据框
                if timestamp_column is not None:
                    temp_df = df.copy()
                    temp_df["TIMESTAMP"] = timestamp_column
                    mask_df = adapt_mask_to_data(mask_df, temp_df)
                else:
                    mask_df = adapt_mask_to_data(mask_df, df)

                print(f"调整后的掩码形状: {mask_df.shape}, 数据形状: {df.shape}")

            # 检查掩码文件是否包含足够的列
            if len(mask_df.columns) < 10:
                print(
                    f"警告: 掩码文件列数少于10 ({len(mask_df.columns)})，可能缺少掩码。"
                )

            # 检查掩码文件长度是否匹配数据长度
            if len(mask_df) != len(df):
                print(
                    f"警告: 掩码文件长度 ({len(mask_df)}) 与数据长度 ({len(df)}) 不匹配!"
                )

                # 确定行数是太多还是太少
                if len(mask_df) > len(df):
                    print(f"掩码文件行数多于数据文件，将只使用前 {len(df)} 行。")
                    mask_df = mask_df.iloc[: len(df)]
                else:
                    print(f"掩码文件行数少于数据文件，将填充剩余行为训练集。")
                    # 创建一个全为1的DataFrame来填充差异
                    padding = pd.DataFrame(
                        1, index=range(len(df) - len(mask_df)), columns=mask_df.columns
                    )
                    mask_df = pd.concat([mask_df, padding], ignore_index=True)
        except Exception as e:
            print(f"错误: 无法读取掩码文件 {mask_path}: {str(e)}")
            continue

        MAEs = np.array([])
        RMSEs = np.array([])
        SLOPEs = np.array([])
        BIASs = np.array([])
        R2s = np.array([])
        inference_times = 0

        for i in range(1, 11):
            mask_sheet = "mask" + str(i)

            # 检查掩码列是否存在
            if mask_sheet not in mask_df.columns:
                print(f"警告: 掩码列 '{mask_sheet}' 不存在于掩码文件中。")
                continue

            mask = mask_df[mask_sheet]

            # 打印掩码的基本信息
            n_train = sum(mask == 1)
            n_test = sum(mask == 0)
            print(f"掩码 {i}: 训练点: {n_train}, 测试点: {n_test}")

            if n_train == 0 or n_test == 0:
                print("警告: 掩码没有训练点或测试点，跳过。")
                continue

            # 复制数据集
            nee_df = df.copy()

            # 获取训练和测试索引
            train_idxs = np.where(mask.values == 1)[0]
            test_idxs = np.where(mask.values == 0)[0]

            print(f"执行随机森林填充 (掩码 {i})...")
            mae, rmse, slope, bias, r2, inference_time = fill_missing_with_rf(
                nee_df, train_idxs, test_idxs, args.model_dir, j, i, args.target_column
            )

            # 如果训练成功
            if mae != 0 or rmse != 0:
                MAEs = np.append(MAEs, mae)
                RMSEs = np.append(RMSEs, rmse)
                SLOPEs = np.append(SLOPEs, slope)
                BIASs = np.append(BIASs, bias)
                R2s = np.append(R2s, r2)
                inference_times += inference_time

                # 保存每个掩码的处理结果
                process_path = os.path.join(
                    args.result_dir, "process", f"{setting}_mask{i}.csv"
                )
                nee_df.to_csv(process_path, index=False)

                print(f"掩码 {i} 结果: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            else:
                print(f"掩码 {i} 处理失败或没有结果。")

        # 如果有任何成功的训练
        if len(MAEs) > 0:
            # 计算平均指标
            MAE = np.mean(MAEs)
            RMSE = np.mean(RMSEs)
            SLOPE = np.mean(SLOPEs)
            BIAS = np.mean(BIASs)
            R2 = np.mean(R2s)

            print("\n平均指标:")
            print(f"  RMSE: {RMSE:.4f}")
            print(f"  MAE: {MAE:.4f}")
            print(f"  Slope: {SLOPE:.4f}")
            print(f"  Bias: {BIAS:.4f}")
            print(f"  R²: {R2:.4f}")
            print(f"  Inference Time: {inference_times:.6f} seconds")

            # 保存总结果
            with open(result_path, "a") as f:
                f.write(setting + "  \n")
                f.write(
                    "rmse:{:.4f}, mae:{:.4f}, slope:{:.4f}, bias:{:.4f}, r2:{:.4f}".format(
                        RMSE, MAE, SLOPE, BIAS, R2
                    )
                )
                f.write("\n")
                f.write(f"Inference Time: {inference_times:.6f} seconds")
                f.write("\n\n")
        else:
            print(f"警告: 设置 {setting} 下没有成功的模型训练。")

    print("\n处理完成!")
    print(f"结果保存在: {result_path}")


if __name__ == "__main__":
    main()
