"""
将2024年shisanling的flux的RF填充结果与REddyProc的结果进行对比
- 使用单条折线代表原始数据
- 使用不同颜色标记RF填充和REddyProc填充的点
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib import font_manager

# 设置中文字体
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Arial Unicode MS",
    "Microsoft YaHei",
    "SimSun",
]  # 优先使用的中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题


def read_data(file_path, is_reddyproc=False, time_col=None):
    """
    读取数据文件

    Args:

    file_path : str
        数据文件路径
    is_reddyproc : bool
        是否为REddyProc数据文件
    time_col : str
        时间列名称

    Return:

    pandas.DataFrame
        读取的数据
    """
    try:
        df = pd.read_csv(file_path)

        if is_reddyproc:  # REddyProc文件
            # 确保时间列被正确转换为datetime
            if time_col and time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                print(f"成功读取REddyProc数据，共{len(df)}行")
            else:
                print(f"警告: 在REddyProc数据中找不到指定的时间列 '{time_col}'")
        else:  # RF文件
            if time_col and time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                print(f"成功读取RF数据，共{len(df)}行")
            else:
                print(f"警告: 在RF数据中找不到指定的时间列 '{time_col}'")

        return df
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return pd.DataFrame()  # 返回空DataFrame


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="对比RF填充和REddyProc填充效果")

    # 添加文件路径参数
    parser.add_argument("--rf-file", type=str, required=True, help="RF填充数据文件路径")
    parser.add_argument(
        "--re-file", type=str, required=True, help="REddyProc填充数据文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../compare_results",
        help="输出结果的目录路径",
    )

    # 添加列名参数
    parser.add_argument(
        "--rf-time-col", type=str, default="TIMESTAMP", help="RF数据时间列名"
    )
    parser.add_argument("--rf-col", type=str, default="NEE_VUT_REF", help="RF数据列名")
    parser.add_argument(
        "--rf-flag-col", type=str, default="NEE_VUT_REF_FLAG", help="RF数据FLAG列名"
    )

    parser.add_argument(
        "--re-time-col", type=str, default="TIMESTAMP", help="REddyProc数据时间列名"
    )
    parser.add_argument(
        "--re-col", type=str, default="nee_ustar_f", help="REddyProc数据通量列名"
    )

    # 添加图表标题参数
    parser.add_argument(
        "--title", type=str, default="RF填充与REddyProc填充数据对比", help="图表标题"
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 打印文件路径，确认正确
    print(f"RF文件路径: {args.rf_file}")
    print(f"REddyProc文件路径: {args.re_file}")
    print(f"RF文件存在: {os.path.exists(args.rf_file)}")
    print(f"REddyProc文件存在: {os.path.exists(args.re_file)}")
    # 读取RF数据
    rf_data = read_data(args.rf_file, is_reddyproc=False, time_col=args.rf_time_col)
    if rf_data.empty:
        print("未能读取RF数据，程序终止")
        exit()

    # 显示RF数据的列名
    print(f"RF数据列名: {list(rf_data.columns)}")

    # 检查必要的列是否存在
    required_rf_cols = [args.rf_time_col, args.rf_col, args.rf_flag_col]
    missing_cols = [col for col in required_rf_cols if col not in rf_data.columns]
    if missing_cols:
        print(f"RF数据中缺少以下列: {missing_cols}")
        print(f"可用的列有: {list(rf_data.columns)}")

    # 读取REddyProc数据
    re_data = read_data(args.re_file, is_reddyproc=True, time_col=args.re_time_col)
    if re_data.empty:
        print("未能读取REddyProc数据，程序终止")
        exit()
    else:
        # 显示REddyProc数据的列名
        print(f"REddyProc数据列名: {list(re_data.columns)}")

        # 检查必要的列是否存在
        if args.re_time_col in re_data.columns and args.re_col in re_data.columns:
            pass
        else:
            print(f"REddyProc数据中缺少必要的列，可用的列有: {list(re_data.columns)}")

    # 检查两组数据的时间范围
    if not re_data.empty and "TIMESTAMP" in re_data.columns:
        print(
            f"RF数据时间范围: {rf_data['TIMESTAMP'].min()} 到 {rf_data['TIMESTAMP'].max()}"
        )
        print(
            f"REddyProc数据时间范围: {re_data['TIMESTAMP'].min() if not re_data.empty else 'N/A'} 到 {re_data['TIMESTAMP'].max() if not re_data.empty else 'N/A'}"
        )

    # 根据TIMESTAMP列进行合并
    merged_data = pd.merge(
        rf_data, re_data, on="TIMESTAMP", how="outer", suffixes=("_RF", "_RE")
    )

    # 检查合并后的数据
    print(f"合并后数据包含{len(merged_data)}行")
    print(f"RF数据中有{rf_data['TIMESTAMP'].nunique()}个唯一时间点")
    print(
        f"RE数据中有{re_data['TIMESTAMP'].nunique() if not re_data.empty else 0}个唯一时间点"
    )
    print(f"合并后数据中有{merged_data['TIMESTAMP'].nunique()}个唯一时间点")

    # 检查指定列的有效值数量
    re_valid_count = (
        merged_data[args.re_col + "_RE"].notna().sum()
        if args.re_col + "_RE" in merged_data.columns
        else 0
    )
    print(
        f"{args.re_col + '_RE'}列有{re_valid_count}个有效值，占比{re_valid_count/len(merged_data)*100:.2f}%"
    )

    # 将数据按照时间排序
    merged_data = merged_data.sort_values(by="TIMESTAMP")

    # 创建一个图形和一个子图
    fig, ax = plt.subplots(figsize=(15, 8))

    # 标识不同类型的数据点
    original_mask = merged_data[args.rf_flag_col] == "original"
    filled_mask = merged_data[args.rf_flag_col] == "filled"  # RF填充点

    # 1. 仅绘制原始数据点的折线图 (不包含填充点)
    original_data = merged_data[original_mask]
    ax.plot(
        original_data["TIMESTAMP"],
        original_data[args.re_col + "_RF"],
        color="black",
        label="原始数据",
        linewidth=1.5,
        zorder=5,
    )

    # 2. 绘制原始数据点 (使用散点图更清晰地标记)
    ax.scatter(
        original_data["TIMESTAMP"],
        original_data[args.rf_col + "_RF"],
        color="green",
        s=20,
        alpha=0.7,
        label="原始数据点",
        zorder=10,
    )

    # 3. 绘制RF填充数据点
    filled_data = merged_data[filled_mask]
    if not filled_data.empty:
        ax.scatter(
            filled_data["TIMESTAMP"],
            filled_data[args.rf_col + "_RF"],
            color="red",
            s=40,
            alpha=0.8,
            label="RF填充",
            marker="^",  # 三角形标记
            zorder=15,
        )

    # 4. 绘制REddyProc填充数据点 (只有在填充位置且不是NaN的情况下)
    filled_re_data = filled_data[filled_data[args.re_col + "_RE"].notna()]
    if not filled_re_data.empty:
        ax.scatter(
            filled_re_data["TIMESTAMP"],
            filled_re_data[args.re_col + "_RE"],
            color="blue",
            s=40,
            alpha=0.8,
            label="REddyProc填充",
            marker="s",  # 方形标记
            zorder=15,
        )
        print(f"绘制了{len(filled_re_data)}个REddyProc填充点")
    else:
        print("没有找到有效的REddyProc填充点")

    # 设置图表标题和标签
    ax.set_title(
        args.title,
        fontsize=16,
        fontproperties="SimHei",
    )
    ax.set_xlabel("日期", fontsize=14, fontproperties="SimHei")
    ax.set_ylabel(
        f"{args.re_col} (μmol m$^{-2}$ s$^{-1}$)", fontsize=14, fontproperties="SimHei"
    )

    # 设置x轴日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)

    # 添加网格线
    ax.grid(True, linestyle="--", alpha=0.7)

    # 添加图例
    legend = ax.legend(loc="best", fontsize=12)
    # 设置图例中的中文字体
    for text in legend.get_texts():
        text.set_fontproperties("SimHei")

    # 计算RF填充和REddyProc填充的差异
    if "NEE_VUT_REF_RE" in merged_data.columns:
        # 找到同时有RF填充和REddyProc填充的点
        common_filled_points = filled_mask & merged_data[args.re_col + "_RE"].notna()

        if common_filled_points.sum() > 0:
            diff_data = merged_data.loc[common_filled_points]
            diff = diff_data[args.rf_col + "_RF"] - diff_data[args.re_col + "_RE"]

            max_diff = diff.abs().max()
            mean_diff = diff.abs().mean()
            rmse = np.sqrt(np.mean(diff**2))

            # 计算相关系数
            correlation = np.corrcoef(
                diff_data[args.rf_col + "_RF"], diff_data[args.re_col + "_RE"]
            )[0, 1]

            print(
                f"填充点比较 - 共有{common_filled_points.sum()}个时间点同时有RF和REddyProc填充数据"
            )
            print(f"填充点比较 - 最大差异: {max_diff:.6f}")
            print(f"填充点比较 - 平均差异: {mean_diff:.6f}")
            print(f"填充点比较 - RMSE: {rmse:.6f}")
            print(f"填充点比较 - 相关系数: {correlation:.6f}")

            # 在图表上添加统计信息
            stats_text = f"填充点统计:\n相关系数: {correlation:.3f}\nRMSE: {rmse:.6f}\n平均差异: {mean_diff:.6f}"
            plt.annotate(
                stats_text,
                xy=(0.02, 0.96),
                xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=12,
                ha="left",
                va="top",
                fontproperties="SimHei",
            )
        else:
            print("没有同时存在RF填充和REddyProc填充的时间点")

    # 调整布局
    plt.tight_layout()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存图表
    output_file = os.path.join(
        args.output_dir, args.rf_col, "rf_vs_reddyproc_comparison.svg"
    )
    plt.savefig(output_file, dpi=300)
    print(f"图表已保存到 {output_file}")

    # 显示图表
    plt.show()

    # 保存合并后的数据
    output_csv = os.path.join(
        args.output_dir, args.rf_col, "rf_vs_reddyproc_merged_data.csv"
    )
    merged_data.to_csv(output_csv, index=False)
    print(f"合并数据已保存到 {output_csv}")

    # 打印统计信息
    print(f"数据总量: {len(merged_data)}")
    print(f"原始数据量: {sum(original_mask)}")
    print(f"RF填充数据量: {sum(filled_mask)}")
    print(
        f"REddyProc填充数据量 (在filled标记处): {sum(filled_mask & merged_data[args.re_col + "_RE"].notna())}"
    )


if __name__ == "__main__":
    main()

"""
pm2_5的对比
python .\compare_with_REddyProc.py --rf-col "pm2_5" --rf-flag-col "pm2_5_FLAG" --re-col "pm2_5" --rf-file "../data/2024_shisanling_rf_fill_pm2_5_result.csv" --re-file "../data/2024_shisanling_re_fill_pm2_5_result.csv"

"""
