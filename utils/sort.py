import pandas as pd
from datetime import datetime


def sort_csv_by_timestamp(input_file, output_file=None):
    """
    按照TIMESTAMP列从小到大排序CSV文件

    参数:
    input_file (str): 输入CSV文件路径
    output_file (str, optional): 输出CSV文件路径，默认为None，将在原文件上修改

    返回:
    None
    """
    # 如果未指定输出文件，则在原文件上修改
    if output_file is None:
        output_file = input_file

    # 读取CSV文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)

    # 显示排序前的前几行数据
    print("\n排序前的前5行数据:")
    print(df.head())

    # 将TIMESTAMP列转换为datetime类型以便正确排序
    print("\n正在将TIMESTAMP转换为日期时间格式...")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%Y/%m/%d %H:%M")

    # 统计排序前的记录数
    records_before = len(df)
    print(f"\n排序前的总记录数: {records_before}")

    # 按TIMESTAMP列排序
    print("\n正在按TIMESTAMP排序...")
    df_sorted = df.sort_values(by="TIMESTAMP")

    # 统计排序后的记录数
    records_after = len(df_sorted)
    print(f"\n排序后的总记录数: {records_after}")

    # 确保排序前后记录数一致
    if records_before != records_after:
        print(
            f"警告: 排序前后记录数不一致! 排序前: {records_before}, 排序后: {records_after}"
        )

    # 将TIMESTAMP转换回原始格式
    df_sorted["TIMESTAMP"] = df_sorted["TIMESTAMP"].dt.strftime("%Y/%m/%d %H:%M")

    # 显示排序后的前几行数据
    print("\n排序后的前5行数据:")
    print(df_sorted.head())

    # 保存排序后的数据
    print(f"\n正在保存排序后的数据到: {output_file}")
    df_sorted.to_csv(output_file, index=False)

    print(f"\n排序完成！文件已保存到: {output_file}")


if __name__ == "__main__":
    # 文件路径
    input_csv = ".\\data\\2024_shisanling_raw_fluxs_ready.csv"

    # 如果想要保存到新文件，可以取消下面一行的注释并指定新文件名
    # output_csv = "2024_shisanling_raw_fluxs_ready_sorted.csv"

    # 在原文件上修改
    sort_csv_by_timestamp(input_csv)

    # 如果想要保存到新文件，可以取消下面一行的注释
    # sort_csv_by_timestamp(input_csv, output_csv)
