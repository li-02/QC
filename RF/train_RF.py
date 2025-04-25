import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def SLOPE(true, pred):
    x_mean = np.mean(pred)
    y_mean = np.mean(true)
    numerator = np.sum((pred - x_mean) * (true - y_mean))
    denominator = np.sum((pred - x_mean) ** 2)
    return numerator / denominator


def mybias(true, pred):
    n = len(pred)  # 数据点数量
    return np.sum(pred - true) / n


def set_season_tag(df):
    # 根据月份计算季节并添加到 DataFrame 中
    df["month"] = df["TIMESTAMP"].dt.month
    df["season"] = (df["month"] % 12 + 3) // 3
    # 删除临时创建的 'month' 列
    df.drop("month", axis=1, inplace=True)
    return df, ["season"]


def set_rg_tag(df, rg):
    # 根据辐射值 rg 将数据点分为四个等级
    df["rg_rank"] = np.select(
        condlist=[df[rg] < 10, (df[rg] >= 10) & (df[rg] < 100), df[rg] >= 100],
        choicelist=[1, 2, 3],
        default=0,
    )
    # 删除 'SW_IN_F' 列
    df.drop("SW_IN_F", axis=1, inplace=True)
    return df, ["rg_rank"]


def set_doy_year_tag(df):
    df["doy"] = df["TIMESTAMP"].dt.dayofyear  # 添加一年中的第几天标签
    # df["year"] = df['TIMESTAMP'].dt.year  # 添加年份标签
    df["hour"] = df["TIMESTAMP"].dt.hour  # 添加小时数标签
    return df, ["year", "doy", "hour"]


def fill_missing_with_rf(df, train_idx, test_idx,dir_path,j,i):
    # 分割数据集
    X_train, X_test, y_train, y_test = (
        df.iloc[train_idx].drop(columns=["NEE_VUT_REF"]),
        df.iloc[test_idx].drop(columns=["NEE_VUT_REF"]),
        df.iloc[train_idx]["NEE_VUT_REF"],
        df.iloc[test_idx]["NEE_VUT_REF"],
    )

    # X_train, X_no, y_train, y_no = train_test_split(df.drop(columns=['NEE_VUT_REF']), df.loc[:,'NEE_VUT_REF'],
    #                                                     test_size=0.25, random_state=42)
    # X_no, X_test, y_no, y_test = df.iloc[train_idx].drop(columns=['NEE_VUT_REF']), df.iloc[test_idx].drop(
    #     columns=['NEE_VUT_REF']), \
    #                                    df.iloc[train_idx]['NEE_VUT_REF'], df.iloc[test_idx]['NEE_VUT_REF']

    # 实例化随机森林回归器
    rf_regressor = RandomForestRegressor(random_state=42)
    # rf_regressor = RandomForestRegressor(n_estimators=193, max_depth=35, random_state=50)
    # rf_regressor = RandomForestRegressor(n_estimators=300, max_depth=32, min_samples_leaf=8, min_samples_split=2, random_state=50)
    # rf_regressor = RandomForestRegressor(n_estimators=300, max_depth=32, min_samples_leaf=8, min_samples_split=2,
    #                                      random_state=50)

    # 训练模型
    # print(X_train.shape)
    # print(X_test.shape)
    start_time = time.time()

    rf_regressor.fit(X_train, y_train)
    model_filename=os.path.join(dir_path, "RF", "model", f"RF_shisanling_{j * 48}_mask{i}.pkl")
    with open(model_filename,'wb') as file:
        pickle.dump(rf_regressor,file)      
    # 预测缺失值
    y_pred = rf_regressor.predict(X_test)

    inference_time = time.time() - start_time

    df.loc[test_idx, "NEE_raw"] = y_test
    df.loc[test_idx, "NEE_pred"] = y_pred

    # 计算模型性能指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    slope = SLOPE(y_test, y_pred)
    bias = mybias(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 输出性能指标
    # print("Mean Absolute Error:", mae)
    # print("Root Mean Squared Error:", rmse)
    # print("R^2 Score:", r2)

    # MAE = "Mean Absolute Error: " + str(mae)
    # RMSE = "Root Mean Squared Error: " + str(rmse)
    # R2 = "R^2 Score: " + str(r2)
    # result = [MAE, RMSE, R2]
    #
    # df.loc[[1, 2, 3], 'RESULT'] = result

    # return df
    return mae, rmse, slope, bias, r2, inference_time



def read_data(csv_file):
    # df = pd.read_csv(csv_file, parse_dates=['TIMESTAMP'],  skiprows=range(1, start_row),
    #                  nrows=17520)  # 读取 CSV 文件并解析时间戳列
    df = pd.read_csv(csv_file, parse_dates=["TIMESTAMP"])  # 读取 CSV 文件并解析时间戳列
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"],format="%d/%m/%Y %H:%M:%S")  # 将时间戳列转换为 datetime 类型
    return df


def main(root_path):
    dir_path = root_path
    site_name = "shisanling"
    data_path = os.path.join(dir_path,"data", "shisanling.csv")

    os.makedirs(os.path.join(dir_path,"RF", "process"), exist_ok=True)
    os.makedirs(os.path.join(dir_path,"RF", "result"), exist_ok=True)
    os.makedirs(os.path.join(dir_path,"RF", "model"), exist_ok=True)

    # data_path = dir_path + '/RF_sequence.csv'
    # start_row = 9095
    df = read_data(data_path)

    # set_season_tag(df)
    # set_rg_tag(df, 'SW_IN_F')
    # set_doy_year_tag(df)
    #
    df.drop("TIMESTAMP", axis=1, inplace=True)
    # print(df.head())

    seg_len = [1, 7, 15, 30, 90]

    for j in seg_len:
        setting = f"RF_{site_name}_{j * 48}"
        print(setting)
        mask_path = os.path.join(dir_path,"RF", "masks", f"masks{j}.csv")
        mask_df = pd.read_csv(mask_path)

        MAEs = np.array([])
        RMSEs = np.array([])
        SLOPEs = np.array([])
        BIASs = np.array([])
        R2s = np.array([])
        inference_times = 0

        for i in range(1, 11):
            mask_sheet = "mask" + str(i)
            mask = mask_df[mask_sheet]
            # print(mask)

            nee_df = df.copy()

            train_idxs = np.where(mask.values == 1)[0]
            test_idxs = np.where(mask.values == 0)[0]

            mae, rmse, slope, bias, r2, inference_time = fill_missing_with_rf(
                nee_df, train_idxs, test_idxs,dir_path,j,i
            )
            MAEs = np.append(MAEs, mae)
            RMSEs = np.append(RMSEs, rmse)
            SLOPEs = np.append(SLOPEs, slope)
            BIASs = np.append(BIASs, bias)
            R2s = np.append(R2s, r2)
            inference_times += inference_time

            # output_path = dir_path + '/RF_output_' + str(i) + '.csv'
            # output_path = dir_path + '/RF_output_' + '90' + '.csv'
            # df.to_csv(output_path, index=False)

            # df.drop('NEE_raw', axis=1, inplace=True)
            # df.drop('NEE_pred', axis=1, inplace=True)
            # df.drop('RESULT', axis=1, inplace=True)

        MAE = np.mean(MAEs)
        RMSE = np.mean(RMSEs)
        SLOPE = np.mean(SLOPEs)
        BIAS = np.mean(BIASs)
        R2 = np.mean(R2s)

        print(
            "rmse:{}, mae:{}, slope:{}, bias:{}, r2:{}".format(
                RMSE, MAE, SLOPE, BIAS, R2
            )
        )
        print(f"Inference Time: {inference_times:.6f} seconds")
        result_path=os.path.join(dir_path,"RF","result", "result_imputation.txt")
        f = open(result_path, "a")
        f.write(setting + "  \n")
        f.write(
            "rmse:{}, mae:{}, slope:{}, bias:{}, r2:{}".format(
                RMSE, MAE, SLOPE, BIAS, R2
            )
        )
        f.write("\n")
        f.write("\n")
        f.close()


if __name__ == "__main__":
    main(os.path.join("D:\\", "Code", "QC"))
