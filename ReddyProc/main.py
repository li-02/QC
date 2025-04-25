import sys
import pandas as pd
import argparse
import os
import datetime
from model.DataQc import DataQc
from utils.validators import validate_args
from utils.log_config import close_logger, setup_logger


def main():
    logger = setup_logger(ftp=args.ftp)
    parser=argparse.ArgumentParser(description="数据质量控制工具")
    parser.add_argument("--file-path","-d",type=str,default="data.csv",help="数据文件路径")
    parser.add_argument("--data-type","-t",type=str,default="flux",help="数据类型")
    parser.add_argument("--ftp","-f",type=str,default="shisanling",help="站点ftp")
    parser.add_argument("--longitude","-lon",type=float,default=0.0,help="经度")
    parser.add_argument("--latitude","-lat",type=float,default=0.0,help="纬度")
    parser.add_argument("--is-strg","-s",type=int,default=0,help="是否做存储项校正")
    parser.add_argument("--despiking-z","-z",type=float,default=4.0,help="去噪声的z值")
    args=parser.parse_args()


    logger.info("数据质量控制工具开始运行")
    logger.info(f"参数信息: file-path={args.file_path}, data-type={args.data_type}, ftp={args.ftp}, longitude={args.longitude}, latitude={args.latitude}")
    logger.info("验证输入参数")
    valid, error_msgs=validate_args(args)
    if not valid:
        logger.error("参数验证失败")
        close_logger(logger,success=False)
        for msg in error_msgs:
            print(msg)
        sys.exit(1)
    else:
        logger.info("参数验证成功")
    task_id=args.ftp+datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logger.info(f"创建任务ID: {task_id}")
    logger.info("开始读取数据文件")
    # step-1：读取数据
    if not os.path.exists(args.file_path):
        logger.info(f"文件 {args.file_path} 不存在")
        return
    try:
        data= pd.read_csv(args.file_path)
        logger.info(f"数据时间范围：{data['record_time'].min()} 至 {data['record_time'].max()}")
    except Exception as e:
        logger.error(f"读取数据文件失败: {str(e)}")
        close_logger(logger,success=False)
        sys.exit(1)
    logger.info(f"执行{args.data_type}类型数据的质量控制")
    # qc_indicators model.Indicators.query_items(query_dict={"is_qc":1})
    try:
        qc_indicators_csv=pd.read_csv("qc_indicators.csv",header=None)
    except Exception as e:
        logger.error(f"读取质量控制指标文件失败: {str(e)}")
        close_logger(logger,success=False)
        sys.exit(1)
    qc_indicators=qc_indicators_csv.iloc[0].tolist()
    qc_indicators.append(["qc_co2_flux","qc_h2o_flux","qc_le_flux","qc_h_flux"])
    dc=DataQc(task_id=task_id,
                   data=data,
                   data_type=args.data_type,
                   ftp=args.ftp,
                   qc_indicators=qc_indicators,
                   qc_flag_list=["0","1","2"],
                   is_strg=args.is_strg,
                   despiking_z=args.despiking_z,
                   longitude=args.longitude,
                   latitude=args.latitude,
                   timezone=8,
                   filename=args.file_path,
                   logger=logger)
    data_pd=dc.data_qc()
    pass


if __name__ == "__main__":
    main()