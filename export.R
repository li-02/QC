library(REddyProc)
library(data.table)
library(lubridate)

# 设置工作目录（请替换为你的数据文件所在目录）
setwd("D:\\Code\\QC")

data <- fread(".\\data\\FilledEddyData_MDS.csv")

# 筛选主要填补结果和必要的列
selected_data <- data[, c(
  "TIMESTAMP",                                # 时间戳
  "Rg_f", "Rn_f", "Tair_f", "VPD_f", "SWC_f", "Tsoil_f", "RH_f"  # 主要填补结果
)]


# 保存最终数据
write.csv(selected_data, ".\\data\\Final_Data_With_Filled_Values.csv", row.names = FALSE)