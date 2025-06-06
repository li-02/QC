# 安装并加载必要的包
if (!require("REddyProc")) install.packages("REddyProc")
if (!require("data.table")) install.packages("data.table")
if (!require("lubridate")) install.packages("lubridate")
library(REddyProc)
library(data.table)
library(lubridate)

# 设置工作目录（请替换为你的数据文件所在目录）
setwd("D:\\Code\\QC")

# 读取数据
data <- fread(".\\data\\2024_shisanling_raw_fluxs_ready.csv")

# 检查数据结构
str(data)
summary(data)

# 处理TIMESTAMP格式
# 将TIMESTAMP列转换为日期时间格式
data$TIMESTAMP <- as.POSIXct(data$TIMESTAMP, format="%Y/%m/%d %H:%M")

# 提取年、月、日、小时、分钟
data$Year <- year(data$TIMESTAMP)
data$Month <- month(data$TIMESTAMP)
data$Day <- day(data$TIMESTAMP)
data$DoY <- yday(data$TIMESTAMP)  # 一年中的第几天
data$Hour <- hour(data$TIMESTAMP) + minute(data$TIMESTAMP)/60  # 小数形式的小时

# 创建符合REddyProc要求的数据框
EddyData <- data.frame(
  DateTime = data$TIMESTAMP,
  Year = data$Year, 
  DoY = data$DoY,
  Hour = data$Hour,
  Rg = data$SW_IN_F,  # 总辐射
  Rn = data$NETRAD,  # 净辐射
  Tair = data$TA_F_MDS,  # 空气温度
  VPD = data$VPD_F_MDS,  # 饱和水汽压差
  SWC = data$SWC_F_MDS_1,  # 土壤水含量
  Tsoil = data$TS_F_MDS_1,  # 土壤温度
  RH = data$RH  # 添加相对湿度
)

# 转换为REddyProc需要的数据类型
EddyDataWithPosix <- fConvertTimeToPosix(EddyData, 'YDH', Year = 'Year', Day = 'DoY', Hour = 'Hour')

# 1. 检查并处理重复的时间戳
duplicate_indices <- which(duplicated(EddyDataWithPosix$DateTime))
if(length(duplicate_indices) > 0) {
  cat("发现", length(duplicate_indices), "个重复的时间戳。移除重复项...\n")
  EddyDataWithPosix <- EddyDataWithPosix[!duplicated(EddyDataWithPosix$DateTime), ]
}

# 2. 创建完整的时间序列并手动构建新的数据框
# 获取开始和结束时间
start_time <- min(EddyDataWithPosix$DateTime, na.rm = TRUE)
end_time <- max(EddyDataWithPosix$DateTime, na.rm = TRUE)

cat("数据开始时间:", as.character(start_time), "\n")
cat("数据结束时间:", as.character(end_time), "\n")

# 创建理想的半小时间隔时间序列
complete_times <- seq(from = start_time, to = end_time, by = "30 min")

# 从头创建一个完整的数据框
complete_data <- data.frame(
  DateTime = complete_times,
  Year = year(complete_times),
  DoY = yday(complete_times),
  Hour = hour(complete_times) + minute(complete_times)/60,
  Rg = NA,
  Rn = NA,
  Tair = NA,
  VPD = NA,
  SWC = NA,
  Tsoil = NA,
  RH = NA  # 添加RH列
)

# 手动填充数据 - 将原始数据的值填入对应时间的行
for (i in 1:nrow(EddyDataWithPosix)) {
  idx <- which(complete_data$DateTime == EddyDataWithPosix$DateTime[i])
  if (length(idx) == 1) {
    complete_data$Rg[idx] <- EddyDataWithPosix$Rg[i]
    complete_data$Rn[idx] <- EddyDataWithPosix$Rn[i]
    complete_data$Tair[idx] <- EddyDataWithPosix$Tair[i]
    complete_data$VPD[idx] <- EddyDataWithPosix$VPD[i]
    complete_data$SWC[idx] <- EddyDataWithPosix$SWC[i]
    complete_data$Tsoil[idx] <- EddyDataWithPosix$Tsoil[i]
    complete_data$RH[idx] <- EddyDataWithPosix$RH[i]  # 添加RH数据
  }
}

# 检查构建后的数据
cat("完整时间序列数据点数:", length(complete_times), "\n")
cat("构建后的数据点数:", nrow(complete_data), "\n")

# 计算缺失值比例
na_count <- sapply(complete_data[, c("Rg", "Rn", "Tair", "VPD", "SWC", "Tsoil", "RH")], function(x) sum(is.na(x)))
total_count <- nrow(complete_data)
na_percent <- na_count / total_count * 100

cat("各变量缺失值比例(%):\n")
print(na_percent)

# 4. 使用规范化的数据创建REddyProc对象
EP <- sEddyProc$new('shisanling', complete_data, c('Rg', 'Rn', 'Tair', 'VPD', 'SWC', 'Tsoil', 'RH'))

# 设置站点位置信息（十三陵的大致位置）
EP$sSetLocationInfo(LatDeg = 40.0, LongDeg = 116.0, TimeZoneHour = 8)

# 5. 进行MDS填补缺失值
cat("开始执行MDS缺失值填补...\n")

# 对每个变量应用MDS方法
cat("填补Rg (总辐射)...\n")
EP$sMDSGapFill('Rg', FillAll = TRUE)

cat("填补Rn (净辐射)...\n")
EP$sMDSGapFill('Rn', FillAll = TRUE)

cat("填补Tair (空气温度)...\n")
EP$sMDSGapFill('Tair', FillAll = TRUE)

cat("填补VPD (饱和水汽压差)...\n")
EP$sMDSGapFill('VPD', FillAll = TRUE)

cat("填补SWC (土壤水含量)...\n")
EP$sMDSGapFill('SWC', FillAll = TRUE)

cat("填补Tsoil (土壤温度)...\n")
EP$sMDSGapFill('Tsoil', FillAll = TRUE)

cat("填补RH (相对湿度)...\n")
EP$sMDSGapFill('RH', FillAll = TRUE)  # 添加对RH的填补

# 6. 获取填补后的数据
FilledEddyData <- EP$sExportResults()

# 检查填补后的数据结构
str(FilledEddyData)

# 7. 评估填补质量
# 计算各变量填补前后的缺失值数量
na_before <- sapply(complete_data[, c("Rg", "Rn", "Tair", "VPD", "SWC", "Tsoil", "RH")], function(x) sum(is.na(x)))
na_after <- sapply(FilledEddyData[, c("Rg_f", "Rn_f", "Tair_f", "VPD_f", "SWC_f", "Tsoil_f", "RH_f")], function(x) sum(is.na(x)))

cat("填补前后缺失值数量对比:\n")
fill_comparison <- data.frame(
  Variable = c("Rg", "Rn", "Tair", "VPD", "SWC", "Tsoil", "RH"),
  Before = na_before,
  After = na_after,
  FillPercentage = round((na_before - na_after) / na_before * 100, 2)
)
print(fill_comparison)

# 8. 保存填补后的数据为CSV文件
write.csv(FilledEddyData, ".\\data\\FilledEddyData_MDS.csv", row.names = FALSE)
cat("填补后的数据已保存至 .\\data\\FilledEddyData_MDS.csv\n")

# 9. 绘制可视化图表
if (require("ggplot2")) {
  library(ggplot2)
  
  # 为每个变量创建可视化
  create_visualizations <- function(var_name) {
    # 准备绘图数据
    original_col <- var_name
    filled_col <- paste0(var_name, "_f")
    
    plot_data <- data.frame(
      DateTime = FilledEddyData$DateTime,
      Original = complete_data[[original_col]],
      Filled = FilledEddyData[[filled_col]],
      Gap = is.na(complete_data[[original_col]])
    )
    
    # 1. 时间序列对比图
    time_series_plot <- ggplot(plot_data, aes(x = DateTime)) +
      geom_line(aes(y = Original), color = "black", alpha = 0.6, na.rm = TRUE) +
      geom_line(aes(y = Filled), color = "red", alpha = 0.8, na.rm = TRUE) +
      labs(title = paste(var_name, "- 原始与填补值对比"),
           x = "时间", y = var_name) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    ggsave(paste0(".\\plots\\", var_name, "_timeseries.png"), time_series_plot, width = 10, height = 6)
    
    # 2. 散点图 - 只对比有原始值的部分
    scatter_plot <- ggplot(plot_data[!plot_data$Gap, ], aes(x = Original, y = Filled)) +
      geom_point(alpha = 0.5) +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      labs(title = paste(var_name, "- 原始值与填补值散点图"),
           x = "原始值", y = "填补值") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    ggsave(paste0(".\\plots\\", var_name, "_scatter.png"), scatter_plot, width = 8, height = 6)
    
    # 3. 填补值分布直方图
    hist_plot <- ggplot(plot_data, aes(x = Filled)) +
      geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
      labs(title = paste(var_name, "- 填补值分布"),
           x = "填补值", y = "频次") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    ggsave(paste0(".\\plots\\", var_name, "_histogram.png"), hist_plot, width = 8, height = 6)
  }
  
  # 创建输出目录
  if (!dir.exists(".\\plots")) {
    dir.create(".\\plots")
  }
  
  # 为每个变量生成可视化
  variables <- c("Rg", "Rn", "Tair", "VPD", "SWC", "Tsoil", "RH")  # 添加RH到可视化变量中
  for (var in variables) {
    cat("为", var, "生成可视化图表...\n")
    create_visualizations(var)
  }
  
  cat("所有可视化图表已保存至 .\\plots\\ 目录\n")
}

cat("MDS缺失值填补过程已完成!\n")