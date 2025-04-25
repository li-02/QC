"""
R脚本模块
"""
import os
# os.environ['R_HOME']='D:\\Env\\R-4.2.2'
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
from rpy2.robjects import vectors

# 激活pandas转换器
pandas2ri.activate()

# R对象别名
my_robjects = robjects
StrVector = vectors.StrVector
FloatVector = vectors.FloatVector
IntVector = vectors.IntVector

# 导入R脚本，确保R函数被定义
from . import r_gap_fill_par
from . import r_co2_flux