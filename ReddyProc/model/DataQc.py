from model import pandas as pd
from model import numpy as np
from utils.data_process import *
from utils.ustar_fill_partitioning import ustar_data

campbell_site = ['aosen', 'badaling']

class DataQc(object):
    """
        对flux aqi sapflow nai 同时做质量控制, 暂时没有flux的站点就先不做质量控制了。
        不同类型的数据: flux_data, aqi_data, sapflow_data, nai_data
        数据保存到本地: flux_filename, aqi_filename, sapflow_filename, nai_filename
        qc_flag_list: 对flux里面的四大通量LE H H2O CO2 选择质量标签
        flux_strg: 对flux 里面的四大通量是否做存储项校正
    """

    def __init__(self,
                 data,
                 filename,
                 longitude,
                 latitude,
                 qc_flag_list: list,
                 is_strg,
                 timezone,
                 qc_indicators,
                 data_type,
                 task_id,
                 ftp,
                 logger,
                 despiking_z=4,
                ):
        self.filename = filename
        self.qc_flag_list = qc_flag_list
        self.is_strg = is_strg
        self.task_id = task_id
        self.despiking_z = despiking_z
        self.longitude = longitude
        self.latitude = latitude
        self.timezone = timezone
        self.ftp = ftp
        self.use_index = [
            'record_time', 'rg_1_1_2', 'ppfd_1_1_1', 'ta_1_2_1', 'tsoil', 'rh',
            'vpd', 'u_', 'short_up_avg', 'rh_12m_avg', 'ta_12m_avg', 'rh_10m_avg'
        ]
        self.qc_indicators = qc_indicators
        self.data_type = data_type

        # data 是需要进行QC的数据
        self.data_start_time = data[0]['record_time']
        self.data_end_time = data[-1]['record_time']
        self.raw_data = pd.DataFrame(data)
        self.logger=logger



    def data_qc(self):
        self.logger.info("数据预处理")
        if "id" in self.raw_data.columns:
            self.logger.info("删除id列")
            del self.raw_data["id"]
        self.logger.info("NAN值处理")
        self.raw_data=self.raw_data.replace(["NaN", "nan", "NAN", "N/A", "N/a", "n/a", "N/A"," ",""], np.NAN)
        self.raw_data=self.raw_data.fillna(np.NAN)
        # 转成 float
        not_convert_list = [
            'record_time', 'qc_co2_flux', 'qc_h2o_flux', 'qc_h', 'qc_le'
        ]
        self.logger.info("数据转换float")
        for i in self.raw_data.columns:
            if i not in not_convert_list:
                self.raw_data[i] = self.raw_data[i].astype('float')
        if self.data_type == "flux":
            self.logger.info("flux数据质量控制")
            for item in self.use_index:
                if item not in self.raw_data.columns.tolist():
                    self.raw_data[item]=np.NAN
            # 按照质量标签筛选数据
            self.logger.info("根据质量标签筛选数据")
            self._filter_by_quality()
            self.logger.info("添加存储项")
            self._add_strg()
            # TODO: cuihu yeyehu yuankeyuan songshan aosen badaling 要做特殊处理，这里先不写
            # 删除不需要的指标
            if 'short_up_avg' in self.raw_data.columns:
                del self.raw_data['short_up_avg']
            if 'rh_12m_avg' in self.raw_data.columns:
                del self.raw_data['rh_12m_avg']
            if 'rh_10m_avg' in self.raw_data.columns:
                del self.raw_data['rh_10m_avg']
            if 'ta_12m_avg' in self.raw_data.columns:
                del self.raw_data['ta_12m_avg']
            # 根据阈值进行数据筛选
            self.logger.info("根据阈值筛选数据")
            self._threshold_limit()
            self.logger.info("插补par光合有效辐射 ppfd_1_1_1")
            self._gap_fill_par()
            self.logger.info("对co2 h2o le h进行despiking")
            self._despiking()
            self.logger.info("异常值过滤")
            self._del_abnormal_value()
            self.logger.info("插补")
            self._ustar_fill_partition()
        else:
            pass





    def _threshold_limit(self):
            self.raw_data = threshold_limit(self.raw_data, self.qc_indicators, self.data_type)

    def _filter_by_quality(self):
        """根据QC标记列表(0,1,2)筛选通量数据"""
        excluded_qc_flags = [
            flag for flag in ["0", "1", "2"] if flag not in self.qc_flag_list
        ]
        # 如果没有排除的QC标记，则直接复制通量列
        # 否则根据QC标记过滤通量数据
        if len(excluded_qc_flags) == 0:
            if self.ftp in campbell_site:
                self.raw_data = handle_campbell_special_case(self.raw_data)
            else:
                self.raw_data = copy_flux_columns_without_qc_filter(self.raw_data)
        else:
            self.raw_data = filter_flux_by_qc_flags(self.raw_data, excluded_qc_flags)

    def _add_strg(self):
        if self.is_strg == '1':
            self.raw_data = do_add_strg(self.raw_data)
        else:
            self.raw_data = not_add_strg(self.raw_data)

    def _gap_fill_par(self):
        """仅对光合有效辐射ppfd_1_1_1进行插补 供四大通量的despiking使用"""
        self.raw_data = gap_fill_par(self.filename, self.longitude,self.latitude, self.timezone, self.raw_data)

    def _despiking(self):
        """对co2 h2o le h进行despiking"""
        self.raw_data = despiking_data(self.raw_data, self.despiking_z)
    
    def _del_abnormal_value(self):
        """
        1、选取冬天的数据(空气温度连续三天均值<5)
        2、冬天的NEE数据夜间(小于-0.2)的删掉, 白天的NEE(大于等于+-1)
        Returns:
        """
        self.raw_data = del_abnormal_data(self.raw_data, nee_name='co2_despiking', par_name='Par_f')

    def _ustar_fill_partition(self):
        """
        对co2 flux 进行u*计算、gapfilling、partitioning
        对其它的指标只进行 gapfilling 
        """
        self.raw_data = ustar_data(self.filename, self.longitude,
                                   self.latitude, self.timezone, self.raw_data,
                                   self.qc_indicators)