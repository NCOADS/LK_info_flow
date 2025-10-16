import numpy as np
from scipy.stats import norm
from .utils import track, generate_pairs, split_matrix, inverse_symmetric_mat, cal_diag_inv_cov, prepare_dataset
from .utils import cal_information_flow, cal_dH_noise, cal_information_flow_std
import warnings

class LinearLKInformationFlow(object):
    def __init__(self, dt=1) -> None:
        """
        Parameters:
        ts_data_list: Time series list(length of time series, number of variables), each elements in the list is supposed to follow the same dynamical system.
        dt: Time step between two consecutive data points.
        euler_step: The step length used in Euler's method to approximate the derivative.
        lag_list: A list of integers representing the lag order.
        segments: A list defining the row and column intervals for dividing the matrix, e.g., [(0,1,2),(3,4,5)]; [[0],[1],[2]]
        significance_test: If True, will perform significance test.
        """
        self.dt = dt

        self.conf_level_99 = norm.ppf(0.995)  # 99% confidence level
        self.conf_level_95 = norm.ppf(0.975)  # 95% confidence level
        self.conf_level_90 = norm.ppf(0.95)   # 90% confidence level



    def data_init(self,ts_data_list,euler_step=1, lag_list=[1], segments=None, significance_test=True, criterion=None, max_lag=10) -> None:
        """
        Initialize data for causality estimation.
        Parameters:
            ts_data_list: Time series list(length of time series, number of variables), each elements in the list is supposed to follow the same dynamical system.
            lag_list: A list of integers representing the lag order.
            segments: A list defining the row and column intervals for dividing the matrix, e.g., [(0,1,2),(3,4,5)]; [[0],[1],[2]]
            significance_test: If True, will perform significance test.
            criterion: 'AIC' or 'BIC' for automatic lag selection. If None, use the provided lag_list.
            max_lag: Maximum lag length for automatic lag selection.
        """
        self.significance_test = significance_test

        if not isinstance(ts_data_list, list):
            ts_data_list = [ts_data_list]

        if criterion is None:
            self.prepare_dataset(ts_data_list,euler_step, lag_list, segments)
            self.linear_dynamic_estimate()
        else:
            self.select_optimal_lag(ts_data_list, max_lag=max_lag, criterion=criterion, euler_step=euler_step, segments=segments)
        
        self.covariance_estimate()



    def prepare_dataset(self,ts_data_list,euler_step=1, lag_list=[1], segments=None) -> None:
        """
        Prepare dataset for causality estimation.
        Parameters:
            ts_data_list: Time series list(length of time series, number of variables), each elements in the list is supposed to follow the same dynamical system.
            lag_list: A list of integers representing the lag order.
            segments: A list defining the row and column intervals for dividing the matrix, e.g., [(0,1,2),(3,4,5)]; [[0],[1],[2]]
        Returns:
            delta_ts_data: The prepared delta time series data.
            ts_data_process: The processed time series data.
            segments: The processed segments.
        """
        ts_length, ts_var_num = ts_data_list[0].shape

        if segments == None:
            segments = generate_pairs(ts_var_num)
        segments = [sorted(item) for item in segments]
        self.original_segments = segments.copy()
        segments_num = len(segments)
        delta_ts_data, ts_data_process, segments = prepare_dataset(
            ts_data_list, segments, euler_step, lag_list, self.dt)

        ts_length = ts_data_process.shape[0]
        assert ts_length > ts_var_num, f"Assertion failed: length of time series ({ts_length}) must be greater than the number of variables ({ts_var_num})."

        self.lag_list = lag_list
        self.delta_ts_data = delta_ts_data
        self.ts_data_process = ts_data_process
        self.segments = segments   # If there's a lag_list, the segments will be appended accordingly. Your can overwrite it in causality_estimate.
        self.segments_num = segments_num
        self.deg_freedom = ts_length - ts_var_num


    def linear_dynamic_estimate(self) -> None:
        # estimator of dynamic system matrix : A
        ones_column = np.ones((self.ts_data_process.shape[0], 1))  # 添加常数列
        ts_data_process_augmented = np.concatenate(
            [self.ts_data_process, ones_column], axis=1)
        self.ts_data_process_augmented = ts_data_process_augmented
        invC_mul_dC, _, _, _ = np.linalg.lstsq(
            ts_data_process_augmented, self.delta_ts_data, rcond=None)
        # self.invC_mul_dC_ = invC_mul_dC.copy()
        # error square mean, related to B
        error_vec = self.delta_ts_data - ts_data_process_augmented@invC_mul_dC
        error_square_mean = error_vec.T@error_vec/self.deg_freedom
        self.invC_mul_dC = invC_mul_dC[:-1, :].T
        self.error_square_mean = error_square_mean
    
    
    def select_optimal_lag(self, ts_data_list, max_lag=10, criterion='BIC', euler_step=1, segments=None):
        """
        使用 AIC 或 BIC 准则自动选择最优 lag 长度
        
        Parameters:
            ts_data_list: 时间序列列表
            max_lag: 最大 lag 长度
            criterion: 'AIC' 或 'BIC'
            euler_step: 欧拉步长
            segments: 分段定义
        
        Returns:
            best_lag_list: 最优的 lag 列表
            ic_values: 每个 lag 对应的信息准则值
        """
        ic_values = []
        lag_configs = []
        
        for lag_len in track(range(1, max_lag + 1), desc="Selecting optimal lag"):
            lag_list = list(range(1, lag_len + 1))
            lag_configs.append(lag_list)
            
            # 准备数据集
            self.prepare_dataset(ts_data_list, euler_step, lag_list, segments)
            
            # 估计动力学系统
            self.linear_dynamic_estimate()
            
            # 计算损失函数（RSS: Residual Sum of Squares）
            n_samples = self.ts_data_process.shape[0]
            n_vars = self.delta_ts_data.shape[1]
            
            # 参数数量 = (变量数 * lag长度) * 输出变量数 + 常数项
            n_params = (self.ts_data_process.shape[1] + 1) * n_vars
            
            # 残差平方和
            rss = np.trace(self.error_square_mean) * self.deg_freedom
            
            # 计算 AIC 或 BIC
            if criterion == 'AIC':
                ic = n_samples * np.log(rss / n_samples) + 2 * n_params
            elif criterion == 'BIC':
                ic = n_samples * np.log(rss / n_samples) + np.log(n_samples) * n_params
            else:
                raise ValueError("criterion must be 'AIC' or 'BIC'")
            
            ic_values.append(ic)
        
        # 选择最小 IC 值对应的 lag
        best_idx = np.argmin(ic_values)
        best_lag_list = lag_configs[best_idx]
        
        # 使用最优 lag 重新准备数据集
        self.prepare_dataset(ts_data_list, euler_step, best_lag_list, segments)
        self.linear_dynamic_estimate()
        
        print(f"Best lag list: {best_lag_list} with {criterion} = {ic_values[best_idx]:.4f}")
        
        return best_lag_list, ic_values


    def covariance_estimate(self) -> None:
        cov = np.cov(self.ts_data_process.T)
        self.cov = cov


    def causality_estimate_takens(self) -> None:
        """
        A quick way to get the causality estimate without a significance_test.
        Mostly used in Takens embedding.
        """
        cov = split_matrix(self.cov, self.original_segments)
        invC_mul_dC = split_matrix(self.invC_mul_dC, self.original_segments)[
            :self.segments_num, :self.segments_num]
        error_square_mean = split_matrix(self.error_square_mean, self.original_segments)[
            :self.segments_num, :self.segments_num]

        # invariance of block diagonal matrix
        diag_inv_cov = cal_diag_inv_cov(cov)
        self.diag_inv_cov = diag_inv_cov
        # calculate informtaion flow

        information_flow = cal_information_flow(
            invC_mul_dC, cov, diag_inv_cov)[:self.segments_num, :]
        self.information_flow = information_flow

        # calculate normalized information flow w.r to local
        dH_noise = cal_dH_noise(
            diag_inv_cov, error_square_mean).reshape(-1, 1)
        normalizer = np.sum(np.abs(
            information_flow), axis=1, keepdims=True) + np.abs(dH_noise)
        normalized_information_flow = information_flow/normalizer
        self.dH_noise = dH_noise
        self.normalizer = normalizer
        self.normalized_information_flow = normalized_information_flow




    def causality_estimate(self) -> None:
        """
        Calculate Liang-Kleeman information flow under linear conditions with significance test. Get the result by calling **get_dict()**.
        Parameters:
            segments_overwite: Overwrite the segments defined in prepare_dataset.
            significance_test: If True, will perform significance test.
        """
        if self.significance_test:
            inv_cov = inverse_symmetric_mat(self.cov)
            self.inv_cov = inv_cov

        cov = split_matrix(self.cov, self.segments)
        invC_mul_dC = split_matrix(self.invC_mul_dC, self.segments)[
            :self.segments_num, :]
        error_square_mean = split_matrix(self.error_square_mean, self.original_segments)[
            :self.segments_num, :self.segments_num]

        # invariance of block diagonal matrix
        diag_inv_cov = cal_diag_inv_cov(cov)
        self.diag_inv_cov = diag_inv_cov
        # calculate informtaion flow

        information_flow = cal_information_flow(
            invC_mul_dC, cov, diag_inv_cov)[:self.segments_num, :]
        self.information_flow = information_flow

        # calculate normalized information flow
        dH_noise = cal_dH_noise(
            diag_inv_cov, error_square_mean).reshape(-1, 1)
        normalizer = np.sum(np.abs(
            information_flow), axis=1, keepdims=True) + np.abs(dH_noise)
        normalized_information_flow = information_flow/normalizer
        self.dH_noise = dH_noise
        self.normalizer = normalizer
        self.normalized_information_flow = normalized_information_flow

        if self.significance_test:
            inv_cov = split_matrix(self.inv_cov, self.segments)
            information_flow_std = cal_information_flow_std(
                invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, self.deg_freedom)
            self.information_flow_std = information_flow_std
            self.p = (1 - norm.cdf(np.abs(self.information_flow /
                      self.information_flow_std))) * 2  # p-value


    def get_dict(self):
        """
        Get the information flow and normalized information flow.
        Returns:
            information_flow: Information flow matrix. (i,j) represents (j → i)'s information flow.
            normalized_information_flow: Normalized information flow matrix.
            segments: Segments of the matrix.
            lag_list: Lag list of the matrix.


            information_flow_std: Standard deviation of information flow.
            information_flow_std_origin: Standard deviation of information flow for original method.
            statistics: Statistics of the information flow.
                p99_critical_value: 99% critical value.
                p95_critical_value: 95% critical value.
                p90_critical_value: 90% critical value.
                p: p-value of the information flow.
        """
        if hasattr(self, 'information_flow'):
            state_dict = {
                "information_flow": self.information_flow,
                "normalized_information_flow": self.normalized_information_flow,
                "segments": self.segments,
                "lag_list": self.lag_list
            }
            if self.significance_test:
                state_dict.update({
                    "information_flow_std": self.information_flow_std,
                    "statistics": {
                        "p99_critical_value": self.information_flow_std*self.conf_level_99,
                        "p95_critical_value": self.information_flow_std*self.conf_level_95,
                        "p90_critical_value": self.information_flow_std*self.conf_level_90,
                        "p": self.p
                    }
                })
            return state_dict
        else:
            return "Causality estimate has not been run yet. Please run 'causality_estimate' first!"
