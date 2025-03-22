import numpy as np
from scipy.stats import norm
from .utils import track, generate_pairs, split_matrix, inverse_symmetric_mat, cal_diag_inv_cov, prepare_dataset
from .utils import cal_information_flow, cal_dH_noise, cal_information_flow_std, cal_information_flow_std_origin
import warnings

class RBFKernel(object):
    def __init__(self, bandwidth=1):
        self.bandwidth = bandwidth

    def __call__(self, x, y):
        return np.exp(-np.sum((x - y)**2) / (2 * self.bandwidth**2))




class KernelLKInformationFlow(object):
    def __init__(self, dt=1, ker = "gaussian", bandwidth = 1, alpha = 0.05):
        '''
        Parameters:
            dt: Time interval.
        '''
        self.dt = dt
        self.ker = ker
        self.bandwidth = bandwidth
        self.alpha = alpha

    def causality_estimate(self, ts_data_list, lag_list=[1], segments=None, significance_test=True) -> None:
        self.significance_test = significance_test
        ts_length, ts_var_num = ts_data_list[0].shape

        if segments == None:
            segments = generate_pairs(ts_var_num)
        segments = [sorted(item) for item in segments]
        segments_num = len(segments)

        delta_ts_data, ts_data_process, segments = prepare_dataset(
            ts_data_list, segments, lag_list, self.dt)

        self.segments = segments
        self.lag_list = lag_list

        ts_length = ts_data_process.shape[0]  # overwrite ts_length
        assert ts_length > ts_var_num, f"Assertion failed: length of time series ({ts_length}) must be greater than the number of variables ({ts_var_num})."

        self.deg_freedom = ts_length - ts_var_num
        cov = np.cov(ts_data_process.T)
        if significance_test:
            inv_cov = inverse_symmetric_mat(cov)
            inv_cov = split_matrix(inv_cov, segments)

        # estimator of dynamic system matrix : A
        ones_column = np.ones((ts_data_process.shape[0], 1))  # 添加常数列
        ts_data_process_augmented = np.concatenate(
            [ts_data_process, ones_column], axis=1)
        self.ts_data_process_augmented = ts_data_process_augmented
        self.delta_ts_data = delta_ts_data
        invC_mul_dC, _, _, _ = np.linalg.lstsq(
            ts_data_process_augmented, delta_ts_data, rcond=None)
        self.invC_mul_dC_ = invC_mul_dC.copy()
        # error square mean
        error_vec = delta_ts_data - ts_data_process_augmented@invC_mul_dC
        error_square_mean = error_vec.T@error_vec/self.deg_freedom
        # constant_term = invC_mul_dC[-1, :].T  # constant term
        invC_mul_dC = invC_mul_dC[:-1, :].T
        self.invC_mul_dC = invC_mul_dC
        self.error_square_mean = error_square_mean

        # split into block matrix
        self.cov = cov
        cov = split_matrix(cov, segments)
        invC_mul_dC = split_matrix(invC_mul_dC, segments)[
            :segments_num, :]
        error_square_mean = split_matrix(error_square_mean, segments)[
            :segments_num, :segments_num]

        # invariance of block diagonal matrix
        diag_inv_cov = cal_diag_inv_cov(cov)
        self.diag_inv_cov = diag_inv_cov
        # calculate informtaion flow
        information_flow = cal_information_flow(
            invC_mul_dC, cov, diag_inv_cov)[:segments_num, :]
        self.information_flow = information_flow

        # calculate normalized information flow
        dH_noise = cal_dH_noise(
            diag_inv_cov, error_square_mean).reshape(-1, 1)
        self.dH_noise = dH_noise
        normalizer = np.sum(np.abs(
            information_flow), axis=1, keepdims=True) + np.abs(dH_noise)
        self.normalizer = normalizer
        normalized_information_flow = information_flow/normalizer

        self.normalized_information_flow = normalized_information_flow


    def bootstrap_estimate(self, ts_data_list, lag_list=[1], segments=None, bootstrap_num=1000, output_all=False) -> dict:
        '''
        bootstrap method, estimate the information flow.
        Parameters:
            ts_data_list: Time series(length of time series, number of variables) list.
            segments: A list defining the row and column intervals for dividing the matrix
            bootstrap_num: Number of bootstrap samples.
            output_all: If True, will output the original list.
        '''
        ts_length, ts_var_num = ts_data_list[0].shape
        if segments == None:
            segments = generate_pairs(ts_var_num)
        segments = [sorted(item) for item in segments]
        segments_num = len(segments)

        # ts_data = (ts_data - np.mean(ts_data, axis=0)) / \
        #     np.std(ts_data, axis=0)

        delta_ts_data, ts_data_process, segments = prepare_dataset(
            ts_data_list, segments, lag_list, self.dt)
        x_centered = ts_data_process - np.mean(ts_data_process, axis=0)
        ts_length = x_centered.shape[0]
        assert ts_length > ts_var_num, f"Assertion failed: length of time series ({ts_length}) must be greater than the number of variables ({ts_var_num})."

        information_flow_list = []
        t = track(range(bootstrap_num), leave=False, desc="Bootstrap Progress")
        for _ in t:
            indices = np.random.choice(
                range(ts_length), size=ts_length, replace=True)
            x_centered_sample = x_centered[indices]
            delta_ts_data_sample = delta_ts_data[indices]
            cov = x_centered_sample.T @ x_centered_sample / \
                (ts_length - 1)  # covariance matrix
            invC_mul_dC, _, _, _ = np.linalg.lstsq(
                x_centered_sample, delta_ts_data_sample, rcond=None)
            cov = split_matrix(cov, segments)
            invC_mul_dC = split_matrix(invC_mul_dC.T, segments)[
                :segments_num, :]
            # invariance of block diagonal matrix
            diag_inv_cov = cal_diag_inv_cov(cov)
            # calculate informtaion flow
            information_flow = cal_information_flow(
                invC_mul_dC, cov, diag_inv_cov)[:segments_num, :]
            information_flow_list.append(information_flow)

        information_flow_mean = np.mean(information_flow_list, axis=0)
        information_flow_std_error = np.std(information_flow_list, axis=0)
        if output_all:
            return {"bootstrap_information_flow_mean": information_flow_mean, "bootstrap_information_flow_std": information_flow_std_error, "bootstrap_information_flow_list": information_flow_list}
        else:
            return {"bootstrap_information_flow_mean": information_flow_mean, "bootstrap_information_flow_std": information_flow_std_error,  "bootstrap_information_flow_list": None}

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
                    # use original significance test method
                    "information_flow_std_origin": self.information_flow_std_origin,
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
