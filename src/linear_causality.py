import numpy
from scipy.stats import norm
import sys
from src.utils import track


class LinearLKInformationFlow(object):
    def __init__(self, xp=numpy, dt=1) -> None:
        """
        Parameters:
            xp: Module(numpy or cupy). defualt: numpy.
            dt: Time gap.
        """
        self.xp = xp
        self.dt = dt

        self.conf_level_99 = norm.ppf(0.995)  # 99% confidence level
        self.conf_level_95 = norm.ppf(0.975)  # 95% confidence level
        self.conf_level_90 = norm.ppf(0.95)   # 90% confidence level

    def _split_matrix(self, matrix, segments):
        slices = [self.xp.s_[start:end] for start, end in segments]
        row_slices, col_slices = self.xp.meshgrid(
            slices, slices, indexing="ij")
        result = self.xp.vectorize(lambda r, c: matrix[r, c], otypes=[
                                   object])(row_slices, col_slices)
        return result

    def _inverse_symmetric_mat(self, mat):
        """
        Calculate the inverse of a symmetric matrix.
        """
        if self.xp.linalg.cond(mat) < 1/sys.float_info.epsilon:
            inverse_mat = self.xp.linalg.inv(mat)
            return (inverse_mat+inverse_mat.T)/2
        else:
            # warning
            print("Warning: Matrix is ill-conditioned. Using pseudo-inverse instead.")
            inverse_mat = self.xp.linalg.pinv(mat)
            return (inverse_mat+inverse_mat.T)/2

    def _generate_pairs(self, N):
        '''
        Generate segements for 1d subspace.
        '''
        return [[i, i + 1] for i in range(N)]

    def _cal_diag_inv_cov(self, cov):
        diag_inv_cov = self.xp.vectorize(lambda x: self._inverse_symmetric_mat(
            x), otypes=[object])(self.xp.diagonal(cov))
        return diag_inv_cov

    def _cal_information_flow(self, invC_mul_dC, cov, diag_inv_cov):
        def cal_block_information_flow_(i, j):
            return self.xp.trace(invC_mul_dC[i, j] @ cov[j, i] @ diag_inv_cov[i])

        rows, cols = self.xp.indices(invC_mul_dC.shape)
        information_flow = self.xp.vectorize(
            cal_block_information_flow_, otypes=[float])(rows, cols)
        return information_flow

    def _cal_dH_noise(self, diag_inv_cov, error_square_mean):
        def cal_block_dH_noise_(i):
            return self.xp.trace(error_square_mean[i, i] @ diag_inv_cov[i])
        dH_noise = self.xp.vectorize(cal_block_dH_noise_, otypes=[float])(
            self.xp.arange(error_square_mean.shape[0])) * 1/2
        return dH_noise

    def _cal_information_flow_std(self, invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, n):
        def cal_block_cal_variance_(i, j):
            temp = cov[i, j].T@diag_inv_cov[i]
            variance = self.xp.trace(invC_mul_dC[i, j].T @ diag_inv_cov[i] @ invC_mul_dC[i, j] @ (cov[j, j] - cov[j, i]@diag_inv_cov[i]@cov[i, j]))\
                + self.xp.trace((temp.T@inv_cov[j, j]@temp@error_square_mean[i, i]))
            return variance/n

        rows, cols = self.xp.indices(invC_mul_dC.shape)
        information_flow_variance = self.xp.vectorize(
            cal_block_cal_variance_, otypes=[float])(rows, cols)
        return self.xp.sqrt(information_flow_variance)

    def _cal_information_flow_std_origin(self, invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, n):
        def cal_block_cal_variance_(i, j):
            temp = cov[i, j].T@diag_inv_cov[i]
            variance = self.xp.trace(
                (temp.T@inv_cov[j, j]@temp@error_square_mean[i, i]))
            return variance/n

        rows, cols = self.xp.indices(invC_mul_dC.shape)
        information_flow_variance = self.xp.vectorize(
            cal_block_cal_variance_, otypes=[float])(rows, cols)
        return self.xp.sqrt(information_flow_variance)

    def _prepare_dataset(self, ts_data, segments, lag_list=[1]):
        '''
        prepare for dataset for causality estimation.
        Parameters:
            ts_data: Time series(length of time series, number of variables).
            segments: A list defining the row and column intervals for dividing the matrix, e.g., [(0, 5), (5, 10)], which devide the matrix into 2 segments.
            lag_list: A list of integers representing the lag order.
        '''
        lag_list_length = len(lag_list)
        assert lag_list_length > 0, f"Assertion failed: lag list must not be empty."
        # sort
        lag_list = sorted(lag_list)
        lag_list_max = lag_list[-1]

        delta_ts_data = (ts_data[lag_list_max:, :] -
                         ts_data[lag_list_max - 1: -1, :]) / (self.dt)

        lag = lag_list[0]
        processed_ts_data = ts_data[lag_list_max-lag:-lag, :]
        processed_segments = segments.copy()

        for i, lag in enumerate(lag_list[1:]):
            processed_ts_data = self.xp.hstack(
                (processed_ts_data, ts_data[lag_list_max-lag:-lag, :]))
            processed_segments += [[x + ts_data.shape[1] *
                                    (i+1), y + ts_data.shape[1]*(i+1)] for x, y in segments]
        return delta_ts_data, processed_ts_data, processed_segments

    def bootstrap_estimate(self, ts_data, lag_list=[1], segments=None, bootstrap_num=1000, output_all=False):
        '''
        bootstrap method
        Parameters:
            ts_data: Time series(length of time series, number of variables).
            segments: A list defining the row and column intervals for dividing the matrix
            bootstrap_num: Number of bootstrap samples.
            output_all: If True, will output the original list.
        '''
        ts_length, ts_var_num = ts_data.shape
        assert ts_length > ts_var_num, f"Assertion failed: length of time series ({ts_length}) must be greater than the number of variables ({ts_var_num})."
        if segments == None:
            segments = self._generate_pairs(ts_var_num)
        segments = [sorted(item) for item in segments]
        segments_num = len(segments)
        delta_ts_data, ts_data_process, segments = self._prepare_dataset(
            ts_data, segments, lag_list)
        x_centered = ts_data_process - self.xp.mean(ts_data_process, axis=0)
        ts_length = x_centered.shape[0]
        information_flow_list = []
        t = track(range(bootstrap_num), leave=False, desc="Bootstrap Progress")
        for _ in t:
            # 生成相同的随机索引
            indices = self.xp.random.choice(
                range(ts_length), size=ts_length, replace=True)
            # 根据生成的索引抽取对应的样本
            x_centered_sample = x_centered[indices]
            delta_ts_data_sample = delta_ts_data[indices]
            cov = x_centered_sample.T @ x_centered_sample / \
                (ts_length - 1)  # covariance matrix
            # TODO: further develop for significance test for subspace causality
            invC_mul_dC, _, _, _ = self.xp.linalg.lstsq(
                x_centered_sample, delta_ts_data_sample, rcond=None)
            cov = self._split_matrix(cov, segments)
            invC_mul_dC = self._split_matrix(invC_mul_dC.T, segments)[
                :segments_num, :]
            # invariance of block diagonal matrix
            diag_inv_cov = self._cal_diag_inv_cov(cov)
            # calculate informtaion flow
            information_flow = self._cal_information_flow(
                invC_mul_dC, cov, diag_inv_cov)[:segments_num, :]
            information_flow_list.append(information_flow)

        information_flow_mean = self.xp.mean(information_flow_list, axis=0)
        information_flow_std_error = self.xp.std(information_flow_list, axis=0)
        if output_all:
            return {"bootstrap_information_flow_mean": information_flow_mean, "bootstrap_information_flow_std": information_flow_std_error, "bootstrap_information_flow_list": information_flow_list}
        else:
            return {"bootstrap_information_flow_mean": information_flow_mean, "bootstrap_information_flow_std": information_flow_std_error,  "bootstrap_information_flow_list": None}

    def causality_estimate(self, ts_data, lag_list=[1], segments=None, significance_test=True):
        """
        Calculate Liang-Kleeman information flow under linear conditions with significance test.
        Parameters:
            ts_data: Time series(length of time series, number of variables).
            segments: A list defining the row and column intervals for dividing the matrix, e.g., [[0, 5], [5, 10]],[[0,1],[1,2],[2,3]].
        """
        self.significance_test = significance_test
        ts_length, ts_var_num = ts_data.shape

        assert ts_length > ts_var_num, f"Assertion failed: length of time series ({ts_length}) must be greater than the number of variables ({ts_var_num})."
        if segments == None:
            segments = self._generate_pairs(ts_var_num)

        segments = [sorted(item) for item in segments]
        segments_num = len(segments)
        # delta_ts_data = (ts_data[1:,:] - ts_data[:ts_length - 1, :]) / (self.dt)
        delta_ts_data, ts_data_process, segments = self._prepare_dataset(
            ts_data, segments, lag_list)
        self.segments = segments
        self.lag_list = lag_list

        x_centered = ts_data_process - self.xp.mean(ts_data_process, axis=0)
        ts_length = x_centered.shape[0]  # overwrite ts_length
        cov = x_centered.T @ x_centered / (ts_length)  # covariance matrix
        if significance_test:
            inv_cov = self._inverse_symmetric_mat(cov)
            inv_cov = self._split_matrix(inv_cov, segments)

        # estimator of dynamic system matrix : A
        # TODO: further develop for significance test for subspace causality
        invC_mul_dC, _, _, _ = self.xp.linalg.lstsq(
            x_centered, delta_ts_data, rcond=None)

        # error square mean
        error_vec = delta_ts_data - x_centered@invC_mul_dC
        error_square_mean = error_vec.T@error_vec/(ts_length-ts_var_num)

        self.invC_mul_dC = invC_mul_dC.T
        self.error_square_mean = error_square_mean

        # split into block matrix
        cov = self._split_matrix(cov, segments)
        invC_mul_dC = self._split_matrix(invC_mul_dC.T, segments)[
            :segments_num, :]
        error_square_mean = self._split_matrix(error_square_mean, segments)[
            :segments_num, :segments_num]

        # invariance of block diagonal matrix
        diag_inv_cov = self._cal_diag_inv_cov(cov)

        # calculate informtaion flow
        information_flow = self._cal_information_flow(
            invC_mul_dC, cov, diag_inv_cov)[:segments_num, :]
        self.information_flow = information_flow

        # calculate normalized information flow
        dH_noise = self._cal_dH_noise(
            diag_inv_cov, error_square_mean).reshape(-1, 1)
        self.dH_noise = dH_noise
        normalizer = self.xp.sum(self.xp.abs(
            information_flow), axis=1, keepdims=True) + self.xp.abs(dH_noise)
        self.normalizer = normalizer
        normalized_information_flow = information_flow/normalizer

        self.normalized_information_flow = normalized_information_flow

        if significance_test:
            information_flow_std = self._cal_information_flow_std(
                invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, (ts_length-ts_var_num))
            information_flow_std_origin = self._cal_information_flow_std_origin(
                invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, (ts_length-ts_var_num))
            self.information_flow_std = information_flow_std
            self.information_flow_std_origin = information_flow_std_origin
            self.p = (1 - norm.cdf(self.xp.abs(self.information_flow /
                      self.information_flow_std))) * 2  # p-value

    def get_dict(self):
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
