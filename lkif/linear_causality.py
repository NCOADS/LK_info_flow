import numpy as np
from scipy.stats import norm
from .utils import track
import warnings

class LinearLKInformationFlow(object):
    def __init__(self, dt=1) -> None:
        """
        Parameters:
            dt: Time gap.
        """
        self.dt = dt

        self.conf_level_99 = norm.ppf(0.995)  # 99% confidence level
        self.conf_level_95 = norm.ppf(0.975)  # 95% confidence level
        self.conf_level_90 = norm.ppf(0.95)   # 90% confidence level

    def _generate_pairs(self, N):
        '''
        Generate segements for 1d subspace.
        '''
        return [[i, i + 1] for i in range(N)]

    def _split_matrix(self, matrix, segments):
        slices = [np.s_[start:end] for start, end in segments]
        row_slices, col_slices = np.meshgrid(
            slices, slices, indexing="ij")
        result = np.vectorize(lambda r, c: matrix[r, c], otypes=[
                                   object])(row_slices, col_slices)
        return result

    def _prepare_dataset(self, ts_data_list, segments, lag_list=[1]):
        '''
        prepare for dataset for causality estimation.
        Parameters:
            ts_data_list: Time series(length of time series, number of variables) list.
            segments: A list defining the row and column intervals for dividing the matrix, e.g., [(0, 5), (5, 10)], which devide the matrix into 2 segments.
            lag_list: A list of integers representing the lag order.
        '''
        lag_list_length = len(lag_list)
        assert lag_list_length > 0, f"Assertion failed: lag list must not be empty."
        # sort
        lag_list = sorted(lag_list)
        lag_list_max = lag_list[-1]
        delta_ts_data_list = []
        processed_ts_data_list = []
        for ts_data in ts_data_list:
            delta_ts_data = (ts_data[lag_list_max:, :] -
                             ts_data[lag_list_max - 1: -1, :]) / (self.dt)
            delta_ts_data_list.append(delta_ts_data)

            lag = lag_list[0]
            processed_ts_data_ = ts_data[lag_list_max-lag:-lag, :]

            for i, lag in enumerate(lag_list[1:]):
                processed_ts_data_ = np.hstack(
                    (processed_ts_data_, ts_data[lag_list_max-lag:-lag, :]))

            processed_ts_data_list.append(processed_ts_data_)

        processed_segments = segments.copy()
        for i, lag in enumerate(lag_list[1:]):
            processed_segments += [[x + ts_data.shape[1] *
                                    (i+1), y + ts_data.shape[1]*(i+1)] for x, y in segments]

        delta_ts_data = np.vstack(delta_ts_data_list)
        processed_ts_data = np.vstack(processed_ts_data_list)
        return delta_ts_data, processed_ts_data, processed_segments

    def _inverse_symmetric_mat(self, mat):
        """
        Calculate the inverse of a symmetric matrix.
        """
        # if numpy.linalg.cond(mat) < 1/sys.float_info.epsilon:
        #     inverse_mat = np.linalg.inv(mat)
        #     return (inverse_mat+inverse_mat.T)/2
        # else:
        #     # warning
        #     print("Warning: Matrix is ill-conditioned. Using pseudo-inverse instead.")
        inverse_mat = np.linalg.pinv(mat)
        return (inverse_mat+inverse_mat.T)/2

    def _cal_diag_inv_cov(self, cov):
        diag_inv_cov = np.vectorize(lambda x: self._inverse_symmetric_mat(
            x), otypes=[object])(np.diagonal(cov))
        return diag_inv_cov

    def _cal_information_flow(self, invC_mul_dC, cov, diag_inv_cov):
        def cal_block_information_flow_(i, j):
            return np.trace(invC_mul_dC[i, j] @ cov[j, i] @ diag_inv_cov[i])

        rows, cols = np.indices(invC_mul_dC.shape)
        information_flow = np.vectorize(
            cal_block_information_flow_, otypes=[float])(rows, cols)
        return information_flow

    def _cal_dH_noise(self, diag_inv_cov, error_square_mean):
        def cal_block_dH_noise_(i):
            return np.trace(error_square_mean[i, i] @ diag_inv_cov[i])
        dH_noise = np.vectorize(cal_block_dH_noise_, otypes=[float])(
            np.arange(error_square_mean.shape[0])) * 1/2
        return dH_noise

    def _cal_information_flow_std(self, invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, n):
        def cal_block_cal_variance_(i, j):
            temp = cov[i, j].T@diag_inv_cov[i]
            variance = np.trace(invC_mul_dC[i, j].T @ diag_inv_cov[i] @ invC_mul_dC[i, j] @ (cov[j, j] - cov[j, i]@diag_inv_cov[i]@cov[i, j]))\
                + np.trace((temp.T@inv_cov[j, j]@temp@error_square_mean[i, i]))
            return variance/n

        rows, cols = np.indices(invC_mul_dC.shape)
        information_flow_variance = np.vectorize(
            cal_block_cal_variance_, otypes=[float])(rows, cols)
        return np.sqrt(information_flow_variance)

    def _cal_information_flow_std_origin(self, invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, n):
        def cal_block_cal_variance_(i, j):
            temp = cov[i, j].T@diag_inv_cov[i]
            variance = np.trace(
                (temp.T@inv_cov[j, j]@temp@error_square_mean[i, i]))
            return variance/n

        rows, cols = np.indices(invC_mul_dC.shape)
        information_flow_variance = np.vectorize(
            cal_block_cal_variance_, otypes=[float])(rows, cols)
        return np.sqrt(information_flow_variance)

    def causality_estimate(self, ts_data_list, lag_list=[1], segments=None, significance_test=True) -> None:
        """
        Calculate Liang-Kleeman information flow under linear conditions with significance test. Get the result by calling **get_dict()**.
        Parameters:
            ts_data_list: Time series list(length of time series, number of variables), each elements in the list is supposed to follow the same dynamical system.
            lag_list: A list of integers representing the lag order.
            segments: A list defining the row and column intervals for dividing the matrix, e.g., [[0, 5], [5, 10]],[[0,1],[1,2],[2,3]].
            significance_test: If True, will perform significance test.
        """
        self.significance_test = significance_test
        ts_length, ts_var_num = ts_data_list[0].shape

        if segments == None:
            segments = self._generate_pairs(ts_var_num)
        segments = [sorted(item) for item in segments]
        segments_num = len(segments)

        delta_ts_data, ts_data_process, segments = self._prepare_dataset(
            ts_data_list, segments, lag_list)

        self.segments = segments
        self.lag_list = lag_list

        ts_length = ts_data_process.shape[0]  # overwrite ts_length
        assert ts_length > ts_var_num, f"Assertion failed: length of time series ({ts_length}) must be greater than the number of variables ({ts_var_num})."

        self.deg_freedom = ts_length - ts_var_num
        cov = np.cov(ts_data_process.T)
        if significance_test:
            inv_cov = self._inverse_symmetric_mat(cov)
            inv_cov = self._split_matrix(inv_cov, segments)

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
        cov = self._split_matrix(cov, segments)
        invC_mul_dC = self._split_matrix(invC_mul_dC, segments)[
            :segments_num, :]
        error_square_mean = self._split_matrix(error_square_mean, segments)[
            :segments_num, :segments_num]

        # invariance of block diagonal matrix
        diag_inv_cov = self._cal_diag_inv_cov(cov)
        self.diag_inv_cov = diag_inv_cov
        # calculate informtaion flow
        information_flow = self._cal_information_flow(
            invC_mul_dC, cov, diag_inv_cov)[:segments_num, :]
        self.information_flow = information_flow

        # calculate normalized information flow
        dH_noise = self._cal_dH_noise(
            diag_inv_cov, error_square_mean).reshape(-1, 1)
        self.dH_noise = dH_noise
        normalizer = np.sum(np.abs(
            information_flow), axis=1, keepdims=True) + np.abs(dH_noise)
        self.normalizer = normalizer
        normalized_information_flow = information_flow/normalizer

        self.normalized_information_flow = normalized_information_flow

        if significance_test:
            information_flow_std = self._cal_information_flow_std(
                invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, self.deg_freedom)
            information_flow_std_origin = self._cal_information_flow_std_origin(
                invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, self.deg_freedom)
            self.information_flow_std = information_flow_std
            self.information_flow_std_origin = information_flow_std_origin
            self.p = (1 - norm.cdf(np.abs(self.information_flow /
                      self.information_flow_std))) * 2  # p-value

    def real_information_flow_linear_case(self, det_mat, sto_mat, deg_freedom=None, segments=None, discrete_lyapunov=True) -> dict:
        if discrete_lyapunov:
            from scipy.linalg import solve_discrete_lyapunov
        else:
            from scipy.linalg import solve_continuous_lyapunov
        """
        Calculate the real information flow for linear case.
        dX = A X dt + B dw
        discrete: X(t+1) = (A dt + I) X(t) + B \epsilon

        Parameters:
            det_mat: Deterministic matrix.
            sto_mat: Stochastic matrix.
            deg_freedom: Degree of freedom.
            segments: A list defining the row and column intervals for dividing the matrix, e.g., [[0, 5], [5, 10]],[[0,1],[1,2],[2,3]].
            discrete_lyapunov: If True, will use discrete lyapunov equation.
        """
        # calculate the covariance matrix
        if segments == None:
            segments = self._generate_pairs(det_mat.shape[0])

        significance_test = True
        if deg_freedom == None:
            # judge the existence of the degree of freedom
            if hasattr(self, 'deg_freedom'):
                deg_freedom = self.deg_freedom
            else:
                significance_test = False
                warnings.warn("Degree of freedom is not specified. Will not calculate Std.", UserWarning)

        Q = sto_mat @ sto_mat.T
        segments = [sorted(item) for item in segments]
        segments_num = len(segments)

        if discrete_lyapunov:
            cov = solve_discrete_lyapunov(
                det_mat*self.dt+np.eye(det_mat.shape[0]), Q*self.dt)
        else:
            cov = solve_continuous_lyapunov(det_mat, -Q)

        inv_cov = self._inverse_symmetric_mat(cov)
        inv_cov = self._split_matrix(inv_cov, segments)
        cov = self._split_matrix(cov, segments)
        det_mat = self._split_matrix(det_mat, segments)[
            :segments_num, :]

        Q = self._split_matrix(Q, segments)[
            :segments_num, :segments_num]
        self.Q = Q
        diag_inv_cov = self._cal_diag_inv_cov(cov)
        self.true_cov = cov
        self.true_diag_inv_cov = diag_inv_cov
        # calculate informtaion flow
        information_flow = self._cal_information_flow(
            det_mat, cov, diag_inv_cov)[:segments_num, :]

        dH_noise = self._cal_dH_noise(
            diag_inv_cov, Q).reshape(-1, 1)
        normalizer = np.sum(np.abs(
            information_flow), axis=1, keepdims=True) + np.abs(dH_noise)

        normalized_information_flow = information_flow/normalizer

        if significance_test:
            information_flow_std = self._cal_information_flow_std(
                det_mat, cov, inv_cov, diag_inv_cov, Q/self.dt, deg_freedom)

            information_flow_std_origin = self._cal_information_flow_std_origin(
                det_mat, cov, inv_cov, diag_inv_cov, Q/self.dt, deg_freedom)
            p = (1 - norm.cdf(np.abs(self.information_flow /
                                        self.information_flow_std))) * 2  # p-value
            
            return {
                    "information_flow": information_flow,
                    "normalized_information_flow": normalized_information_flow,
                    "segments": segments,
                    "lag_list": [1],
                    "information_flow_std": information_flow_std,
                    # use original significance test method
                    "information_flow_std_origin": information_flow_std_origin,
                    "statistics": {
                        "p99_critical_value": information_flow_std*self.conf_level_99,
                        "p95_critical_value": information_flow_std*self.conf_level_95,
                        "p90_critical_value": information_flow_std*self.conf_level_90,
                        "p": p
                    }
                }
        
        return {
                "information_flow": information_flow,
                "normalized_information_flow": normalized_information_flow,
                "segments": segments,
                "lag_list": [1]
            }

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
            segments = self._generate_pairs(ts_var_num)
        segments = [sorted(item) for item in segments]
        segments_num = len(segments)

        # ts_data = (ts_data - np.mean(ts_data, axis=0)) / \
        #     np.std(ts_data, axis=0)

        delta_ts_data, ts_data_process, segments = self._prepare_dataset(
            ts_data_list, segments, lag_list)
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
            cov = self._split_matrix(cov, segments)
            invC_mul_dC = self._split_matrix(invC_mul_dC.T, segments)[
                :segments_num, :]
            # invariance of block diagonal matrix
            diag_inv_cov = self._cal_diag_inv_cov(cov)
            # calculate informtaion flow
            information_flow = self._cal_information_flow(
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
