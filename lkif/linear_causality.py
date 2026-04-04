import numpy as np
from scipy.stats import norm
from .utils import (
    track, generate_pairs, split_matrix, inverse_symmetric_mat,
    cal_diag_inv_cov, prepare_dataset,
    cal_information_flow, cal_dH_noise, cal_information_flow_std
)


class LinearLKInformationFlow:
    """
    Estimates Liang-Kleeman (LK) information flow under linear stochastic dynamics.

    The underlying model is a multivariate linear SDE:
        dX = A·X dt + B dW
    where A is estimated via least squares and B·B^T is inferred from residuals.

    Usage
    -----
    model = LinearLKInformationFlow(dt=0.1)
    model.data_init(ts_data_list, lag_list=[1])
    model.causality_estimate()
    result = model.get_dict()
    """

    def __init__(self, dt: float = 1) -> None:
        """
        Parameters
        ----------
        dt : float
            Time step between consecutive observations.
        """
        self.dt = dt
        self.conf_level_99 = norm.ppf(0.995)
        self.conf_level_95 = norm.ppf(0.975)
        self.conf_level_90 = norm.ppf(0.95)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def data_init(
        self,
        ts_data_list,
        euler_step: int = 1,
        lag_list: list = [1],
        segments=None,
        significance_test: bool = True,
        criterion=None,
        max_lag: int = 10,
        lag_interval: int = 1,
        ridge_lambda='auto',
        target_cond: float = 1e3,
    ) -> None:
        """
        Prepare data and estimate the linear dynamics.

        Parameters
        ----------
        ts_data_list : array or list of arrays, shape (T, N)
            One or more time series from the same dynamical system.
        euler_step : int
            Step size used for finite-difference derivative approximation.
        lag_list : list of int
            Lag orders to include as regressors.
        segments : list of list of int, optional
            Variable groupings, e.g. [[0,1],[2,3]]. Defaults to individual variables.
        significance_test : bool
            Whether to compute standard errors and p-values.
        criterion : {'AIC', 'BIC'} or None
            If given, selects the optimal lag automatically; `lag_list` is ignored.
        max_lag : int
            Maximum lag length considered during automatic selection.
        lag_interval : int
            Stride between lags during automatic selection.
        ridge_lambda : float or 'auto'
            Ridge regularization for matrix inversion. 'auto' chooses the
            smallest value that brings the condition number below `target_cond`.
        target_cond : float
            Condition number threshold used when `ridge_lambda='auto'`.
        """
        self.significance_test = significance_test
        self.ridge_lambda = ridge_lambda
        self.target_cond = target_cond

        if not isinstance(ts_data_list, list):
            ts_data_list = [ts_data_list]

        if criterion is None:
            self.prepare_dataset(ts_data_list, euler_step, lag_list, segments)
            self.linear_dynamic_estimate()
        else:
            self.select_optimal_lag(
                ts_data_list, max_lag=max_lag, criterion=criterion,
                euler_step=euler_step, segments=segments, lag_interval=lag_interval,
            )

        self.covariance_estimate()

    def causality_estimate(self) -> None:
        """
        Compute LK information flow T_{j->i} for all variable pairs.

        Results are stored internally and accessible via `get_dict()`.
        If `significance_test=True`, also computes standard errors and p-values.
        """
        cov = split_matrix(self.cov, self.segments)
        invC_mul_dC = split_matrix(self.invC_mul_dC, self.segments)[:self.segments_num, :]
        error_square_mean = split_matrix(
            self.error_square_mean, self.original_segments
        )[:self.segments_num, :self.segments_num]

        diag_inv_cov = cal_diag_inv_cov(cov)
        self.diag_inv_cov = diag_inv_cov

        # Information flow T_{j->i}
        self.information_flow = cal_information_flow(
            invC_mul_dC, cov, diag_inv_cov
        )[:self.segments_num, :]

        # Noise contribution to entropy change
        dH_noise = cal_dH_noise(diag_inv_cov, error_square_mean, self.dt).reshape(-1, 1)
        normalizer = np.abs(self.information_flow).sum(axis=1, keepdims=True) + np.abs(dH_noise)
        self.dH_noise = dH_noise
        self.normalizer = normalizer
        self.normalized_information_flow = self.information_flow / normalizer

        if self.significance_test:
            # Full-matrix ridge inverse (needed for variance formula)
            lam = self._find_min_lambda(self.cov)
            inv_cov_full = inverse_symmetric_mat(self.cov, lam)
            inv_cov = split_matrix(inv_cov_full, self.segments)

            self.information_flow_std = cal_information_flow_std(
                invC_mul_dC, cov, inv_cov, diag_inv_cov,
                error_square_mean, self.deg_freedom,
            )
            self.p = (
                1 - norm.cdf(np.abs(self.information_flow / self.information_flow_std))
            ) * 2

    def get_dict(self) -> dict:
        """
        Return estimation results as a dictionary.

        Returns
        -------
        dict with keys:
            information_flow, normalized_information_flow, segments, lag_list,
            used_ridge_lambda, and (if significance_test) information_flow_std
            and statistics (p99/p95/p90 critical values, p-values).
        """
        if not hasattr(self, 'information_flow'):
            return "Run `causality_estimate()` first."

        result = {
            "information_flow": self.information_flow,
            "normalized_information_flow": self.normalized_information_flow,
            "segments": self.segments,
            "lag_list": self.lag_list,
            "used_ridge_lambda": (
                "auto (adaptive)" if str(self.ridge_lambda).lower() == 'auto'
                else self.ridge_lambda
            ),
        }
        if self.significance_test:
            result.update({
                "information_flow_std": self.information_flow_std,
                "statistics": {
                    "p99_critical_value": self.information_flow_std * self.conf_level_99,
                    "p95_critical_value": self.information_flow_std * self.conf_level_95,
                    "p90_critical_value": self.information_flow_std * self.conf_level_90,
                    "p": self.p,
                },
            })
        return result

    # ------------------------------------------------------------------
    # Internal estimation steps
    # ------------------------------------------------------------------

    def prepare_dataset(self, ts_data_list, euler_step=1, lag_list=[1], segments=None) -> None:
        """Build regressor matrix and finite-difference targets from raw time series."""
        ts_length, ts_var_num = ts_data_list[0].shape

        if segments is None:
            segments = generate_pairs(ts_var_num)
        segments = [sorted(item) for item in segments]
        self.original_segments = segments.copy()

        delta_ts_data, ts_data_process, segments = prepare_dataset(
            ts_data_list, segments, euler_step, lag_list, self.dt
        )

        ts_length = ts_data_process.shape[0]
        assert ts_length > ts_var_num, (
            f"Time series length ({ts_length}) must exceed variable count ({ts_var_num})."
        )

        self.lag_list = lag_list
        self.delta_ts_data = delta_ts_data
        self.ts_data_process = ts_data_process
        self.segments = segments
        self.segments_num = len(self.original_segments)
        self.deg_freedom = ts_length - ts_var_num

    def linear_dynamic_estimate(self) -> None:
        """
        Estimate the drift matrix A and noise covariance B·B^T via OLS.

        Solves:  [X | 1] · [A; b]^T ≈ dX/dt
        Stores `invC_mul_dC` (= A^T) and `error_square_mean` (proportional to B·B^T).
        """
        ones = np.ones((self.ts_data_process.shape[0], 1))
        X_aug = np.hstack([self.ts_data_process, ones])

        coeffs, _, _, _ = np.linalg.lstsq(X_aug, self.delta_ts_data, rcond=None)
        residuals = self.delta_ts_data - X_aug @ coeffs

        self.invC_mul_dC = coeffs[:-1, :].T          # drop bias row; shape (N, N·lags)
        self.error_square_mean = residuals.T @ residuals / self.deg_freedom

    def covariance_estimate(self) -> None:
        """Estimate the sample covariance of the regressor matrix."""
        self.cov = np.cov(self.ts_data_process.T)

    def select_optimal_lag(
        self, ts_data_list, max_lag=10, criterion='BIC',
        euler_step=1, segments=None, lag_interval=3,
    ):
        """
        Grid-search over lag lengths and pick the one minimising AIC or BIC.

        Parameters
        ----------
        ts_data_list : array or list of arrays, shape (T, N)
            One or more time series from the same dynamical system.
        max_lag : int
            Maximum number of distinct lags to consider.
        criterion : {'AIC', 'BIC'}
            Information criterion used for model selection.
        euler_step : int
            Step size used for finite-difference derivative approximation.
        segments : list of list of int, optional
            Variable groupings. Defaults to individual variables.
        lag_interval : int
            Stride between successive lag values.

        Returns
        -------
        best_lag_list : list of int
            Optimal lag configuration.
        ic_values : list of float
            Information criterion value for each candidate lag length.
        """
        ic_values = []
        lag_configs = []

        for lag_len in track(range(1, max_lag + 1), desc=f"Lag selection ({criterion})"):
            lag_list = list(range(1, lag_len * lag_interval + 1, lag_interval))
            lag_configs.append(lag_list)

            self.prepare_dataset(ts_data_list, euler_step, lag_list, segments)
            self.linear_dynamic_estimate()

            n = self.ts_data_process.shape[0]
            k = (self.ts_data_process.shape[1] + 1) * self.delta_ts_data.shape[1]
            rss = np.trace(self.error_square_mean) * self.deg_freedom

            if criterion == 'AIC':
                ic = n * np.log(rss / n) + 2 * k
            elif criterion == 'BIC':
                ic = n * np.log(rss / n) + np.log(n) * k
            else:
                raise ValueError("criterion must be 'AIC' or 'BIC'")

            ic_values.append(ic)

        best_idx = int(np.argmin(ic_values))
        best_lag_list = lag_configs[best_idx]

        self.prepare_dataset(ts_data_list, euler_step, best_lag_list, segments)
        self.linear_dynamic_estimate()

        return best_lag_list, ic_values

    def _find_min_lambda(self, mat: np.ndarray) -> float:
        """
        Find the smallest ridge coefficient that reduces cond(mat) below `self.target_cond`.

        Returns 0.0 if the matrix is already well-conditioned, or `self.ridge_lambda`
        directly if it is not set to 'auto'.
        """
        if not (isinstance(self.ridge_lambda, str) and self.ridge_lambda.lower() == 'auto'):
            return float(self.ridge_lambda)

        if np.linalg.cond(mat) <= self.target_cond:
            return 0.0

        max_diag = np.max(np.abs(np.diag(mat))) or 1.0
        for lam in np.logspace(-8, 0, 200):
            if np.linalg.cond(mat + np.eye(mat.shape[0]) * lam * max_diag) <= self.target_cond:
                return float(lam)

        return 1.0  # fallback for severely ill-conditioned matrices