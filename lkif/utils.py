import matplotlib
import numpy as np
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Global cluster mode flag
# When enabled, suppresses interactive tqdm bars and switches matplotlib to
# a non-interactive backend suitable for headless environments.
# ---------------------------------------------------------------------------

CLUSTER = False


def set_cluster(is_cluster: bool) -> None:
    """
    Configure whether the code is running in a headless cluster environment.

    Parameters
    ----------
    is_cluster : bool
        If True, disables tqdm progress bars and uses the 'Agg' matplotlib backend.
    """
    global CLUSTER
    CLUSTER = is_cluster
    if CLUSTER:
        matplotlib.use('Agg')


def is_cluster() -> bool:
    """Return True if cluster mode is active."""
    return CLUSTER


def track(iterator, **kwargs):
    """
    Wrap an iterator with tqdm in interactive mode; pass through unchanged in cluster mode.

    Parameters
    ----------
    iterator : iterable
    **kwargs : passed directly to tqdm.
    """
    if not CLUSTER:
        return tqdm(iterator, **kwargs)
    return iterator


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def generate_pairs(N: int) -> list:
    """
    Generate default single-variable segments for N variables.

    Returns
    -------
    list of list of int
        [[0], [1], ..., [N-1]]
    """
    return [[i] for i in range(N)]


def prepare_dataset(ts_data_list, segments, euler_step=1, lag_list=[1], dt=1):
    """
    Build the finite-difference target matrix and lagged regressor matrix.

    For each time series X of shape (T, N), computes:
        dX  ≈  (X[t + euler_step] - X[t]) / (dt * euler_step)
        X_t =  [X[t-lag_1], X[t-lag_2], ...]  (horizontally stacked)

    Parameters
    ----------
    ts_data_list : list of ndarray, shape (T, N)
        One or more time series assumed to share the same dynamics.
    segments : list of list of int
        Variable groupings used for block-wise analysis.
    euler_step : int
        Finite-difference step size for derivative approximation.
    lag_list : list of int
        Sorted list of lag orders to include as regressors.
    dt : float
        Observation time step.

    Returns
    -------
    delta_ts_data : ndarray, shape (T', N)
        Stacked finite-difference targets across all time series.
    processed_ts_data : ndarray, shape (T', N * len(lag_list))
        Stacked lagged regressor matrices.
    processed_segments : list of list of int
        Extended segments accounting for the additional lag columns.
    """
    lag_list = sorted(lag_list)
    lag_max = lag_list[-1]

    delta_list, regressor_list = [], []

    for ts_data in ts_data_list:
        # Finite-difference approximation of dX/dt
        delta = (
            ts_data[lag_max + euler_step - 1:, :]
            - ts_data[lag_max - 1: -euler_step, :]
        ) / (dt * euler_step)
        delta_list.append(delta)

        # Lagged regressors: start with the first lag, then hstack remaining lags
        regressors = ts_data[lag_max - lag_list[0]: -lag_list[0] - euler_step + 1, :]
        for lag in lag_list[1:]:
            regressors = np.hstack(
                (regressors, ts_data[lag_max - lag: -lag - euler_step + 1, :])
            )
        regressor_list.append(regressors)

    # Extend segments to cover the repeated columns introduced by each additional lag
    N = ts_data_list[0].shape[1]
    processed_segments = segments.copy()
    for i in range(len(lag_list) - 1):
        processed_segments += [
            [x + N * (i + 1) for x in seg] for seg in segments
        ]

    return np.vstack(delta_list), np.vstack(regressor_list), processed_segments


# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

def split_matrix(matrix: np.ndarray, segments: list) -> np.ndarray:
    """
    Partition a matrix into an object array of blocks defined by `segments`.

    Parameters
    ----------
    matrix : ndarray, shape (M, K)
    segments : list of list of int
        Each entry lists the row/column indices belonging to that segment.

    Returns
    -------
    result : ndarray of object, shape (n_segments, n_segments)
        result[i, j] is the sub-matrix formed by rows segments[i] and columns segments[j].
        Indices that exceed matrix dimensions are silently dropped.
    """
    n = len(segments)

    def extract_block(i, j):
        rows = [r for r in segments[i] if r < matrix.shape[0]]
        cols = [c for c in segments[j] if c < matrix.shape[1]]
        if not rows or not cols:
            return np.empty((len(rows), len(cols)))
        return matrix[np.ix_(rows, cols)]

    idx = np.indices((n, n))
    return np.vectorize(extract_block, otypes=[object])(idx[0], idx[1])


def inverse_symmetric_mat(mat: np.ndarray, ridge_lambda: float = 0) -> np.ndarray:
    """
    Compute the inverse of a symmetric matrix with optional ridge regularization.

    The regularized matrix is:  mat + lambda * max(diag(mat)) * I

    Falls back to the Moore-Penrose pseudo-inverse if the standard inverse fails.
    The result is symmetrized to suppress floating-point asymmetry.

    Parameters
    ----------
    mat : ndarray, shape (N, N)
    ridge_lambda : float
        Regularization coefficient (applied relative to the largest diagonal entry).

    Returns
    -------
    ndarray, shape (N, N)
    """
    max_diag = np.max(np.diag(mat))
    if max_diag == 0 or np.isnan(max_diag):
        max_diag = 1.0

    mat_reg = mat + np.eye(mat.shape[0]) * ridge_lambda * max_diag

    try:
        inv = np.linalg.inv(mat_reg)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(mat_reg)

    return (inv + inv.T) / 2


def cal_diag_inv_cov(cov: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of each diagonal block of a split covariance matrix.

    Parameters
    ----------
    cov : ndarray of object, shape (n_segments, n_segments)
        Output of `split_matrix` applied to a covariance matrix.

    Returns
    -------
    ndarray of object, shape (n_segments,)
        diag_inv_cov[i] = inverse of cov[i, i].
    """
    return np.vectorize(inverse_symmetric_mat, otypes=[object])(np.diagonal(cov))


# ---------------------------------------------------------------------------
# Information flow calculations
# ---------------------------------------------------------------------------

def cal_information_flow(invC_mul_dC, cov, diag_inv_cov) -> np.ndarray:
    """
    Compute the LK information flow T_{j->i} for all (i, j) pairs.

    T_{j->i} = tr( A[i,j] · C[j,i] · C[i,i]^{-1} )

    where A = invC_mul_dC (drift matrix blocks) and C = cov (covariance blocks).

    Parameters
    ----------
    invC_mul_dC : ndarray of object, shape (n, m)
    cov : ndarray of object, shape (n, m)
    diag_inv_cov : ndarray of object, shape (n,)

    Returns
    -------
    ndarray, shape (n, m)
    """
    def _block(i, j):
        return np.trace(invC_mul_dC[i, j] @ cov[j, i] @ diag_inv_cov[i])

    idx = np.indices(invC_mul_dC.shape)
    return np.vectorize(_block, otypes=[float])(idx[0], idx[1])


def cal_dH_noise(diag_inv_cov, error_square_mean, dt: float) -> np.ndarray:
    """
    Compute the noise contribution to the rate of entropy change for each variable.

    dH_noise[i] = (1/2) * dt * tr( Q[i,i] · C[i,i]^{-1} )

    where Q = error_square_mean (proportional to B·B^T).

    Parameters
    ----------
    diag_inv_cov : ndarray of object, shape (n,)
    error_square_mean : ndarray of object, shape (n, n)
    dt : float

    Returns
    -------
    ndarray, shape (n,)
    """
    def _block(i):
        return np.trace(error_square_mean[i, i] @ diag_inv_cov[i])

    return np.vectorize(_block, otypes=[float])(np.arange(error_square_mean.shape[0])) * 0.5 * dt


def cal_information_flow_std(invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, n: int) -> np.ndarray:
    """
    Compute the asymptotic standard error of T_{j->i} for all (i, j) pairs.

    Based on the delta method applied to the plug-in estimator of the information flow.

    Parameters
    ----------
    invC_mul_dC : ndarray of object, shape (n_seg, m_seg)
    cov : ndarray of object, shape (n_seg, m_seg)
    inv_cov : ndarray of object, shape (n_seg, m_seg)
        Inverse of the full (possibly regularized) covariance matrix, split into blocks.
    diag_inv_cov : ndarray of object, shape (n_seg,)
    error_square_mean : ndarray of object, shape (n_seg, n_seg)
    n : int
        Degrees of freedom used to normalize the variance estimate.

    Returns
    -------
    ndarray, shape (n_seg, m_seg)
        Standard error of T_{j->i}.
    """
    def _block_variance(i, j):
        temp = cov[i, j].T @ diag_inv_cov[i]
        variance = (
            np.trace(
                invC_mul_dC[i, j].T @ diag_inv_cov[i]
                @ invC_mul_dC[i, j]
                @ (cov[j, j] - cov[j, i] @ diag_inv_cov[i] @ cov[i, j])
            )
            + np.trace(temp.T @ inv_cov[j, j] @ temp @ error_square_mean[i, i])
        )
        return variance / n

    idx = np.indices(invC_mul_dC.shape)
    variance = np.vectorize(_block_variance, otypes=[float])(idx[0], idx[1])
    return np.sqrt(np.abs(variance))