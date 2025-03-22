from tqdm.auto import tqdm
import matplotlib
import numpy as np

CLUSTER = False

def set_cluster(is_cluster):
    global CLUSTER 
    CLUSTER = is_cluster
    if CLUSTER:
        matplotlib.use('Agg')

def is_cluster():
    return CLUSTER

def track(iterator, **kwargs):
    if not CLUSTER:
        return tqdm(iterator, **kwargs)
    else:
        return iterator

def generate_pairs(N):
        '''
        Generate segements for 1d subspace.
        '''
        return [[i, i + 1] for i in range(N)]

def split_matrix(matrix, segments):
    slices = [np.s_[start:end] for start, end in segments]
    row_slices, col_slices = np.meshgrid(
        slices, slices, indexing="ij")
    result = np.vectorize(lambda r, c: matrix[r, c], otypes=[
                                object])(row_slices, col_slices)
    return result

def inverse_symmetric_mat(mat):
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

def cal_diag_inv_cov(cov):
    diag_inv_cov = np.vectorize(lambda x: inverse_symmetric_mat(
        x), otypes=[object])(np.diagonal(cov))
    return diag_inv_cov

def prepare_dataset(ts_data_list, segments, lag_list=[1], dt=1):
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
                            ts_data[lag_list_max - 1: -1, :]) / (dt)
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

def cal_information_flow(invC_mul_dC, cov, diag_inv_cov):
    def cal_block_information_flow_(i, j):
        return np.trace(invC_mul_dC[i, j] @ cov[j, i] @ diag_inv_cov[i])

    rows, cols = np.indices(invC_mul_dC.shape)
    information_flow = np.vectorize(
        cal_block_information_flow_, otypes=[float])(rows, cols)
    return information_flow

def cal_dH_noise(diag_inv_cov, error_square_mean):
    def cal_block_dH_noise_(i):
        return np.trace(error_square_mean[i, i] @ diag_inv_cov[i])
    dH_noise = np.vectorize(cal_block_dH_noise_, otypes=[float])(
        np.arange(error_square_mean.shape[0])) * 1/2
    return dH_noise

def cal_information_flow_std(invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, n):
    def cal_block_cal_variance_(i, j):
        temp = cov[i, j].T@diag_inv_cov[i]
        variance = np.trace(invC_mul_dC[i, j].T @ diag_inv_cov[i] @ invC_mul_dC[i, j] @ (cov[j, j] - cov[j, i]@diag_inv_cov[i]@cov[i, j]))\
            + np.trace((temp.T@inv_cov[j, j]@temp@error_square_mean[i, i]))
        return variance/n

    rows, cols = np.indices(invC_mul_dC.shape)
    information_flow_variance = np.vectorize(
        cal_block_cal_variance_, otypes=[float])(rows, cols)
    return np.sqrt(information_flow_variance)

def cal_information_flow_std_origin(invC_mul_dC, cov, inv_cov, diag_inv_cov, error_square_mean, n):
    def cal_block_cal_variance_(i, j):
        temp = cov[i, j].T@diag_inv_cov[i]
        variance = np.trace(
            (temp.T@inv_cov[j, j]@temp@error_square_mean[i, i]))
        return variance/n

    rows, cols = np.indices(invC_mul_dC.shape)
    information_flow_variance = np.vectorize(
        cal_block_cal_variance_, otypes=[float])(rows, cols)
    return np.sqrt(information_flow_variance)