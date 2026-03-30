from tqdm.auto import tqdm
import matplotlib
import numpy as np
import scipy.linalg as la
import warnings
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
        return [[i] for i in range(N)]


def split_matrix(matrix, segments):
    n_segments = len(segments)
    result = np.empty((n_segments, n_segments), dtype=object)
    
    # 使用 vectorize 处理索引对
    def extract_block(i, j):
        row_indices = segments[i]
        col_indices = segments[j]
        
        # 过滤掉超出矩阵边界的索引
        valid_rows = [idx for idx in row_indices if idx < matrix.shape[0]]
        valid_cols = [idx for idx in col_indices if idx < matrix.shape[1]]
        
        # 如果没有有效索引，返回空数组
        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return np.array([]).reshape(len(valid_rows), len(valid_cols))
        
        return matrix[np.ix_(valid_rows, valid_cols)]
    
    rows, cols = np.indices((n_segments, n_segments))
    result = np.vectorize(extract_block, otypes=[object])(rows, cols)
    
    return result

def inverse_symmetric_mat(mat, ridge_lambda=0):
    max_diag = np.max(np.diag(mat))
    if max_diag == 0 or np.isnan(max_diag):
        max_diag = 1.0

    mat_reg = mat + np.eye(mat.shape[0]) * ridge_lambda * max_diag
    
    try:
        inverse_mat = np.linalg.inv(mat_reg)
    except Exception as e:
        inverse_mat = np.linalg.pinv(mat_reg)
        
    return (inverse_mat + inverse_mat.T) / 2

def cal_diag_inv_cov(cov, ridge_lambda=0.):
    diag_inv_cov = np.vectorize(lambda x: inverse_symmetric_mat(x, ridge_lambda), otypes=[object])(np.diagonal(cov))
    return diag_inv_cov

def prepare_dataset(ts_data_list, segments, euler_step=1, lag_list=[1], dt=1):
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
        delta_ts_data = (ts_data[lag_list_max+euler_step-1:, :] -
                            ts_data[lag_list_max - 1: -euler_step, :]) / (dt*euler_step)
        delta_ts_data_list.append(delta_ts_data)

        lag = lag_list[0]
        processed_ts_data_ = ts_data[lag_list_max-lag:-lag-euler_step+1, :]

        for i, lag in enumerate(lag_list[1:]):
            processed_ts_data_ = np.hstack(
                (processed_ts_data_, ts_data[lag_list_max-lag:-lag-euler_step+1, :]))

        processed_ts_data_list.append(processed_ts_data_)

    processed_segments = segments.copy()
    for i, lag in enumerate(lag_list[1:]):
        processed_segments += [[x + ts_data.shape[1] * (i+1) for x in segment] 
                            for segment in segments]

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

def cal_dH_noise(diag_inv_cov, error_square_mean, dt):
    def cal_block_dH_noise_(i):
        return np.trace(error_square_mean[i, i] @ diag_inv_cov[i])
    dH_noise = np.vectorize(cal_block_dH_noise_, otypes=[float])(
        np.arange(error_square_mean.shape[0])) * 1/2 * dt
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
    return np.sqrt(np.abs(information_flow_variance))