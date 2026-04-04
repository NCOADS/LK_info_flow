# LK_info_flow

[中文文档](./README_zh.md)

A Python library for estimating **Liang-Kleeman (LK) information flow** under linear stochastic dynamics, with support for panel data, automatic lag selection, ridge regularization, and significance testing.

---

## Installation

```bash
git clone git@github.com:NCOADS/LK_info_flow.git
cd LK_info_flow
conda create -n lkif python=3.9
conda activate lkif
pip install -r requirements.txt
```

---

## Quick Start

### 1. Initialize the model

```python
from lkif import LinearLKInformationFlow
import numpy as np

dt = 1  # time step between consecutive observations
model = LinearLKInformationFlow(dt)
```

### 2. Load data

The input must be a **list of 2-D arrays**, where each array has shape `(T, N)` — `T` time steps and `N` variables. All arrays in the list are assumed to follow the same underlying dynamical system (panel data). A bare array is also accepted and will be automatically wrapped into a list.

```python
# data: list of arrays, each with shape (T, N)
model.data_init(
    data,
    lag_list=[1],          # lag orders to use as regressors
    segments=None,          # variable groupings; defaults to individual variables
    significance_test=True, # compute standard errors and p-values
    criterion="BIC",        # 'AIC' or 'BIC' for automatic lag selection; None to use lag_list
    max_lag=3,              # maximum lag length to search when criterion is set
    lag_interval=1,         # stride between candidate lag values
    ridge_lambda='auto',    # ridge regularization; 'auto' selects the minimum stabilizing value
    target_cond=1e3,        # condition number threshold for 'auto' ridge selection
)
```

**`segments` parameter** — controls how variables are grouped for block-wise analysis:

| Value | Meaning |
|-------|---------|
| `None` (default) | Each variable is its own group: `[[0], [1], ..., [N-1]]` |
| `[[0,1], [2,3]]` | Variables 0–1 form group 1; variables 2–3 form group 2 |

### 3. Estimate information flow

```python
model.causality_estimate()
```

### 4. Retrieve results

```python
result = model.get_dict()

# Key outputs
result["information_flow"]             # T_{j->i}, shape (n_segments, n_segments)
result["normalized_information_flow"]  # normalized by total entropy budget

# Available when significance_test=True
result["information_flow_std"]         # standard error of T_{j->i}
result["statistics"]["p"]              # two-sided p-values
result["statistics"]["p95_critical_value"]  # 95% critical values
```

---

## Full Example

```python
import numpy as np
from lkif import LinearLKInformationFlow

# Simulate a simple 3-variable system
np.random.seed(0)
T, N = 500, 3
data = np.cumsum(np.random.randn(T, N), axis=0)

# Estimate information flow
model = LinearLKInformationFlow(dt=1)
model.data_init([data], lag_list=[1], significance_test=True)
model.causality_estimate()

result = model.get_dict()
print("Information flow:\n", result["information_flow"])
print("p-values:\n", result["statistics"]["p"])
```

---

## API Reference

### `LinearLKInformationFlow(dt)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dt` | float | `1` | Time step between observations |

### `data_init(...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ts_data_list` | array or list of arrays | — | Input time series, each shape `(T, N)` |
| `euler_step` | int | `1` | Finite-difference step for derivative approximation |
| `lag_list` | list of int | `[1]` | Lag orders used as regressors |
| `segments` | list of list of int | `None` | Variable groupings; `None` = one group per variable |
| `significance_test` | bool | `True` | Whether to compute standard errors and p-values |
| `criterion` | `'AIC'`, `'BIC'`, or `None` | `None` | Automatic lag selection criterion; `None` uses `lag_list` |
| `max_lag` | int | `10` | Maximum lag length searched when `criterion` is set |
| `lag_interval` | int | `1` | Stride between candidate lag values |
| `ridge_lambda` | float or `'auto'` | `'auto'` | Ridge regularization coefficient |
| `target_cond` | float | `1e3` | Condition number threshold for `'auto'` ridge selection |

### `causality_estimate()`

Estimates T_{j→i} for all variable pairs. Must be called after `data_init()`.

### `get_dict()`

Returns a dictionary with the following keys:

| Key | Description |
|-----|-------------|
| `information_flow` | Raw information flow matrix T_{j→i} |
| `normalized_information_flow` | Normalized by the total entropy budget |
| `segments` | Segment definitions used |
| `lag_list` | Lag list used |
| `used_ridge_lambda` | Actual ridge coefficient applied |
| `information_flow_std` | Standard errors *(significance_test only)* |
| `statistics` | Dict of p-values and critical values *(significance_test only)* |