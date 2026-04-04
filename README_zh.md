# LK_info_flow

基于线性随机动力学的 **Liang-Kleeman（LK）信息流**估计库，支持面板数据、自动滞后阶数选择、岭正则化及显著性检验。

---

## 安装

```bash
git clone git@github.com:NCOADS/LK_info_flow.git
cd LK_info_flow
conda create -n lkif python=3.9
conda activate lkif
pip install -r requirements.txt
```

---

## 快速上手

### 1. 初始化模型

```python
from lkif import LinearLKInformationFlow
import numpy as np

dt = 1  # 相邻观测之间的时间步长
model = LinearLKInformationFlow(dt)
```

### 2. 输入数据

输入需为 **二维数组的列表**，每个数组的形状为 `(T, N)`，即 `T` 个时间步、`N` 个变量。列表中的所有数组应服从相同的动力学系统（面板数据）。若直接传入单个数组，会自动包装为列表。

```python
# data: 列表，每个元素形状为 (T, N)
model.data_init(
    data,
    lag_list=[1],           # 用作回归量的滞后阶数
    segments=None,           # 变量分组方式；默认每个变量独立为一组
    significance_test=True,  # 是否计算标准误和 p 值
    criterion="BIC",         # 'AIC' 或 'BIC' 自动选择滞后阶数；None 则使用 lag_list
    max_lag=3,               # 自动选择时搜索的最大滞后长度
    lag_interval=1,          # 候选滞后值之间的步长
    ridge_lambda='auto',     # 岭正则化系数；'auto' 自动选取使条件数达标的最小值
    target_cond=1e3,         # 'auto' 模式下的条件数阈值
)
```

**`segments` 参数** — 控制变量的分组方式，用于分块分析：

| 取值 | 含义 |
|------|------|
| `None`（默认） | 每个变量独立为一组：`[[0], [1], ..., [N-1]]` |
| `[[0,1], [2,3]]` | 变量 0–1 为第一组，变量 2–3 为第二组 |

### 3. 估计信息流

```python
model.causality_estimate()
```

### 4. 获取结果

```python
result = model.get_dict()

# 主要输出
result["information_flow"]             # T_{j->i}，形状为 (n_segments, n_segments)
result["normalized_information_flow"]  # 按总熵预算归一化后的信息流

# 当 significance_test=True 时可用
result["information_flow_std"]              # T_{j->i} 的标准误
result["statistics"]["p"]                   # 双侧 p 值
result["statistics"]["p95_critical_value"]  # 95% 显著性水平临界值
```

---

## 完整示例

```python
import numpy as np
from lkif import LinearLKInformationFlow

# 模拟一个简单的三变量系统
np.random.seed(0)
T, N = 500, 3
data = np.cumsum(np.random.randn(T, N), axis=0)

# 估计信息流
model = LinearLKInformationFlow(dt=1)
model.data_init([data], lag_list=[1], significance_test=True)
model.causality_estimate()

result = model.get_dict()
print("信息流矩阵：\n", result["information_flow"])
print("p 值：\n", result["statistics"]["p"])
```

---

## API 参考

### `LinearLKInformationFlow(dt)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dt` | float | `1` | 相邻观测之间的时间步长 |

### `data_init(...)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ts_data_list` | 数组或数组列表 | — | 输入时间序列，每个形状为 `(T, N)` |
| `euler_step` | int | `1` | 求导数的有限差分步长 |
| `lag_list` | int 列表 | `[1]` | 用作回归量的滞后阶数 |
| `segments` | 列表的列表 | `None` | 变量分组；`None` 表示每个变量独立为一组 |
| `significance_test` | bool | `True` | 是否计算标准误和 p 值 |
| `criterion` | `'AIC'`、`'BIC'` 或 `None` | `None` | 自动选择滞后阶数的准则；`None` 表示直接使用 `lag_list` |
| `max_lag` | int | `10` | 自动选择时搜索的最大滞后长度 |
| `lag_interval` | int | `1` | 候选滞后值之间的步长 |
| `ridge_lambda` | float 或 `'auto'` | `'auto'` | 岭正则化系数 |
| `target_cond` | float | `1e3` | `'auto'` 模式下的条件数阈值 |

### `causality_estimate()`

对所有变量对估计 T_{j→i}，须在 `data_init()` 之后调用。

### `get_dict()`

返回包含以下键的字典：

| 键 | 说明 |
|----|------|
| `information_flow` | 原始信息流矩阵 T_{j→i} |
| `normalized_information_flow` | 按总熵预算归一化后的结果 |
| `segments` | 实际使用的变量分组 |
| `lag_list` | 实际使用的滞后阶数列表 |
| `used_ridge_lambda` | 实际应用的岭正则化系数 |
| `information_flow_std` | 标准误 *（仅 significance_test 模式）* |
| `statistics` | 包含 p 值及各显著性水平临界值的字典 *（仅 significance_test 模式）* |