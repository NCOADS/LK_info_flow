# LK_info_flow


### intsall

    git clone git@github.com:NCOADS/LK_info_flow.git
    cd LK_info_flow
    conda create -n lkif python=3.9
    conda activate lkif
    pip install -r requirements.txt


### Instruction

1. Class 实例化

    ```
    from lkif import LinearLKInformationFlow
    import numpy as np # cupy
    dt = 1 # time interval 间隔时间
    lkif = LinearLKInformationFlow(np,dt)
    ```


2. 数据输入要求
    
    + 输入为数据列表list，用于兼容 Panel Data
        
        + 数据列表中的每一个 element 都服从同一个 dynamical system 

        + 每一个 element 的 shape 都是 (时间序列长度 * 变量个数)

3. 基于显著性检验的检验

    ```
    ## XX为数据 (变量个数 * 时间序列长度)
    lkif.causality_estimate([XX[:,-15000:].T], lag_list=[1])

    ## Panel Data 输入, 滞后考虑 -1,-2
    lkif.causality_estimate([XX[:,-15000:].T,XX[:,-30000:-15000].T], lag_list=[1,2])

    ## 子空间检验，子空间 (0,1,2,3,4) 和 (5,6,7,8,9)
    segments = [(0,5),(5,10)]
    lkif.causality_estimate([XX[:,-15000:].T,XX[:,-30000:-15000].T], lag_list=[1,2], segments = segments)
    ```

4. 获取结果

    ```
    result_dict = lkif.get_dict()
    ```

5. 附加：bootstrap方法 & 真实信息流（已知动力系统，稳态下的信息流）