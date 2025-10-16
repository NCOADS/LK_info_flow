# LK_info_flow


### intsall

    git clone git@github.com:NCOADS/LK_info_flow.git
    cd LK_info_flow
    conda create -n lkif python=3.9
    conda activate lkif
    pip install -r requirements.txt


### Instruction

1. Class

    ```
    from lkif import LinearLKInformationFlow
    import numpy as np 
    dt = 1 # time interval 
    lkif_linear = LinearLKInformationFlow(dt)
    ```


2. Data input requirements
    
   + The input is a data list (`list`) (if you input an array, it will be processed as [array]), designed to support panel data. 

        + Each element in the list follows the same dynamical system.  

        + The shape of each element is (time series length × number of variables).
    ```
    lkif_linear.data_init(data, segments=segments, criterion="BIC", max_lag=3)
    ``` 
3. Significance test

    ```
    ## XX is the data with shape (number of variables × time series length)
    lkif_linear.causality_estimate()
    ```

4. Results

    ```
    result_dict = lkif_linear.get_dict()
    ```