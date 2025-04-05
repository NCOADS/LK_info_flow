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
    
   + The input is a data list (`list`), designed to support panel data. 

        + Each element in the list follows the same dynamical system.  

        + The shape of each element is (time series length × number of variables).

3. Significance test

    ```
    ## XX is the data with shape (number of variables × time series length)
    lkif_linear.causality_estimate([XX[:,-15000:].T], lag_list=[1])

    ## panel data input with lag terms of -1 and -2 considered.
    lkif_linear.causality_estimate([XX[:,-15000:].T,XX[:,-30000:-15000].T], lag_list=[1,2])

    ## significance test of subspace-wise information flow with subspaces α₁ = (0,1,2,3,4) and α₂ = (5,6,7,8,9).
    segments = [(0,5),(5,10)]
    lkif_linear.causality_estimate([XX[:,-15000:].T,XX[:,-30000:-15000].T], lag_list=[1,2], segments = segments)
    ```

4. Results

    ```
    result_dict = lkif_linear.get_dict()
    ```