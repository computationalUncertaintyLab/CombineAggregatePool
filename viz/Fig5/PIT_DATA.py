#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    pits = pd.read_csv("../../score_ensemble_models/all_pits.csv")

    def most_recent(x):
        most_recent_release = x.releaseEW.max()
        return x.loc[x.releaseEW==most_recent_release]
    pits = pits.groupby(["Location","EW_forecast_week", "MW_forecast_week"]).apply(most_recent)
    pits = pits.reset_index(drop=True)
    
    def produce_cdfs(x):
        def cdf(x):
            N = len(x)
            x,cx = np.sort(x), np.arange(0,N)/N 
            return x,cx
        x,px = cdf(x.PIT)
        return pd.DataFrame({"cdf_x":x, "cdf_y":px})

    def produce_structured_cdf(x):
        N = len(x)

        dom = np.arange(0,1+0.05,0.05 )
        counts = pd.value_counts(x.PIT,bins = dom)
        cuum   = np.cumsum(counts).values / N

        avg_dom = [0.5*(x+y) for (x,y) in zip(dom,dom[1:]) ]
        return pd.DataFrame({"left":dom[:-1], "right":dom[1:], "avg": avg_dom, "px":cuum})
    
    cdf_pits = pits.groupby(["Location","Target","compute_model_id"]).apply(produce_structured_cdf)
    cdf_pits = cdf_pits.reset_index()

    # cdf_pits["loc_ind"] = [1 if x in {1,5,8} else 0 for x in cdf_pits.Location]

    # loc_one  = cdf_pits.loc[ cdf_pits.loc_ind==1 ]
    # loc_zero = cdf_pits.loc[ cdf_pits.loc_ind==0 ]
    
    model2color = {-1:"b",-2:"r",-3:"g",-7:"k",-14:"y"}
   
    #models = [-1,-2,-3,-7,-14]
    models = [-1,-3, 320,321]
    model2color = {-1:'blue', -2:'blue', -3:'blue', 320:'red', 321:'red'}

    
    #--group by model type
    def ensemble_type(x):
        if x["compute_model_id"] in {-1,320}:
            return "Equal" 
        elif x["compute_model_id"] in {-2}:
            return "Static"
        elif  x["compute_model_id"] in {-3,321}:
            return "Adaptive" 
        else:
            return np.nan
    cdf_pits["ensemble_type"] = cdf_pits.apply(ensemble_type,1) 

    #--group by model type
    def algorithm_type(x):
        if x["compute_model_id"] in {-1,-2,-3}:
            return "Original Alg." 
        elif x["compute_model_id"] in {320,321}:
            return "Plus Alg."
        else:
            return np.nan
    cdf_pits["algorithm_type"] = cdf_pits.apply(algorithm_type,1) 

    cdf_pits = cdf_pits.loc[ cdf_pits.compute_model_id.isin(models) ]

    
    cdf_pits.to_csv("./cdf_pits.csv",index=False)
    


