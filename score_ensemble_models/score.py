#mcandrew

import sys
sys.path.append("../")

import numpy as np
import pandas as pd

from epiweeks import Week

from mods.index import index

def add_forecast_week(x):

    x = x.drop(columns=["FW"])
    
    epiweeks = x.EW.unique()
    ew2forecast = {"EW":[], "Target":[], "FW":[] }

    for ew in epiweeks:
        fw = Week.fromstring(str(ew))
        
        for target in range(4):
            ew2forecast["EW"].append(ew)
            ew2forecast["Target"].append(target)

            FW = (fw+1+target).cdcformat()
            ew2forecast["FW"].append(FW)
    ew2forecast = pd.DataFrame(ew2forecast)
    ew2forecast["FW"] = ew2forecast.FW.astype(int)
    
    return x.merge(ew2forecast, on = ["EW","Target"])

if __name__ == "__main__":

    #--download forecast data
    idx = index("../analysisdata/")
    D = idx.grabEpiData_f()

    #--FluSight rules round wili values to two decimals
    D["wili"] = np.round(D.wili,1)

    #--import ensemble forecasts
    all_ensembles = pd.read_csv("../build_ensemble_models/all_ensembles.csv")
    
    #--add in forecast epidemic week
    all_ensembles = add_forecast_week(all_ensembles)

    n=0
    for release_ew, d in D.groupby(["releaseEW"]):
        forecast_and_scores = all_ensembles.merge( d, left_on = ["Location","FW"], right_on = ["Location","EW"] )

        if len(forecast_and_scores)==0:
            continue

        #--find the bin that the truth falls into
        bins_that_contain_truth = forecast_and_scores[(forecast_and_scores.wili>=forecast_and_scores["Bin_start_incl"]) & (forecast_and_scores.wili < forecast_and_scores["Bin_end_notincl"])]

        bins_that_contain_truth["logscore"] = np.log(bins_that_contain_truth.Value)

        #--rename week columns
        bins_that_contain_truth = bins_that_contain_truth.rename(columns={"EW_x":"EW_forecast_week","MW_x":"MW_forecast_week"
                                                                          ,"EW_y":"EW_target_week"  ,"MW_y":"MW_target_week"})
        
        #--keep only columns that we need
        scores = bins_that_contain_truth[["Location","Target","compute_model_id","static_plus","releaseEW","EW_forecast_week","EW_target_week","MW_forecast_week","MW_target_week","logscore"]]

        if n==0:
            scores.to_csv("all_scores.csv", mode="w",index=False,header=True)
            n=1
        else:
            scores.to_csv("all_scores.csv", mode="a",index=False,header=False)

    
