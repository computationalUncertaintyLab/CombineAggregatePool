#mcandrew

import sys
sys.path.append("../")

import numpy as np
import pandas as pd

from epiweeks import Week

from mods.index import index

#import concurrent.futures
#from multiprocessing import Pool

from joblib import Parallel, delayed

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

    #--most recent releases only
    def most_recent(x):
        most_recent_release = x.releaseEW.max()
        return x.loc[x.releaseEW==most_recent_release]
    most_recent = D.groupby(["Location","EW","MW"]).apply(most_recent)
    most_recent = most_recent.reset_index(drop=True)

    #--import ensemble forecasts
    all_ensembles = pd.read_csv("../build_ensemble_models/all_ensembles.csv")
    
    #--add in forecast epidemic week
    all_ensembles = add_forecast_week(all_ensembles)

    forecast_and_scores = all_ensembles.merge( most_recent, left_on = ["Location","FW"], right_on = ["Location","EW"] )

    #--add season
    ew2season = idx.grabEW2Season()
    forecast_and_scores = forecast_and_scores.merge(ew2season, left_on = ["EW_x"], right_on = ["EW"])

    def compute_brs(model_data,threshold):
        def prob_above(x,threshold):
            prob_ab    = x.loc[x.Bin_start_incl > threshold,"Value"].sum()

            wili = float(x.iloc[0].wili)
            wili_above = 1 if wili > threshold else 0

            brier = (prob_ab - wili_above )**2

            return pd.Series({"prob_above":prob_ab, "wili_above": wili_above, "brier":brier })
        prob_above = model_data.groupby(["EW_x"]).apply( lambda x: prob_above(x, threshold) )
        brS = prob_above.brier.mean()
        return pd.Series({"brS":brS})

    def thresholdling(threshold):
        brs_threshold = forecast_and_scores.groupby(["Location","Target","Season","compute_model_id"]).apply( lambda x: compute_brs(x,threshold) )
        brs_threshold = brs_threshold.reset_index()
        brs_threshold["threshold"] = threshold
       
        #if threshold==0:
        #    brs_threshold.to_csv("all_briers_new.csv", mode="w",index=False,header=True)
        #else:
        brs_threshold.to_csv("all_briers_new_{:f}.csv".format(threshold), mode="w",index=False,header=True)
        return 0

    ts = np.arange(0,10+0.1,0.1)

    res = Parallel(
        n_jobs=10
    )(
        delayed(thresholdling)(x) for x in ts
    )
