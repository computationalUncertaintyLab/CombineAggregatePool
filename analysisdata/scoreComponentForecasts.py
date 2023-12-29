#mcandrew

import sys
sys.path.append("../")

import numpy as np
import pandas as pd

from mods.index import index

import dask.dataframe as dd

if __name__ == "__main__":

    idx = index("./")
    epidata   = idx.grabEpiData_f()
    forecasts = idx.grabForecasts_f()

    weekAheadForecasts = forecasts.loc[forecasts.Target.isin([0,1,2,3])]
    weekAheadForecasts = weekAheadForecasts.drop(columns=["Unnamed: 0"])
    
    weekAheadForecasts["MWtargetWeek"] = weekAheadForecasts.MW+(weekAheadForecasts.Target+1)

    weekAheadForecasts["Bin_start_incl"] = weekAheadForecasts.Bin_start_incl.astype(float)
    weekAheadForecasts["Bin_end_notincl"] = weekAheadForecasts.Bin_end_notincl.astype(float)

    # add Bins to epidata for merging
    epidata["Bin_start_incl"]  = np.floor(epidata.wili*10)/10
    epidata["Bin_end_notincl"] = np.ceil(epidata.wili*10)/10
    
    epidata            = epidata.drop(columns=["EW","lag"]).set_index(["Location","MW","Bin_start_incl","Bin_end_notincl"])
    
    scores = weekAheadForecasts.merge(epidata,left_on = ["Location","MWtargetWeek","Bin_start_incl","Bin_end_notincl"]
                                             , right_on = ["Location","MW","Bin_start_incl","Bin_end_notincl"])

    scores["logscore"] = np.round(np.log(scores.Value),3)
    
    scores["Value"] = np.round(scores.Value,3)
    
    scores = scores[["Location","component_model_id", "Target","MW","MWtargetWeek","releaseEW","Bin_start_incl","Bin_end_notincl","Value","wili","logscore"]]
    scores.to_csv("scores.csv",index=False)
