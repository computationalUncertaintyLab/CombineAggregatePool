#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots

if __name__ == "__main__":

    d = pd.read_csv("./cluster_data.csv")
    d["compute_model_id"] = (d.Combine.astype(int).astype(str)+d.Aggregate.astype(int).astype(str)+d.Pool.astype(int).astype(str)).astype(int)
    
    epidata = pd.read_csv("../../analysisdata/epidataFormated.csv.gz")
    epidata = epidata.groupby(["EW","Location"]).apply(lambda x: x.sort_values("releaseEW").iloc[-1] ).reset_index(drop=True)
    
    seasons = pd.read_csv("../../analysisdata/EWsandSeasons.csv")
    #seasons = seasons.merge(epidata[["EW","MW"]].drop_duplicates(), on = "EW")

    epidata = epidata.merge(seasons,on=["EW"], how="left")
    
    d = d.merge(epidata, left_on = ["Location","Modelweek"], right_on=["Location","MW"], indicator="I", how="outer")
    d = d.loc[d.I=="both"]
    
    def find_peak(x):

        peak_week = np.argmax(x.wili.values)

        mws = x["MW"].values

        max_mw = np.max(mws)
        min_mw = np.min(mws)
        
        peak  = mws[peak_week]
        #weeks = np.clip( mws[peak_week-2:peak_week+2], min_mw, max_mw  )
        return pd.Series({"peak_week":peak})
    
    peaks = epidata.groupby(["Season","Location"]).apply(find_peak).reset_index()
    peaks = peaks.drop_duplicates()

    d = d.merge( peaks, left_on = ["Location","Season"],right_on = ["Location","Season"])

    d["MW_target_week"] = d["MW"] + d["Target"]+1
    d["peak_times"] = d["MW_target_week"] - d["peak_week"]


    d = d.loc[d.compute_model_id==321]

    d[d.peak_times <= -20]["nclusters"].mean()
    d[(d.peak_times <= 1) & (d.peak_times >=-1)]["nclusters"].mean()
    d[(d.peak_times >=20)]["nclusters"].mean()
    
    d[d.peak_times <= -20]["entropy_ratio"].mean()
    d[(d.peak_times <= 1) & (d.peak_times >=-1)]["entropy_ratio"].mean()
    d[(d.peak_times >=20)]["entropy_ratio"].mean()
 
