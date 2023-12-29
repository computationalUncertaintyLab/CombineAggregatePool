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

    #--change entropy from ratio ro percent
    d["entropy_ratio"] = 100*d.entropy_ratio

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    plt.style.use("science")
            
    colors = sns.color_palette("tab10",2)
    fig,ax = plt.subplots()
    
    sns.lineplot(x="peak_times",y="nclusters"    , data=d.loc[d.compute_model_id.isin([321])],ax=ax,color=colors[0])

    twin = ax.twinx()
#    make_patch_spines_invisible(twin)

    twin.spines["left"].set_position(("axes", -0.10))
    twin.spines["left"].set_visible(True)
    
    twin.yaxis.set_label_position('left')
    twin.yaxis.set_ticks_position('left')

    ax.yaxis.label.set_color(colors[0])
    twin.yaxis.label.set_color(colors[1])

    sns.lineplot(x="peak_times",y="entropy_ratio", data=d.loc[d.compute_model_id.isin([321])],ax=twin,color=colors[1])

    twin.set_ylabel("Percent Entropy",fontsize=10)
    ax.set_ylabel("Number of clusters",fontsize=10)

    ax.set_xlabel("Epidemic week minus peak week",fontsize=10)
    
    fig.set_size_inches(6.5/1,9.5/4)
    
    plt.savefig("./algorithm_extras.pdf")
    plt.close()
        
    
 
    
