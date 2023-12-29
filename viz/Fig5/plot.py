#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

from matplotlib.patches import Rectangle
import matplotlib as mpl

def stamp(ax,s):
    ax.text(0.0125,0.95,s=s,ha="left",va="top",transform=ax.transAxes,fontsize=10,fontweight="bold")

if __name__ == "__main__":

    # d = pd.read_csv("../../score_ensemble_models/all_scores.csv")
    # d = d.loc[ d["static_plus"] == 0 ]

    model2name = {-1:"Equal",-2:"Static",-3:"Adapt",320:"CAP Equal", 321:"CAP Adapt"}
    
    # most_recent = d.groupby(["Location","Target","compute_model_id","MW_forecast_week"]).apply( lambda x: x.sort_values(["releaseEW"]).iloc[-1] ).reset_index(drop=True)

    # d = most_recent.loc[ most_recent.compute_model_id.isin([-1,-2,-3,320,321])]

    # d["compute_model_id"] = d.compute_model_id.replace(model2name)
    # d["logscore"] = np.clip(d.logscore, -10, np.inf)

    # d.to_csv("./dataset.csv",index=False)
    
    d = pd.read_csv("./dataset.csv")

    d["ensemble_type"]  = d.compute_model_id.replace({"Adapt":"Adapt"   ,"CAP Adapt":"Adapt","Equal":"Equal","CAP Equal":"Equal"})
    d["algorithm_type"] = d.compute_model_id.replace({"Adapt":"Original","CAP Adapt":"CAP","Equal":"Original","CAP Equal":"CAP"})
    
    plt.style.use("science")

    fig    = plt.figure(constrained_layout=True)
    spec  = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
    
    #--BOX PLOT---------------------------------------------------
    ax = fig.add_subplot(spec[0, 0])

    order = ["Equal","CAP Equal","Adapt","CAP Adapt"]
    d = d.loc[d.compute_model_id.isin(order)]

    color = sns.color_palette("tab10",2)
    solidcolor=["blue","darkgoldenrod"]
    xpoint=0

    
    for ens_n,(ens_type) in enumerate(["Equal","Adapt"]):
        subset_ens = d.loc[d.ensemble_type==ens_type]

        for alg_n,(alg_type) in enumerate(["Original","CAP"]):

            subset = subset_ens.loc[subset_ens.algorithm_type==alg_type]
            
            _10,_25,_50,_75,_90 = np.percentile(subset.logscore,[10,25,50,75,90])

            ax.plot([xpoint]*2,[_10,_90] ,color=color[alg_n%2],lw=10,alpha=0.5,zorder=0,markeredgecolor="black")

            lw = 0.15

            if ens_type=="Equal":
                line_style="--"
            else:
                line_style = "-"
            
            ax.add_patch(Rectangle( (xpoint-(lw) ,_10-(2.*lw)) , 2*(lw), _90+(lw)  -(_10-(lw))+2*lw  , fill=None,alpha=1,ls=line_style ) )
            ax.add_patch(Rectangle( (xpoint-(lw) ,_25-(2.*lw)) , 2*(lw), (_75+(lw) - (_25-(lw)))+2*lw, fill=None,alpha=1,ls=line_style ) )

            ax.plot([xpoint]*2,[_25,_75] ,color=color[alg_n%2],lw=10,alpha=0.5,zorder=1)

            ax.scatter([xpoint],_50,s=10 ,color=solidcolor[alg_n%2],zorder=2)

            xpoint+=0.5
        xpoint+=1.
    ax.set_xlim(-0.5,3)

    ax.text(x=-0.15,y=-2.1,s="Non-CAP",va="top",fontsize=8,ha="right",rotation=90)
    ax.text(x =1.85,y=-2.1,s="Non-CAP",va="top",fontsize=8,ha="right",rotation=90)

    
    ax.text(x=0.50 ,y=-2.,s="CAP",va="bottom",fontsize=8,ha="center",rotation=0)
    ax.text(x =2.5,y=-2.,s="CAP",va="bottom",fontsize=8,ha="center",rotation=0)

    ax.set_xlabel("")
    ax.set_ylabel("Log score (Nowcast)",fontsize=8)

    ax.set_xticks([0.25,2.25])
    ax.set_xticklabels(["Equal","Adaptive"])

    ax.set_yticks([-6,-3,-1])
    ax.set_yticklabels([-6,-3,-1],fontsize=6)

    stamp(ax,"A.")

    #---------------------------------------------------------------

    #--PIT SCORES----------------------------------------------------
    ax = fig.add_subplot(spec[0, 1])

    cdf_pits = pd.read_csv("./cdf_pits.csv")
    models = [-1,-3,320,321]
    
    subset                     = cdf_pits.loc[cdf_pits.compute_model_id.isin(models)]
    subset["compute_model_id"] = subset.compute_model_id.replace(model2name)

    subset = subset.loc[subset.Target==1]
    
    g = sns.lineplot( x ="right", y="px"
                     , style   = "ensemble_type"
                     , hue = "algorithm_type"
                     #, col   = "Target"
                     #, row   = "ensemble_type"
                     , data  = subset
                      ,palette= sns.color_palette("tab10",2)
                     #, kind="line"
                     #, hue_order = ["Original Alg.","Plus Alg."]
                     #, row_order = ["Equal","Static","Adaptive"]
                     #, label = "compute_model_id"
                     , alpha=0.80
                     ,ax=ax
                     , lw=1
    )
    ax.plot([0,1], [0,1], color = "k", lw=1, ls="-")

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.set_xticks(np.arange(0,1+0.1,0.2))
    ax.set_yticks(np.arange(0,1+0.1,0.2))

    ax.tick_params(which="both",labelsize=8)

    ax.set_title("")
            #ax.text(0.01,0.99,s="{:d} wk ahead".format(n+1),fontsize=10
            #        ,ha="left",va="top",transform=ax.transAxes)

    ax.set_ylabel("Cumul. Prob", fontsize=8)
    ax.set_xlabel("Prob. Int. Transf.",fontsize=8)
    
    h,l = g.axes.get_legend_handles_labels()

    ax.get_legend().remove()

    stamp(ax,"B.")
    #---------------------------------------------------------------

    #--Brier SCORES----------------------------------------------------
    ax = fig.add_subplot(spec[0, 2])

    d = pd.read_csv("../../score_ensemble_models/all_briers.csv")
    d["Location"] = d.Location.astype(int)
    d["Target"]   = d.Target.astype(int)

    models = [-1,-3,320,321]
    
    d             = d.loc[d.compute_model_id.isin(models)]
    d["compute_model_id"] = d.compute_model_id.replace(model2name)

    d["ensemble_type"]  = d.compute_model_id.replace({"Adapt":"Adapt"   ,"CAP Adapt":"Adapt","Equal":"Equal","CAP Equal":"Equal"})
    d["algorithm_type"] = d.compute_model_id.replace({"Adapt":"Original","CAP Adapt":"CAP","Equal":"Original","CAP Equal":"CAP"})

    d = d.loc[d.Target==1]

    sns.lineplot(x="threshold",y="brS",hue="algorithm_type",style="ensemble_type",data=d, palette= sns.color_palette("tab10",2))
    ax.set_ylabel("Brier Score",fontsize=8)
    ax.set_xlabel("ILI (\%)",fontsize=8)

    ax.set_yticks([0,0.2,0.3])
    ax.set_xticks([0,2.5,5.,10.])
    
    ax.get_legend().remove()
    stamp(ax,"C.")
    
    #---------------------------------------------------------------

    
    #--Over time----------------------------------------------------
    # ax = fig.add_subplot(spec[1, :])

    # d = pd.read_csv("./dataset.csv")

    # epidata = pd.read_csv("../../analysisdata/epidataFormated.csv.gz")
    # epidata = epidata.groupby(["EW","Location"]).apply(lambda x: x.sort_values("releaseEW").iloc[-1] ).reset_index(drop=True)
    
    # seasons = pd.read_csv("../../analysisdata/EWsandSeasons.csv")
    # epidata = epidata.merge(seasons,on=["EW"], how="left")
    
    # d = d.merge(seasons, left_on = ["EW_target_week"], right_on=["EW"], indicator="I", how="outer")
    # d = d.loc[d.I=="both"]
    
    # def find_peak(x):

    #     peak_week = np.argmax(x.wili.values)

    #     mws = x["MW"].values

    #     max_mw = np.max(mws)
    #     min_mw = np.min(mws)
        
    #     peak  = mws[peak_week]
    #     #weeks = np.clip( mws[peak_week-2:peak_week+2], min_mw, max_mw  )
    #     return pd.Series({"peak_week":peak})
    
    # peaks = epidata.groupby(["Season","Location"]).apply(find_peak).reset_index()
    # peaks = peaks.drop_duplicates()

    # d = d.merge( peaks, left_on = ["Location","Season"],right_on = ["Location","Season"])

    # d["peak_times"] = d["MW_target_week"] - d["peak_week"]

    # def starts(x):
    #     x["start"] = np.arange(len(x))
    #     return x
    # d = d.groupby(["Location","Season","Target","compute_model_id"]).apply(starts)

    # d = d.loc[d.compute_model_id.isin(["Equal","Adapt","CAP Equal","CAP Adapt"])]

    # d = d.loc[d.Target==1] #--Now cast

    # d["ensemble_type"]  = d.compute_model_id.replace({"Adapt":"Adapt"   ,"CAP Adapt":"Adapt","Equal":"Equal","CAP Equal":"Equal"})
    # d["algorithm_type"] = d.compute_model_id.replace({"Adapt":"Original","CAP Adapt":"CAP","Equal":"Original","CAP Equal":"CAP"})

    # sns.lineplot(  x    ="start"
    #              , y    ="logscore"
    #              , hue  ="algorithm_type"
    #              , style="ensemble_type" 
    #              , data =d
    #              , estimator =np.mean
    #              , errorbar=("se", 2)
    #              , err_kws={"alpha": .2})
    # ax.set_xlabel("Epidemic week",fontsize=8)
    # ax.set_ylabel("Log score (Nowcast)",fontsize=8)

    # ax.set_ylim(-6,-2)
    
    # #---------------------------------------------------------------

    
    #--By PEAK----------------------------------------------------
    ax = fig.add_subplot(spec[1, :])

    d = pd.read_csv("./dataset.csv")

    epidata = pd.read_csv("../../analysisdata/epidataFormated.csv.gz")
    epidata = epidata.groupby(["EW","Location"]).apply(lambda x: x.sort_values("releaseEW").iloc[-1] ).reset_index(drop=True)
    
    seasons = pd.read_csv("../../analysisdata/EWsandSeasons.csv")
    epidata = epidata.merge(seasons,on=["EW"], how="left")
    
    d = d.merge(seasons, left_on = ["EW_target_week"], right_on=["EW"], indicator="I", how="outer")
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

    d["peak_times"] = d["MW_target_week"] - d["peak_week"]

    def starts(x):
        x["start"] = np.arange(len(x))
        return x
    d = d.groupby(["Location","Season","Target","compute_model_id"]).apply(starts)

    d = d.loc[d.compute_model_id.isin(["Equal","Adapt","CAP Equal","CAP Adapt"])]

    d = d.loc[d.Target==1] #--Now cast

    d["ensemble_type"]  = d.compute_model_id.replace({"Adapt":"Adapt"   ,"CAP Adapt":"Adapt","Equal":"Equal","CAP Equal":"Equal"})
    d["algorithm_type"] = d.compute_model_id.replace({"Adapt":"Original","CAP Adapt":"CAP","Equal":"Original","CAP Equal":"CAP"})
    
    sns.lineplot(  x    ="peak_times"
                 , y    ="logscore"
                 , hue  ="algorithm_type"
                 , style = "ensemble_type" 
                 , data =d
                   ,palette= sns.color_palette("tab10",2)
                 , estimator =np.mean
                 , errorbar=("se", 2)
                 ,err_kws={"alpha": .2})
    ax.set_xlabel("Epidemic week minus peak week",fontsize=8)
    ax.set_ylabel("Log score (Nowcast)",fontsize=8)

    ax.get_legend().remove()
    
    ax.set_ylim(-6,-2)

    stamp(ax,"D.")
    
    #---------------------------------------------------------------
    
    fig.set_size_inches(6.5,9.5/3)
    plt.subplots_adjust(wspace=0.01,hspace=0.01)
    
    plt.savefig("./all_scores.pdf")
    plt.close()
        
    
    





    
