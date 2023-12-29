#mcandrew

import sys
sys.path.append("../../")

import numpy as np
import pandas as pd

from mods.index    import index

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def extractWeek(x):
    return int(str(x["EW"])[-2:])

def stamp(ax,letter):
    ax.text(0.01,0.99,letter,fontweight="bold",fontsize=10,ha="left",va="top",transform=ax.transAxes)


if __name__ == "__main__":

    idx = index("../../analysisdata/")
    scores = idx.grabForecastScores()
    
    #--only take the scores values from the most recent release on that EW
    def chooseMostRecentValue(d):
        return d.sort_values(["releaseEW"]).iloc[-1] # bottom row is most recent
    mostRecent_scores = scores.groupby(["Location","Target","component_model_id","MW"]).apply(chooseMostRecentValue)
    mostRecent_scores = mostRecent_scores.reset_index(drop=True)

    mostRecent_scores = mostRecent_scores.replace(-np.inf,-10.)
    mostRecent_scores["logscore"] = [ -10 if _<-10. else _ for _ in mostRecent_scores["logscore"].values]

    #--subset to a specific location, target
    target1_loc_10 = mostRecent_scores.loc[(mostRecent_scores.Location==10)&(mostRecent_scores.Target==1)]

    #--subset to one target
    target1 = mostRecent_scores.loc[mostRecent_scores.Target==1]

    #--average over log scores recorded by location for each component model and at each modelweek 
    avgOverComponentModels = target1.groupby(["component_model_id","MW"]).apply(lambda x: pd.Series({"logscore":np.mean(x.loc[:,"logscore"])}))
    avgOverComponentModels = avgOverComponentModels.reset_index()

    #--add in season
    ewAndSeason = idx.grabEW2Season()
    ewAndMW     = idx.grabEW2MW()

    avgOverComponentModels = avgOverComponentModels.merge(ewAndMW, on=["MW"])
    avgOverComponentModels = avgOverComponentModels.merge(ewAndSeason, on=["EW"])

    #--grab ili data
    idx = index("../../analysisdata/")
    epidata = idx.grabEpiData_f()

    # for each week, plot the most recent ILI value available
    def chooseMostRecentValue(d):
        return d.sort_values(["releaseDate"]).iloc[-1] # bottom row is most recent
    mostRecentEpidata = epidata.groupby(["Location","EW"]).apply( chooseMostRecentValue )
    mostRecentEpidata = mostRecentEpidata.reset_index(drop=True)

    ew2season = idx.grabEW2Season()
    
    mostRecentEpidata = mostRecentEpidata.merge(ew2season, on = ["EW"])
    mostRecentEpidata["week"] = mostRecentEpidata.apply(extractWeek,1)

    #--bring in correlation matrix data and store in wide_corrs
    letters = ["A.","B.","C.","D."]
    MWchoices = [235,245,255,264]

    wide_corrs = []
    for letter,choice in zip(letters,MWchoices):
        corrs = pd.read_csv("../../compute_correlations_between_models/correlation_matrix_over_time.csv")
                        #,dtype={"Location":str,"Target":int,"releaseMW":int,"model1":int,"model2":int,"corrtype":str,"corr":float})
        corrs2017Loc10 = corrs.loc[(corrs.Location==10)&(corrs.Target==2)&(corrs.releaseMW==choice)]
        wcorr = pd.pivot_table( index="model1",columns="model2",values="corr",data=corrs2017Loc10)

        wide_corrs.append( (letter, wcorr) )
  
    #--plot
    def ticks(ax):
        ax.tick_params(which="both",labelsize=8)
 
    plt.style.use("fivethirtyeight")

    fig  = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig, height_ratios = [2,2,1])

    #--prefill axs with plot axes
    axs=[]
    for r in range(2):
        for c in range(2):
            axs.append(fig.add_subplot(spec[r,c]))
            
    #--loop through four plots        
    n=0
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    for r in range(2):
        for c in range(2):
            letter,corr = wide_corrs[n]

            if c==1:
                sns.heatmap(corr, vmin=-1, vmax=1, cmap = cmap,linewidths= 0.00, linecolor="black", cbar=True,ax=axs[n])
                cbar = axs[n].collections[0].colorbar
                cbar.ax.tick_params(labelsize=8)
            else:
                sns.heatmap(corr, vmin=-1, vmax=1, cmap = cmap,linewidths= 0.00, linecolor="black", cbar=False,ax=axs[n])
                
            ax=axs[n]
            ticks(ax)

            #--label x and y axes
            if c==0:
                ax.set_ylabel("Component model",fontsize=8)
            else:
                ax.set_ylabel("",fontsize=8)
            if r==1:
                ax.set_xlabel("Component model",fontsize=8)
            else:
                ax.set_xlabel("")

            stamp(axs[n],"{:s}".format(letter))         
            n+=1

    ax = fig.add_subplot(spec[2,:])
    ili2017 = mostRecentEpidata.loc[(mostRecentEpidata.Season=="2017/2018") & (mostRecentEpidata.Location==10)]
    ax.plot(ili2017.MW.values,ili2017.wili.values, color="black", alpha=0.9, lw=2)

    for mw,letter in zip(MWchoices,letters):
        ax.text(mw,ili2017.loc[ili2017.MW==mw,"wili"],"{:s}".format(letter)
                ,fontsize=10, fontweight="bold",ha="left",va="bottom")

    xticks = np.arange(235,265+5,5)
    ax.set_xticks(xticks)
    ax.set_xticklabels( ewAndMW.loc[ewAndMW.MW.isin(xticks),"EW"].values, fontsize=8 )

    ax.set_ylabel("ILI",fontsize=8)
    
    ticks(ax)
    stamp(ax,"E.")         
            
    #fig.set_tight_layout(True)
    def mm2inch(x):
        return x/25.4

    w = mm2inch(183)
    fig.set_size_inches( w,w/1.5 )
    
    plt.savefig("correlation_over_time.pdf")
    plt.close()
