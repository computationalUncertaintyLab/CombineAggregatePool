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

import scienceplots

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
    avgOverComponentModels = target1_loc_10.groupby(["component_model_id","MW"]).apply(lambda x: pd.Series({"logscore":np.mean(x.loc[:,"logscore"])}))
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

    #--bring in correlation matrix data
    MWchoice=235
    corrs = pd.read_csv("../../compute_correlations_between_models/correlation_matrix_over_time.csv")
                        #,dtype={"Location":str,"Target":int,"releaseMW":int,"model1":int,"model2":int,"corrtype":str,"corr":float})
    corrs2017Loc10 = corrs.loc[(corrs.Location==10)&(corrs.Target==2)&(corrs.releaseMW==MWchoice)]
    wide_corr1 = pd.pivot_table( index="model1",columns="model2",values="corr",data=corrs2017Loc10)

    MWchoice = 247
    corrs2017Loc10 = corrs.loc[(corrs.Location==10)&(corrs.Target==2)&(corrs.releaseMW==MWchoice)]
    wide_corr2 = pd.pivot_table( index="model1",columns="model2",values="corr",data=corrs2017Loc10)
    
    #--plot
    def ticks(ax):
        ax.tick_params(which="both",labelsize=8)
 
    plt.style.use("science")
    fig  = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    #--panel1
    #--plot the mean logscore over epidemic week per component model

    def add_seasonweek(x):
        x = x.sort_values(["MW"])
        x["seasonweek"] = np.arange(len(x))
        return x
    avgOverComponentModels_ = avgOverComponentModels.groupby(["component_model_id"]).apply(add_seasonweek).reset_index()

    color = sns.color_palette("tab10",1)[0]
    
    ax = fig.add_subplot(spec[0,:])
    for c,subset in avgOverComponentModels_.groupby(["component_model_id"]):
        for s,season_subset in subset.groupby(["Season"]):
            ax.plot( season_subset.MW.values, season_subset.logscore.values, alpha= 0.40, lw=1, color = color)
    ticks(ax)
    ax.set_ylabel("logarithmic score",fontsize=10)

    xticks = np.arange(0,450,50)
    xticks = [0, 52, 103, 151,203,253,302,352,402]
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(ewAndMW.iloc[xticks]["EW"])
    stamp(ax,"A.")
    
    #--panel2
    ax = fig.add_subplot(spec[1,:])
    ax2= ax.twinx()
    
    season2017 = avgOverComponentModels.loc[avgOverComponentModels.Season=="2017/2018"]
    for c,subset in season2017.groupby(["component_model_id"]):
        ax.plot( subset.MW.values, subset.logscore.values, alpha= 0.40, lw=1,color=color )

    ili2017 = mostRecentEpidata.loc[(mostRecentEpidata.Season=="2017/2018") & (mostRecentEpidata.Location==10)]
    ax2.plot(ili2017.MW.values,ili2017.wili.values, color="black", alpha=0.9, lw=2)

    xlocator = 30
    ax2.text(ili2017.MW.values[xlocator],ili2017.wili.values[xlocator],"ILI",fontsize=10, fontweight="bold",ha="left",va="bottom")

    ticks(ax)
    ticks(ax2)

    ax.set_ylabel("logarithmic score",fontsize=10)
    ax2.set_yticks([])

    stamp(ax,"B.")

    xticks = ax.get_xticks()
    ax.set_xticklabels(ewAndMW.iloc[xticks]["EW"])

    ax.set_xlabel("Epidemic week",fontsize=10)

    #--panel3
    # ax = fig.add_subplot(spec[2,:])

    # wide_corr = wide_corr1.replace(np.nan,0)+wide_corr2.replace(np.nan,0).T
    # np.fill_diagonal(wide_corr.values, 0)
    
    # cmap = sns.diverging_palette(220, 20, as_cmap=True)
    # sns.heatmap(wide_corr, vmin=-1, vmax=1, cmap = cmap,linewidths= 0.005, linecolor="black", cbar=False)

    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=8)

    #ax.set_xticks(np.arange(0,26+1))
    
    #ticks(ax)
    #ax.set_xlabel("")
    #ax.set_ylabel("")

    #stamp(ax,"C.")

    #fig.set_tight_layout(True)

    fig.set_size_inches(6.5, 9.5/3 )
    
    plt.savefig("logscores_summary.pdf")
    plt.close()
