#mcandrew

import sys
sys.path.append("../../")

import json

import numpy as np
import pandas as pd

from mods.index    import index
from mods.tsTools  import tsTools
from mods.plotTools import culpl

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from epiweeks import Week

def extractWeek(x):
    return int(str(x["EW"])[-2:])

def compute_quantiles(d):
    from scipy.interpolate import interp1d
    from scipy.optimize import root_scalar as root

    pmf = {x:y for x,y in zip(d.Bin_start_incl.values, d.Value.values)}

    #-- build cmf
    s=0
    xs,probs = [],[]
    for x,prob in sorted(pmf.items()):
        xs.append(x)

        probs.append(s)
        s+=prob

    xs.append(100)
    probs.append(1.)
    cmf_func = interp1d(xs,probs) 

    #--compute quantiles for a model
    quantile = { "quantile":[], "value":[] }
    for q in [0.1,0.25,0.50,0.75,0.90]:
        #-- find bracket
        left,right=-1,np.inf
        for x in xs:
            if cmf_func(x) < q:
                left=x
            if cmf_func(x) > q:
                right=min(x,right)

        #--find quantile
        F = lambda x: cmf_func(x) - q
        rslt = root(F,bracket=(left,right))

        quant = rslt.root

        quantile["quantile"].append(q)
        quantile["value"].append(quant)
    quantile = pd.DataFrame(quantile)
    return quantile

if __name__ == "__main__":

    idx = index("../../analysisdata/")
    
    epidata = idx.grabEpiData_f()
    
    forecastdata = idx.grabForecasts()
    forecastdata = forecastdata.loc[forecastdata.Type=="Bin"] # bins only.
    forecastdata = forecastdata.loc[forecastdata.Target.str.contains("wk ahead") ]
    
    #--polish epidata

    # for each week, plot the most recent ILI value available
    def chooseMostRecentValue(d):
        return d.sort_values(["releaseDate"]).iloc[-1] # bottom row is most recent
    mostRecentEpidata = epidata.groupby(["Location","EW"]).apply( chooseMostRecentValue )
    mostRecentEpidata = mostRecentEpidata.reset_index(drop=True)

    ew2season = idx.grabEW2Season()
    
    mostRecentEpidata = mostRecentEpidata.merge(ew2season, on = ["EW"])
    mostRecentEpidata["week"] = mostRecentEpidata.apply(extractWeek,1)

    #--choose one HHS region
    hhs = 2

    mostRecentEpidata = mostRecentEpidata.loc[mostRecentEpidata.Location==hhs]
    forecastdata = forecastdata.loc[forecastdata.Location=="HHS Region {:d}".format(hhs)]

    #--choose one season
    season = "2017/2018"

    mostRecentEpidata = mostRecentEpidata.loc[mostRecentEpidata.Season==season]
    
    forecastdata = forecastdata.merge(ew2season,on=["EW"])
    forecastdata = forecastdata.loc[forecastdata.Season==season]

    #--format columns
    forecastdata["Bin_start_incl"] = forecastdata.Bin_start_incl.astype(float)
    
    model_quantiles = forecastdata.groupby(["component_model_id","Location","Target","EW"]).apply(compute_quantiles)
    model_quantiles = model_quantiles.reset_index()

    plt.style.use("fivethirtyeight")   
    fig,axs = plt.subplots(1,2)

    #--forecast week in terms of model week
    forecastweek = 12

    ew_str = str(mostRecentEpidata.EW.values[forecastweek])
    forecast_epi_week = Week.fromstring(ew_str)
    
    modelweek = np.arange(0,len(mostRecentEpidata))
    ilis      = mostRecentEpidata.wili.values

    ilibefore = ilis[:forecastweek+1]
    iliafter  = ilis[forecastweek+1:]

    for ax in axs:
        ax.plot( modelweek[:forecastweek+1], ilibefore , color="black", lw=2)
        ax.plot( modelweek[forecastweek+1:], iliafter , color="black", alpha=0.50, ls="--" , lw=2)

    #--left plot shows all forecasts at a single ew
    ax=axs[0]
    
    #--all model quantiles for specific forecastweek and plot median and IQR
    quantiles = model_quantiles.loc[model_quantiles.EW== int(ew_str) ]
    quantiles["increment"] = quantiles.Target.str.slice(0,1).astype(int)

    for model,subset in quantiles.groupby(["component_model_id"]):

        #--skip this model and mention in the legend
        if model=="Delphi_Uniform":
            continue
    
        forecast_data = pd.pivot_table(subset, index=["increment"], columns=["quantile"],values=["value"])
        forecast_data.columns = [y for (x,y) in forecast_data.columns]
        
        ax.plot(forecastweek+forecast_data.index.values, forecast_data[0.50], lw=1)
        ax.fill_between(forecastweek+forecast_data.index.values, forecast_data[0.25],forecast_data[0.75], alpha=0.20 )

    ax.tick_params(which="both",labelsize=10)
    ax.set_xlabel("Epidemic week")
    ax.set_ylabel("ILI (%)")

    xticks=np.arange(0,30+10,10)
    ax.set_xticks(xticks)
    ax.set_xticklabels( mostRecentEpidata.EW.values[xticks], fontsize=10 )
 

    #--right plot shows a single model at a few different weeks
    ax = axs[1]
    model = "Protea_Springbok"

    for forecastweek in [12,18,25]:
        ew_str = str(mostRecentEpidata.EW.values[forecastweek])
        forecast_epi_week = Week.fromstring(ew_str)
        
        quantiles = model_quantiles.loc[ (model_quantiles.EW== int(ew_str)) & (model_quantiles.component_model_id==model) ]
        quantiles["increment"] = quantiles.Target.str.slice(0,1).astype(int)

        forecast_data = pd.pivot_table(quantiles, index=["increment"], columns=["quantile"],values=["value"])
        forecast_data.columns = [y for (x,y) in forecast_data.columns]

        ax.plot(forecastweek+forecast_data.index.values, forecast_data[0.50], lw=1, color="blue")
        ax.fill_between(forecastweek+forecast_data.index.values, forecast_data[0.25],forecast_data[0.75], alpha=0.20, color="blue" )

    ax.tick_params(which="both",labelsize=10)
    ax.set_xlabel("Epidemic week")
    ax.set_ylabel("ILI (%)")

    xticks=np.arange(0,30+10,10)
    ax.set_xticks(xticks)
    ax.set_xticklabels( mostRecentEpidata.EW.values[xticks], fontsize=10 )

    #--stamp
    for ax,letter in zip(axs,["A.","B."]):
        ax.text(0.01,0.99,letter,fontsize=10,fontweight="bold",ha="left",va="top",transform=ax.transAxes)
        
    #--write out
    fig.set_tight_layout(True)

    def mm2inch(x):
        return x/25.4

    w = mm2inch(183)
    fig.set_size_inches( w,w/1.5 )
    
    plt.savefig("forecasting_example.pdf")
    plt.close()
