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

def extractWeek(x):
    return int(str(x["EW"])[-2:])

if __name__ == "__main__":

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

    loc2num = idx.grabLocationDict()
    num2loc = { y:x for x,y in loc2num.items() }
    
    pl = culpl()
    pl.style()

    fig,axs = plt.subplots(3,3)
    axs_flat = axs.flatten()
    
    for axidx,(season,subset) in enumerate(mostRecentEpidata.groupby(["Season"])):
        print(axidx)
        ax = axs_flat[axidx]

        for n,(loc,sub_subset) in enumerate(subset.groupby(["Location"])):
            modelweek = np.arange(0,len(sub_subset))

            if n==0:
                p = ax.plot( modelweek, sub_subset.wili, lw=1 )
                clr = p[0].get_color()
            else:
                if loc==10:
                    clr="black"
                    ax.plot( modelweek, sub_subset.wili, lw=1, color=clr, label = "US National" )
                ax.plot( modelweek, sub_subset.wili, lw=1, color=clr )
                
            ax.tick_params(which="both",labelsize=8)
            ax.set_ylim(0,13)
            ax.set_yticks([2.5,5,7.5,10,12.5])

            ax.set_xlim(0,31)

            if axidx<8:
                xticks = list(np.arange(0,25,10)) + [ modelweek[-1]]
            else:
                xticks = list(np.arange(0,25,10))
            ax.set_xticks(xticks)
        print(xticks)
        ax.set_xticklabels( sub_subset.EW.values[xticks],fontsize=8 )
            
        ax.text(0.99,0.99,"{:s}".format(season),ha="right",va="top",fontsize=10,transform=ax.transAxes)

    for i in range(3):
        for j in range(3):
            if i<2:
                axs[i,j].set_xlabel("")
                #axs[i,j].set_xticklabels([])

            if j>0:
                axs[i,j].set_yticklabels([])

            if j==0:
                axs[i,j].set_ylabel("ILI",fontsize=10)

            if i==2:
                axs[i,j].set_xlabel("Epidemic week",fontsize=10)
                
    axs[1,2].legend(frameon=False,fontsize=10,loc="center")
                
    fig.set_tight_layout(True)

    def mm2inch(x):
        return x/25.4

    w = mm2inch(183)
    fig.set_size_inches( w,w/1.5 )
    
    plt.savefig("ILIbySeason.pdf")
    plt.close()
