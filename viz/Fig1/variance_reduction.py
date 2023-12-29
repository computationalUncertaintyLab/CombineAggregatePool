#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats

from scipy.stats import entropy as KL
import matplotlib.gridspec as gridspec

from matplotlib.patches import Rectangle
import scienceplots

if __name__ == "__main__":

    norm = scipy.stats.norm

    normal0 = norm( 0  , 0.5 ).pdf
    normal1 = norm( 3/4, 0.5 ).pdf

    plt.style.use("science")

    
    fig = plt.figure()
    spec  = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, left=0.05, right=0.40 )

    gs2 = fig.add_gridspec(nrows=1, ncols=2, left=0.55, right=0.98,hspace=0.05, width_ratios = [10,1])
    
    ax = fig.add_subplot(spec[0])
    
    dom = np.linspace(-3,3,100)
    ax.plot( dom, normal0(dom), color = "blue", ls="-"  , label= "N(0,1/2)")
    ax.plot( dom, normal1(dom), color = "black", ls="--", label= "N(3/4,1/2)")
    
    ax.plot( dom, 0.5*normal0(dom) + 0.5*normal1(dom), color="green", ls="-.", label = "Ensemble" )

    ax.legend(frameon=False,loc="upper left", fontsize=8)

    ax.set_xlim(-3,3)
    ax.set_xlabel("Values")
    ax.set_ylabel("Density")

    ax.text(0.99,0.96,"A.",fontsize=10,fontweight="bold",ha="right",va="top",transform=ax.transAxes)
    
    ax = fig.add_subplot(spec[1])

    xs    = np.arange(3/4,6/4,0.05)
    
    data = {"kl":[], "var":[]}
    for x in xs:
        
        normal0 = norm( 3/4  , 0.5 ).pdf
        normal1 = norm( x  , 0.5).pdf
    
        kl = KL(normal0(dom),normal1(dom))

        vals = []
        for y in range(2*10**3):
            if np.random.random()<0.5:
               vals.append( np.random.normal(3/4,0.5) )
            else:
                vals.append( np.random.normal(x,0.5) )
        var = np.var(vals)

        data["kl"].append(kl) 
        data["var"].append(var)
    data = pd.DataFrame(data)
        
    sns.regplot(x="kl",y="var",data=data,ax=ax)
    
    lims = ax.get_xlim()
    ax.set_xlim( lims[-1], lims[-2]  )

    ax.set_xlabel("Kullback-Leibler Divergence", fontsize=10)
    ax.set_ylabel("Ensemble variance",fontsize=10)

    ax.text(0.99,0.96,"B.",fontsize=10,fontweight="bold",ha="right",va="top",transform=ax.transAxes)
    

    ax = fig.add_subplot(gs2[0])
    cbar_ax = fig.add_subplot(gs2[1])
                         
    d = pd.read_csv("./all_KLS.csv")

    average_KL = d.groupby(["model0","model1"]).apply( lambda x: x.KL.median() ).reset_index()
    table = pd.pivot_table( index=["model0"],columns=["model1"],data=average_KL )

    g = sns.heatmap( table, robust = True, ax=ax, cmap = "viridis", linecolor="white"
                     , cbar_kws = dict(  use_gridspec=False,location="right",label="KL Divergence"), cbar=True, cbar_ax=cbar_ax)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.add_patch(Rectangle((0, 0), 7, 7, fill=False, edgecolor='white', lw=0.5))
    ax.add_patch(Rectangle((7, 7), 6, 6, fill=False, edgecolor='white', lw=0.5))
    ax.add_patch(Rectangle((13, 13), 2, 2, fill=False, edgecolor='white', lw=0.5))
    ax.add_patch(Rectangle((15, 15), 2, 2, fill=False, edgecolor='white', lw=0.5))
    ax.add_patch(Rectangle((17, 17), 1, 1, fill=False, edgecolor='white', lw=0.5))
    ax.add_patch(Rectangle((18, 18), 3, 3, fill=False, edgecolor='white', lw=0.5))
    ax.add_patch(Rectangle((21, 21), 5, 5, fill=False, edgecolor='white', lw=0.5))
    ax.add_patch(Rectangle((26, 26), 1, 1, fill=False, edgecolor='white', lw=0.5))
    
    ax.set_xticks(np.arange(0,27))
    ax.set_yticks(np.arange(0,27))

    ax.tick_params(which="both",direction="in",color="white")
    
    ax.set_xticklabels( [] )
    ax.set_yticklabels( [] )

    ax.text(0,6/2         ,"CU"        ,fontsize=8, ha="right", va="top")
    ax.text(0,7+0.5*(12-7),"Delphi"    ,fontsize=8, ha="right", va="top")
    ax.text(0,13+0.25*(1)  ,"FluOutlook",fontsize=8, ha="right", va="top")
    ax.text(0,15+0.5*(1)  ,"FluX"      ,fontsize=8, ha="right", va="top")
    ax.text(0,17.75          ,"LANL"      ,fontsize=8, ha="right", va="top")
    ax.text(0,18+3/2+0.25      ,"Protea"    ,fontsize=8, ha="right", va="top")
    ax.text(0,21+5/2      ,"Reichlab"  ,fontsize=8, ha="right", va="top")
    ax.text(0,26      ,"UA"            ,fontsize=8, ha="right", va="top")

    ax.text(6/2         ,27         ,"CU"  ,fontsize=8  , ha="right", va="top",rotation=45)
    ax.text(7+0.5*(12-7),27,"Delphi"       ,fontsize=8  , ha="right", va="top",rotation=45)
    ax.text(13+0.5*(1)  ,27  ,"FluOutlook" ,fontsize=8  , ha="right", va="top",rotation=45)
    ax.text(15+0.5*(1)  ,27  ,"FluX"       ,fontsize=8  , ha="right", va="top",rotation=45)
    ax.text(17          ,27    ,"LANL"     ,fontsize=8  , ha="right", va="top",rotation=45)
    ax.text(18+3/2      ,27     ,"Protea"  ,fontsize=8  , ha="right", va="top",rotation=45)
    ax.text(21+5/2      ,27    ,"Reichlab" ,fontsize=8  , ha="right", va="top",rotation=45)
    ax.text(26          ,27 ,"UA"          ,fontsize=8  , ha="right", va="top",rotation=45)

    
    ax.text(0.0,0.99,"C.",fontsize=10,fontweight="bold",ha="right",va="top",transform=ax.transAxes)
    
    #fig.set_tight_layout(True)

    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    fig.set_size_inches(6.5, 6.5/2)
    plt.savefig("./variance_reduction.pdf")
    
    
    
    
    
    
    


    

