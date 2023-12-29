#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots
from matplotlib.gridspec import GridSpec

if __name__ == "__main__":

    cdf_2 = pd.read_csv("./CDF_sample_2.csv")
    cdf_3 = pd.read_csv("./CDF_sample_3.csv")

    cdf_2 = cdf_2.sort_values(["Cluster_status","component_model_id","Bin_start_incl"])
    
    from_cluster_to_color = sns.color_palette("Set2",7)

    plt.style.use("science")
    
    #fig,axs = plt.subplots(2,4)

    fig = plt.figure(layout="constrained")
    gs  = GridSpec(3, 4, figure=fig)

    axs = []
    for row in np.arange(2):
        new_row = []
        for col in np.arange(4):
            new_row.append( fig.add_subplot(gs[row, col]) )
        axs.append(new_row)
    axs = np.array(axs)
            
    g = sns.lineplot(x="Bin_start_incl", y="CDF", hue="Cluster_status", hue_order=[1,2,3,4,5,6,7], data = cdf_2,palette = from_cluster_to_color, ax=axs[0,0])
    axs[0,0].get_legend().remove()
    
    axs_f = axs.flatten()
    for cluster in np.arange(1,7+1):
        cluster_data = cdf_2.loc[cdf_2.Cluster_status==cluster]

        print(cluster)
        print(cluster_data.component_model_id.unique())
    
        ax = axs_f[cluster]
        ax.text(0.96,0.04,s="Clust. {:1d}".format(cluster), fontsize=8,ha="right",va="bottom",transform=ax.transAxes)
        
        for model, model_data in cluster_data.groupby("component_model_id"):

            representative = int(model_data.Top_status.unique())

            if representative:
                ax.plot( model_data.Bin_start_incl.values, model_data.CDF, color=from_cluster_to_color[cluster-1], ls="--", lw=2, alpha=1.  )
            else:
                ax.plot( model_data.Bin_start_incl.values, model_data.CDF, color=from_cluster_to_color[cluster-1], lw=1.5, alpha=0.8  )
            

    for r,row in enumerate(axs):
        for c,col in enumerate(row):

            axs[r,c].set_xticks([0,2,5,10,13])
            axs[r,c].set_yticks([0,0.25,0.50,0.75,1.0])
            
            if r in [0,1] and c==0:
                axs[r,c].set_ylabel("CDF",fontsize=8)
            if r ==0:
                axs[r,c].set_xlabel("")
            if r == 1:
                axs[r,c].set_xlabel("ILI",fontsize=8)
            if r == 2:
                axs[r,c].axis("off")

            if r==0:
                axs[r,c].set_xticklabels([])
            if c>0:
                axs[r,c].set_yticklabels([])


    #--matrix below
    ax = fig.add_subplot(gs[2, :])

    matrix = np.full( (7,27) , -1 )

    cluster = [0,4,7,8,9,15,16,17,22,24,25]
    matrix[0,cluster] = [ 1, 1, 0, 0, 0, 0,0,1,0,0,0 ] 
                
    cluster = [1,2,3,4,5,6]
    matrix[1,cluster] = [1, 1,1,1,1,1]
    
    cluster = [12]
    matrix[2,cluster] = [ 0 ] 

    cluster = [21,23]
    matrix[3,cluster] = [0,0 ]
    
    cluster = [26]
    matrix[4,cluster] = [ 1 ] 

    cluster = [10,11,13,14,19]
    matrix[5,cluster] = [ 0,0,1,1,0 ] 

    cluster = [18,20]
    matrix[6,cluster] = [ 0,0 ]

    colors = sns.color_palette("tab10",3)
    sns.heatmap(matrix, linewidths=0.5, cmap = colors , cbar=False,ax=ax)

    ax.axvline(7,color="black")
    ax.axvline(7+6,color="black")
    ax.axvline(7+6+2,color="black")
    ax.axvline(7+6+2+2,color="black")
    ax.axvline(7+6+2+2+1,color="black")
    ax.axvline(7+6+2+2+1+3,color="black")
    ax.axvline(7+6+2+2+1+3+5,color="black")
    ax.axvline(7+6+2+2+1+3+5+1,color="black")
    
    ax.set_xticks([ 7/2
                   ,7+6/2
                   ,7+6+2/2-0.5
                    ,7+6+2+2/2-0.15
                    ,7+6+2+2+1/2+0.1
                    ,7+6+2+2+1+3/2
                    ,7+6+2+2+1+3+5/2
                    ,7+6+2+2+1+3+5+1/2])
    ax.set_xticklabels(["CU","Delphi","FluOutlook","FluX","LANL","Protea","Reichlab","UA"],rotation=0,fontsize=10)

    #print(ax.get_yticks())
    ax.set_yticks(np.arange(0,6+1)+0.5)
    ax.set_yticklabels(["Clust. {:d}".format(x) for x in np.arange(1,7+1)],fontsize=8,rotation=0,va="center")


    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color= colors[1], lw=4), Line2D([0], [0], color=colors[2], lw=4)]
    ax.legend(custom_lines, ['Statistical', 'Mechansitic model'], fontsize=8,ncols=2,bbox_to_anchor = (0.60,0.9095), borderpad=0., columnspacing=0.75)

    
    #fig.set_tight_layout(True)

    plt.subplots_adjust(hspace=0. , wspace=-1)
    fig.set_size_inches( 6.5/1 , 8/3 )
    plt.savefig("./cluster_example_with_properties.pdf")
    plt.close()
    

    
    # for model,model_data in cdf_2.groupby("component_model_id"):
    #     cluster = int(model_data.Cluster_status.unique())    
    #     ax.plot( model_data.Bin_start_incl.values, model_data.CDF, color=from_cluster_to_color[cluster], lw=2, alpha=0.5  )
    # plt.show()
    
    

    

    

