#mcandrew

import sys
sys.path.append("../")

import numpy as np
import pandas as pd

import mpltern
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats

from scipy.optimize import minimize

from mods.index import index

if __name__ == "__main__":

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    })

    def obj( weights, data,  models, scale=1 ):
        weights = np.array(weights)/scale

        e = 10**-6

        weights = weights+e
        weights = weights/sum(weights)

        m = np.array([scipy.stats.norm(m,s).pdf(data) for (m,s) in models]) #--rows are model and column is each datapoint
        m = m.T

        s = m.dot(weights)       

        return -1*np.log(s).sum(0)

    # one per row in first columns

    from mpltern.datasets import get_triangular_grid
    ts,ls,rs = get_triangular_grid(200)

    tstar,lstar,rstar = [],[],[]
    for (t,l,r) in zip(ts,ls,rs):
        if (0.1 < t < 0.8) and (0.1 < l < 0.8) and (0.1 < r < 0.8):
            tstar.append(t)
            lstar.append(l)
            rstar.append(r)
    t,l,r = tstar, lstar, rstar
        

    orig_map=plt.cm.get_cmap('viridis',50)
    reversed_map = orig_map.reversed()
    
    #fig,axs = plt.subplots(3,2)
    fig = plt.figure()
    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10, wspace=0.2, hspace=0.5)
    
    for row,x in enumerate([2.5,0.50,0.01]):
        
        #--BUILD DATASET----------------------------------------------------
        model1 = [0,1]
        model2 = [x,1]
        model3 = [5,1]

        models = [model1,model2,model3]
        weights = np.array([0.2,0.5,0.3])
        truth = weights

        choices = np.random.choice( np.arange(3), p=weights, size=(1000,) )

        dataset = []
        for choice in choices:
            m,v = models[choice] 
            dataset.append(np.random.normal( m,v))
        #------------------------------------------------------------------
            
        #--BUILD LOGLIKELIHOOD SURFACE------------------------------------
        scale=1
        f = lambda p: obj(p,dataset,models,scale)

        f_s = f( np.stack([t,l,r]))
        f_s = np.log10(np.array(f_s))
        #------------------------------------------------------------------

        vmin,vmax = np.min(f_s), np.max(f_s)
        levels = np.linspace(vmin,vmax,25)
        
        ax = fig.add_subplot(2,3, row+1 , projection='ternary')
        cs = ax.tripcolor( t,l,r, f_s, shading = "gouraud", cmap = "Blues" )

        ax.tricontour(t, l, r, f_s, colors="k", levels = np.arange(3.10,3.50,0.025), linewidths=0.5)
        
        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")

        ax.set_tlabel(r"$\mathcal{N}(0,1/2)$",fontsize=8)
        ax.set_llabel(r"$\mathcal{N}$"+"$({:.1f},1/2)$".format(x),fontsize=8)
        
        if row==2:
            pass
            #cax = ax.inset_axes([-0.05, -0.5, 0.9, 0.10], transform=ax.transAxes)
            #colorbar = fig.colorbar(cs, cax=cax, orientation = 'horizontal')
            #colorbar.set_label(r"$- \ell \ell$", rotation=0, va="baseline")
            #cax.tick_params(direction="in")

        else:
            ax.set_rlabel(r"$\mathcal{N}(5,1/2)$",fontsize=8)

        def compute_hessian(weights,data,models):
            nmodels = len(models)

            m = np.array([scipy.stats.norm(m,s).pdf(data) for (m,s) in models]) #--rows are model and column is each datapoint
            m = m.T

            denom = m.dot(weights)**(-2)

            H = np.zeros( (nmodels,nmodels) )
            for model1 in range(nmodels):
                for model2 in range(nmodels):
                    H[model1,model2] = sum(-1*(denom)*m[:,model1]*m[:,model2])
            return -1*H #--negative log likelihood

        start_pos = np.array([0.33]*3)
        cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
        bnds = tuple((10**-5,1-10**-5) for x in start_pos)

        def save(x):
            global P
            P.append(x)
        
        ends = []

        #--plot the min and hessian
        start_pos = np.random.dirichlet([1]*3)

        rslts = minimize( lambda weights: obj(weights, dataset, models, scale=1), x0 = start_pos, method='SLSQP', bounds=bnds ,constraints=cons)
        sol = rslts.x

        H   = compute_hessian( sol, dataset, models )
        Cov = np.linalg.inv(H)

        rv = scipy.stats.multivariate_normal( sol, Cov )

        ax.tripcolor( t,l,r, rv.pdf( np.stack([t,l,r]).T ),cmap = "Reds")#, color="red", alpha=1 )

        ax.set_tlim(0.1, 0.8)
        ax.set_llim(0.1, 0.8)
        ax.set_rlim(0.1, 0.8)
        
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.scatter(truth[0],truth[1],truth[2], color='k', s=20, marker="x")

        ax.taxis.set_tick_params(labelsize=6, tickdir="in")
        ax.laxis.set_tick_params(labelsize=6, tickdir="in")
        ax.raxis.set_tick_params(labelsize=6, tickdir="in")
 
        

    #--EMPIRICAL REPRESNETATIONS

    #--real data
    kls = pd.read_csv("./all_KLS.csv")

    overlap = kls.loc[(kls.KL>0) & (kls.Target<=3)].sort_values('KL')

    overlap1 = overlap.iloc[6107]
    overlap2 = overlap.iloc[342880]
    overlap3 = overlap.iloc[73888]   

    models = [ [22,23], [25,18], [21,15] ] 
    
    overlaps = [overlap1, overlap2, overlap3]
    
    sols = pd.read_csv("solutions_for_three_overlaps.csv")
        
    for n,(idx,sol) in enumerate(sols.groupby("overlap")):
        ax = fig.add_subplot(2,3, 4+n , projection='ternary')   
        
        overlap = overlaps[n]
        model0,model1 = models[n]

        model_dict = {"component_model_id":[], "modelgroup":[]}
        for model in range(27):
            model_dict["component_model_id"].append(model)
            if model==model0:
                model_dict["modelgroup"].append(0)
            elif model==model1:
                model_dict["modelgroup"].append(1) 
            else:
                model_dict["modelgroup"].append(2)
        model_dict = pd.DataFrame(model_dict)

        sol = sol.merge(model_dict, on = ["component_model_id"])

        sol = sol.groupby(["sim","modelgroup"]).apply(lambda x: pd.Series({"w":x.weight.sum()})).reset_index()

        sol = pd.pivot_table(index="sim",columns="modelgroup",values="w",data=sol)
    
        t,l,r = sol[0],sol[1],sol[2]
        ax.hexbin( t,l,r, edgecolors="none", gridsize=10 )
        ax.scatter( t,l,r, s=5, color="white",alpha=0.50 )

        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")
        
        ax.set_tlabel(r"$w_{1}$",fontsize=8)
        ax.set_llabel(r"$w_{2}$",fontsize=8)
        ax.set_rlabel(r"1 - $\sum_{m=1}^{2} w_{m}$",fontsize=8)

        ax.taxis.set_tick_params(labelsize=6, tickdir="in")
        ax.laxis.set_tick_params(labelsize=6, tickdir="in")
        ax.raxis.set_tick_params(labelsize=6, tickdir="in")
 
    fig.set_size_inches( 6.5 , 8.5/2 )
    plt.savefig("training_and_redundancy.pdf")
    plt.close()
    


        

        
