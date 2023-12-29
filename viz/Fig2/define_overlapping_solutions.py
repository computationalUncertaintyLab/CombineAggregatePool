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

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

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

if __name__ == "__main__":

    #--real data
    kls = pd.read_csv("./all_KLS.csv")

    overlap = kls.loc[(kls.KL>0) & (kls.Target<=3)].sort_values('KL').reset_index()
    

    overlap1 = overlap.iloc[6107]
    overlap2 = overlap.iloc[342880]
    overlap3 = overlap.iloc[73888]   

    #--download forecast data
    idx = index("../analysisdata/")
    F = idx.grabForecasts_f()

    F = F.drop(columns = ["Unnamed: 0"])
    F = F.loc[F.Unit=="percent"]
    F["Bin_start_incl"] = F.Bin_start_incl.astype(float)
    F["Bin_end_notincl"] = F.Bin_end_notincl.astype(float)

    #S = idx.grabForecastScores()

    epi_data = idx.grabEpiData_f()

    #--collect latest release date
    epi_data = epi_data.groupby(["Location","MW"]).apply( lambda x: x.sort_values("releaseEW").iloc[-1] ).reset_index(drop=True)
    
    def compute_ensemble(weights,d, MW, Location, Target):
        E = d.dot(weights).reset_index()

        E.columns = ["MW","Bin_start_incl","Bin_end_notincl","Value"]
            
        E["Target"]   = Target
        E["Location"] = Location

        return E
        
    def score_ensemble(E,truth):
        S = E.merge( truth, left_on = ["Location","forecast_MW"], right_on = ["Location","MW"])
        S = S.loc[ (S.wili >=S.Bin_start_incl) & (S.wili < S.Bin_end_notincl)]
        return -1*np.log(S.Value).sum()

    def objective(weights,F,MW,Location,Target,truth):
        E = compute_ensemble(weights, F, MW, Location, Target)
        E["forecast_MW"] = E.MW+E.Target

        S = score_ensemble(E,truth)
        return S


    def jacobian(weights, d, truth, MW, Location, Target ):

        E = compute_ensemble( weights, d, MW, Location, Target )
        E["forecast_MW"] = E["MW"]+Target
        
        S = E.merge( truth, left_on = ["Location","forecast_MW"], right_on = ["Location","MW"])
        S = S.loc[ (S.wili >=S.Bin_start_incl) & (S.wili < S.Bin_end_notincl)]

        S = S[["MW_x","Value","Target","Location","EW","wili"]]
        S = S.rename(columns = {"Value":"ensemble_value"})
        
        truth = truth.loc[truth.Location==Location]
        truth["forecast_mw"] = truth.MW-Target
        truth= truth[["wili","forecast_mw"]]
        
        s = subset.merge( truth, left_on = ["MW"], right_on = ["forecast_mw"])
        s = s.loc[ (s.wili >=s.Bin_start_incl) & (s.wili < s.Bin_end_notincl)]

        scores = s.merge(S, left_on = ["MW","Location","Target", "wili"], right_on = ["MW_x","Location","Target", "wili"] )
        scores["fraction"] = scores["Value"] / scores["ensemble_value"]

        jac= scores.groupby("component_model_id").apply(lambda x: x.fraction.sum() ).reset_index().sort_values("component_model_id")[0].values
        return -1*jac
 
    overlaps   = [overlap1,overlap2,overlap3]
    overlap_df = pd.DataFrame()
    for overlap_num,overlap in enumerate(overlaps):

        MW,Location,Target = overlap.MW,overlap.Location,overlap.Target

        subset = F.loc[ (F.MW<=MW) & (F.Location==Location) & (F.Target==Target) ]
        d = pd.pivot_table( index = ["MW","Bin_start_incl","Bin_end_notincl"], columns = ["component_model_id"], values = ["Value"], data = subset )

        o = lambda w: objective(w, d, overlap.MW, overlap.Location, overlap.Target, epi_data)
        jac = lambda x: jacobian(x,d,epi_data,MW,Location,Target)
        
        def output_a_solution(o,jacobian):
            start_pos = np.random.dirichlet([1./27]*27)
            cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
            bnds = tuple((10**-5,1-10**-5) for x in start_pos)

            rslts = minimize( o , jac = jacobian, x0 = start_pos, method='trust-constr', bounds=bnds ,constraints=cons, options={'disp': True, 'maxiter':800,'verbose':2, 'xtol': 10**-4})
            sol = rslts.x

            return sol
            
        solutions = Parallel(n_jobs=25)( delayed(output_a_solution)(o,jac) for i in range(100) )

        solution_df = {"component_model_id":[], "weight":[],"sim":[]}
        for sim,weights in enumerate(solutions):
            solution_df["component_model_id"].extend( np.arange(1,27+1) )
            solution_df["weight"].extend( weights )
            solution_df["sim"].extend([sim]*27)
            
        solution_df = pd.DataFrame(solution_df)
        solution_df["MW"]       = MW
        solution_df["Location"] = Location
        solution_df["Target"]   = Target
        solution_df["overlap"]  = overlap_num

        overlap_df = pd.concat([overlap_df,solution_df])
    overlap_df.to_csv("solutions_for_three_overlaps.csv",index=False)
    
