#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d
from scipy.integrate import quad

if __name__ == "__main__":

    model2name = {-1:"Equal",-2:"Static",-3:"Adapt",320:"CAP Equal", 321:"CAP Adapt"}
    
    #--PIT---------------------------------------------------------------------------------------------------------------------
    cdf_pits = pd.read_csv("./cdf_pits.csv")
    models = [-1,-3,320,321]
    
    subset                     = cdf_pits.loc[cdf_pits.compute_model_id.isin(models)]
    subset["compute_model_id"] = subset.compute_model_id.replace(model2name)

    subset = subset.loc[subset.Target==1]

    data = {"model":[],"location":[], "area":[] }
    for (model,location), model_data in subset.groupby(["compute_model_id","Location"]):
        f = interp1d( model_data.avg, model_data.px - model_data.avg )
        area, precision = quad(f,0.025,0.975)

        data["model"].append(model)
        data["location"].append(location)
        data["area"].append(area)
    data = pd.DataFrame(data)
    #---------------------------------------------------------------------------------------------------------------------------

    #--Brier score---------------------------------------------------------------------------------------------------------------
    d = pd.read_csv("../../score_ensemble_models/all_briers.csv")
    d["Location"] = d.Location.astype(int)
    d["Target"]   = d.Target.astype(int)

    models = [-1,-3,320,321]
    
    d             = d.loc[d.compute_model_id.isin(models)]
    d["compute_model_id"] = d.compute_model_id.replace(model2name)

    d["ensemble_type"]  = d.compute_model_id.replace({"Adapt":"Adapt"   ,"CAP Adapt":"Adapt","Equal":"Equal","CAP Equal":"Equal"})
    d["algorithm_type"] = d.compute_model_id.replace({"Adapt":"Original","CAP Adapt":"CAP","Equal":"Original","CAP Equal":"CAP"})

    d = d.loc[d.Target==1]

    data = {"model":[],"location":[], "season":[], "area":[], "area2":[] }
    for (model,location,season), model_data in d.groupby(["compute_model_id","Location","Season"]):
    
        model_data = model_data.sort_values(["threshold"])

        f = interp1d( model_data.threshold, model_data.brS )
        area, precision = quad(f,0.,10,limit=100)
        area2, precision2 = quad(f,2.,3.,limit=100)
        

        data["model"].append(model)
        data["location"].append(location)
        data["season"].append(season)
        data["area"].append(area)
        data["area2"].append(area2)
    data = pd.DataFrame(data)
    #-----------------------------------------------------------------------------------------------------------------------------
    
        
        
    
        
    
