#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob

from scipy.stats import entropy
from scipy.special import rel_entr

if __name__ == "__main__":

    cluster_data = {"Location":[],"Target":[],"Modelweek":[],"Combine":[],"Aggregate":[],"Pool":[],"nclusters":[],"entropy_ratio":[]}
    for fil in glob("../../build_ensemble_models/cluster_data/*ensemble_membership*.csv"):
        d       = pd.read_csv(fil)

        weights = pd.read_csv(fil.replace("membership","weights"))

        name,LTM,ensemble_params = fil.split("__")
        combine,agg,weight_csv = ensemble_params.split("_")
        weight = weight_csv.split(".")[0]
        
        combine = int(combine)
        agg     = int(agg)
        weight  = int(weight)

        location,target,modelweek = [int(x) for x in LTM.split("_")]
        
        nclusters = d.cluster.max()+1 

        e = entropy(weights.weight.values)
        ref_entropy = entropy(np.array([1./nclusters]*nclusters))

        entropy_ratio = e/ref_entropy
    
        cluster_data["Location"].append(location)
        cluster_data["Target"].append(target)
        cluster_data["Modelweek"].append(modelweek)
        cluster_data["Combine"].append(combine)
        cluster_data["Aggregate"].append(agg)
        cluster_data["Pool"].append(weight)
        cluster_data["nclusters"].append(nclusters)
        cluster_data["entropy_ratio"].append(entropy_ratio)
        
    cluster_data = pd.DataFrame(cluster_data)
    cluster_data.to_csv("cluster_data.csv",index=False)
    

