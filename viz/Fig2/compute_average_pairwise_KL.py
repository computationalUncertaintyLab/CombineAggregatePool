#mcandrew

import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from mods.index import index

from scipy.stats import entropy as KL

if __name__ == "__main__":

    #--download forecast data
    idx = index("../analysisdata/")
    F = idx.grabForecasts_f()

    F = F.drop(columns = ["Unnamed: 0"])
    F = F.loc[F.Unit=="percent"]
    F["Bin_start_incl"] = F.Bin_start_incl.astype(float)

    eps = np.finfo(float).eps

    data = {"MW":[],"Location":[],"Target":[],"model0":[], "model1":[], "KL":[]}

    groups = F.groupby(["MW","Location","Target"])
    N = groups.ngroups
    
    for n, ((mw,location,target), subset) in enumerate(groups):
        print("{:05d}/{:05d}".format(n,N))
        
        d = pd.pivot_table(index=["Bin_start_incl"], columns = ["component_model_id"], values = ["Value"], data = subset)
        d.columns = [y for (x,y) in d.columns]
        
        d = d+eps
        d = d / d.sum(0)

        models = d.columns
        
        for model0 in models:
            for model1 in models:
                two_models = d[[model0,model1]]

                kl = KL(two_models.iloc[:,0], two_models.iloc[:,1])

                data["MW"].append(mw)
                data["Location"].append(location)
                data["Target"].append(target)
                data["model0"].append(model0)
                data["model1"].append(model1)
                data["KL"].append(kl)
                
    data = pd.DataFrame(data)
    data.to_csv("./all_KLS.csv",index=False)
