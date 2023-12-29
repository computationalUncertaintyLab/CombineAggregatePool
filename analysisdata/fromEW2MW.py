#mcandrew

import sys
sys.path.append("../")

import numpy as np
import pandas as pd

from mods.index import index

if __name__ == "__main__":

    idx = index("./")
    T = idx.grabEpiData()
    EWs = sorted(list(T.epiweek.unique()))

    EWandMW = pd.DataFrame( { "EW":EWs, "MW":np.arange(0,len(EWs)) } )
    EWandMW.to_csv("EWsandMWs.csv",index=False)
   
 
