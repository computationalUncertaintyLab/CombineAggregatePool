#mcandrew

import sys
import numpy as np
import pandas as pd

from glob import glob

if __name__ == "__main__":


    for n,fil in enumerate(glob("./all_briers_new_*.csv")):
        d = pd.read_csv(fil)
        d = d[["Location", "Target", "Season", "compute_model_id", "brS", "threshold"]]
        
        if n == 0:
            d.to_csv("all_briers.csv", mode="w",index=False,header=True)
        else:
            d.to_csv("all_briers.csv", mode="a",index=False,header=False)

