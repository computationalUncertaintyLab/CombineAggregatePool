#mcandrew

import sys
sys.path.append("../")

import json

import numpy as np
import pandas as pd

from mods.index import index

def writeDataDict(outfl,dct):
    fout = open(outfl,"w")
    json.dump(dct,fout,indent=6)
    fout.close()

if __name__ == "__main__":

    idx = index(".")
    epidata = idx.grabEpiData()

    #--subset to columns of interest
    epidata = epidata[ ["release_date","region","issue","epiweek","lag","wili"] ]

    #--rename columns
    epidata = epidata.rename(columns = {"region":"Location", "epiweek":"EW", "release_date":"releaseDate", "issue":"releaseEW" })
    #epidata = epidata[epidata.lag>=1]
    
    EW2MW = index("./").grabEW2MW()
    epidata = epidata.merge(EW2MW,on="EW")

    epidata["Location"] = epidata["Location"].replace({loc:n for n,loc in enumerate(epidata.Location.unique())})
    epidata.to_csv("epidataFormated.csv",index=False)
