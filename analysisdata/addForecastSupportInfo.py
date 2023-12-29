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

    idx = index("./")
    forecasts = idx.grabForecasts()

    forecasts = forecasts.drop(columns=["Unnamed: 0"])
    
    forecasts = forecasts.loc[ forecasts.Type=="Bin",:]
    forecasts = forecasts.drop(columns = ["Type"])
    
    EW2MW = index("./").grabEW2MW()
    forecasts = forecasts.merge(EW2MW,on=["EW"])

    def convertString2Nums(column,outfile,forecasts):
        strings = sorted(forecasts[column].unique())
    
        fromString2Number = { loc:n  for n,loc in enumerate(strings) }
        writeDataDict(outfile,fromString2Number)
   
        forecasts[column] = forecasts[column].replace(fromString2Number)
        return forecasts

    for col in ["Location","Target","component_model_id"]:
        forecasts = convertString2Nums(col,"from{:s}2Number.json".format(col),forecasts)
    forecasts.to_csv("forecastsFormatted.csv")
