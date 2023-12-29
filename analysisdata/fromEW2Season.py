#mcandrew

import sys
sys.path.append("../")

import numpy as np
import pandas as pd

from mods.index import index

def fromEw2Season(ew):
    ew = str(ew)
    yr,week = ew[:4],ew[-2:]
    yr,week = int(yr), int(week)

    if week >=40 and week<=53:
        return "{:d}/{:d}".format(yr,yr+1)
    else:
        return "{:d}/{:d}".format(yr-1,yr)

if __name__ == "__main__":

    idx = index("./")
    forecasts = idx.grabForecasts()

    EWs = sorted(list(forecasts.EW.unique()))

    seasons = []
    for ew in EWs:
        season = fromEw2Season(ew)
        seasons.append(season)
    EWandSeasons = pd.DataFrame( { "EW":EWs, "Season":seasons} )

    #--add a column that orders the seasons using an integer. The earliest season is assigned the int 0.
    season2int = {s:n for n,s in enumerate(sorted(EWandSeasons.Season.unique())) }
    EWandSeasons["Season_int"] = EWandSeasons.Season.replace(season2int)
    
    EWandSeasons.to_csv("EWsandSeasons.csv",index=False)
