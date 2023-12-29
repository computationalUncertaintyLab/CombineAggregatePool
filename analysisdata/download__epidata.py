#mcandrew

import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from mods.index import index

from delphi_epidata import Epidata

def generateEpiWeeks():
    from epiweeks import Week

    epiweeks = []
    for year in np.arange(2010,2018+1):
        yr=year
        
        ew = Week(year,40)
        while True:
            epiweeks.append( ew.cdcformat() )
            ew+=1

            yr = ew.year

            if yr == (year+1) and ew.week>20:
                break
    return epiweeks

if __name__ == "__main__":

    epiweeks = generateEpiWeeks()
    
    n=1
    for location in list(np.arange(1,10+1))+ ['nat']:
        print(location)
        for epiweek in ['201741']: #epiweeks:

            #--find maxlag
            if location=='nat':
                maxlag = Epidata.fluview(['nat'], [epiweek])['epidata'][0]['lag']
            else:
                maxlag = Epidata.fluview(['HHS{:d}'.format(location)], [epiweek])['epidata'][0]['lag']
            
            lag_miss_counter=0
            for lag in np.arange(0,maxlag+1):
                if location == 'nat':
                    data = Epidata.fluview(['nat'], [epiweek], lag = lag)
                else:
                    data = Epidata.fluview(['HHS{:d}'.format(location)], [epiweek], lag = lag)

                if lag_miss_counter==5:
                    break
                
                if data['result']==-2:
                    lag_miss_counter+=1
                    continue
                
                epidata = data['epidata'][0]
                epidata = {k:[v] for k,v in epidata.items()}

                epidata = pd.DataFrame(epidata)

                if n==0:
                    epidata.to_csv("./epidata__original.csv",header=True,mode="w",index=False)
                    n=1
                else:
                    epidata.to_csv("./epidata__original.csv",header=False,mode="a",index=False)
