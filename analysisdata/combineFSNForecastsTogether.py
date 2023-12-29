#mcandrew

import re
import sys
import os

import numpy as np
import pandas as pd

from glob import glob

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--FSNForecasts",help = 'folder containing FSN component model forecasts')
    
    args = parser.parse_args()
    FSNPATH = args.FSNForecasts
    FSNPATHS = os.listdir(FSNPATH)

    ttl = len(FSNPATHS)
    n,m=1,1

    for forecastFolder in FSNPATHS:
        modelName = forecastFolder
        
        for foreCastFile in sorted(glob('{:s}{:s}/*.csv'.format(FSNPATH,modelName))):
            EW = re.findall('EW\d{2}-\d{4}',foreCastFile)[0]
            sys.stdout.write('\x1b[2K\r ({:d}/{:d}) {:s}-{:s}\r'.format(n,ttl,modelName,EW))
            sys.stdout.flush()
            
            forecasts = pd.read_csv(foreCastFile)
            forecasts['component_model_id'] = modelName

            EW,year = re.findall('(\d+)-(\d+)',EW)[0]
            forecasts['EW'] = int('{:04d}{:02d}'.format(int(year),int(EW)))

            if m==1:
                forecasts.to_csv('./0_data/FSNforecasts.csv', header=True, mode='a')
            else:
                forecasts.to_csv('./0_data/FSNforecasts.csv', header=False, mode='a')
            m+=1
        n+=1 
