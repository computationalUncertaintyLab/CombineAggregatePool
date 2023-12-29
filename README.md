# Combine Aggregate Pool (CAP) Ensemble 
**Ningxi Wei,1 Xinze Zhou,1 Wei-Min Huang,1 and Thomas McAndrew2**
**1 Department of Mathematics, College of Arts and Science, Lehigh University, Bethlehem, Pennsylvania, United States of America**
**2 Department of Community and Population health, College of Health, Lehigh University, Bethlehem, Pennsylvania, United States of America**

ILI data can be downloaded from https://cmu-delphi.github.io/delphi-epidata/api/flusurv.html
The python script analysisdata/download__epidata.py can be used to download ILI data 

Component model forecasts can be cloned from https://github.com/FluSightNetwork/cdc-flusight-ensemble
The script analysisdata/combineFSNForecastsTogether.py can be used to combine forecasts into a single dataset. 

The script _build_ensemble_models/adaptive_plus__cluster__selection.py_ can be used to run a CAP algorithm. There are at present several choices for each of the C, A, and P approaches. 
Efforts in the future will be made to produce an easy to use python package that implements the CAP algorithm.

The ./score_ensemble_models folder contains code to produce logscores, Brier scores, PIT scores. 
