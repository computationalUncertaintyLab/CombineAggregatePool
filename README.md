# Combine Aggregate Pool (CAP) Ensemble 

**Ningxi Wei,1 Xinze Zhou,1 Wei-Min Huang,1 and Thomas McAndrew2**
**1 Department of Mathematics, College of Arts and Science, Lehigh University, Bethlehem, Pennsylvania, United States of America**
**2 Department of Community and Population health, College of Health, Lehigh University, Bethlehem, Pennsylvania, United States of America**

### Tutorial 
A tutorial of the exact CAP algporithm that was implemented in the above manuscript can be viewed as a notebook at notebook/tutorial.ipynb. 
In that tutorial, the "ground truth" data is simulated from a SIR model with demographic stochasticity using Gillespie's tau-leap algorithm. 
Three models are trained: a SIR, SEIR, and Kalman Filter. 
The CAP ensemble is trained and a plot is generated. 

### How to run code

A Makefile is included in this repository to run the code that was used in the manuscript.
The Makefile includes two tags that can be run. 
The first tag (from_scratch) is not recommended as this tag runs the code from scratch and will take a long time (on the order of a week or more). 
The second tag (quick_run) is recommended. The datasets needed are downloaded via Zenodo and the code to run all plots is executed.  

## Data 

### Zenodo Dataset links
1. Forecasts [link](https://zenodo.org/records/15233040)
2. Component Model scores [link](https://zenodo.org/records/15258854)
3. Ensemble Model scores [link](https://zenodo.org/uploads/15258864)

ILI data can be downloaded from https://cmu-delphi.github.io/delphi-epidata/api/flusurv.html
The python script analysisdata/download__epidata.py can be used to download ILI data 

Component model forecasts can be cloned from https://github.com/FluSightNetwork/cdc-flusight-ensemble
The script analysisdata/combineFSNForecastsTogether.py can be used to combine forecasts into a single dataset. 

The script _build_ensemble_models/adaptive_plus__cluster__selection.py_ can be used to run a CAP algorithm. There are at present several choices for each of the C, A, and P approaches. 
Efforts in the future will be made to produce an easy to use python package that implements the CAP algorithm.

The ./score_ensemble_models folder contains code to produce logscores, Brier scores, PIT scores. 
