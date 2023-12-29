class index(object):

    def __init__(self,root):
        self.root = root
        
    def grab(self,fl):
        import os
        import pandas as pd
        
        fl = os.path.join(self.root,fl)
        print("Importing {:s}".format(fl))
        return pd.read_csv(fl)

    def grabForecasts(self):
        return self.grab("FSNforecasts.csv")

    def grabEpiData(self):
        return self.grab("epidata__original.csv")

    def grabForecasts_f(self):
        return self.grab("forecastsFormatted.csv.gz")

    def grabEpiData_f(self):
        return self.grab("epidataFormated.csv.gz")

    def grabForecastScores(self):
        return self.grab("scores.csv.gz")

    def grabEW2MW(self):
        return self.grab("EWsandMWs.csv")

    def grabEW2Season(self):
         return self.grab("EWsandSeasons.csv")

    def grabLocationDict(self):
         import os
         import json
         fl = os.path.join(self.root,"fromlocation2Number.json")
         return json.load(open(fl,"r"))
        
