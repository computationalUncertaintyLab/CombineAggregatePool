#mcandrew

import sys
sys.path.append("../")

import numpy as np
import pandas as pd

import argparse

from mods.index import index

class adaptive_plus(object):
    
    def __init__(self,data, options,thresh=None):
        self.F    = data["F"]
        self.R    = data["R"]
        self.F_tm = data["F_tm"]
        self.T    = data["T"]

        self.COMBINE   = options["COMBINE"]
        self.AGGREGATE = options["AGGREGATE"]
        self.WEIGHT    = options["WEIGHT"]

        self.LOCATION = options["LOCATION"]
        self.TARGET   = options["TARGET"]
        self.UNIT     = "percent"

        self.FORECAST_WEEK = self.F.iloc[0].forecast_week

        self.combination_params = {}
        self.aggregation_params = {}
        self.weight_params      = {}

        self.num_of_models_for_training = len(self.F_tm.component_model_id.unique())
        
    def has_training_data(self):
        #--when there are no available scores
        if len(self.R)==0 or len(self.R.truth_week.unique()) <=2:
            return 0
        #--when scores are available
        else:
            return 1
        return ensemble_forecast

    #--equally weighted ensemble model
    def equal_weights_nodata(self):
        F = self.F
        
        #--take an equally weighted average over bins (ie equal weights)
        def computeAverage(d):
            return pd.Series({"Value":d.Value.mean()})
        ensemble = F.groupby(["Location","Target","Unit","Bin_start_incl","Bin_end_notincl","forecast_week","FW"]).apply(computeAverage)
        ensemble = ensemble.reset_index()

        ensemble = ensemble.rename(columns = {"forecast_week":"EW"})
        
        component_models = list(F.component_model_id.unique())
        M = len(component_models)
        component_models_and_weights = pd.DataFrame({"cluster":component_models, "weight":[1/M]*M})

        #--add in identifying info for weights
        info = F.iloc[0]

        for var in ["Location","Target","forecast_week"]:
            component_models_and_weights[var] =info[var]

        return ensemble, component_models_and_weights

    #---------------------------------------------------------------------------------------------------------
    # COMBINATION METHODS - START
    #---------------------------------------------------------------------------------------------------------
    #--helper scripts for combinations
    def build_score_matrix(self):
        S = self.R.copy()
        S["logscore"] = np.log(S.Value)
        S = S.replace(-np.inf,-10)
        
        #--compute score matrix
        scoreMatrix = pd.pivot_table(S, index=["truth_week"], columns = ["component_model_id"], values = ["logscore"])
        scoreMatrix = scoreMatrix.replace(np.nan,-10)

        self.scoreMatrix_it = scoreMatrix
        
        #--MAY NEED TO ADD THE FOLLOWING CODE
        #scoreMatrix = scoreMatrix[ scoreMatrix < -10 ] = -10.
        
        return scoreMatrix
    
    def build_corr_matrix(self):
        scoreMatrix = self.build_score_matrix()
        
        component_models = [y for (x,y) in scoreMatrix.columns]

        #--compute correlation matrix
        corr_matrix = scoreMatrix.corr(method="pearson")
        corr_matrix = corr_matrix.replace(np.nan,0)

        #--remove diagonals (correlatin of 1)
        np.fill_diagonal(corr_matrix.values,0)
        
        return corr_matrix
    
    #--main combination method that is called.
    def combine(self):
        def parse_cluster_info(cluster_results):
            component_model_2_cluster = cluster_results[0]
            cluster_extras            = cluster_results[-1]

            self.component_model_2_cluster = component_model_2_cluster
            return component_model_2_cluster, cluster_extras
        
        if self.COMBINE==0:
            #--build correlation matrix that is needed for network clustering
            corr_matrix = self.build_corr_matrix()

            self.combination_params["F"] = self.F
            self.combination_params["corr_matrix"] = corr_matrix
            #--also need a key "phi"
            
            cluster_results = self.build_network_and_clusters()
            component_model_2_cluster, cluster_extras = parse_cluster_info(cluster_results)
            
        elif self.COMBINE==1:
            #--build matrix of log scores for factor analysis
            score_matrix = self.build_score_matrix()

            self.combination_params["score_matrix"]=score_matrix
            #--also need a key called "n_components"
            
            cluster_results = self.build_factor_analysis()
            component_model_2_cluster, cluster_extras = parse_cluster_info(cluster_results)
            
        elif self.COMBINE==2:
            #--build matrix of log scores for KMEANS
            score_matrix = self.build_score_matrix()

            self.combination_params["score_matrix"] = score_matrix
            #--also need a key called "n_clusters"
            
            cluster_results = self.build_kmeans_analysis()
            component_model_2_cluster, cluster_extras = parse_cluster_info(cluster_results)

        elif self.COMBINE==3:
            #--build correlation matrix that is needed for all level clustering
            corr_matrix = self.build_corr_matrix()

            self.combination_params["corr_matrix"] = corr_matrix
            #--also need a key called threshold
            
            cluster_results = self.build_all_the_same_level_clusters()
            component_model_2_cluster, cluster_extras = parse_cluster_info(cluster_results)

        elif self.COMBINE==4:
            #--build correlation matrix that is needed for all level clustering
            corr_matrix = self.build_corr_matrix()

            self.combination_params["corr_matrix"] = corr_matrix
            #--also need a key called threshold
            
            cluster_results = self.build_all_the_same_level_clusters()
            component_model_2_cluster, cluster_extras = parse_cluster_info(cluster_results)
            
    #--combination algorithms
    def build_all_the_same_level_clusters(self):
        corr_matrix = self.combination_params["corr_matrix"]
        thresh      = self.combination_params["threshold"][0]

        thresh      = float(thresh)

        #--removing multi index with label "logscore" so that the columns and rows are labeled by model
        corr_matrix.index = [y for x,y in corr_matrix.index]
        corr_matrix.columns = [y for x,y in corr_matrix.columns] 
        
        #--pick random model
        models = corr_matrix.index

        random_model_to_start = np.random.choice(models)
        models = models[models!=random_model_to_start]
        
        clusters = [ [random_model_to_start] ]

        while len(models) >0:
            random_model = np.random.choice(models)

            #--decide if this model can belong to any of the current clusters.
            cluster_assigned=0
            for cluster_number, cluster in enumerate(clusters):
                if np.all( np.abs(corr_matrix.loc[random_model, cluster]) >thresh):
                    clusters[cluster_number].append(random_model)
                    cluster_assigned=1
                    
            #--if not then generate a new cluster
            if cluster_assigned==0:
                clusters.append( [random_model] )
            models = models[ models != random_model]

        #--build dataframe of clusters
        component_model_2_cluster = { "component_model_id":[], "cluster":[] }
        for cluster_num,cluster in enumerate(clusters):
            for node in cluster:
                component_model_2_cluster["cluster"].append(cluster_num)
                component_model_2_cluster["component_model_id"].append( node )
        component_model_2_cluster = pd.DataFrame(component_model_2_cluster)

        return component_model_2_cluster, clusters
    
    def build_network_and_clusters(self):
        import networkx as nx

        #--import parameter data
        corr_matrix = self.combination_params["corr_matrix"]
        F           = self.combination_params["F"]
        phi         = self.combination_params["phi"]

        #--format params
        phi = float(phi)

        #--build adjacency matrix and network
        A = (corr_matrix>phi).to_numpy()
        network = nx.from_numpy_matrix( A )

        from_node_number_2_model = { node:score_model[-1] for (node,score_model) in enumerate(corr_matrix.columns)}

        #--compute connected components (ie clusters)
        clusters = nx.connected_components(network)

        component_model_2_cluster = { "component_model_id":[], "cluster":[] }
        for cluster_num,cluster in enumerate(clusters):
            for node in cluster:
                component_model_2_cluster["cluster"].append(cluster_num)
                component_model_2_cluster["component_model_id"].append( from_node_number_2_model[node])
        component_model_2_cluster = pd.DataFrame(component_model_2_cluster)

        #E =  F.merge(component_model_2_cluster, on = ["component_model_id"])

        return component_model_2_cluster, network

    def build_factor_analysis(self):
        from sklearn.decomposition import FactorAnalysis as FA
        from scipy.spatial import distance_matrix

        score_matrix = self.combination_params["score_matrix"]
        N = score_matrix.shape[0]
        
        n_components = self.combination_params["n_components"]

        n_components = max(1, int(n_components))
        n_components = min(N, int(n_components) )

        number2model = { num:model for num,(idx,model) in enumerate(score_matrix.columns)}
        
        model = FA(n_components = n_components)
        factors = model.fit_transform( score_matrix )
            
        distances = distance_matrix(factors.T,score_matrix.T)
        clusters = np.argmin(distances,0)

        component_model_2_cluster = { "component_model_id":[], "cluster":[] }
        for num,cluster in enumerate(clusters):
            component_model_2_cluster["component_model_id"].append( number2model[num]  )
            component_model_2_cluster["cluster"].append( cluster )
        component_model_2_cluster = pd.DataFrame(component_model_2_cluster)
        
        return component_model_2_cluster, factors

    def build_kmeans_analysis(self):
        from sklearn.cluster import KMeans as KM

        score_matrix = self.combination_params["score_matrix"]
        N = score_matrix.shape[0]
        
        n_clusters   = self.combination_params["n_clusters"]

        n_clusters = max(1, int(n_clusters) )
        n_clusters = min(N, int(n_clusters) )
        
        number2model = { num:model for num,(idx,model) in enumerate(score_matrix.columns)}

        model   = KM(n_clusters = n_clusters)
        clusters = model.fit_predict(score_matrix.T) # transpose so that observations are models
        centers = model.cluster_centers_
        
        component_model_2_cluster = { "component_model_id":[], "cluster":[] }
        for num,cluster in enumerate(clusters):
            component_model_2_cluster["component_model_id"].append( number2model[num]  )
            component_model_2_cluster["cluster"].append( cluster )
        component_model_2_cluster = pd.DataFrame(component_model_2_cluster)
         
        return component_model_2_cluster, centers

    def add_cluster_data_to_forecasts(self, past_forecasts=True ):
        if past_forecasts:
            self.forecasts_and_clusters = self.F_tm.merge(self.component_model_2_cluster, on = ["component_model_id"])
        else:
            self.forecasts_and_clusters = self.F.merge(self.component_model_2_cluster, on = ["component_model_id"])
        return self.forecasts_and_clusters
    #---------------------------------------------------------------------------------------------------------
    # COMBINATION METHODS - END
    #---------------------------------------------------------------------------------------------------------
   
    #----------------------------------------------------------------------------------------------------------
    # AGGREGATION METHODS - START
    #----------------------------------------------------------------------------------------------------------
   
    def aggregate(self):
        if self.AGGREGATE == 0:
            self.cluster_forecasts = self.median_aggregation()
            
        elif self.AGGREGATE == 1:
            self.cluster_forecasts = self.mean_aggregation()

        elif self.AGGREGATE == 2:
            score_matrix = self.build_score_matrix()
            self.aggregation_params["score_matrix"] = score_matrix
            
            self.cluster_forecasts = self.top_aggregation()
            
    def median_aggregation(self):
        G = pd.pivot_table(index=["forecast_week","FW","cluster","component_model_id"],columns=["Bin_start_incl","Bin_end_notincl"],values="Value",data=self.forecasts_and_clusters)
        G = G.groupby(level=[0,1,2]).apply(lambda x: x.median(0))

        G = G.melt(ignore_index=False,value_name="Value").reset_index()

        def normalize(x):
            x["Value"] = x.Value.values/x.Value.values.sum()
            return x
        return G.groupby(["forecast_week","FW","cluster"]).apply(normalize).reset_index(drop=True)

    def mean_aggregation(self):
        G = pd.pivot_table(index=["forecast_week","FW","cluster","component_model_id"],columns=["Bin_start_incl","Bin_end_notincl"],values="Value",data=self.forecasts_and_clusters)
        G = G.groupby(level=[0,1,2]).apply(lambda x: x.mean(0))
        return G.melt(ignore_index=False,value_name="Value").reset_index()

    def top_aggregation(self,present=True):
        score_matrix = self.aggregation_params["score_matrix"]
        score_matrix = score_matrix.apply( lambda x: pd.Series({"med_logscore": np.median(x)}), 0).melt( value_name = "med_logscore" )

        score_matrix = score_matrix.merge( self.component_model_2_cluster, on = ["component_model_id"])

        #--if using present forecast restrict top models to those that are present
        if present:
            present_forecasts = self.collect_present_forecasts()
            score_matrix = score_matrix.merge(present_forecasts, on = ["component_model_id"])

        def find_best_logscore(x):
            best_score = x.med_logscore.max()
            return x.loc[ x.med_logscore == best_score, ["cluster","component_model_id"] ] #there may be more than one model per cluster that is "best"
        top_models_per_cluster = score_matrix.groupby(["cluster"]).apply( find_best_logscore ).reset_index(drop=True)

        #--subset to top models only
        top_forecasts = self.forecasts_and_clusters.merge( top_models_per_cluster, on = ["component_model_id","cluster"] )

        cluster_forecasts = top_forecasts.groupby(["cluster","forecast_week","FW","Bin_start_incl","Bin_end_notincl"]).apply( lambda x:  pd.Series({"Value":x.Value.mean()})).reset_index()
        return cluster_forecasts

    def collect_present_forecasts(self):
        return pd.DataFrame({"component_model_id": self.forecasts_and_clusters.component_model_id.unique()})
    
    def score_cluster_forecasts(self):
        d = self.cluster_forecasts
        d = d.merge(self.T, left_on = ["FW"], right_on = ["EW"] )
        
        truth_matrix = d.loc[ (d.wili > d.Bin_start_incl) & (d.wili <= d.Bin_end_notincl) ]
        truth_matrix["logscore"] = np.log(d.Value)

        truth_matrix_wide = pd.pivot_table(truth_matrix, index=["forecast_week"], columns = ["cluster"], values = ["logscore"])
        truth_matrix_wide = truth_matrix_wide.replace(np.nan,-10)

        self.truth_matrix_wide = truth_matrix_wide
        return truth_matrix_wide

    #----------------------------------------------------------------------------------------------------------
    # AGGREGATION METHODS - END
    #----------------------------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------------------------
    # WEIGHT METHODS - START
    #----------------------------------------------------------------------------------------------------------
    def weight(self):
        if self.WEIGHT==0:
            self.ensemble_forecast = self.equal_weights()
        
        elif self.WEIGHT==1: 
            cluster_score_matrix = self.score_cluster_forecasts()
            self.weight_params = {"rho": 0.08, "score_matrix":cluster_score_matrix }

            self.adaptive_weights()
            self.ensemble_forecast = self.adaptive_forecast()

    def forecast_present_time(self):
        self.add_cluster_data_to_forecasts(False) # call in present forecast
        self.aggregate()                          # create cluster forecasts

        if self.WEIGHT==0:
            return self.equal_weights()

        elif self.WEIGHT==1:
            return self.adaptive_forecast()

    def add_info_to_cluster_weights(self):
        self.cluster_2_weights["LOCATION"]      = self.LOCATION
        self.cluster_2_weights["TARGET"]        = self.TARGET
        self.cluster_2_weights["forecast_week"] = self.FORECAST_WEEK
            
    def equal_weights(self):
        f = self.cluster_forecasts

        clusters = list(f.cluster.unique())
        C = len(clusters)
        
        cluster_2_weights = pd.DataFrame({"cluster": clusters , "weight": [1/C]*C })
        self.cluster_2_weights = cluster_2_weights
        
        def compute_mean(x):
            cluster_by_bins = pd.pivot_table( index = ["cluster"], columns = ["Bin_start_incl", "Bin_end_notincl"], values = ["Value"], data = x )
            ensemble_forecast = cluster_by_bins.apply( lambda x: pd.Series({"Value": np.mean(x)}), 0)
            ensemble_forecast = ensemble_forecast.melt(value_name="Value")[["Bin_start_incl","Bin_end_notincl","Value"]]
            return ensemble_forecast

        groups = f.groupby(["forecast_week","FW"])
        ensemble_forecast = groups.apply(compute_mean).reset_index()
        
        if groups.ngroups==1:
            ensemble_forecast["forecast_week"] = f.forecast_week.unique()[0]
            ensemble_forecast["FW"]            = f.FW.unique()[0]
        else:
            pass
        ensemble_forecast = ensemble_forecast[["forecast_week","FW","Bin_start_incl","Bin_end_notincl","Value"]]
        ensemble_forecast = ensemble_forecast.rename(columns = {"forecast_week":"EW"})

        ensemble_forecast["Location"] = self.LOCATION
        ensemble_forecast["Target"]   = self.TARGET
        ensemble_forecast["Unit"]     = self.UNIT
        
        return ensemble_forecast

    def adaptive_weights(self):
        import stan
        rho = self.weight_params["rho"]
        score_matrix = self.weight_params["score_matrix"]

        #--if scores are negative inf then change them to -10
        score_matrix[score_matrix==-1*np.inf] = -10
        
        N,M = score_matrix.shape
        
        if M==1:
            weights = [1.] # just one model so no need to assign weights
            
        else:
            model = '''
            data {
               int N;
               int M;
               real alpha;
               matrix [N,M] S;
            }
            parameters {
               simplex [M] pi;
            }
            model {
                vector[M] logpi = log(pi);

                //add in prior
                pi~dirichlet( rep_vector(alpha,M) );

                for (n in 1:N){
                    for (m in 1:M){
                       logpi[m] += S[n,m];
                    }
                }
                target+= log_sum_exp(logpi);
            }
            '''

            #--fit model
            data = {"N":N , "M":M, "alpha": rho*N/M , "S":score_matrix.values}
            posterior = stan.build(model, data=data)
            fit       = posterior.sample(num_samples=5*10**3,num_chains=1)

            #--weights are the mean of the samples
            weights = fit.get("pi").mean(1)

        #--build dataframe of clusters and their corresponding weights
        cluster_2_weights = {"cluster":[], "weight":[]} 
        for cluster, weight in enumerate(weights):
            cluster_2_weights["cluster"].append(cluster)
            cluster_2_weights["weight"].append(weight)
        cluster_2_weights = pd.DataFrame(cluster_2_weights)
        
        self.cluster_2_weights = cluster_2_weights 
        return self.cluster_2_weights
        
    def adaptive_forecast(self):
        #--merge weights with cluster forecasts
        cluster_forecasts = self.cluster_forecasts.merge( self.cluster_2_weights, on = ["cluster"] )
        cluster_forecasts["weighted_value"] = cluster_forecasts.Value * cluster_forecasts.weight
        
        def sum_weighted_values(x):
            return pd.Series({"Value": x.weighted_value.sum()})
        ensemble_forecast = cluster_forecasts.groupby(["forecast_week","FW","Bin_start_incl", "Bin_end_notincl"]).apply(sum_weighted_values).reset_index()

        ensemble_forecast["Location"] = self.LOCATION
        ensemble_forecast["Target"]   = self.TARGET
        ensemble_forecast["Unit"]     = self.UNIT

        ensemble_forecast = ensemble_forecast.rename(columns = {"forecast_week":"EW"})
    
        self.ensemble_forecast = ensemble_forecast
        return self.ensemble_forecast
        
    def score_ensemble_forecasts(self):
        d = self.ensemble_forecast
        d = d.merge(self.T, left_on = ["FW"], right_on = ["EW"] )
        
        truth_matrix = d.loc[ (d.wili > d.Bin_start_incl) & (d.wili <= d.Bin_end_notincl) ]
        truth_matrix["logscore"] = np.log(d.Value)

        return truth_matrix
        
    #----------------------------------------------------------------------------------------------------------
    # WEIGHT METHODS - END
    #----------------------------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------------------------
    # OPTIMIZE METHODS - START
    #----------------------------------------------------------------------------------------------------------
    def optimize(self):
        from scipy.optimize import brute
        
        def optimize_params(self, combination_param):
            print(combination_param)
            if self.COMBINE==0:
                self.combination_params = { "phi"         : combination_param }
            elif self.COMBINE==1:
                self.combination_params = { "n_components": combination_param }
            elif self.COMBINE==2:
                self.combination_params = { "n_clusters"  : combination_param}
            elif self.COMBINE==3:
                self.combination_params = { "threshold"      : combination_param}
            elif self.COMBINE==4:
                self.combination_params = { "threshold"      : [0.80]} #--USER SHOULD SPECIFY
                
            #--COMBINE
            self.combine()
            self.add_cluster_data_to_forecasts()

            #--AGGREGATE
            self.aggregate()
            self.score_cluster_forecasts()

            #--WEIGHT
            self.weight()

            #--ASSESS
            ensemble_scores = self.score_ensemble_forecasts()
            self.ensemble_scores = ensemble_scores

            print(ensemble_scores.logscore.mean())
            return ensemble_scores.logscore.mean()

        if self.COMBINE==0:
            rrange = [(0,1)]
        elif self.COMBINE==1:
            #--compute number of models with previous scores
            rrange = [(1,self.num_of_models_for_training)] 
        elif self.COMBINE==2:
            rrange = [(1,self.num_of_models_for_training)]
        elif self.COMBINE==3:
            rrange = [(0,1)]
        elif self.COMBINE==4:
            pass
            
        if self.COMBINE==0:
            optimal_cluster_param = brute( lambda combination_param: -1.*optimize_params(self,combination_param), ranges = rrange, Ns=50 )
        elif self.COMBINE==3:
            optimal_cluster_param = brute( lambda combination_param: -1.*optimize_params(self,combination_param), ranges = rrange, Ns=20, finish=None ) #optimize_params(self,self.thresh)  #--USER SELECTED THRESHOLD
        elif self.COMBINE==4:
            optimal_cluster_param = 0.80 #--USER SHOULD BE ABLE TO SPECIFY
        else:
            optimal_cluster_param = brute( lambda combination_param: -1.*optimize_params(self,combination_param), ranges = rrange, Ns=self.num_of_models_for_training )
            
        optimal_cluster_param = float(optimal_cluster_param)
        self.optimal_cluster_param = optimal_cluster_param
        print(optimal_cluster_param)
        
        #--create optimal clusterfrom above
        optimize_params(self,[optimal_cluster_param])

    def viz_scores(self):

        def send_back_scores(self,combination_param):
            if self.COMBINE==0:
                self.combination_params = { "phi"         : combination_param }
            elif self.COMBINE==1:
                self.combination_params = { "n_components": combination_param }
            elif self.COMBINE==2:
                self.combination_params = { "n_clusters"  : combination_param}

            #--COMBINE
            self.combine()
            self.add_cluster_data_to_forecasts()

            #--AGGREGATE
            self.aggregate()
            self.score_cluster_forecasts()

            #--WEIGHT
            self.weight()

            #--ASSESS
            ensemble_scores = self.score_ensemble_forecasts()
            self.ensemble_scores = ensemble_scores
            return ensemble_scores.logscore.mean()

        params_scores = [] 
        for param in np.linspace(0,1,50):
            scor = send_back_scores(self, param )
            params_scores.append( (param,scor) )
        return params_scores

    #----------------------------------------------------------------------------------------------------------
    # OPTIMIZE METHODS - END
    #----------------------------------------------------------------------------------------------------------

    #--decide whether to fit adaptive or equal weights
    def fit(self):
        if self.has_training_data():
            print("llama")
            self.optimize()
            
            forecast = self.forecast_present_time()
            return forecast, self.cluster_2_weights
        else:
            self.ensemble, self.cluster_2_weights = self.equal_weights_nodata()

            component_model_2_cluster = self.cluster_2_weights
            component_model_2_cluster = component_model_2_cluster.rename( columns = {"cluster":"component_model_id"} )
            
            component_model_2_cluster["cluster"] = np.arange(len(component_model_2_cluster))
            component_model_2_cluster = component_model_2_cluster[["cluster","component_model_id"]]
            
            self.component_model_2_cluster = component_model_2_cluster

            self.optimal_cluster_param = -1
            
            return self.ensemble, self.cluster_2_weights
#----CLASS END

def add_forecast_week(x):
    from epiweeks import Week

    epiweeks = x.EW.unique()
    ew2forecast = {"EW":[], "Target":[], "FW":[] }

    for ew in epiweeks:
        fw = Week.fromstring(str(ew))

        for target in range(4):
            ew2forecast["EW"].append(ew)
            ew2forecast["Target"].append(target)

            FW = (fw+1+target).cdcformat()
            ew2forecast["FW"].append(FW)
    ew2forecast = pd.DataFrame(ew2forecast)
    ew2forecast["FW"] = ew2forecast.FW.astype(int)

    return x.merge(ew2forecast, on = ["EW","Target"])

def most_recent_release(x):
    return x.groupby(["EW"]).apply( lambda y: y.sort_values("releaseMW").iloc[-1] ).reset_index(drop=True)

if __name__ == "__main__":

    #--download forecast data
    idx = index("../analysisdata/")
    F = idx.grabForecasts_f()

    #--remove unneeded column
    F = F.drop(columns = ["Unnamed: 0"])

    #--only keep the week ahead targets
    F = F.loc[F.Target.isin([0,1,2,3]),:]
    
    #--some bins are considered character and others are floats.
    #--Changing this to always be float
    F["Bin_start_incl"]  = F.Bin_start_incl.astype(float)
    F["Bin_end_notincl"] = F.Bin_end_notincl.astype(float)

    #--download score data
    idx = index("../analysisdata/")
    S = idx.grabForecastScores()
    S = S.drop_duplicates()
    
    #--cap logscores at -10
    S = S.replace(-np.inf,-10)

    #--add in EW alongside the MW
    ewAndMW = idx.grabEW2MW()
    S = S.merge(ewAndMW,on=["MW"])

    #--download truth data
    idx = index("../analysisdata/")
    T = idx.grabEpiData_f()
    
    #--add in a release model week
    ewAndMW = ewAndMW.rename(columns={"EW":"releaseEW","MW":"releaseMW"})
    S = S.merge(ewAndMW,on=["releaseEW"])
    S = S.sort_values(["Target","Location","MW","releaseEW"])

    T = T.merge(ewAndMW,on=["releaseEW"])
    
    #--add in season to F and to S
    ew2season = idx.grabEW2Season()

    F = F.merge(ew2season, on = ["EW"])
    S = S.merge(ew2season, on = ["EW"])
    T = T.merge(ew2season, on = ["EW"])
    
    #--add forecast week to F
    F = add_forecast_week(F)
    
    #--include parameters
   
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--START',type=int)
    # parser.add_argument('--END'  ,type=int)

    #--capture arguments
    #args = parser.parse_args()
    
    #START     = args.START
    #END       = args.END

    #END=0
    #for START in np.arange(0,12541): 
    #    print("START = {:03d}".format(START))
    #START=70
       
    def run_this(START,END=0):
        if END==0:
            if START==0:
                runs = pd.read_csv("allruns.csv", skiprows=START, nrows=1)
            else:
                runs = pd.read_csv("allruns.csv", skiprows=START, nrows=1, names=["Location","Target","MW","Season","Season_int","COMBINE","AGGREGATE","WEIGHT"])
        else:
            runs = pd.read_csv("allruns.csv",skiprows=START)

        print(START)
        for idx, (LOCATION,TARGET,MW,SEASON,SEASON_INT, COMBINE, AGGREGATE,WEIGHT) in runs.iterrows():
            #--subset to within season forecasts
            F_t = F.loc[ (F.Location==LOCATION) & (F.Target==TARGET) & (F.MW==MW) & (F.Season_int==SEASON_INT)] # season here is redundant 
            F_t = F_t.rename(columns = {"EW":"forecast_week"})

            #--include scores and the true ILI values for this location, target, and any epidemic week that has a score
            F_tm = F.loc[ (F.Location==LOCATION) & (F.Target==TARGET) & (F.MW<MW) & (F.Season_int==SEASON_INT)]

            #--add in truth data at release EW to all previous forecasts
            T_t = T.loc[ (T.Location==LOCATION) & (T.releaseMW <= MW) & (T.Season_int==SEASON_INT)]
            if len(T_t) >0:
                T_t = most_recent_release(T_t)
            else:
                pass

            F_tm = F_tm.merge(T_t, left_on = "FW", right_on="EW" )
            F_tm = F_tm.rename( columns = {"EW_x":"forecast_week", "EW_y": "truth_week" } )

            training_matrix = F_tm.loc[ (F_tm.wili >= F_tm.Bin_start_incl) & (F_tm.wili < F_tm.Bin_end_notincl)  ] # subset to bin that contains truth
            training_matrix = training_matrix[["truth_week","component_model_id","Bin_start_incl", "Bin_end_notincl","Value","wili"]]

            data = { "F": F_t
                    ,"F_tm": F_tm
                    ,"R": training_matrix
                    ,"T":T_t}

            options = { "COMBINE"  :COMBINE
                       ,"AGGREGATE":AGGREGATE
                       ,"WEIGHT"   :WEIGHT
                       ,"LOCATION" :LOCATION
                       ,"TARGET"   :TARGET}
            
            #--equally weighted model assigns weights of 1/number of forecasts for each model.
            ensemble_model = adaptive_plus(data, options)

            adaptive_forecast, component_models_and_weights = ensemble_model.fit()
            
            adaptive_forecast =adaptive_forecast[["Location", "Target", "EW", "FW", "Unit", "Bin_start_incl", "Bin_end_notincl", "Value"]] 

            ensemble_model.add_info_to_cluster_weights()
            component_models_and_weights = ensemble_model.cluster_2_weights

            #--save ensemble forecast
            adaptive_forecast.to_csv("./cluster_data/adaptive_plus_selection_ensemble__{:d}_{:d}_{:d}__{:d}_{:d}_{:d}.csv".format(LOCATION,TARGET,MW,COMBINE,AGGREGATE,WEIGHT)
                                     ,header=True,index=False,mode="w")
            #--save weights
            component_models_and_weights.to_csv("./cluster_data/adaptive_plus_selection_ensemble_weights__{:d}_{:d}_{:d}__{:d}_{:d}_{:d}.csv".format(LOCATION,TARGET,MW,COMBINE,AGGREGATE,WEIGHT)
                                    ,header=True,index=False,mode="w")

            #--save component model membership
            component_model_membership           = ensemble_model.component_model_2_cluster
            component_model_membership["cutoff"] = ensemble_model.optimal_cluster_param

            component_model_membership.to_csv("./cluster_data/adaptive_plus_selection_ensemble_membership__{:d}_{:d}_{:d}__{:d}_{:d}_{:d}.csv".format(LOCATION,TARGET,MW,COMBINE,AGGREGATE,WEIGHT)
                                    ,header=True,index=False,mode="w")
        
    from joblib import Parallel, delayed
    Parallel(n_jobs=30)(delayed(run_this)(i) for i in np.arange(0,12541))
 
