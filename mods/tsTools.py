class tsTools(object):

    def __init__(self,xt):
        self.xt = xt
    
    def computeAutorCorrs(self,k=10):
        import numpy as np

        ccs = []
        for _ in range(1,k+1):
            ccs.append(np.corrcoef( self.xt[_:], self.xt[:-_] )[0,1])
        self.ccs = ccs

        lags = np.arange(1,k+1)
        return lags, ccs

    def computeRunningMean(self,window=5):
        from scipy.ndimage.filters import uniform_filter1d
        rMean = uniform_filter1d(self.xt,window)
        
        self.rMean = rMean
        self.resids = self.xt - rMean
        
        return rMean,self.resids
        

    
 
