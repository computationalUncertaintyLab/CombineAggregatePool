#mcandrew


class culpl(object):
    
    def __init__(self):
        pass
    
    def style(self):
        import matplotlib.pyplot as plt
        plt.style.use("fivethirtyeight")

    def styleTicks(self,axs):
        for ax in axs:
             ax.tick_params(which="both",labelsize=8)
             
    def styleLbls(self,axs):
        for ax in axs:
             xlbl = ax.get_xlabel()
             ax.set_xlabel(xlbl,fontsize=10)
             
             ylbl = ax.get_ylabel()
             ax.set_ylabel(ylbl,fontsize=10)
    
