# import plotters
import getdist.plots as gdplt
from getdist import loadMCSamples
import matplotlib.pyplot as plt

# values
burn_in = 0.3
filename = 'plots.png'

# path to chain
#chain0 = 'chains/CMB-SPA/CMB-SPA'      # CMB-SPA
chain1 = 'chains/PPS+DESI/PPS+DESI'    # PPS + DESI 
chain2 = 'chains/PP+DESI/PP+DESI'      # PP + DESI
#chain3 = 'chains/CMB-SPA+PPS+DESI'     # CMB-SPA + PPS + DESI
#chain4 = 'chains/CMB-SPA+PP+DESI'      # CMB-SPA + PP + DESI

# load and plot samples
try:
    #samples0 = loadMCSamples(chain0, settings={'ignore_rows':burn_in})
    samples1 = loadMCSamples(chain1, settings={'ignore_rows':burn_in})
    samples2 = loadMCSamples(chain2, settings={'ignore_rows':burn_in})
    #samples3 = loadMCSamples(chain3, settings={'ignore_rows':burn_in})
    #samples4 = loadMCSamples(chain4, settings={'ignore_rows':burn_in})

    g = gdplt.get_subplot_plotter()
    g.triangle_plot([samples1, samples2], 
                    ['H0','ombh2','omch2','w','wa'],
                    filled=False,
                    legend_labels=['Pantheon+ w/ SH0ES + DESI',
                                   'Pantheon+ + DESI'])

    plt.legend()

    # save figure
    plt.savefig(filename, dpi=300)
    print(f"Figure saved as {filename}")

except Exception as e:
    print(f"Error plotting chains: {e}")