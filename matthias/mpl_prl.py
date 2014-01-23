import scipy.io
import matplotlib
from mpl_toolkits.axes_grid.inset_locator import inset_axes # to have inset
from numpy import mean, std

def setrcparam(prl=False):
    matplotlib.rcParams['figure.figsize']=(13.0,8.0)
    matplotlib.rcParams['font.size']=15
    matplotlib.rcParams['svg.fonttype']='none'
    matplotlib.rcParams['text.latex.unicode']=True

    matplotlib.rcParams['font.family']='serif'
    prl=False
    if prl:
        matplotlib.rcParams['font.serif']='Times'
        matplotlib.rcParams['text.usetex']=True
        fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inches
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        golden_mean = (sqrt(5)-1.0)/2.0
        fig_height =fig_width*golden_mean       # height in inches
        fig_size = [fig_width,fig_height]
        fntsze = 36
        matplotlib.rcParams['font.size']=fntsze*0.8
        matplotlib.rcParams['legend.fontsize']= fntsze
        matplotlib.rcParams['figure.figsize']=(fig_width*2,fig_height*2)
        #          'font.size' : 10,
        matplotlib.rcParams['axes.labelsize'] = fntsze
        #      'font.size' : 10,
        #      'legend.fontsize': 10,
        matplotlib.rcParams['xtick.labelsize'] = fntsze
        matplotlib.rcParams['ytick.labelsize'] = fntsze


def doMinAndStd(data,flags):
    """function that taking two list, return the average and the std of the values of the first by group of the second."""
    #convert flag to a set:
    flags_set=set(list(flags));
    aligned_data =  zip(data,flags);
    rv = [];
    for flag in flags_set:
        view = [ d for d,x in aligned_data if x == flag ]
        rv.append((mean(view),std(view),flag))
    return rv
