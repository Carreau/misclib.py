# -*- coding: utf-8 -*-
from matplotlib import pyplot
from numpy import ceil
def plot_n(plotfunc,data,by=3,w=3, fig=None):
    """ apply a plot function to N dataset and plot in grid
        
        Parameters :
        plotfunc : function handle that take one parameter[1]
        data : iterable of parameters taken by `plotfunc`
        by : overlay n curves by graph
        w : number of column
        
        trace toutes less courbes passées en premier argument sur des
        subfigures dans un tableau *w* (3 par ddefaut) par N (determiné automatiquement)
        avec *by* courbes par graph, et accesoiremetn en rouge si la courbe est dans *slist*.
        affiche la detectin du zero si *nf0* == True
    """
    if not fig:
        fig=pyplot.figure()
    nbr = len(data)
    h=int(ceil(nbr/w)+1)
    fig.set_figheight(h*4)
    for i in range(nbr):
        pyplot.subplot(h,w,int(i)+1)
        plotfunc(data[i])

