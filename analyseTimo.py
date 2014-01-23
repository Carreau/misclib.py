from scipy.io import loadmat
from matplotlib.pyplot import plot,xlim,ylim,title,xlabel,ylabel
import numpy
from numpy import array
from pylab import *


def loadAndCat():
	files=(
	'30_mars_10cp_25apr_fit_res.mat',
	'30_mars_30cp_25apr_fit_res.mat',
	'30_mars_50cp_25apr_fit_res.mat',
	'8mars25Arp50cp_fit_res.mat',
	'8mars25arp00cp_fit_res.mat',
	'run1_1-mars-2011_25arp-10cp_fit_res.mat',
	'run3_1-mars-2011_25arp-30cp_fit_res.mat',
	'run4_1-mars-2011_25arp-30cp_fit_res.mat',
	)

	struct= numpy.ndarray(0,dtype=object)

	for sfile in files :
		temp = loadmat(sfile, struct_as_record=False,squeeze_me=True);
		struct = numpy.append(struct,temp['fit_res'])
	return struct

def plotData(data):
	time_array=array([u.time_m for u in data])
	plot([u.cp + y/3.0 for u,y in zip(data,time_array)],[u.alpha for u in data],'+')
	title('power law exponent versus (cp+time)')
	xlabel('[cp] + time/3 (min)')
	ylabel('powerlaw exponant')

	figure(2)
	lmean=[mean([x.alpha for x in data if x.cp==beta]) for beta in set(array([x.cp for x in data]))]
	lstd=[std([x.alpha for x in data if x.cp==beta]) for beta in set(array([x.cp for x in data]))]
	cplist= list(set(array([x.cp for x in data])))
	plot([u.cp + y/3.0 for u,y in zip(data,time_array)],[u.alpha for u in data],'+')
	errorbar(cplist,lmean,yerr=lstd,fmt='ro')
	title('power law exponent versus (cp+time)')
	xlabel('[cp] + time/3 (min)')
	ylabel('powerlaw exponant')
	xlim(xmin=-1)
	ylim(ymax=1);
	ylim(ymin=-13);
	ylim(ymin=-12);

	figure(3)
	plot([u.cp + y/3.0 for u,y in zip(data,time_array)],[u.d0 for u in data],'+')
	plot([u.cp + 0*y/3.0 for u,y in zip(data,time_array)],[u.d0 for u in data],'+')
	clf()
	plot([u.cp + 0*y/3.0 for u,y in zip(data,time_array)],[u.d0 for u in data],'+')
	xlim((-1,51))
	ylim(ymax=10)
	d0lstd=[std([x.d0 for x in data if x.cp==beta]) for beta in set(array([x.cp for x in data]))]
	d0lmean=[mean([x.d0 for x in data if x.cp==beta]) for beta in set(array([x.cp for x in data]))]
	errorbar(cplist,d0lmean,yerr=d0lstd,fmt='ro')
	clf 
	clf()
	errorbar(cplist,d0lmean,yerr=d0lstd,fmt='ro')
	plot([u.cp + y/5.0 for u,y in zip(data,time_array)],[u.d0 for u in data],'+')
	xlim(xmin=-1)
	title('do evolution with cp and time')
	title('d0 evolution with cp and time')
	xlabel('cp+time/5')
	ylabel('d0')
	ylim((-1,10))
