# -*- coding: utf-8 -*-

from __future__ import division
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from numpy import shape
from numpy import sum
import numpy as np
from collections import namedtuple
import scipy.ndimage

import pyximport
pyximport.install()
import dmask
import io


__all__ = ['proj3',
        'light_table',
        'make_posmap',
        'projxy'
        ]

def proj3b(im, extent, **kwargs):
    return proj3(im,extent.xm,extent.ym,extent.zm, **kwargs)


def projxy(ax, arr,  cmap='gray', xm=None, ym=None, alpha=1):
    assert(xm is not None)
    assert(ym is not None)
    fig = None
    if not ax: 
        fig, ax = plt.subplots()

    ax.imshow(sum((arr), axis=0), cmap=cmap, extent=[0,xm,0,ym], axes='equal' , origin='lower',alpha=alpha)
    ax.set_xlabel(r'x \mu m')
    ax.set_ylabel(u'y µm')
    return fig,ax



def proj3(arr, xm, ym, zm, cmap=None, circles=None, axes=None, alpha=1):
    if not axes:
        fig = figure()
        fig.set_figwidth(15)
        fig.set_figheight(15    )
        gs = gridspec.GridSpec(2, 2, hspace=0, wspace=0)
        
        ax3 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[1,0], sharex=ax3)
        ax1 = plt.subplot(gs[1,1], sharey=ax2)    
        ax3.set_xlim(0,xm)
    #ax3.set_ylim(-ym,ym*2)
    else:
        ax1,ax2,ax3 = axes
   
    projxy(ax3, arr, cmap=cmap, xm=xm,ym=ym, alpha=alpha)

    ax2.imshow(sum((arr), axis=1), cmap=cmap, extent=[0,xm,0,zm], axes='equal', origin='lower',alpha=alpha)
    ax2.set_xlabel(u'x µm')
    ax2.set_ylabel(u'z µm')

    
    ax1.imshow(sum((arr), axis=2), cmap=cmap, extent=[0,ym,0,zm], axes='equal', origin='lower',alpha=alpha)
    ax1.set_xlabel(u'y µm')
    ax1.set_ylabel(u'z µm')
    ax1.yaxis.tick_right()

    if circles is not None :
        x,y,z,r = circles[0:4]
        circ = plt.Circle((x, y), radius=r, color=(0,0,0,0), ec='r', lw=2)
        ax3.add_patch(circ)
        circ = plt.Circle((x, z), radius=r, color=(0,0,0,0), ec='r', lw=2)
        ax2.add_patch(circ)
        circ = plt.Circle((y, z), radius=r, color=(0,0,0,0), ec='r', lw=2)
        ax1.add_patch(circ)
        if len(circles)>4:
            x,y,z,r = circles[4:8]
            circ = plt.Circle((x, y), radius=r, color=(0,0,0,0), ec='r', lw=2)
            ax3.add_patch(circ)
            circ = plt.Circle((x, z), radius=r, color=(0,0,0,0), ec='r', lw=2)
            ax2.add_patch(circ)
            circ = plt.Circle((y, z), radius=r, color=(0,0,0,0), ec='r', lw=2)
            ax1.add_patch(circ)

    return (ax1,ax2,ax3)


def light_table(im,xm,ym,zm, n=5):
    h = shape(im)[0]//n
    zl,xl,yl = shape(im)
    fig, axes = plt.subplots(h,n)
    fig.set_figwidth(15)
    fig.set_figheight(h*2.5)
    faxes = axes.flatten()
    for i in range(0, shape(im)[0],1):
        faxes[i].imshow(im[i,::3,::3], extent=[0,xm,0,ym], cmap='gray', axes='equal')
        faxes[i].xaxis.set_ticks([])
        faxes[i].yaxis.set_ticks([])
    #faxes[-1].imshow(im[:,:,300], extent=[0,xm,0,zm], axes='equal')
    return fig,axes


def make_posmap(profile, dx, dy, dz):
    """workaround wile I don't have the scale in X/Y/Z
    """
    
    pshape = list(np.shape(profile))
    pshape.append(3)
    posmap = np.zeros(pshape)

    for i in range(pshape[0]):
        posmap[i,:,:,0] = dz*i
    
    for i in range(pshape[1]):
        posmap[:,i,:,1] = dy*i
        
    for i in range(pshape[2]):
        posmap[:,:,i,2] = dx*i
    
    return posmap

def dgauss(mean, x, spread=5):
    """ Generate a gausssian profile around `mean` and
    with variation `spread`. Not normalized though
        
    """
    return np.exp(-(x-mean)**2/spread)

import skimage.io

ImShape = namedtuple('ImShape', ['z','y','x'])
ImRes =   namedtuple('ImRes',   ['dz','dy','dx'])
ImExtent =namedtuple('ImRes',   ['zm','ym','xm'])

DoubletData = namedtuple('DoubletData',['im','shape','res','extent','posmap', 'nim'])
def im_data(name=None, zm=0, im=None, zcut=None):
    if im is None:
        im = skimage.io.imread(name, plugin='tifffile')

    olen = len(im)
    if zcut :
        im = im[zcut]


    imshape = ImShape(*shape(im))
    print 'openned image of', imshape
    dx = 0.129#0.108333
    dy = dx#0.108333
    #zm = 31.62#*1.2 ## um in 35
    #assert( imshape.z ==35)
    zm = zm*(len(im)/olen)
    xm = imshape.x*dx
    ym = imshape.y*dy
    dz = zm / (imshape.z-1)

    posmap = make_posmap(im, dx, dy, dz)

    return DoubletData(
            im,
            imshape,
            ImRes(dz,dy,dx),
            ImExtent(zm,ym,xm),
            posmap,
            normalize_intens(im)
            )

def makegaussd(x0,y0,z0,r0,
               x1,y1,z1,r1,
               cu,thick, blur=1):
    m = dmask.makemask_doublet2(x0,y0,z0,r0,
                   x1,y1,z1,r1,
                   cu.posmap,thick)
    d= blur
    mm = scipy.ndimage.filters.gaussian_filter1d(m,d/cu.res.dz,axis=0)
    mm = scipy.ndimage.filters.gaussian_filter1d(mm,d/cu.res.dy,axis=1)
    mm = scipy.ndimage.filters.gaussian_filter1d(mm,d/cu.res.dx,axis=2)
    return mm


def normalize_intens(raw_image):
    
    ufun = lambda x: max(x,0)
    ufun = np.vectorize(ufun)
    nprofiles = np.array(raw_image)
    nprofiles = ufun(nprofiles-nprofiles.min())
    nprofiles = nprofiles.astype('float32')
    nprofiles = nprofiles/nprofiles.max()
    return nprofiles

def get_amaris_info(infon):
    with io.open(infon) as f:
        for l in  f.readlines():
            if l.startswith('x :'):
                print l,
#            elif l.startswith('y :'):
#                print l,
#            elif l.startswith('Z :'):
#                print l
    
            elif l.strip().startswith('Repeat Z'):
#                print l.strip()
                zzm= abs(float(l.strip()[11:17])),
            else :
                pass
                #print l,
    return {'zzm':zzm[0]}
