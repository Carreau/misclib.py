
from __future__ import division
from collections import namedtuple
from numpy import arccos, sqrt, cos, array
from matplotlib.pyplot import subplots
import  matplotlib.pyplot as plt
import numpy as np


Point = namedtuple('Point', 'x y z'.split(' '))
Sphere = namedtuple('Sphere', 'x y z r'.split(' '))


def _cercle_intersect(d, R, r):
    """
    helper return the intersection coordinate of (one of) the intersection of two circle
    
    retrun (X, Y)
    
    d = distance between two circle center
    R = radius of circle at (0, 0)
    r = radius of circle at (d, 0)
    """
    if R+r < d : 
        return (None, None)
    print('d, R, r :',d, R, r)
    x = (d**2+R**2-r**2)/(d*2)
    y = sqrt(R**2-x**2)
    return (x,y)

def cercle_intersect(xa, ya, ra, xb, yb, rb):
    uab = array([xb-xa, yb-ya])
    dcircle = sqrt(sum(uab**2))
    print('dcircle :',dcircle)
    print('uab :',uab)
    locx, locy = _cercle_intersect(dcircle, ra, rb)
    uab = uab/ dcircle

    Hx = xa+locx*uab[0] - locy*uab[1]
    Hy = ya+locx*uab[1] + locy*uab[0]
    Hxb = xa+locx*uab[0] + locy*uab[1]
    Hyb = ya+locx*uab[1] - locy*uab[0]
    print('Hx, Hy :',Hx, Hy)
    return ((Hx,Hy),(Hxb,Hyb))

def draw_tangeants(ax, x,y,r, X, Y, R,l=10, dirs=[]):
    (Ap,Bp), (Cp,Dp) = cercle_intersect(x,y,r,X,Y,R)
    ax.scatter(Ap, Bp)
    ax.scatter(Cp, Dp)
    ax.scatter(X, Y)
    ax.scatter(x, y)

    tx, ty = tan_seg(x,y,Ap,Bp,length=l)
    ax.plot(tx,ty)
    tx1, ty1 = tan_seg(X,Y,Ap,Bp, length=l, reverse=True)
    ax.plot(tx1,ty1)
    bisx, bisy = (tx+tx1)/2, (ty+ty1)/2
    ax.plot(bisx,bisy)


    tx, ty = tan_seg(X,Y,Cp,Dp, length=l, reverse=False)
    ax.plot(tx,ty)

    tx1, ty1 = tan_seg(x,y,Cp,Dp, length=l, reverse=True)
    ax.plot(tx1,ty1)
    bisx, bisy = (tx+tx1)/2, (ty+ty1)/2
    ax.plot(bisx,bisy)

def tan_seg(cx, cy, px, py,length=3, reverse=False):
    """return a secment tangean at circle on point"""
    
    
    ucp = np.array([px -cx, py -cy])
    norm = sqrt(sum(ucp**2))
    ucp = ucp / norm 
    sign = 1 if not reverse else -1
    
    rpx = px-sign*length*ucp[1]
    rpy = py+sign*length*ucp[0]
    return np.array((px, rpx)),np.array((py, rpy))

def drawinterface(ax, z, spheres):
    
    Cg = Sphere(  *spheres[0:4])
    Cp = Sphere(  *spheres[4:8])

    if Cg.r < Cp.r :
        Cg,Cp = Cp,Cg
    # Cg, grande sphere
    # Cp, petite sphere
    
    # D
    D = sqrt(
         (Cp.x-Cg.x)**2
        +(Cp.y-Cg.y)**2
        +(Cp.z-Cg.z)**2)

    beta = arccos(
        (D**2+ Cp.r**2-Cg.r**2)
        /(2*D*Cp.r)
        )

    alpha = arccos(
        (D**2+ Cg.r**2-Cp.r**2)
        /(2*D*Cg.r)
        )

    CpH = Cp.r*cos(beta)
    CgH = Cg.r*cos(alpha)

    Ri = (Cg.r*Cp.r)/(Cg.r-Cp.r)
    uvec = array((Cp.x-Cg.x, Cp.y-Cg.y, Cp.z-Cg.z))/D


    CiH = sqrt(Ri**2-Cp.r**2+CpH**2)

    semihaut = sqrt(Cp.r**2-CpH**2)


    H = array((Cp[0:3]-CpH*uvec))
    Hi1 =H+semihaut*array([-uvec[1],uvec[0],uvec[2]])
    Hi2 =H-semihaut*array([-uvec[1],uvec[0],uvec[2]])

    Ci = list(H+CiH*uvec)
    Ci.append(Ri)


    def pcirc(ax, ztarget,  *args):
        """
        trace sur `ax` l'intersection des spheres `*args` aver le plan `z-target`

        """

        if not ax:
            fig, ax = subplots()
            ax.set_xlim(0,60)
            ax.set_ylim(0,40)
        ax.set_aspect('equal')
        tanparam = []
        for c,param in zip(['r','g','m'],args[0:]):
            if c == 'm':
                continue
            x = param[0]
            y = param[1]
            z = param[2]
            r = sqrt(param[3]**2-(z-ztarget)**2)
            tanparam.extend([x,y,r])
            circ = plt.Circle((x, y), radius=r, color=(0,0,0,0), ec=c, lw=1)
            ax.add_patch(circ)
            ax.scatter((x),(y), c=c)

        draw_tangeants(ax, *tanparam[0:6], l=5)
        return ax
    z= z if z is not None else (Cp.z+Cg.z)/2
    ax= pcirc(ax, z ,Cp,Cg,Ci)

#    ax.scatter(*H,c='gray')
#    ax.scatter(*Hi1[0:2],s=50,c='gray')
#    ax.scatter(*Hi2[0:2],s=50,c='gray')
