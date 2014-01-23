
from collections import namedtuple
from numpy import arccos, sqrt, cos, array
from matplotlib.pyplot import subplots
import  matplotlib.pyplot as plt

Point = namedtuple('Point', 'x y z'.split(' '))
Sphere = namedtuple('Sphere', 'x y z r'.split(' '))

def drawinterface(ax, z, spheres):
    
    Cg = Sphere(  *spheres[0:4])
    Cp = Sphere(  *spheres[4:8])
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


    def pcirc(ax, *args):
        if not ax:
            fig, ax = subplots()
            ax.set_xlim(0,60)
            ax.set_ylim(0,40)
        ztarget = args[0]
        ax.set_aspect('equal')
        for c,param in zip(['r','g','b'],args[1:]):
            x = param[0]
            y = param[1]
            z = param[2]
            r = sqrt(param[3]**2-(z-ztarget)**2)
            circ = plt.Circle((x, y), radius=r, color=(0,0,0,0), ec=c, lw=1)
            ax.add_patch(circ)
            ax.scatter((x),(y), c=c)
        return ax
    z= z if z else (Cp.z+Cg.r)/2
    ax= pcirc(ax, z ,Cp,Cg,Ci)

    ax.scatter(*H,c='gray')
    ax.scatter(*Hi1[0:2],s=50,c='gray')
    ax.scatter(*Hi2[0:2],s=50,c='gray')

