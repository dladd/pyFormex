#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

import elements
from plugins import formex_menu
from examples import WireStent

from plugins.inertia import *

reset()
wireframe()
view('front')



def unitAxes():
    """Create a set of three axes."""
    Hx = Formex(pattern('1'),5).translate([-0.5,0.0,0.0])
    Hy = Hx.rotate(90)
    Hz = Hx.rotate(-90,1)
    Hx.setProp(4)
    Hy.setProp(5)
    Hz.setProp(6)
    return Formex.concatenate([Hx,Hy,Hz])    


Axes = unitAxes()

def showPrincipal1(F):
    """Show the principal axes."""
    clear()
    C,I = inertia(F.f)
    GD.message("Center: %s" % C)
    GD.message("Inertia tensor: %s" % I)
    Iprin,Iaxes = principal(I)
    GD.debug("Principal Values: %s" % Iprin)
    GD.debug("Principal Directions:\n%s" % Iaxes)

    siz = F.dsize()
    H = Axes.scale(siz).affine(Iaxes.transpose(),C)
    Ax,Ay,Az = Iaxes[:,0],Iaxes[:,1],Iaxes[:,2]
    G = Formex([[C,C+Ax],[C,C+Ay],[C,C+Az]],3)
    draw([F,G,H])
    sleep(2)
    return C,I,Iprin,Iaxes


#F = Formex(pattern('1')).replic(2,2,1).replic(2,2,2).scale(2)
nx,ny,nz = 2,3,4
dx,dy,dz = 4,3,2
F = Formex([[[0,0,0]]]).replic(nx,dx,0).replic(ny,dy,1).replic(nz,dz,2)

Fr = F
C,I,Ip,Ia = showPrincipal1(Fr)

Fr = F.rotate(30,0).rotate(45,1).rotate(60,2)
C,I,Ip,Ia = showPrincipal1(Fr)


sleep(2)
Fo = Formex([[C]])
Fc = connect([Fo,Fr],loop=True)
draw(Fc)



