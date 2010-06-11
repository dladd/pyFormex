#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""Inertia

level = 'beginner'
topics = ['geometry']
techniques = ['color','axes']

"""

from plugins import inertia

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
    C,I = inertia.inertia(F.coords)
    GD.message("Center: %s" % C)
    GD.message("Inertia tensor: %s" % I)
    Iprin,Iaxes = inertia.principal(I)
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



