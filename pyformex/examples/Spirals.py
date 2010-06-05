#!/usr/bin/env pyformex --gui
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
"""Spirals

level = 'normal'
topics = ['geometry']
techniques = ['curve','transform']

See also the Sweep example for a more sophisticated Spirals application
"""

from plugins import curve

m = 100 # number of cells along spiral
a = 1. # number of 360 degree turns

linewidth(2)
clear()

F = Formex(origin()) # base pattern, here a point
F = F.replic(m,1.,0)
s = a*2*pi/m
F = F.scale(s)
draw(F)

def spiral(X,dir=[0,1,2],rfunc=lambda x:1,zfunc=lambda x:1):
    """Perform a spiral transformation on a coordinate array"""
    theta = X[...,dir[0]]
    r = rfunc(theta) + X[...,dir[1]]
    x = r * cos(theta)
    y = r * sin(theta)
    z = zfunc(theta) + X[...,dir[2]]
    X = hstack([x,y,z]).reshape(X.shape)
    return Coords(X)


nwires=6

phi = 30.
alpha2 = 70.
c = 1.
a = c*tand(phi)
b = tand(phi) / tand(alpha2)
 

zf = lambda x : c * exp(b*x)
rf = lambda x : a * exp(b*x)


S = spiral(F.coords,[0,1,2],rf)#.rosette(nwires,360./nwires)

PL = curve.PolyLine(S[:,0,:])

clear()
draw(PL,color=red)
draw(PL.coords,color=red)


if ack("Spread point evenly?"):
    at = PL.atLength(PL.nparts)
    X = PL.pointsAt(at)
    PL = curve.PolyLine(X)
    clear()
    draw(PL,color=blue)
    draw(PL.coords,color=blue)

# End

