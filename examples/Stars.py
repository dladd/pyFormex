#!/usr/bin/env python pyformex.py
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
"""Stars"""
from numpy import random

nstars = 200 # number of stars
minpoints = 5 # minimum number of points in the stars
maxpoints = 15# maximum number of points in the stars
noise = 0.2 # relative amplitude of noise in the shape of the star
displ = sqrt(nstars)*3.0 # relative displacement of the stars
maxrot = 70. # maximum rotation angle (in degrees)

def star(n,noise=0.,prop=0):
    """Create a regular n-pointed star, possibly with noise and properties.

    n should be odd and >= 3. With 3 however, the result is a triangle,
    so at least 5 is recommended. If an even number is given, 1 is added.
    A noise parameter can be given to vary the regular shape.
    A prop can be set too.
    """
    if n < 3:
        n = 3
    if n % 2 == 0:
        n += 1
    f = Formex([[[0,1]]]).rosette(n,(n/2)*360./n).data()
    if noise != 0.:
        f = f + noise * random.random(f.shape)
    P = Formex(concatenate([f,f[:1]]))
    return connect([P,P],bias=[0,1]).setProp(prop)

# create random number of points, rotation and translation
npts = random.randint(minpoints-1,maxpoints,(nstars,))
rot = random.random((nstars,3))
ang = random.random((nstars,)) * maxrot
trl = random.random((nstars,3)) * displ
# create the stars
Stars = Formex.concatenate([ star(n,noise,i).rotate(a,r).translate(t) for i,n,a,r,t in zip(range(nstars),npts,ang,rot,trl) ])
# draw them with random colors
colors = random.random((nstars,3))
clear()
draw(Stars,colormap=colors)

# End
