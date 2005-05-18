#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.2.1 Release Fri Apr  8 23:30:39 2005
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where otherwise stated 
##
#
"""Stars"""
from numarray import random_array
nstars = 100 # number of stars
npoints = 7 # number of points in the star
noise = 0.3 # relative amplitude of noise
displ = nstars*0.6 # relative displacement
def star(n,noise=0.,prop=0):
    """Create a regular n-pointed star, possibly with noise and properties.

    n should be odd and > 3.
    A noise parameter can be given to vary the regular shape.
    A prop can be set too.
    """
    m = n/2
    f = Formex([[[0,1]]]).rosette(n,m*360./n).data()
    if noise != 0.:
        f = f + noise * random_array.random(f.shape)
    P = Formex(concatenate([f,f[:1]]))
    return Formex.connect([P,P],bias=[0,1]).setProp(prop)
Stars = Formex.concatenate([ star(npoints,noise,i).translate(displ*random_array.random((3,))) for i in range(nstars) ])
clear()
draw(Stars)
