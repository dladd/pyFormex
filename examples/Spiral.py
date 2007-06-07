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
"""Spiral"""
# This constructs the same example as torus.py, but shows all steps
m = 36 # number of cells along torus big circle
n = 10 # number of cells along torus small circle
# First create a long rectangle

setDrawingOptions({'clear':True})
F = Formex(pattern("164"),[1,2,3]); draw(F)
F = F.replic(m,1,0); draw(F)
F = F.replic(n,1,1); draw(F)
F = F.translate(2,1); draw(F,view='iso')
F = F.cylindrical([2,1,0],[1.,360./n,1.]); draw(F)
F = F.replic(5,m,2); draw(F)
F = F.rotate(-10,0); draw(F)
F = F.translate(0,5); draw(F)
F = F.cylindrical([0,2,1],[1.,360./m,1.]); draw(F)
draw(F,view='right')
