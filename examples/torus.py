#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
## Distributed under the General Public License, see file COPYING for details
##
#
"""Torus"""
clear()
m = 36 # number of cells along torus big circle
n = 10 # number of cells along torus small circle
# First create a long rectangle
F = Formex(pattern("164"),[1,2,3]).replicate(m,0,1).replicate(n,1,1)
drawProp(F);sleep()
# Fold it into a tube
G = F.translate1(2,1).cylindrical([2,1,0],[1.,360./n,1.])
clear();drawProp(G,side='right');sleep()
# Bend the tube into a torus with (mean) radius 5
H = G.translate1(0,5).cylindrical([0,2,1],[1.,360./m,1.])
clear();drawProp(H,side='iso');sleep()
# Something extra: extend the tube, tilt it a bit and then bend it
K = G.replicate(5,2,m).rotate(-10,0).translate1(0,5).cylindrical([0,2,1],[1.,360./m,1.])
clear();drawProp(K,side='iso')# Amazed?
