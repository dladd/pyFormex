#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.2 Release Mon Jan  3 14:54:38 2005
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Copyright (C) 2004 Benedict Verhegghe (benedict.verhegghe@ugent.be)
## Copyright (C) 2004 Bart Desloovere (bart.desloovere@telenet.be)
## Distributed under the General Public License, see file COPYING for details
##
#
"""Torus"""
m = 36 # number of cells along torus big circle
n = 10 # number of cells along torus small circle
message("Create a triangle with three colored members")
F = Formex(pattern("164"),[1,2,3])
clear();drawProp(F);sleep()
message("Replicate it into a rectangular pattern")
F = F.generate2(m,n,0,1,1,1)
clear();drawProp(F);sleep()
message("Fold the rectangle into a tube")
G = F.translate1(2,1).cylindrical([2,1,0],[1.,360./n,1.])
clear();drawProp(G,side='right');sleep()
message("Bend the tube into a torus with mean radius 5")
H = G.translate1(0,5).cylindrical([0,2,1],[1.,360./m,1.])
clear();drawProp(H,side='iso');sleep()
