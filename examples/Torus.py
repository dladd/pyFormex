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
"""Torus"""
m = 36 # number of cells along torus big circle
n = 10 # number of cells along torus small circle
message("Create a triangle with three colored members")
F = Formex(pattern("164"),[1,2,3])
clear();draw(F);
message("Replicate it into a rectangular pattern")
F = F.replic2(m,n,1,1)
clear();draw(F);
message("Fold the rectangle into a tube")
G = F.translate(2,1).cylindrical([2,1,0],[1.,360./n,1.])
clear();draw(G,view='right');
message("Bend the tube into a torus with mean radius 5")
H = G.translate(0,5).cylindrical([0,2,1],[1.,360./m,1.])
clear();draw(H,view='iso');
