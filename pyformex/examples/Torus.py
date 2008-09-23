#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Torus

level = 'beginner'
topics = ['geometry']
techniques = ['colors']

"""
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
