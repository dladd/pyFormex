#!/usr/bin/env pyformex --gui
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
"""Lamella Dome"""
clear()
nx=12   # number of modules in circumferential direction
ny=8    # number of modules in meridional direction
rd=100  # radius of the sphere cap
t=50    # slope of the dome at its base (= half angle of the sphere cap)
a=2     # size of the top opening
rings=False # set to True to include horizontal rings
e1 = Formex([[[0,0],[1,1]]],1).rosette(4,90).translate([1,1,0]) # diagonals
e2 = Formex([[[0,0],[2,0]]],0) # border
f1 = e1.replic2(nx,ny,2,2)
if rings:
    f2 = e2.replic2(nx,ny+1,2,2)
else:
    f2 = e2.replic2(nx,2,2,2*ny)
g = (f1+f2).translate([0,a,1]).spherical(scale=[180/nx,t/(2*ny+a),rd],colat=True)
draw(e1+e2)

draw(f1+f2)

clear()
draw(g)
