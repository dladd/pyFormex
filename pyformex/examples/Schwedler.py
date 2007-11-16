#!/usr/bin/env pyformex --gui
# $Id: Schwedler.py 154 2006-11-03 19:08:25Z bverheg $
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Schwedler Dome"""
clear()
nx=16   # number of modules in circumferential direction
ny=8    # number of modules in meridional direction
rd=100  # radius of the sphere cap
base=50 # slope of the dome at its base (= half angle of the sphere cap)
top=5   # slope of the dome at its top opening (0 = no opening) 
a=ny*float(top)/(base-top)
e1 = Formex(pattern("54"),[1,3]) # diagonals and meridionals
e2 = Formex(pattern("1"),0)      # horizontals
f1 = e1.replic2(nx,ny,1,1)
f2 = e2.replic2(nx,ny+1,1,1)
g = (f1+f2).translate([0,a,1]).spherical(scale=[360./nx,base/(ny+a),rd],colat=True)
draw(e1+e2)

draw(f1+f2)

clear()
draw(g)
h = g.withProp([0,3]) # only horizontals and meridionals
clear()
draw(g+h.translate([2*rd,0,0]))
