#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
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
g = (f1+f2).translate([0,a,1]).cospherical(scale=[360./nx,base/(ny+a),rd])
draw(e1+e2)

draw(f1+f2)

clear()
draw(g)
h = g.withProp(0)+g.withProp(3) # only horizontals and meridionals

clear()
draw(g+h.translate([2*rd,0,0]))
