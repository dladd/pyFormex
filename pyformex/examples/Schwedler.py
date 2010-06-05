#!/usr/bin/env pyformex --gui
# $Id: Schwedler.py 154 2006-11-03 19:08:25Z bverheg $
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""Schwedler Dome

level = 'normal'
topics = ['geometry','domes']
techniques = ['colors']

"""

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
