#!/usr/bin/env pyformex --gui
# $Id: Sphere.py 154 2006-11-03 19:08:25Z bverheg $
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
"""Sphere

level = 'normal'
topics = ['geometry','surface']
techniques = ['dialog', 'colors']

"""

clear()
wireframe()
nx=32   # number of modules in circumferential direction
ny=32   # number of modules in meridional direction
rd=100  # radius of the sphere cap
top=70 # latitude angle at the top (90 = closed)
bot=-70 # latitude angle of the bottom (-90 = closed)

a = ny*float(top)/(bot-top)

# First, a line based model

base = Formex(pattern("543"),[1,2,3]) # single cell
draw(base)

d = base.select([0]).replic2(nx,ny,1,1)   # all diagonals
m = base.select([1]).replic2(nx,ny,1,1)   # all meridionals
h = base.select([2]).replic2(nx,ny+1,1,1) # all horizontals
f = m+d+h
draw(f)

g = f.translate([0,a,1]).spherical(scale=[360./nx,bot/(ny+a),rd])
clear()
draw(g)

# Second, a surface model

clear()
base = Formex( [[[0,0,0],[1,0,0],[1,1,0]],
                [[1,1,0],[0,1,0],[0,0,0]]],
               [1,3] )
draw(base)

f = base.replic2(nx,ny,1,1)
draw(f)

h = f.translate([0,a,1]).spherical(scale=[360./nx,bot/(ny+a),rd])
clear()
draw(h)

# Both

g = g.translate([-rd,0,0])
h = h.translate([rd,0,0])
clear()
bb = bbox([g,h])
draw(g,bbox=bb)
draw(h,bbox=bb)


##if ack('Do you want to see the spheres with smooth rendering?'):
##    smooth()
##    GD.canvas.update()
