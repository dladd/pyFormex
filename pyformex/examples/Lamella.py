# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Lamella Dome

level = 'beginner'
topics = ['geometry','domes']
techniques = ['color']

"""

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
