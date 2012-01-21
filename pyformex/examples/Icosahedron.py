# $Id$  *** pyformex ***
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

"""Icosahedron

Draws an icosahedron and its projection on a sphere.

level = 'normal'
topics = ['Mesh','Geometry',Sphere']
techniques = ['subdivide','project','animation']
"""

clear()

from arraytools import golden_ratio as phi
s = sqrt(1.+phi*phi)
a = sqrt(3.)/6.*(3.+sqrt(5.))

I = Mesh(eltype='icosa').getBorderMesh()
M = I.subdivide(10)
S = M.projectOnSphere()

delay(0)
draw(S,color='red')

a0 = 1./a
a1 = 1./s - a0
A = draw(I.scale(a0),color='yellow')
zoomAll()
n = 100
delay(0.05)
for i in arange(n+1)/float(n):
    B = draw(I.scale(a0+i*a1),color='yellow',bbox='last')
    undraw(A)
    A = B

delay(2)
wait()
draw(I.scale(1./phi),color='yellow')
undraw(A)
delay(0)
# End
