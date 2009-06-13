#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

"""ConnectMesh

level = 'normal'
topics = ['geometry','surface']
techniques = ['colors']

"""

import simple
from connectivity import reverseUniqueIndex
from plugins.mesh import *
        
def drawMesh(mesh,ncolor='blue',ecolor='red'):
    if ncolor:
        draw(mesh.coords,color=ncolor)
    if ecolor:
        draw(mesh,color=ecolor,bbox='last')

nx = 4
ny = 3
nz = 7
F = simple.rectangle(nx,ny).setProp(1)

c1,e1 = F.feModel()
c2 = c1.rotate(45,0).translate([1.,-1.,nz])

G = Formex(c2[e1]).setProp(3)
draw([F,G])


e1 = e1[1:-2]

m1 = Mesh(c1,e1)
m2 = Mesh(c2,e1)

clear()
drawMesh(m1)
drawMesh(m2)
sleep(1)

m = connectMesh(m1,m2,nz)

m.eltype = 'hex8'

clear()
drawMesh(m)

# End
