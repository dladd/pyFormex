#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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

"""ConnectMesh

level = 'normal'
topics = ['mesh']
techniques = ['connect','color']

"""

import simple
from mesh import Mesh

clear()
smoothwire()

nx = 4
ny = 3
nz = 7

delay(2)

# A rectangular mesh
M1 = simple.rectangle(nx,ny).toMesh().setProp(1)
# Same mesh, rotated and translated
M2 = M1.rotate(45,0).translate([1.,-1.,nz]).setProp(3)
draw([M1,M2])

# Leave out the first and the last two elements
sel = arange(M1.nelems())[1:-2]
m1 = M1.select(sel)
m2 = M2.select(sel)
clear()
draw([m1,m2],view=None)

# Connect both meshes to a hexaeder mesh
m = m1.connect(m2,nz)
clear()
draw(m,color=red,view=None)

# End
