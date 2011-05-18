#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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

"""ConnectMesh

level = 'normal'
topics = ['geometry','surface']
techniques = ['color']

"""

import simple
from mesh import Mesh,connectMesh

clear()
smoothwire()

nx = 4
ny = 3
nz = 7

# A rectangular mesh
print '\n1 : Test connectMesh quad4 to hex8\n'
M1 = simple.rectangle(nx,ny).toMesh().setProp(1)
# Same mesh, rotated and translated
M2 = M1.rotate(45,0).translate([1.,-1.,nz]).setProp(3)
draw([M1,M2])
sleep(1)

# leave out the first and the last two elements
e1 = M1.elems[1:-2]
m1 = Mesh(M1.coords,e1)
m2 = Mesh(M2.coords,e1)
clear()
draw([m1,m2],view=None)
sleep(1)

# Connect both meshes to a hexaeder mesh
m = connectMesh(m1,m2,nz)
clear()
draw(m,view=None)

pause()
clear()

print '\n2 : Test connectMesh quad8 to hex20\n'
M1 = simple.rectangle(nx,ny).toMesh().convert('quad8').setProp(1)
# Same mesh, rotated and translated
M2 = M1.rotate(45,0).translate([1.,-1.,nz]).setProp(3)
draw([M1,M2])
drawNumbers(Formex(M1.select([0]).coords))
sleep(1)
m = connectMesh(M1,M2,nz,eltype='hex20')
clear()
draw(m,view=None)
drawNumbers(Formex(m.renumber().select([0]).coords))
pause()
clear()

print '\n3 : Test simple.circle line3, connectMesh line3 to quad8, extrude quad8 to hex20\n'
M1 = simple.circle(a3=60.,a1=10,eltype='line3').toMesh().setProp(1)
# Same mesh, rotated and translated
M2 = M1.rotate(10.,1).translate([1.,-1.,nz]).setProp(3)
draw([M1,M2])
sleep(1)
m = connectMesh(M1,M2,nz,eltype='quad8')
draw(m)
sleep(1)
m=m.extrude(5,step=1.,dir=0,eltype='hex20')
draw(m)
drawNumbers(Formex(m.renumber().select([0]).coords))
pause()
clear()

print '\n4 : Test default values for simple.circle line2, connectMesh line2 to quad8, extrude quad4 to hex8\n'
M1 = simple.circle(a3=60.,a1=10).toMesh().setProp(1)
# Same mesh, rotated and translated
M2 = M1.rotate(10.,1).translate([1.,-1.,nz]).setProp(3)
draw([M1,M2])
sleep(1)
m = connectMesh(M1,M2,nz)
draw(m)
sleep(1)
m=m.extrude(5,step=1.,dir=0)
draw(m)
drawNumbers(Formex(m.renumber().select([0]).coords)) 
pause()
clear()

print '\n5 : Test connectMesh quad8 to hex20,getBorderMesh for Quadratic elements\n'
M1 = simple.circle(a3=60.,a1=10,eltype='line3').toMesh().setProp(1)
# Same mesh, rotated and translated
M2 = M1.rotate(10.,1).translate([1.,-1.,nz]).setProp(3)
draw([M1,M2])
sleep(1)
m = connectMesh(M1,M2,nz,eltype='quad8')
draw(m)
sleep(1)
m=m.extrude(5,step=1.,dir=0,eltype='hex20')
m=m.getBorderMesh()
m=Mesh(m.coords,m.elems,eltype='quad8').setProp(5)
#~ draw(Mesh(m.coords,m.elems,eltype='quad8'))
draw(m)
drawNumbers(Formex(m.renumber().select([0]).coords))
