#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
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
"""MeshMatch

level = 'normal'
topics = ['mesh']
techniques = ['draw','replicate','match']

"""
transparent()
smoothwire()
clear()
from mesh import Mesh

n=5
nx=4*n
ny=2*n

M = Formex('4:0123').replic2(nx,ny).cselect(arange(4*nx,int(7.5*nx))).toMesh().setProp(1)
draw(M)
drawNumbers(M.coords,color=red)

M1 = Formex('3:012').replic2(int(0.6*nx),int(0.45*ny),bias=1,taper=-2).toMesh().scale(2).trl(1,1.).setProp(2)
draw(M1)
zoomAll()
drawNumbers(M1.coords,color=yellow,trl=[0.,-0.25,0.])

match = M.matchCoords(M1)

m = match>=0
n1=arange(len(match))

print "List of the %s matching nodes" % m.sum()
print column_stack([match[m],n1[m]])

draw(M.coords[match[m]],marksize=10,bbox='last')


# End
