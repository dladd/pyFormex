#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Mon Dec 29 15:32:01 2008
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
from plugins.mesh import connectMesh

class Mesh(object):
    """A general FE style geometry data class."""

    def __init__(self,coords,elems):
        """Create a new Mesh from given coordinates and connectivity arrays.

        """
        if coords.ndim != 2 or coords.shape[-1] != 3 or elems.ndim != 2 or \
               elems.max() >= coords.shape[0] or elems.min() < 0:
            raise ValueError,"Invalid mesh data"
        self.coords = asarray(coords,dtype=Float)
        self.elems = asarray(elems,dtype=Int)

    def data(self):
        """Return the mesh data as a tuple (coords,elems)"""
        return self.coords,self.elems

    def compact(self):
        """Renumber the mesh and remove unconnected nodes."""
        nodes = unique1d(self.elems)
        if nodes[-1] >= nodes.size:
            self.coords = self.coords[nodes]
            self.elems = reverseUniqueIndex(nodes)[self.elems]
        
    def draw(self):
        draw(Formex(self.coords))
        draw(Formex(self.coords[self.elems]))
        


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
m1.draw()
m2.draw()

x,e = connectMesh(c1,c2,e1,nz)

F = Formex(x[e])
F.setProp(1)
F.eltype = 'hex8'
clear()
draw(F)

# End
