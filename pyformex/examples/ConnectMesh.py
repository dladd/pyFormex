#!/usr/bin/env pyformex --gui
# $Id$

"""ConnectMesh

level = 'normal'
topics = ['geometry','surface']
techniques = ['colors']

"""

import simple
from plugins.connectivity import reverseUniqueIndex
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
