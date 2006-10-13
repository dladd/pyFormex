#!/usr/bin/env pyformex
# $Id$
#
# Show a tetraeder model from tetgen output
from plugins import tetgen

base = 'klauw-voor.1'
nodes = tetgen.readNodes(base+'.node')
elems = tetgen.readElems(base+'.ele')

print nodes.shape
print elems.shape
print elems
F = Formex(nodes[elems-1])

draw(F,eltype='tet',color='random')

