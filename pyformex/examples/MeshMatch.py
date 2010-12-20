#!/usr/bin/env pyformex --gui
# $Id$
"""MeshMatch

level = 'normal'
topics = ['mesh']
techniques = ['draw','replicate','match']

"""
transparent()
smoothwire()
clear()
from plugins.mesh import Mesh

n=5
nx=4*n
ny=2*n

M = Formex(mpattern('123')).replic2(nx,ny).cselect(arange(4*nx,int(7.5*nx))).toMesh().setProp(1)
draw(M)
drawNumbers(M.coords,color=red)

M1 = Formex(mpattern('12')).replic2(int(1.2*nx),int(0.9*ny),bias=1,taper=-2).toMesh().setProp(2)
draw(M1)
drawNumbers(M1.coords,color=yellow,trl=[0.,-0.25,0.])

match = M.matchCoords(M1)

m = match>=0
n1=arange(len(match))

print "List of the %s matching nodes" % m.sum()
print column_stack([match[m],n1[m]])

draw(M.coords[match[m]],marksize=10)

# End
