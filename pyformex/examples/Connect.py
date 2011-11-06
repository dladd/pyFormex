#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
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

"""Connect

level = 'normal'
topics = ['formex','surface']
techniques = ['connect','color']

"""

import simple
clear()
linewidth(2)
flatwire()
setDrawOptions({'bbox':'auto'})

# A tapered grid of points
F = Formex([0.]).replic2(10,5,taper=-1)
draw(F)

# Split in parts by testing y-position; note use of implicit loop!
G = [ F.clip(F.test(dir=1,min=i-0.5,max=i+0.5)) for i in range(5) ]
print [ Gi.nelems() for Gi in G ]

def annot(char):
    [ drawText3D(G[i][0,0]+[-0.5,0.,0.],"%s%s"%(char,i)) for i,Gi in enumerate(G) ]
   

# Apply a general mapping function : x,y,x -> [ newx, newy, newz ]
G = [ Gi.map(lambda x,y,z:[x,y+0.01*float(i+1)**1.5*x**2,z]) for i,Gi in enumerate(G) ]
clear()
annot('G')
draw(G)

setDrawOptions({'bbox':'last'})

# Connect G0 with G1
H1 = connect([G[0],G[1]])
draw(H1,color=blue)

# Connect G1 with G2 with a 2-element bias 
H2 = connect([G[1],G[2]],bias=[0,2])
draw(H2,color=green)

# Connect G3 with G4 with a 1-element bias plus loop
H2 = connect([G[3],G[4]],bias=[1,0],loop=True)
draw(H2,color=red)

# Create a triangular grid of bars
clear()
annot('G')
draw(G)
# Connect Gi[j] with Gi[j+1] to create horizontals   
K1 = [ connect([i,i],bias=[0,1]) for i in G ]
draw(K1,color=blue)

# Connect Gi[j] with Gi+1[j] to create verticals 
K2 = [ connect([i,j]) for i,j in zip(G[:-1],G[1:]) ]
draw(K2,color=red)

# Connect Gi[j+1] with Gi+1[j] to create diagonals 
K3 = [ connect([i,j],bias=[1,0]) for i,j in zip(G[:-1],G[1:]) ]
draw(K3,color=green)

# Create triangles
clear()
annot('G')
draw(G)

L1 = [ connect([i,i,j],bias=[0,1,0]) for i,j in zip(G[:-1],G[1:]) ] 
draw(L1,color=red)
L2 = [ connect([i,j,j],bias=[1,0,1]) for i,j in zip(G[:-1],G[1:]) ] 
draw(L2,color=green)

# Connecting multiplex Formices using bias
clear()
annot('K')
draw(K1)
L1 = [ connect([i,i,j],bias=[0,1,0]) for i,j in zip(K1[:-1],K1[1:]) ] 
draw(L1,color=red)
L2 = [ connect([i,j,j],bias=[1,0,1]) for i,j in zip(K1[:-1],K1[1:]) ] 
draw(L2,color=green)

# Connecting multiplex Formices using nodid
clear()
annot('K')
draw(K1)
L1 = [ connect([i,i,j],nodid=[0,1,0]) for i,j in zip(K1[:-1],K1[1:]) ]
draw(L1,color=red)
L2 = [ connect([i,j,j],nodid=[1,0,1]) for i,j in zip(K1[:-1],K1[1:]) ] 
draw(L2,color=green)

# Add the missing end triangles
L3 = [ connect([i,i,j],nodid=[0,1,1],bias=[i.nelems()-1,i.nelems()-1,j.nelems()-1])  for i,j in zip(K1[:-1],K1[1:]) ]
draw(L3,color=magenta)

# Collect all triangles in a single Formex
L = (Formex.concatenate(L1)+Formex.concatenate(L3)).setProp(1) + Formex.concatenate(L2).setProp(2)
clear()
draw(L)

# Convert to a Mesh
print "nelems = %s, nplex = %s, coords = %s" % (L.nelems(),L.nplex(),L.coords.shape)
M = L.toMesh()
print "nelems = %s, nplex = %s, coords = %s" % (M.nelems(),M.nplex(),M.coords.shape)
clear()
draw(M,color=yellow,mode=flatwire)
drawNumbers(M)
draw(M.getBorderMesh(),color=black,linewidth=6)

# Convert to a surface
from plugins.trisurface import TriSurface
S = TriSurface(M)
print "nelems = %s, nplex = %s, coords = %s" % (S.nelems(),S.nplex(),S.coords.shape)
clear()
draw(S)
print "Total surface area: %s" % S.area()
export({'surface-1':S})
setDrawOptions({'bbox':'auto'})

# End
