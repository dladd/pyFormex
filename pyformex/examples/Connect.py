#!/usr/bin/env pyformex
# $Id$

"""Connect

level = 'normal'
topics = ['formex','surface']
techniques = ['connect','color']

"""

import simple
#bgcolor(white)
clear()
flat()
linewidth(2)

# A tapered grid of points
F = Formex([0.]).replic2(10,5,taper=-1)
draw(F)

# Split in parts by testing y-position; note use of implicit loop!
G = [ F.clip(F.test(dir=1,min=i-0.5,max=i+0.5)) for i in range(5) ]
print [ Gi.nelems() for Gi in G ]

# Apply a general mapping function : x,y,x -> [ newx, newy, newz ]
G = [ Gi.map(lambda x,y,z:[x,y+0.05*x**2,z]) for Gi in G ]
clear()
draw(G)

# Connect G0 with G1
H1 = connect([G[0],G[1]])
draw(H1,color=blue)

# Connect G1 with G2 with a 2-element bias 
H2 = connect([G[1],G[2]],bias=[0,2])
draw(H2,color=green)

# Create a triangular grid of bars
clear()
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
draw(G)

L1 = [ connect([i,i,j],bias=[0,1,0]) for i,j in zip(G[:-1],G[1:]) ] 
draw(L1,color=red)
L2 = [ connect([i,j,j],bias=[1,0,1]) for i,j in zip(G[:-1],G[1:]) ] 
draw(L2,color=green)

# Connecting multiplex Formices using bias
clear()
draw(K1)
L1 = [ connect([i,i,j],bias=[0,1,0]) for i,j in zip(K1[:-1],K1[1:]) ] 
draw(L1,color=red)
L2 = [ connect([i,j,j],bias=[1,0,1]) for i,j in zip(K1[:-1],K1[1:]) ] 
draw(L2,color=green)

# Connecting multiplex Formices using nodid
clear()
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
draw(M)
draw(M.getBorderMesh(),color=black,linewidth=6)

# Convert to a surface
from plugins.trisurface import TriSurface
S = TriSurface(L)
print "nelems = %s, nplex = %s, coords = %s" % (S.nelems(),S.nplex(),S.coords.shape)
clear()
draw(S)
print "Surface area: %s" % S.area()

export({'surface-1':S})

# End
