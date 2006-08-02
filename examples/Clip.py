#!/usr/bin/env pyformex
# $Id$

from simple import unitcircle


clear()
n = 16

# These are triangles
F = Formex([[[0,0,0],[1,0,0],[0,1,0]],[[1,0,0],[1,1,0],[0,1,0]]],0).replic2(n,n,1,1)

# Novation (Spots)
m = 4
h = 0.15*n
r = n/m
s = n/r
a = [ [r*i,r*j,h]  for j in range(1,s) for i in range(1,s) ]

for p in a:
    F = F.bump(2,p, lambda x:exp(-0.75*x),[0,1])

draw(F)


# Define a plane
p = [3.2,3.0,0.0]
n = [2.0,1.0,0.0]
#number of nodes above/below the plane
dist = distanceFromPlane(F.f,p,n)
above = sum(dist>0.0,-1)
below = sum(dist<0.0,-1) 

sel = [ F.where(nodes=0,dir=0,xmin=1.5,xmax=3.5),
        F.where(nodes=[0,1],dir=0,xmin=1.5,xmax=3.5),
        F.where(nodes=[0,1,2],dir=0,xmin=1.5,xmax=3.5),
        F.where(nodes='all',dir=1,xmin=1.5,xmax=3.5),
        F.where(nodes='any',dir=1,xmin=1.5,xmax=3.5),
        F.where(nodes='none',dir=1,xmin=1.5),
        (above > 0) * (below > 0 )
        ]

txt = [ 'First node has x between 1.5 and 3.5',
        'First two nodes have x between 1.5 and 3.5',
        'First 3 nodes have x between 1.5 and 3.5',
        'All nodes have y between 1.5 and 3.5',
        'Any node has y between 1.5 and 3.5',
        'No node has y larger than 1.5',
        'Touching the plane P = [3.0,3.0,0.0], n = [2.0,1.0,0.0]'
        ]
col = GD.cfg.propcolors

p = zeros(F.f.shape[:1])
for i,s in enumerate(sel):
    p[s] = i + 1 # omit the black
    F.setProp(p)
    clear()
    message('%s (%s): %s' % (col[i+1],sum(s),txt[i]))
    draw(F)

clear()
message('Clip Formex to last selection')
draw(F.clip(s),view=None)

clear()
message('Clip complement')
draw(F.cclip(s))
