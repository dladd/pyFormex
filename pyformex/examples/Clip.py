#!/usr/bin/env pyformex --gui
# $Id: Clip.py 154 2006-11-03 19:08:25Z bverheg $
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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

"""Clip

level = 'beginner'
topics = ['geometry']
techniques = ['colors']

"""

resetAll()
setDrawOptions({'clear':True})

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
plane_p = [3.2,3.0,0.0]
plane_n = [2.0,1.0,0.0]
#number of nodes above/below the plane
dist = F.distanceFromPlane(plane_p,plane_n)
above = sum(dist>0.0,-1)
below = sum(dist<0.0,-1) 

# Define a line by a point and direction
line_p = [0.0,0.0,0.0]
line_n = [1.,1.,1./3]
d = F.distanceFromLine(line_p,line_n)
#number of nodes close to line 
close = sum(d < 2.2,-1)



sel = [ F.test(nodes=0,dir=0,min=1.5,max=3.5),
        F.test(nodes=[0,1],dir=0,min=1.5,max=3.5),
        F.test(nodes=[0,1,2],dir=0,min=1.5,max=3.5),
        F.test(nodes='all',dir=1,min=1.5,max=3.5),
        F.test(nodes='any',dir=1,min=1.5,max=3.5),
        F.test(nodes='none',dir=1,min=1.5),
        (above > 0) * (below > 0 ),
        close == 3,
        ]

txt = [ 'First node has x between 1.5 and 3.5',
        'First two nodes have x between 1.5 and 3.5',
        'First 3 nodes have x between 1.5 and 3.5',
        'All nodes have y between 1.5 and 3.5',
        'Any node has y between 1.5 and 3.5',
        'No node has y larger than 1.5',
        'Touching the plane P = [3.2,3.0,0.0], n = [2.0,1.0,0.0]',
        '3 nodes close to line through [0.0,0.0,0.0] and [1.0,1.0,1.0]',
        ]

color = getcfg('canvas/colormap')[1:] # omit the black
while len(color) < len(sel):
    color.extend(color)
color[0:0] = ['black'] # restore the black
prop = zeros(F.nelems())
i = 1
for s,t in zip(sel,txt):
    prop[s] = i
    F.setProp(prop)
    message('%s (%s): %s' % (color[i],sum(s),t))
    draw(F)
    i += 1

message('Clip Formex to last selection')
draw(F.clip(s),view=None)

message('Clip complement')
draw(F.cclip(s))

# End
