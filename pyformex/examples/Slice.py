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

"""Slice

level = 'advanced'
topics = ['surface']
techniques = ['colors','widgets']

"""

from plugins.surface import TriSurface


def askSlices(bb):
    res = askItems([['Direction',0],
                    ['# slices',25],
                    ['total rot',70.],
                   ],caption = 'Define the slicing planes')
    if res:
        axis = res['Direction']
        nslices = res['# slices']
        totalrot = res['total rot']
        xmin,xmax = bb[:,axis]
        dx =  (xmax-xmin) / nslices
        x = arange(nslices+1) * dx
        N = unitVector(axis)
        P = [ bb[0]+N*s for s in x ]
        return P,N,totalrot
    else:
        return [],[]

reset()
smooth()
lights(True)
transparent(False)
setView('horse',[20,20,0])
chdir('/home/bene/prj/pyformex/stl')
S = TriSurface.read('horse-upright.gts')
bb = S.bbox()

t = -0.3
bb[0] = (1.0-t)*bb[0] + t*bb[1]
draw(S,bbox=bb,view='front')

P,n,t = askSlices(S.bbox())
a = t/len(P)

F = S.toFormex()
G = []
old = seterr(all='ignore')
setDrawOptions({'bbox':None})
for i,p in enumerate(P):
    F1 = F.cutAtPlane(p,-n)
    F1.setProp(i)
    G = [ g.rot(a,around=p) for g in G ] 
    G.append(F1)
    F = F.cutAtPlane(p,n)
    clear()
    draw(F)
    draw(G)

seterr(**old)
    
x = GD.canvas.width()/2
y = GD.canvas.height() - 30
drawtext("No animals got hurt during the making of this movie!",x,y,font='tr24',adjust='center')

# End
