#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8 Release Mon Jun  8 11:56:55 2009
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
"""HorseTorse

level = 'advanced'
topics = ['geometry','surface']
techniques = ['animation', 'colors']

Torsing a horse is like horsing a torse.
"""
from plugins.surface import TriSurface

def drawSurf(F,surface=False,**kargs):
    """Draw a Formex as surface or not."""
    if surface:
        F = TriSurface(F)
    return draw(F,**kargs)

reset()
smooth()
lights(True)
chdir(GD.cfg['curfile'])

surf=True
F = Formex.read(GD.cfg['pyformexdir']+'/examples/horse.formex')
F = F.translate(-F.center())
xmin,xmax = F.bbox()

F = F.scale(1./(xmax[0]-xmin[0]))
FA = drawSurf(F,surf)

angle = 360.
n = 120
da = rad*angle/n

F.setProp(1)
for i in range(n+1):
    a = i*da
    torse = lambda x,y,z: [x,cos(a*x)*y-sin(a*x)*z,cos(a*x)*z+sin(a*x)*y]
    G = F.map(torse)
    GA = drawSurf(G,surf)
    undraw(FA)
    FA = GA

elong = 2.0
nx = 50
dx = elong/nx

for i in range(nx+1):
    s = 1.0+i*dx
    H = G.scale([s,1.,1.])
    GA = drawSurf(H,surf,bbox=None)
    undraw(FA)
    FA = GA
