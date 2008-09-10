#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
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
