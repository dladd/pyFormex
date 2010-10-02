#!/usr/bin/env pyformex --gui
# $Id$
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
"""Novation

level = 'normal'
topics = ['geometry','surface']
techniques = ['dialog', 'persistence', 'color']

"""

reset()


basechoices = ['Triangles','Quadrilaterals']
renderchoices = ['wireframe','flat','flatwire','smooth','smoothwire']
res = askItems([
    ('baseGeom',None,'radio',{'choices':basechoices,'text':'Type of surface element'}),
    ('nbumps',3,{'text':'Number of bumps'}),
    ('rendermode',None,'select',{'choices':renderchoices,'text':'Render mode'}),
    ('transparent',False,{'text':'Transparent'}),
    ('bottom',False,{'text':'Add a bottom plate'}),
    ('shrink',False,{'text':'Shrink elements'}),
    ('export',False,{'text':'Export to .stl'}),
    ])
if not res:
    exit()

globals().update(res)

n = 10*nbumps

if baseGeom == 'Triangles':
    # The base consists of two triangles
    e = Formex([[[0,0,0],[1,0,0],[0,1,0]],[[1,0,0],[1,1,0],[0,1,0]]],1).replic2(n,n,1,1)
else:
    # The base is one quadrilateral
    e = Formex([[[0,0,0],[1,0,0],[1,1,0],[0,1,0]]],1).replic2(n,n,1,1)

# These are lines forming quadrilaterals
#e = Formex([[[0,0,0],[1,0,0]]]).rosad(.5,.5).rinid(n,n,1,1)

# Novation (Spots)
s = nbumps+1
r = n/s
h = 12
a = [ [r*i,r*j,h]  for j in range(1,s) for i in range(1,s) ]

if bottom:
    # create a bottom
    b = e.reverse()
    #b.setProp(2)
    
# create the bumps
for p in a:
    e = e.bump(2,p, lambda x:exp(-0.5*x),[0,1])
if shrink:
    e = e.shrink(0.8)

renderMode(rendermode)
if transparent:
    GD.canvas.alphablend = True
if bottom:
    draw(b,color=yellow,alpha=1.0)

draw(e,alpha=0.8)

if export and checkWorkdir():
    from plugins import trisurface
    f = file('novation.stl','w')
    F = e # + b
    # Create triangles
    G = F.selectNodes([0,1,2])
    # If polygones, add more triangles
    for i in range(3,F.nplex()):
        G += F.selectNodes([0,i-1,i])
    clear()
    draw(G)
    surface.write_stla(f,G.coords)
    f.close()

# End
