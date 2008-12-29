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

"""Projection

level = 'normal'
topics = ['geometry','surface']
techniques = ['colors']

"""

import simple
from gui.canvas import *

nx,ny = 20,10

F = simple.rectangle(nx,ny)
F = F.trl(-F.center()+[0.,0.,nx/2])
draw(F)

x = F.f.projectOnSphere(ny)
G = Formex(x)
draw(G,color=red)

x = F.f.rotate(30).projectOnCylinder(ny)
H = Formex(x)
draw(H,color=blue)

smooth()
n=200
for i in range (n):
    v = float(i)/(2*n)
    #print "\n\nNEW %s" % v
    #GD.canvas.ambient = v
    GD.canvas.specular = v
    #GD.canvas.emission = v
    #GD.canvas.shininess = v
    GD.canvas.update()
    GD.app.processEvents()
    #sleep(1)

GD.canvas.resetLighting()

#End
