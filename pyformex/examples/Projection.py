#!/usr/bin/env pyformex --gui
# $Id$

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
