#!/usr/bin/env pyformex
##
## This file is part of pyFormex 0.7.1 Release Fri May  9 08:39:30 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Viewports.py

Demonstrate multiple viewports.
"""

nsl = 0
F = Formex.read(GD.cfg['pyformexdir']+'/examples/horse.formex')

layout(1)
FA = draw(F,view='front')
sleep(nsl)

layout(3)
draw(F,color='green')
sleep(nsl)


viewport(1)
linkViewport(1,0)
sleep(nsl)

layout(4,2)
viewport(0)
sleep(nsl)


for i in range(1,4):
    linkViewport(i,0)

colors=['indianred','olive','coral','yellow']

for i,v in enumerate(['front','right','top','iso']):
    viewport(i)
    view(v)
    bgcolor(colors[i])
    GD.canvas.setBgColor(GD.canvas.settings.bgcolor)
    GD.canvas.display()
    GD.canvas.update()

sleep(nsl)
viewport(3)
G = F.cutAtPlane([0.,0.,0.],[-1.,0.,0.],side='+')
clear()
draw(G) # this draws in the 4 viewports !
GD.gui.viewports.updateAll()


sleep(nsl)
smooth()
GD.gui.viewports.updateAll()

exit()
from gui import canvas
sleep(nsl)
canvas.glLine()
canvas.glFlat()
GD.gui.viewports.updateAll()




#End
