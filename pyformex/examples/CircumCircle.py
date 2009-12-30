#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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
"""CircumCircle

level = 'beginner'
topics = ['geometry']
techniques = ['function','import','mpattern','dialog','viewports']
"""
import simple
from examples.Cube import cube_tri
from plugins.geomtools import triangleCircumCircle

#
def drawCircles(F):
    for r,C,n in zip(*triangleCircumCircle(F.f)):
        c = simple.circle().swapAxes(0,2).scale(r).rotate(rotMatrix(n)).trl(C)
        draw(c)
        zoomAll()   

# create two viewports
layout(2)

# draw in viewport 0
viewport(0)
clear()
F = Formex(mpattern('16-32'),[0,1]).scale([2,1,0])
draw(F)
drawCircles(F)

# draw in viewport 1
viewport(1)
clear()
F,c = cube_tri()
draw(F)
drawCircles(F)

if not ack("Keep both viewports ?"):
    print "Removing a viewport"
    # remove last viewport
    removeViewport()

# End

