#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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
techniques = ['function','import','mpattern','dialog','viewport']
"""
import simple
from examples.Cube import cube_tri
from plugins.geomtools import *


def draw_circles(circles,color=red):
    for r,c,n in circles:
        C = simple.circle(r=r,n=n,c=c)
        draw(C,color=color)


def drawCircles(F,func,color=red):
    r,c,n = func(F.coords)
    draw(c,color=color)
    draw_circles(zip(r,c,n),color=color)
    
    
layout(2)
wireframe()

# draw in viewport 0
viewport(0)
view('front')
clear()
rtri = Formex(mpattern('16-32')).scale([1.5,1,0])
F = rtri + rtri.shear(0,1,-0.5).trl(0,-4.0) + rtri.shear(0,1,0.75).trl(0,3.0)
draw(F)

drawCircles(F,triangleCircumCircle,color=red)
zoomAll()   
drawCircles(F,triangleInCircle,color=blue)
drawCircles(F,triangleBoundingCircle,color=black)
zoomAll()   


# draw in viewport 1
viewport(1)
view('iso')
clear()
F,c = cube_tri()
draw(F)
drawCircles(F,triangleInCircle)
zoomAll()   

if not ack("Keep both viewports ?"):
    print "Removing a viewport"
    # remove last viewport
    removeViewport()

# End

