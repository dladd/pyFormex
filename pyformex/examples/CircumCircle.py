#!/usr/bin/env pyformex
# $Id$
"""CircumCircle

level = 'beginner'
topics = ['geometry']
techniques = ['function','import','mpattern','dialog','viewport']
"""
import simple
from examples.Cube import cube_tri
from plugins.geometry import triangleCircumCircle

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

