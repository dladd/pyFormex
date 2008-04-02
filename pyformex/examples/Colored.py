#!/usr/bin/env pyformex
# $Id$

from gui.actors import *

smooth()
lights(False)

Shape = { 'triangle':'16',
          'quad':'123',
          }
res = askItems([('Shape',None,'select',Shape.keys())])

if not res:
    exit()
    
shape = res['Shape']

F = Formex(mpattern(Shape[shape])).replic2(8,4)

color0 = None  # no color: current fgcolor
color1 = red   # single color
color2 = array([red,green,blue]) # 3 colors: will be repeated
color3 = resize(color2,F.shape()) # full color


for c in [color0,color1,color2,color3]:
    clear()
    FA = FormexActor(F,color=c)
    drawActor(FA)
    zoomAll()
    sleep(0.5)


# End
