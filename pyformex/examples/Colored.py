#!/usr/bin/env pyformex
# $Id$

from gui.actors import *

smooth()
lights(False)

Rendermode = [ 'smooth','flat' ]
Lights = [ False, True ]
Shape = { 'triangle':'16',
          'quad':'123',
          }


color0 = None  # no color: current fgcolor
color1 = red   # single color
color2 = array([red,green,blue]) # 3 colors: will be repeated

for shape in Shape.keys():
    F = Formex(mpattern(Shape[shape])).replic2(8,4)
    color3 = resize(color2,F.shape()) # full color
    for mode in Rendermode:
        renderMode(mode)
        for c in [ color0,color1,color2,color3]:
            clear()
            FA = FormexActor(F,color=c)
            drawActor(FA)
            zoomAll()
            for light in Lights:
                lights(light)
                sleep(1)


# End
