#!/usr/bin/env pyformex
# $Id$

from gui.actors import *

wireframe()

F = Formex(mpattern('16')).replic2(10,4)

color0 = None  # no color: current fgcolor
color1 = red   # single color
color2 = array([red,green,blue]) # 3 colors: will be repeated
color3 = resize(color2,(F.nelems(),)+color2.shape) # nelems * 3 colors


for c in [color0,color1,color2,color3]:
    clear()
    FA = FormexActor(F,color=c)
    drawActor(FA)
    zoomAll()
    sleep(2)


# End
