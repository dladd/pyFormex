#!/usr/bin/env pyformex
# $Id$
"""Props

A demonstration of propagating property numbers.
Also shows the use of multiple viewports.
"""

def vp(i):
    viewport(i)
    smooth()
    lights(False)
    clear()
    
if __name__ == "draw":

    layout(4)
    F0 = Formex(mpattern('12-34'),[1,3])
    F1 = F0.replic2(2,2)
    F2 = F1 + F1.mirror(1)
    F3 = F2 + F2.rotate(180.,1)
    
    for i,F in enumerate([F0,F1,F2,F3]):
        vp(i)
        draw(F)
        drawtext("F%s"%i,10,10,'hv18')
    
    
# End
