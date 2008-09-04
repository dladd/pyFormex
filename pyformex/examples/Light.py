#!/usr/bin/env pyformex
# $Id$

from gui.prefMenu import setRender

smooth()

Shape = { 'triangle':'16',
          'quad':'123',
          }
color2 = array([red,green,blue]) # 3 base colors
F = Formex(mpattern(Shape['triangle'])).replic2(8,4)
color3 = resize(color2,F.shape())
draw(F,color=color3)


setRender()

for a in [ 'ambient', 'specular', 'emission', 'shininess' ]:
    v = getattr(GD.canvas,a)
    print "  %s: %s" % (a,v)

# End
