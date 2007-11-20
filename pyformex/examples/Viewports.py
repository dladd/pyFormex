#!/usr/bin/env pyformex
"""Viewports.py

Demonstrate multiple viewports.
"""

nsl = 2        
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

for i,v in enumerate(['front','right','top','iso']):
    viewport(i)
    view(v)

#End
