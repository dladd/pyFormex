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

colors=['indianred','limegreen','coral','yellow']

for i,v in enumerate(['front','right','top','iso']):
    viewport(i)
    bgcolor(colors[i])
    GD.canvas.setBgColor(GD.canvas.settings.bgcolor)
    GD.canvas.display()
    GD.canvas.update()
    #view(v)
    print "Viewport %d = %s" % (i,GD.canvas)
    print GD.canvas.actors
    print GD.canvas.settings

#End
