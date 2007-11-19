#!/usr/bin/env pyformex --gui
# $Id$
"""Horse

This script reads horse.formex, transforms it into a surface,
loads the surface plugin and cuts the horse in a number of surfaces.
"""

from plugins.surface import Surface

reset()
wireframe()
chdir(GD.cfg['curfile'])

x = 20
y = GD.canvas.height()-20

def say(text):
    global y
    drawtext(text,x,y)
    y -=20

GD.message('Click Step to continue')

say('A Horse Story...')
y -= 10
F = Formex.read('horse.formex')
A = draw(F)
pause()

say('It\'s rather sad, but')
smooth()
pause()


say('the horse was badly cut;')
T = F.cutAtPlane([0.,0.,0.],[-1.,0.,0.])
undraw(A)
A = draw(T)
pause()


say('to keep it stable,')
undraw(A)
A = draw(T.rotate(-80))
pause()


say('the doctors were able')
undraw(A)
A = draw(T)
pause()


say('to add a mirrored clone:')
T += T.reflect(0)
undraw(A)
A = draw(T)
pause()

say('A method as yet unknown!')
colors = 0.1 * random.random((10,3))
for color in colors:
    B = draw(T,color=color)
    undraw(A)
    A = B
    sleep(0.5)
