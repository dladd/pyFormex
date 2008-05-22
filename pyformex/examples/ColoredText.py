#!/usr/bin/env pyformex
# $Id$
"ColoredText"

n = 40
T = ['Python','NumPy','OpenGL','QT4','pyFormex']

w,h = GD.canvas.width(), GD.canvas.height()
a = random.random((n,2)) * array([w,h])
a = a.astype(int)
colors = random.random((n,3))
t = (random.random((n,)) * len(T)).astype(int)
clear()

bgcolor(white)
lights(False)
TA = None
for i in range(n):
    fgcolor(red)
    TB = drawtext(T[t[i]],a[i][0],a[i][1],'tr24',color=list(colors[i]))
    sleep(0.2)
    breakpt()
    undecorate(TA)
    TA = TB

# End
