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
for i in range(n):
    TA = drawtext(T[t[i]],a[i][0],a[i][1],'hv18',color=list(colors[i]))
    sleep(0.2)
    breakpt()
    undecorate(TA)

# End
