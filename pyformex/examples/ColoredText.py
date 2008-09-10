#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
""""ColoredText

level = 'beginner'
topics = []
techniques = ['colors']

"""

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
    #drawTextQt(T[t[i]],a[i][0],a[i][1])
    #GD.canvas.update()

# End
