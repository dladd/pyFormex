#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""ColoredText

level = 'beginner'
topics = []
techniques = ['colors','text']

"""

n = 40
T = ['Python','NumPy','OpenGL','QT4','pyFormex']
font = 'times'
ftmin,ftmax = 12,36

r = random.random((n,7))
w,h = GD.canvas.width(), GD.canvas.height()
a = r[:,:2] * array([w,h]).astype(int)
size = (ftmin + r[:,2] * (ftmax-ftmin)).astype(int)
colors = r[:,3:6]
t = (r[:,6] * len(T)).astype(int)
clear()

bgcolor(white)
lights(False)
TA = None

for i in range(n):
    # fgcolor(red)
    TB = drawText(T[t[i]],a[i][0],a[i][1],font=font,size=size[i],color=list(colors[i]))
    sleep(0.2)
    breakpt()
    if i < n/2:
        undecorate(TA)
    TA = TB
    #drawTextQt(T[t[i]],a[i][0],a[i][1])
    #GD.canvas.update()

# End
