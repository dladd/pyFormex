#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Mon Dec 29 15:32:01 2008
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
