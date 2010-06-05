#!/usr/bin/env pyformex --gui
# $Id: Sphere2.py 154 2006-11-03 19:08:25Z bverheg $
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
"""Sphere2

level = 'normal'
topics = ['geometry','surface']
techniques = ['colors']

"""

from simple import sphere2,sphere3

reset()

nx = 4
ny = 4
m = 1.6
ns = 6

smooth()
setView('front')
for i in range(ns):
    b = sphere2(nx,ny,bot=-90,top=90).translate(0,-1.0)
    s = sphere3(nx,ny,bot=-90,top=90)
    s = s.translate(0,1.0)
    s.setProp(3)
    clear()
    bb = bbox([b,s])
    draw(b,bbox=bb,wait=False)
    draw(s,bbox=bb)#,color='random')
    nx = int(m*nx)
    ny = int(m*ny)

