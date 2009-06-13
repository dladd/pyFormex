#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
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
"""Interpolate

level = 'beginner'
topics = ['geometry']
techniques = ['colors']

"""

def demo_interpolate():
    clear()
    a = Formex([[[0,0,0],[1,0,0]],[[1,0,0],[2,0,0]]])
    b = Formex([[[0,1,0],[1,1,0]],[[1,1,0],[2,1,0]]])
    message("Two lines")
    draw(a+b)

    n = 10
    v = 1./n * arange(n+1)
    p = arange(n)
    
    c = interpolate(a,b,v)
    c.setProp(p)
    message("Interpolate between the two")
    draw(c)
    drawNumbers(c)

    sleep(2)
    d = interpolate(a,b,v,swap=True)
    d.setProp(p)
    clear()
    message("Interpolate again with swapped order")
    draw(d)
    drawNumbers(d)
    exit()

    sleep(2)
    f = c.divide(v)
    f.setProp((1,2))
    clear()
    message("Divide the set of lines")
    draw(f)

if __name__ == "draw":
    wireframe()
    demo_interpolate()
    
