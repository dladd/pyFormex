# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

Displays subsequent approximations of a sphere. In each step two spheres
are drawn: the left ons is a frame structure (simple.sphere2), the right
one a triangulated surface (simple.sphere3).

Remark that simple.sphere3 may contain degenerate triangles at the north and
south pole.
"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['geometry','surface','sphere']
_techniques = ['color']

from gui.draw import *
from simple import sphere2,sphere3

def run():
    reset()
    nx = 4   # initial value for number of modules
    ny = 4
    m = 1.6  # refinement at each step
    ns = 6   # number of steps

    smooth()
    setView('front')
    for i in range(ns):
        print("nx=%s, ny=%s" % (nx,ny))
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
        pause(2)

if __name__ == 'draw':
    run()
# End

