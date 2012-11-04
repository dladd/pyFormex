# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Random

This example shows how to create random coordinates and property numbers,
and how to connect points to create lines, triangles and quads.

The application first creates 30 random property numbers from 0 to 7 and
30x3 coordinates, and collects them in a Formex P of plexitude 1. When drawn,
30 colored points are shown.

Then the Formex of points is used in a connect operation with itself
(repeated), to construct Formices of plexitude 1,2,3,4 (i.e. points, lines,
triangles, quads). The subsequent versions of P in the Formex list [P]*n
(which is equal to [P,P,...] with P repeated n times) are used
with an increasing bias (0,1,...). This means that to construct the 2-plex
Formex, point 0 is connected to point 1, point 1 to point 2, etc., while for
the 3-plex Formex the elements are formed from point (0,1,2), (1,2,3), and
so on. Because the loop parameter is set True, the list of points is wrapped
around when its end is reached, and the number of multiplex elements is thus
always equal to the number of points.
"""
from __future__ import print_function
_status = 'checked'
_level = 'beginner'
_topics = ['formex']
_techniques = ['color','random','connect']

from gui.draw import *

def run():
    resetAll()
    flat()
    delay(2)
    setDrawOptions({'clear':True})
    npoints = 30
    p = random.randint(0,7,(npoints,))
    P = Formex(random.random((npoints,1,3)),p)
    draw(P)
    smooth()
    for n in range(1,5):
        F = connect([P] * n,
                    bias=[i*(n-1) for i in range(n)],
                    loop=True)
        F.setProp(p)
        draw(F)

if __name__ == 'draw':
    run()
# End
