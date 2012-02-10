# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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

"""Nurbs

This example shows the Nurbs curves of degree 1..MAX for a given set of
control points.

The control points are constructed from an input string using the 'pattern'
function. The number of predefined pattern strings are provided, but the
user can also input a custom value.

If there are N control points, the maximum Nurbs degree is N-1. However,
due to OpenGL limitations, the maximum Nurbs degree that can be shown
directly is 7. Higher order curves are approximated by a PolyLine.

For the last (highest degree) Nurbs curve, also a set of 100 points along
the curve are shown.
"""
_status = 'checked'
_level = 'normal'
_topics = ['geometry', 'curve']
_techniques = ['nurbs']

from gui.draw import *
from plugins.nurbs import NurbsCurve

resetAll()

# Some strings defining line patterns
predefined = [
    '2584',
    '58',
    '214',
    '184',
    '514',
    '1234',
    '51414336',
    '5858585858',
    '12345678',
    'custom']

# Default values
pattern = None
custom = ''


def run():
    
    res = askItems([
        dict(name='pattern',value=pattern,choices=predefined),
        dict(name='custom',value=custom),
        ],enablers=[('pattern','custom','custom')])

    if not res:
        return

    globals().update(res)
    if pattern == 'custom':
        pat = custom
    else:
        pat = pattern

    if not pat.startswith('l:'):
        pat = 'l:' + pat
    C = Formex(pat).toCurve()

    clear()
    linewidth(2)
    flat()

    draw(C.coords)
    drawNumbers(C.coords)
    setDrawOptions({'bbox':None})

    cmap = colormap() * 2
    print cmap
    n = min(len(C.coords),cmap)
    dmax = 7 # maximum Nurbs degree we can draw in OpenGL
    for d in range(1,n):
        print d
        c = cmap[ (d-1) % len(cmap) ] # wrap around if color map is too short
        N = NurbsCurve(C.coords,degree=d)
        if d <= dmax:
            draw(N,color=c)
            draw(N.knotPoints(),color=c,marksize=10)
        else:
            draw(N.approx(),color=c)
            draw(N.knotPoints(),color=c,marksize=5)

    n = 100
    u = arange(n+1)*1.0/n
    x = N.pointsAt(u)
    draw(x)
    
if __name__ == 'draw':
    run()
# End

