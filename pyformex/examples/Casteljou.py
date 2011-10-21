#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
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

"""Casteljou

level = 'normal'
topics = ['geometry', 'curve']
techniques = ['nurbs']

.. Description

Casteljou
=========
This example illustrates the deCasteljou algorithm for constructing a point
on a  Bezier curve. It also draws Bezier points computed from Bernstein
polynomials, as in example BezierCurve.
"""

from plugins.curve import *
from plugins.nurbs import *

predefined = [
    '2584',
    '184',
    '514',
    '1234',
    '51414336',
    '5858585858',
    '12345678',
    'custom']

pat = None
custom = ''
casteljou = 0.5
showNurbs = False

res = askItems([
    dict(name='pat',value=pat,text='pattern',choices=predefined),
    dict(name='custom',value=custom),
    dict(name='casteljou',value=casteljou),
    dict(name='showNurbs',value=showNurbs),
    ])

if not res:
    exit()

globals().update(res)

if not pat.startswith('l:'):
    pat = 'l:' + pat
C = Formex(pat).toCurve()

clear()
linewidth(2)
flat()
delay(0)

draw(C,bbox='auto',view='front')
draw(C.coords)
drawNumbers(C.coords)
setDrawOptions({'bbox':None})


if showNurbs:
    n = min(len(C.coords),len(colormap()))
    for d,c in zip(range(1,n),colormap()[:n-1]):
        N = NurbsCurve(C.coords,degree=d)
        draw(N,color=c)
        draw(N.knotPoints(),color=c,marksize=15)
        print d
    print d
else:
    u = casteljou
    Q = deCasteljou(C.coords,u)
    delay(1)
    for q in Q[1:-1]:
        draw(PolyLine(q),color=red)
    draw(Q[-1],marksize=10)

delay(0)
n = 100
u = arange(n+1)*1.0/n

if showNurbs:
    N = NurbsCurve(C.coords)
    x = N.pointsAt(u)
    draw(x)
else:
    P = pointsOnBezierCurve(C.coords,u)
    print P.shape
    draw(Coords(P))
    
# End

