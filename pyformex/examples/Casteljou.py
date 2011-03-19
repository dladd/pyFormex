#!/usr/bin/env pyformex --gui
# $Id$

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

C = Formex(pattern(pat)).toCurve()

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

