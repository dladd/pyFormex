#!/usr/bin/env pyformex --gui
# $Id$

"""BezierCurve

level = 'normal'
topics = ['geometry', 'curve']
techniques = []

.. Description

BezierCurve
===========
This example illustrates the use of Bernstein polynomials to evaluate points
on a Bezier curve.
"""

from plugins.curve import *
from plugins.nurbs import *

predefined = ['514','1234','51414336','custom']

res = askItems([
    dict(name='pattern',choices=predefined),
    dict(name='custom',value=''),
    ])

if not res:
    exit()

s = res['pattern']
if s == 'custom':
    s = res['custom']

C = Formex(pattern(s)).toCurve()

clear()
linewidth(2)
flat()

draw(C,bbox='auto',view='front')
draw(C.coords)
drawNumbers(C.coords)

setDrawOptions({'bbox':None})
n = 100
u = arange(n+1)*1.0/n
P = pointsOnBezierCurve(C.coords,u)
draw(P)

# End

