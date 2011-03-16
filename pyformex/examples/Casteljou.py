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

predefined = ['2584','514','1234','51414336','custom']

res = askItems([
    dict(name='pattern',choices=predefined),
    dict(name='custom',value=''),
    dict(name='casteljou',value=0.5),
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
for u in arange(n+1)*1.0/n:
    P = pointsOnBezierCurve(C.coords,u)
    draw(P)

def deCasteljou(P,u):
    """Compute points on a Bezier curve using deCasteljou algorithm

    Parameters:
    P is an array with n+1 points defining a Bezier curve of degree n.
    u is a single parameter value between 0 and 1.

    Returns:
    A list with point sets obtained in the subsequent deCasteljou
    approximations. The first one is the set of control points, the last one
    is the point on the Bezier curve.
    """
    n = P.shape[0]-1
    C = [P]
    for k in range(n):
        Q = C[-1]
        Q = (1.-u) * Q[:-1] + u * Q[1:]
        C.append(Q)
    return C

u = res['casteljou']
Q = deCasteljou(C.coords,u)
for q in Q[1:-1]:
    draw(PolyLine(q),color=red)
draw(Q[-1],marksize=10)
# End

