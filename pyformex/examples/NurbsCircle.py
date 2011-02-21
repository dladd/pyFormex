#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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

"""NurbsCircle

level = 'advanced'
topics = ['geometry', 'curve']
techniques = ['nurbs','border']

.. Description

Nurbs Circle
============

The image shows a number of closed nurbs curves that were generated
from the same set of 8 control points (shown in black, numbered 0..7).

The nurbs curve are defined by the following parameters:

:black: order 2 (degree 1: linear)
:red: order 3 (degree 2: quadratic)
:green: order 4 (degree 3: cubic)
:blue: order 3, but using weights 1, sqrt(2)/2 for the midside,corner points respectively. This results in a perfect circle. The blue points on the curve are evaluated from the nurbs formulation, by dividing the parameter space in 20 equidistance parts.
  
The yellow curve is created with simple.circle and uses 180 line segments.
"""


from OpenGL import GL, GLU

import simple
from plugins.curve import *
from plugins.nurbs import *
from gui.actors import Actor
from gui.drawable import drawNurbsCurves
import olist
import lib._nurbs_ as nu

class NurbsCurve():

    """A NURBS curve

    order (2,3,4,...) = degree+1 = min. number of control points
    ncontrol >= order
    nknots = order + ncontrol >= 2*order

    convenient solutions:
    OPEN:
      nparts = (ncontrol-1) / degree
      nintern = 
    """
    
    def __init__(self,control,wts=None,knots=None,closed=False,order=0):
        self.closed = closed
        nctrl = len(control)
        
        if order <= 0:
            if knots is None:
                order = min(nctrl,3)
            else:
                order = len(knots) - nctrl
        if order <= 0:
            raise ValueError,"Length of knot vector (%s) must be larger than number of control points (%s)" % (len(knots),nctrl)

        control = Coords4(control)
        if wts is not None:
            control.deNormalize(wts)
        #print "CONTROL",control

        if closed:
            if knots is None:
                nextra = order-1
            else:
                nextra = len(knots) - nctrl - order
            nextra1 = (nextra+1) // 2
            nextra2 = nextra-nextra1
            print "extra %s = %s + %s" % (nextra,nextra1,nextra2) 
            control = Coords.concatenate([control[-nextra1:],control,control[:nextra2]])

        nctrl = len(control)

        if nctrl < order:
            raise ValueError,"Number of control points (%s) must not be smaller than order (%s)" % (nctrl,order)

        if knots is None:
            if closed:
                knots = unitRange(nctrl+order)
            else:
                degree = order-1
                nparts = (nctrl-1) / degree
                print nparts
                knots = [0.] + [ [float(i)]*degree for i in range(nparts+1) ] + [float(nparts)]
                knots = olist.flatten(knots)
        nknots = len(knots)                                                  

        print "Nurbs curve of order %s with %s control points and %s knots" % (order,nctrl,nknots)
        print "KNOTS",knots

        if nknots != nctrl+order:
            
            raise ValueError,"Length of knot vector (%s) must be equal to number of control points (%s) plus order (%s)" % (nknots,nctrl,order)


       
        self.control = control
        self.knots = asarray(knots)
        #self.order = order
        self.closed = closed


    def order(self):
        return len(self.knots)-len(self.control)
        
    def bbox(self):
        return self.control.toCoords().bbox()


    def pointsAt(self,u=None,n=10):
        if u is None:
            umin = self.knots[0]
            umax = self.knots[-1]
            u = umin + arange(n+1) * (umax-umin) / n
        
        ctrl = self.control.astype(double)
        knots = self.knots.astype(double)
        u = asarray(u).astype(double)

        try:
            pts = nu.bspeval(self.order()-1,ctrl,knots,u)
            if isnan(pts).any():
                print "We got a NaN"
                raise RuntimeError
        except:
            raise RuntimeError,"Some error occurred during the evaluation of the Nurbs curve"

        if pts.shape[-1] == 4:
            pts = Coords4(pts).toCoords()
            print pts.shape
        else:
            pts = Coords(pts)
            print pts.shape
        return pts
        

    def actor(self,**kargs):
        """Graphical representation"""
        return NurbsActor(self,**kargs)


class NurbsActor(Actor):

    def __init__(self,data,color=None,**kargs):
        from gui.drawable import saneColor
        Actor.__init__(self)
        self.object = data
        self.color = saneColor(color)
        self.samplingTolerance = 1.0

        
    def bbox(self):
        return self.object.bbox()

        
    def drawGL(self,**kargs):
        drawNurbsCurves(self.object.control,self.object.knots,color=self.color)



def askCurve():
    default = 'Angle'
    res = askItems([('curve_type',default,'radio',['Circle','Angle']),('closed',False),('circle_npts',6)])

    if not res:
        exit()

    clear()
    globals().update(res)

    if curve_type == 'Circle':
        circle = simple.circle(a1=360./circle_npts)
        draw(circle,color='magenta')
        pts = circle[:,0]


    elif curve_type == 'Angle':
        F = Formex(simple.pattern('41'))
        draw(F,color='magenta')
        pts = F.coords.reshape(-1,3)

    clear()
    drawNumbers(pts)
    print "Number of points: %s"%len(pts)
    print "POLY"
    PL = PolyLine(pts,closed=closed)
    d = PL.directions()
    dm = PL.avgDirections()
    #w = PL.doubles()+1
    #print PL.endOrDouble()
    curve = BezierSpline(pts,closed=closed)
    draw(curve.approx(100),color='red')
    zoomAll()
    

def unitRange(n):
    """Divide the range 0..1 in n equidistant points"""
    if n > 1:
        return arange(n) * (1.0/(n-1))
    elif n == 1:
        return [0.5]
    else:
        return []


def drawThePoints(N,n,color=None):
    degree = N.order()-1
    umin = N.knots[degree]
    umax = N.knots[-degree-1]
    #umin = N.knots[0]
    #umax = N.knots[-1]
    print "Umin = %s, Umax = %s" % (umin,umax)
    u = umin + arange(n+1) * (umax-umin) / float(n)
    print u
    #u = [0.25,0.5,0.75]
    P = N.pointsAt(u)
    print P
    draw(P,color=color)
    drawNumbers(P,color=color)

                         
clear()
linewidth(2)
flat()


sq2 = sqrt(0.5)
closed=False
allpts = Coords([
    [1.,0.,0.],
    [1.,1.,0.],
    [0.,1.,0.],
    [-1.,1.,0.],
    [-1.,0.,0.],
    [-1.,-1.,0.],
    [0.,-1.,0.],
    [1.,-1.,0.],
    [1.,0.,0.],
    [2.,0.,0.],
    [3.,0.,0.],
    [4.,0.,0.],
    ])

#    3*0    -     2*1     -    3*2    : 8 = 5+3
#    nctrl = nparts * degree + 1 
#    nknots = nctrl + degree + 1
#    nknots = (nparts+1) * degree + 2
#
# degree  nparts  nctrl   nknots
#    2      1       3        6
#    2      2       5        8
#    2      3       7       10
#    2      4       9       12
#    3      1       4        8
#    3      2       7       11
#    3      3      10       14
#    3      4      13       17
#    4      1       5       10 
#    4      2       9       14
#    4      3      13       18
#    5      1       6       12
#    5      2      11       17
#    5      3      16       22
#    6      1       7       14       
#    6      2      13       20
#    7      1       8       16
#    8      1       9       18

orders = [ 2,3,4 ]
weight = [0.,0.5,sqrt(0.5),1.,sqrt(2.),2,10]
colors = [red,green,blue,cyan,magenta,yellow,white]

# This should be a valid combination of ntrl/degree
# drawing is only done if degree <= 7

def demo_weights():
    nctrl = 8
    degree = 7
    pts = allpts[:nctrl]
    knots = None
    L = {}
    draw(pts)
    drawNumbers(pts)
    draw(PolyLine(pts))
    for w,c in zip(weight,colors):
        qc = Coords4(pts)
        for i in range(1,degree):
            qc[i::degree].deNormalize(w)
        print qc
        C = NurbsCurve(qc,knots=knots,order=degree+1,closed=False)
        draw(C,color=c)
        drawThePoints(C,10,color=c)
        L["wt-%s" % w] = C
    export(L)

def demo_knots():
    C = Formex(pattern('21412')).toCurve()
    pts = C.coords
    print pts
    draw(C)
    nctrl = C.ncoords()
    degree = 2
    nparts = (nctrl-1)/degree
    knots = [
#        None,
#        [0.,0.,0.,1.,1.,2.,2.,3.,3.,3.],
#        [0.,0.,0.,1.,2.,3.,4.,5.,5.,5.],
        [0.,0.,0.,1.,2.,3.,4.,4.,4.],
        ]
    L = {}
    draw(pts)
    drawNumbers(pts)
    draw(PolyLine(pts))
    for k,c in zip(knots,colors):
        C = NurbsCurve(pts,knots=k,order=degree+1,closed=False)
        draw(C,color=c)
        drawThePoints(C,4*nparts,color=c)


demo_knots()

zoomAll()
exit()

draw(simple.circle(2,4),linewidth=3)
xm = 0.5*(pts[0]+pts[2])
xn = 0.5*(xm+pts[1])
draw([xm,pts[1],xn],marksize=10)

pause()
clear()
wireframe()
transparent(False)
lights(False)
grid = simple.rectangle(4,3,diag='d')
draw(grid,color=blue)
print grid.shape()
grid.eltype='nurbs'
draw(grid,color=red)
print grid.shape()
grid.eltype='curve'
draw(grid,color=yellow)
print grid.shape()

exit()
## for order,color in zip(range(2,5),[cyan,magenta,yellow]):
##     if order > len(pts):
##         continue
##     N[order] = NurbsActor(pts,closed=closed,order=order,color=color)
##     drawActor(N[order])
for order,color in zip(range(2,5),[red,green,blue]):
    if order > len(pts):
        continue
    print wts.shape
    N[order] = NurbsActor(pts,wts=wts,closed=False,order=order,color=color)
    drawActor(N[order])

exit()
drawNumbers(pts*wts,color=red)
for order,color in zip(range(2,5),[cyan,magenta,yellow]):
    N[order] = NurbsActor(pts*wts,closed=closed,order=order,color=color)
    drawActor(N[order])

exit()


NO = NurbsActor(pts*wts,closed=closed,order=3,color=blue)
drawActor(NO)

NO = NurbsActor(pts*wts,closed=closed,order=2,color=cyan)
drawActor(NO)

## def distance(C):
##     d = C.approx(100).pointsOn().distanceFromPoint(origin())
##     return 1.-d.min(),1.-d.max()

## clear()
## S = NaturalSpline(pts*wts,closed=True)
## draw(S,color=magenta)
## draw(S.pointsOn(),color=magenta)
## drawNumbers(S.pointsOn(),color=magenta)
## print distance(S)

## #from gui.colors import *
## for c,curl in zip([black,red,green,blue],[ 0.375055, 0.375058, 0.37506  ]):
##     C = BezierSpline(pts*wts,closed=True,curl=curl)
##     draw(C,color=c)
##     print distance(C)


zoomAll()

# En1
