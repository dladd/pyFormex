#!/usr/bin/env pyformex --gui
# $Id$
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
from gui.actors import Actor
import lib._nurbs_ as nu

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


class NurbsActor(Actor):

    def __init__(self,control,knots=None,closed=False,order=0,color=black):
        Actor.__init__(self)
        self.closed = closed
        nctrl = len(control)
        
        if order <= 0:
            if knots is None:
                order = min(nctrl,3)
            else:
                order = len(knots) - nctrl
        if order <= 0:
            raise ValueError,"Length of knot vector (%s) must be larger than number of control points (%s)" % (len(knots),nctrl)

        if closed:
            if knots is None:
                nextra = order-1
            else:
                nextra = len(knots) - nctrl - order
                #print len(knots)
                #print nctrl
            print "extra %s" % nextra 
            control = Coords.concatenate([control[-nextra:],control])
            nctrl = len(control)
            #print nctrl

        if nctrl < order:
            raise ValueError,"Number of control points (%s) must not be smaller than order (%s)" % (nctrl,order)

        if knots is None:
            if closed:
                knots = unitRange(nctrl+order)
            else:
                knots = concatenate([[0.]*(order-1),unitRange(nctrl-order+2),[1.]*(order-1)])
        nknots = len(knots)                                                  
        if nknots != nctrl+order:
            
            raise ValueError,"Length of knot vector (%s) must be equal to number of control points (%s) plus order (%s)" % (nknots,nctrl,order)

        print "Nurbs curve of order %s with %s control points and %s knots" % (order,nctrl,nknots)

        self.control = control
        self.knots = knots
        self.order = order
        self.closed = closed
        self.samplingTolerance = 1.0
        self.color = color
        print self.control
        print self.knots
        #print self.control.shape
        #print self.knots.shape

        
    def bbox(self):
        return self.control.bbox()

        
    def drawGL(self,mode,color=None):
        GL.glColor3fv(self.color)
        nurb = GLU.gluNewNurbsRenderer()
        GLU.gluNurbsProperty(nurb,GLU.GLU_SAMPLING_TOLERANCE,self.samplingTolerance)
        GLU.gluBeginCurve(nurb)
        GLU.gluNurbsCurve(nurb,self.knots,self.control,GL.GL_MAP1_VERTEX_3)
        GLU.gluEndCurve(nurb)


    def pointsAt(self,u=None):
        if u is None:
            u = self.knots[order:]

        ctrl = self.control.astype(double)
        knots = self.knots.astype(double)
        u = u.astype(double)

        try:
            print "U",u
            pts = nu.bspeval(self.order-1,ctrl.transpose(),knots,u)
            print pts.shape
            print pts
            return Coords(pts.transpose())
        except:
            print "SOME ERROR OCCURRED"
            return Coords()

                         
clear()
#transparent()
linewidth(2)

## askCurve()
## zoomAll()
## exit()

closed=False
pts = Coords([
    [0.,1.,0.],
    [0.,0.,0.],
    [1.,0.,0.],
    [1.,-1.,0.],
    ])
knots = None

closed=True
pts = Coords([
    [1.,0.,0.],
    [1.,1.,0.],
    [0.,1.,0.],
    [-1.,1.,0.],
    [-1.,0.,0.],
    [-1.,-1.,0.],
    [0.,-1.,0.],
    [1.,-1.,0.],
    ])
draw(pts)
drawNumbers(pts)

sq2 = sqrt(0.5)
wts = array([
    1.0,
    sq2,
    1.0,
    sq2,
    1.0,
    sq2,
    1.0,
    sq2,
    ]).reshape(-1,1)


P = PolyLine(pts,closed=closed)
#draw(P,color=magenta)

Q = simple.circle()
draw(Q,color=yellow)
#drawNumbers(Q,color=red)

N = {}
for order,color in zip(range(2,5),[black,green,red]):
    N[order] = NurbsActor(pts,closed=closed,order=order,color=color)
    drawActor(N[order])

print pts.shape
print wts.shape
order = 3
NO = NurbsActor(pts*wts,knots=knots,closed=closed,order=order,color=blue)
draw(pts*wts,color=magenta)
drawActor(NO)

print "KNOTS",NO.knots
n = 20
degree = order-1
umin = NO.knots[degree]
umax = NO.knots[-degree-1]
print "Umin = %s, Umax = %s" % (umin,umax)
u = umin + arange(n+1) * (umax-umin) / n
print u
P = NO.pointsAt(u)
print P
draw(P,color=blue)
drawNumbers(P,color=blue)
print P.distanceFromPoint(origin())

S = NaturalSpline(pts*wts,closed=True)
draw(S,color=magenta)

zoomAll()

# End
