#!/usr/bin/env pyformex --gui
# $Id$

"""NurbsCircle

level = 'advanced'
topics = ['geometry', 'curve']
techniques = ['nurbs','border']

.. Description

Nurbs Circle
============

The image shows (in black) a number of closed nurbs curves that were generated
from the same set of 8 control points (numbered 0..7). From outer to inner the
curves are defined by the following parameters:

- order 2 (degree 1: linear)
- order 3 (degree 2: quadratic)
- order 4 (degree 3: cubic)
- order 3, but using weights 1, 0.75 for the midside,corner points respectively.
  This results in a perfect circle.
  
The red curve is created with simple.circle and uses 180 line segments.
"""

from OpenGL import GL, GLU

import simple
from plugins.curve import *
from gui.actors import Actor
import lib._nurbs_ as nu

def askCurve():
    default = 'Angle'
    res = askItems([('curve_type',default,'radio',['Circle','Angle']),('circle_npts',6)])

    if not res:
        exit()

    clear()
    globals().update(res)

    if curve_type == 'Circle':
        closed = True
        circle = simple.circle(a1=360./circle_npts)
        draw(circle,color='magenta')
        pts = circle[:,0]
        drawNumbers(pts)


    elif curve_type == 'Angle':
        closed = False
        F = Formex(simple.pattern('41'))
        draw(F,color='magenta')
        pts = F.coords.reshape(-1,3)

    curve = BezierSpline(pts,closed=closed)
    draw(curve,color='red')

    print "Number of points: %s",len(pts)


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
            control = Coords.concatenate([control,control[:nextra]])
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
        fgcolor(self.color)
        nurb = GLU.gluNewNurbsRenderer()
        GLU.gluNurbsProperty(nurb,GLU.GLU_SAMPLING_TOLERANCE,self.samplingTolerance)
        GLU.gluBeginCurve(nurb)
        GLU.gluNurbsCurve(nurb,self.knots,self.control,GL.GL_MAP1_VERTEX_3)
        GLU.gluEndCurve(nurb)

    def pointsAt(self,u=None):
        if u is None:
            u = self.knots

        ctrl = self.control.astype(double)
        knots = self.knots.astype(double)
        u = u.astype(double)

        try:
            pts = nu.bspeval(self.order-1,ctrl.transpose(),knots,u)
        except:
            print "SOME ERROR OCCURRED"
        return pts

                         
clear()
transparent()
linewidth(2)

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
draw(P,color=magenta)

Q = simple.circle()
draw(Q,color=red)
#drawNumbers(Q,color=red)

N = {}
for order in range(2,5):
    N[order] = NurbsActor(pts,closed=closed,order=order)
    drawActor(N[order])

#knots = [0.,0.,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.0]
## NN = NurbsActor(pts,knots=knots,closed=closed,order=3)
## drawActor(NN)

print pts.shape
print wts.shape
NO = NurbsActor(pts*wts,knots=knots,closed=closed,order=3)
drawActor(NO)

## pts = Coords([
##     [0.,-0.5,0.],
##     [0.5,-0.5,0.],
##     [0.25,-0.0669873,0.],
##     [0.,0.3880254,0.],
##     [-0.25,-0.0669873,0.],
##     [-0.5,-0.5,0.],
##     [0.,-0.5,0.],
##     ])

## sq2 = 0.5 ** (1./3.)
## wts = array([
##     1.0,
##     sq2,
##     1.0,
##     sq2,
##     1.0,
##     sq2,
##     1.0,
##     ]).reshape(-1,1)
## knots = [0.,0.,0.,1.,1.,2.,2.,3.,3.,3.]
## NN = NurbsActor(pts*wts,knots=knots,closed=True,order=3)
## drawActor(NN)

## draw(pts,color=red)
## draw(pts*wts,color=blue)
    
zoomAll()

## P = NO.pointsAt()
## print P
## draw(P,color=blue)

# End
