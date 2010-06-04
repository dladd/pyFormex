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


    elif curve_type == 'Angle':
        closed = True
        F = Formex(simple.pattern('41'))
        draw(F,color='magenta')
        pts = F.coords.reshape(-1,3)

    clear()
    drawNumbers(pts)
    curve = CardinalSpline(pts,closed=closed)
    draw(curve.approx(100),color='red')
    print "Number of points: %s",len(pts)
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
transparent()
linewidth(2)

askCurve()
zoomAll()
exit()

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
drawActor(NO)

print "KNOTS",NO.knots
n = 10
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

zoomAll()

# End
