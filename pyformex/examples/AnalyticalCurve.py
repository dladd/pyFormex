# $Id$ *** pyformex app ***
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
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
"""Analytical Curves

Play with analytical curves.
"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['curve']
_techniques = ['function']

from gui.draw import *

from plugins.nurbs import *

class AnalyticalCurve(object):
    """Analytical curve

    An analytical curve can be defined by a function of a parameter
    returning three values [x,y,z] for each parameter value in the
    parameter range.

    - `func`: A function taking a float input value and returning
      three float values (x,y,z), being the cartesian coordinates of a point
      on the curve. The function should be defined in the full range of the
      parameter, as specified by `trange`. The function may also take
      other parameters, enabling the definition of classes of parametric
      curves.
    - `trange`: range of the parameter t. This is a tuple (tmin,tmax).
      If unspecified, the range is -inf to +inf.
    - `args`: a tuple of arguments to pass to the function, may be an
      empty tuple.
    - `closed`: bool. Specifies whether the resulting curve should be
      closed.
    """
    def __init__(self,func,trange,args=(),closed=False):
        self._func = func
        self._range = trange
        self._args = args
        self._closed = closed
    def points(self,n,trange=None):
        """Compute n points anlong the curve"""
        if trange is None:
            trange = self._range
        if self._closed:
            n += 1
        t = trange[0] + (trange[1]-trange[0])*arange(n)/float(n-1)
        return Coords(column_stack(self._func(t,*self._args)))


def Viviani(t):
    """Compute points on a Viviani curve at parameters t.

    t ranges between 0 and 360
    """
    return [(1+cos(t))/2, sin(t)/2, sin(t/2)]


def Lemniscate(t):
    """Compute points on a Lemniscate curve at parameters t.

    t ranges between 0 and 360
    """
    return [ cos(t/2), sin(t)/2, 0*t ]

def Tschirnhausen(t):
    return [ 1-3*t**2, t*(3-t**2), 0*t ]


def Lissajous(t,a,d):
    return [ sin(a*t+d), sin(t), 0*t ]



def Hypocycloid(t,k):
    return [ (k-1)*cos(t) + cos((k-1)*t),
             (k-1)*sin(t) - sin((k-1)*t),
             0*t ]



def Epicycloid(t,k):
    return [ (k+1)*cos(t) - cos((k+1)*t),
             (k+1)*sin(t) - sin((k+1)*t),
             0*t ]


def Trochoid(t,l):
    return [ t-l*sin(t), 1-l*cos(t), 0*t ]


def Cycloid(t):
    return Trochoid(t,1.0)


def Spirograph(t,k,l):
    return [ (1-k)*cos(t) + k*l*cos((1-k)/k*t),
             (1-k)*sin(t) - k*l*sin((1-k)/k*t),
             0*t ]


def Involute(t):
    pass


def Curve(func,trange,args,closed,degree,npoints):
    """Approximate an analytical curve

    The approximation is a BezierSpline of the specified degree
    through npoints points on the analytical curve.
    """
    A = AnalyticalCurve(func,trange,args,closed)
    X = A.points(npoints)
    N = globalInterpolationCurve(X,degree=degree)
    drawText(func.__name__,80,y0,size=18)
    return N


def drawCurve(*args,**kargs):
    draw(Curve(*args),**kargs)
    drawText(args[0].__name__,80,y0,size=18,color=kargs.get('color',None))



def run():
    global y0
    clear()
    wireframe()
    linewidth(3)
    delay(1)
    y0=80
    draw(Curve(Tschirnhausen,(-3.,3.),(),False,3,10),view='front',clear=True)
    draw(Curve(Viviani,(0,4*pi),(),True,4,10),view='right',clear=True)
    draw(Curve(Lemniscate,(-pi,3*pi),(),True,4,10),view='front',clear=True)
    draw(Curve(Lemniscate,(-pi,3*pi),(),True,4,12),view='front',color=red)

    NB = None
    n = 100
    delay(0.1)
    clear()
    for a in arange(n+1)/float(n):
        print(a)
        NA = draw(Curve(Lissajous,(0,12*pi),(a,0),True,4,120),color=red)
        if NB:
            undraw(NB)
        NB = NA
        breakpt()

    delay(1)
    wait()
    draw(Curve(Spirograph,(0,40*pi),(2*0.12345678,0.5),True,4,800),clear=True)
    drawCurve(Spirograph,(0,40*pi),(2*0.12345678,1.0),True,4,800,clear=True)
    y0 = 60
    drawCurve(Hypocycloid,(0,40*pi),(300/99.,),True,4,800,color=blue,clear=True)
    y0 = 40
    drawCurve(Epicycloid,(0,40*pi),(300/99.,),True,4,800,color=red)


    # extrude curve
    clear()
    y0=80
    C = Curve(Cycloid,(0,2*pi),(),True,4,20)
    M = C.approx(20).toMesh().extrude(10,dir=2)
    smoothwire()
    draw(M,color=red)


if __name__ == 'draw':
    run()

# End
