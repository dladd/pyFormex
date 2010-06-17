#!/usr/bin/env pyformex
# $Id$
"""SplineSurface

level = 'advanced'
topics = ['geometry','surface']
techniques = ['spline']

.. Description

SplineSurface
-------------
This example illustrates some advanced geometrical modeling tools using
spline curves and surfaces.

The script first creates a set of closed BezierSpline curves. Currently
two sets of curves are predefined:

- a set of transformations of a unit circle. The circle is scaled
  non-uniformously, resulting in an ellips, which is then rotated and
  translated.

- a set of curves obtained by cutting a triangulated surface model
  with a series of parallel planes. The original surface model was
  obtained from medical imaging processes and represents a human
  artery with a kink. These curves are read from a geometry file
  'splines.pgf' include in the pyFormex distribution. 

In the first case, the number of curves will be equal to the specified
number.  In the latter case, the number can not be large than the
number of curves in the file.

The set of splines are then used to create a QuadSurface (a surface
consisting of quadrilaterals). The number of elements along the
splines can be chosen. The number of elements across the splines is
currently unused.
"""

"""Definition of surfaces in pyFormex.

This module defines classes and functions specialized for handling
two-dimensional geometry in pyFormex. 
"""

# I wrote this software in my free time, for my joy, not as a commissioned task.
# Any copyright claims made by my employer should therefore be considered void.

import numpy as np
from geometry import Geometry
from plugins.curve import *


##############################################################################

#
#  !!!!!!!!!!! THIS IS WORK I PROGRESS ! DO NOT USE !!!!!!!!!!!!!!!!!!
#
#

def rollCurvePoints(curve,n=1):
    """Roll the points of a closed curve.

    Rolls the points of a curve forward over n positions. Thus point 0
    becomes point 1, etc. The function does not return a value. The
    curve is changed inplace.

    This only works for PolyLine and BezierSpline (and derived) classes.
    """
    if (isinstance(curve,PolyLine) or isinstance(curve,BezierSpline)) and curve.closed:
        curve.coords = roll(curve.coords,n,axis=0)
    else:
        raise ValueError,"Expected a closed PolyLine or BezierSpline."
    

def alignCurvePoints(curve,axis=1,max=True):
    """Roll the points of a closed curved according to some rule.

    The points of a closed curve are rotated thus that the starting (and
    ending) point is the point with the maximum or minimum value of the
    specified coordinate.

    The function returns nothing: the points are rolled inplace.
    """
    if not curve.closed:
        raise ValueError,"Expected a closed curve."
    if max:
        ind = curve.pointsOn()[:,axis].argmax()
    else:
        ind = curve.pointsOn()[:,axis].argmin()
    #print curve.pointsOn()
    print "ALIGNING ON POINT %s" % ind
    #drawNumbers(curve.pointsOn())
    rollCurvePoints(curve,-ind)
    #print curve.pointsOn()
    #drawNumbers(curve.pointsOn(),color=red)
    

class SplineSurface(Geometry):
    """A surface created by a sequence of splines.

    The surface consists of a list of curves. The parametric value of
    the curves is called 'u', while 'v' is used for the parametric value
    across the splines.

    Two sets of parametric curves can be drawn: in u and in v direction.
    """

    def __init__(self,curves):
        self.curves = curves
        self.grid = None
        self.ccurves = None


    def bbox(self):
        return bbox(self.curves)


    def createGrid(self,nu,nv=None):
        print "Creating grid %s x %s" % (nu,nv)
        if nv is None:
            nv = self.curves[0].nparts()

        CA = [ C.approx(ntot=nu) for C in self.curves ]
        print "Curves have %s points" % CA[0].coords.shape[0]
        print "There are %s curves" % len(CA)
        self.grid = Coords(stack([CAi.coords[:nu] for CAi in CA]))
        print "Created grid %s x %s" % self.grid.shape[:2]


    def uCurves(self):
        return [ BezierSpline(self.grid[:,i,:],curl=0.375) for i in range(self.grid.shape[1]) ]


    def vCurves(self):
        return [ BezierSpline(self.grid[i,:,:],curl=0.375) for i in range(self.grid.shape[0]) ]


    def toMesh(self):
        """Convert the Grid Surface to a Quad Mesh"""
        nu = self.grid.shape[1]
        nv = self.grid.shape[0] -1 
        elems = array([[ 0,1,nu+1,nu ]])
        elems = concatenate([(elems+i) for i in range(nu-1)],axis=0)
        elems = concatenate([elems,[[ nu-1,0,nu,2*nu-1]]],axis=0)
        #print elems
        #print nu
        elems = concatenate([(elems+i*nu) for i in range(nv)],axis=0)

        x = self.grid.reshape(-1,3)
        #print nu,nv
        #print x.shape
        #print elems.shape
        #print elems.min(),elems.max()
        M = Mesh(self.grid.reshape(-1,3),elems)
        #print M.elems
        #drawNumbers(M.coords)
        return M
    

    def actor(self,**kargs):
        return [ draw(c,**kargs) for c in self.curves ] 



def createCircles(n):
    """Create a set of BezierSpline curves.

    The curves are transformations of a unit circle.
    They are non-uniformously scaled to yield ellipses, and then rotated
    and translated.
    """
    C = circle()
    t = arange(n+1) /float(n)
    CL = [ C.scale([1.,a,0.]) for a in 0.5 + arange(n+1) /float(n) ]
    CL = [ Ci.rot(a,2) for Ci,a in zip(CL,arange(n+1)/float(n)*45.) ]
    CL = [ Ci.trl(2,a) for Ci,a in zip(CL,arange(n+1)/float(n)*4.) ]
    return CL


def readSplines():
    """Read spline curves from a geometry file.

    The geometry file splines.pgf is provided with the pyFormex distribution.
    """
    from geomfile import GeometryFile
    fn = getcfg('datadir')+'/splines.pgf'
    f = GeometryFile(fn)
    obj = f.read()
    T = obj.values()
    print len(T)
    print [len(Si.coords) for Si in T]
    return T

    
def removeInvalid(CL):
    """Remove the curves that contain NaN values.

    NaN values are invalid numerical values.
    This function removes the curves containing such values from a list
    of curves.
    """
    nc = len(CL)
    CL = [ Ci for Ci in CL if not isnan(Ci.coords).any() ]
    nd = len(CL)
    if nc > nd:
        print "Removed %s invalid curves, leaving %s" % (nc-nd,nd)
    return CL


def area(C,nroll=0):
    """Compute area inside spline

    The curve is supposed to be in the (x,y) plane.
    The nroll parameter may be specified to roll the coordinates
    appropriately.
    """
    print nroll
    from plugins.section2d import planeSection
    F = C.toFormex().rollAxes(nroll)
    S = planeSection(F)
    C = S.sectionChar()
    return C['A']

###############################################################

clear()
from gui.widgets import simpleInputItem as I

res = askItems([
    I('base',itemtype='vradio',choices=['Circles and Ellipses','Kinked Artery']),
    I('ncurves',value=24,text='Number of spline curves'),
    I('nu',value=36,text='Number of cells along splines'),
    I('nv',value=12,text='Number of cells across splines'),
    I('align',False),
    I('aligndir',1),
    I('alignmax',False),
    ], legacy=False)

globals().update(res)

if base == 'Circles and Ellipses':
    CL = createCircles(n=ncurves)
    nroll = 0
else:
    CL = readSplines()
    nroll = -1

ncurves = len(CL)
print "Created %s BezierSpline curves" % ncurves
CL = removeInvalid(CL)
    
areas = [ area(Ci,nroll) for Ci in CL ]
print areas
for i,a in enumerate(areas):
    if a < 0.0:
        print "Reversing curve %s" % i
        CL[i] = CL[i].reverse()

if align:
    for Ci in CL:
        alignCurvePoints(Ci,aligndir,alignmax)

draw(CL)
export({'splines':CL})
print "Number of points in the curves:",[ Ci.coords.shape[0] for Ci in CL]

PL = [Ci.approx(1) for Ci in CL]

createPL = False
if createPL:
    export({'polylines':PL})
    draw(PL,color=red)
    print "Number of points in the PolyLines:",[ Ci.coords.shape[0] for Ci in PL]


S = SplineSurface(CL)

S.createGrid(nu,nv)
draw(S.grid)
print S.grid.shape

#Cu = S.uCurves()
#Cv = S.vCurves()
#draw(Cu,color=red)
#draw(Cv,color=blue)

smoothwire()
M = S.toMesh()
draw(M,color=yellow,bkcolor='steelblue')
export({'quadsurface':M})





##############################################################################

# End
