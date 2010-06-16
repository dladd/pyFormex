#!/usr/bin/env pyformex
# $Id$
"""SplineSurface

level = 'advanced'
topics = ['geometry','surface']
techniques = ['spline']

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
    print "ALIGNING ON POINT %s"
    print curve.coords
    rollCurvePoints(curve,-ind)
    print curve.coords
    

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


from geomfile import GeometryFile

## S = named('splines')
## print len(S)
## print [len(Si.coords) for Si in S]
## #exit()
## ## S = named('splines-000')
## clear()
## draw(S)
## print os.getcwd()
## G = GeometryFile('splines.pgf','w')
## G.write(S,'splines')
## G.reopen()
## obj = G.read()
## T = obj.values()
## print len(T)
## print [len(Si.coords) for Si in T]
## draw(T,color=red)

## exit()

import os


fn = getcfg('datadir')+'/horse.pgf'
G = GeometryFile(fn,'r')
G.convert()
exit()

clear()

def createCircles(n=4):
    C = circle()
    t = arange(n+1) /float(n)
    CL = [ C.scale([1.,a,0.]) for a in 0.5 + arange(n+1) /float(n) ]
    CL = [ Ci.trl(2,a) for Ci,a in zip(CL,arange(n+1)/float(n)*4.) ]
    CL = [ Ci.rot(a,2) for Ci,a in zip(CL,arange(n+1)/float(n)*45.) ]

def readSplines():
    fn = getcfg('datadir')+'/splines.pgf'
    f = GeometryFile(fn)
    obj = f.read()
    T = obj.values()
    print len(T)
    print [len(Si.coords) for Si in T]
    return T


    m = 36

CL = named('splines')
print len(CL)

print isnan(CL[0].coords).any()

CL = [ Ci for Ci in CL if not isnan(Ci.coords).any() ]
print len(CL)

def area(C):
    """Compute area inside spline"""
    from plugins.section2d import planeSection
    F = C.toFormex().rollAxes(-1)
    S = planeSection(F)
    C = S.sectionChar()
    return C['A']
    
areas = [ area(Ci) for Ci in CL ]
for i,a in enumerate(areas):
    if a < 0.0:
        print "Reversing section %s" % i
        CL[i] = CL[i].reverse()

draw(CL)

for Ci in CL:
    alignCurvePoints(Ci)

PL = [Ci.approx(1) for Ci in CL]

export({'polylines':PL,'splines':CL})
draw(PL,color=red)

print [ Ci.coords.shape[0] for Ci in CL]
print [ Ci.coords.shape[0] for Ci in PL]


S = SplineSurface(CL)

S.createGrid(m,n)
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
