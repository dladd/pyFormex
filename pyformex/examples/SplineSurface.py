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
        if nv is None:
            nv = self.curves[0].nparts()

        CA = [ C.approx(ntot=nu) for C in self.curves ]
        print CA[0].coords.shape
        print len(CA)
        self.grid = stack([CAi.coords for CAi in CA])
        print self.grid.shape


    def uCurves(self):
        return [ BezierSpline(self.grid[:,i,:],curl=0.375) for i in range(self.grid.shape[1]) ]


    def vCurves(self):
        return [ BezierSpline(self.grid[i,:,:],curl=0.375) for i in range(self.grid.shape[0]) ]


    def toMesh(self):
        """Convert the Grid Surface to a Quad Mesh"""
        nu = self.grid.shape[1] -1
        nv = self.grid.shape[0] -1 
        elems = array([[ 0,1,nu+2,nu+1 ]])
        elems = concatenate([(elems+i) for i in range(nu)],axis=0)
        elems = concatenate([(elems+i*(nu+1)) for i in range(nv)],axis=0)

        x = self.grid.reshape(-1,3)
        print nu,nv
        print x.shape
        print elems.shape
        print elems.min(),elems.max()
        M = Mesh(self.grid.reshape(-1,3),elems)
        print M.elems
        return M
    

    def actor(self,**kargs):
        return [ draw(c,**kargs) for c in self.curves ] 

    

clear()
C = circle()
n = 10
m = 36
t = arange(n+1) /float(n)
CL = [ C.scale([1.,a,0.]) for a in 0.5 + arange(n+1) /float(n) ]
CL = [ Ci.trl(2,a) for Ci,a in zip(CL,arange(n+1)/float(n)*2) ]

S = SplineSurface(CL)

S.createGrid(m,n)

#Cu = S.uCurves()
#Cv = S.vCurves()
#draw(Cu,color=red)
#draw(Cv,color=blue)

smoothwire()
M = S.toMesh()
draw(M,color=yellow)
export({'quadsurface':M})


##############################################################################

# End
