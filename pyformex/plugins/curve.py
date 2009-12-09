#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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

"""Definition of curves in pyFormex.

This module defines classes and functions specialized for handling
one-dimensional geometry in pyFormex. These may be straight lines, polylines,
higher order curves and collections thereof. In general, the curves are 3D,
but special cases may be created for handling plane curves.
"""

# I wrote this software in my free time, for my joy, not as a commissioned task.
# Any copyright claims made by my employer should therefore be considered void.
# Acknowledgements: Gianluca De Santis

#from pyformex import debug
from numpy import *
from formex import *


##############################################################################
# THIS IS A PROPOSAL !
#
# Common interface for curves:
# attributes:
#    coords: coordinates of points defining the curve
#    parts:  number of parts (e.g. straight segments of a polyline)
#    closed: is the curve closed or not
#    range: [min,max] : range of the parameter: default 0..1
# methods:
#    subPoints(t,j): returns points with parameter values t of part j
#    points(ndiv,extend=[0.,0.]): returns points obtained by dividing each
#           part in ndiv sections at equal parameter distance.             


class Curve(object):
    """Base class for curve type classes.

    This is a virtual class intended to be subclassed.
    It defines the common definitions for all curve types.
    The subclasses should at least define the following::
    
      sub_points(t,j)
    """

    N_approx = 10
    
    def sub_points(self,t,j):
        """Return the points at values t in part j

        t can be an array of parameter values, j is a single segment number.
        """
        raise NotImplementedError

    def sub_points_2(self,t,j):
        """Return the points at values,parts given by zip(t,j)

        t and j can both be arrays, but should have the same length.
        """
        raise NotImplementedError

    def lengths(self):
        raise NotImplementedError


    def pointsAt(self,t):
        """Returns the points at parameter values t.

        Parameter values are floating point values. Their integer part
        is interpreted as the curve segment number, and the decimal part
        goes from 0 to 1 over the segment.
        """
        t = asarray(t).ravel()
        ti = floor(t).clip(min=0,max=self.nparts-1)
        t -= ti
        i = ti.astype(Int)
        try:
            allX = self.sub_points_2(t,i)
        except:
            allX = concatenate([ self.sub_points(tj,ij) for tj,ij in zip(t,i)])
        return Coords(allX)
        
    
    def subPoints(self,div=10,extend=[0., 0.]):
        """Return a series of points on the PolyLine.

        The parameter space of each segment is divided into ndiv parts.
        The coordinates of the points at these parameter values are returned
        as a Coords object.
        The extend parameter allows to extend the curve beyond the endpoints.
        The normal parameter space of each part is [0.0 .. 1.0]. The extend
        parameter will add a curve with parameter space [-extend[0] .. 0.0]
        for the first part, and a curve with parameter space
        [1.0 .. 1 + extend[0]] for the last part.
        The parameter step in the extensions will be adjusted slightly so
        that the specified extension is a multiple of the step size.
        If the curve is closed, the extend parameter is disregarded. 
        """
        # Subspline parts (without end point)
        if type(div) == int:
            u = arange(div) / float(div)

        else:
            u = array(div).ravel()
            div = len(u)
        parts = [ self.sub_points(u,j) for j in range(self.nparts) ]

        if not self.closed:
            nstart,nend = round_(asarray(extend)*div,0).astype(Int)

            # Extend at start
            if nstart > 0:
                u = arange(-nstart, 0) * extend[0] / nstart
                parts.insert(0,self.sub_points(u,0))

            # Extend at end
            if nend > 0:
                u = 1. + arange(0, nend+1) * extend[1] / nend
            else:
                # Always extend at end to include last point
                u = array([1.])
            parts.append(self.sub_points(u,self.nparts-1))

        X = concatenate(parts,axis=0)
        return Coords(X) 


    def length(self):
        """Return the total length of the curve.

        This is only available for curves that implement the 'lengths'
        method.
        """
        return self.lengths().sum()


    def approx(self,ndiv=N_approx):
        """Return an approximate PolyLine"""
        return PolyLine(self.subPoints(ndiv),closed=self.closed)


    # This allows us to draw approximations of curves that do not specify
    # their own (hopefully better) Formex representation
    def toFormex(self):
        return self.approx().toFormex()
        
  

##############################################################################
#
class PolyLine(Curve):
    """A class representing a series of straight line segments."""

    def __init__(self,coords=[],closed=False):
        """Initialize a PolyLine from a coordinate array.

        coords is a (npts,3) shaped array of coordinates of the subsequent
        vertices of the polyline (or a compatible data object).
        If closed == True, the polyline is closed by connecting the last
        point to the first. This does not change the vertex data.
        """
        coords = Coords(coords)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError,"Expected an (npoints,3) coordinate array"
        self.coords = coords
        self.nparts = self.coords.shape[0]
        if not closed:
            self.nparts -= 1
        self.closed = closed
    

    def toFormex(self):
        """Return the polyline as a Formex."""
        x = self.coords
        F = connect([x,x],bias=[0,1],loop=self.closed)
        return F

    def sub_points(self,t,j):
        """Return the points at values t in part j"""
        j = int(j)
        t = asarray(t).reshape(-1,1)
        n = self.coords.shape[0]
        X0 = self.coords[j % n]
        X1 = self.coords[(j+1) % n]
        X = (1.-t) * X0 + t * X1
        return X


    def sub_points2(self,t,j):
        """Return the points at value,part pairs (t,j)"""
        j = int(j)
        t = asarray(t).reshape(-1,1)
        n = self.coords.shape[0]
        X0 = self.coords[j % n]
        X1 = self.coords[(j+1) % n]
        X = (1.-t) * X0 + t * X1
        return X


    def vectors(self):
        """Return the vectors of the points to the next one.

        The vectors are returned as a Coords object.
        If not closed, this returns one less vectors than the number of points.
        """
        x = self.coords
        y = roll(x,-1,axis=0)
        if not self.closed:
            n = self.coords.shape[0] - 1
            x = x[:n]
            y = y[:n]
        return y-x


    def directions(self):
        """Returns unit vectors in the direction of the next point."""
        return normalize(self.vectors())


    def avgDirections(self,normalized=True):
        """Returns average directions at the inner nodes.

        If open, the number of directions returned is 2 less than the
        number of points.
        """
        d = self.directions()
        if self.closed:
            d1 = d
            d2 = roll(d,1,axis=0)
        else:
            d1 = d[:-1]
            d2 = d[1:]
        d = 0.5*(d1+d2)
        return d
    

    def lengths(self):
        """Return the length of the parts of the curve."""
        return length(self.vectors())


    def atLength(self, div):
        """Returns the parameter values for relative curve lengths div.
        
        ``div`` is a list of relative curve lengths (from 0.0 to 1.0).
        As a convenience, an single integer value may be specified,
        in which case the relative curve lengths are found by dividing
        the interval [0.0,1.0] in the specified number of subintervals.

        The function returns a list with the parameter values for the points
        at the specified relative lengths.
        """
        lens = self.lengths().cumsum()
        rlen = concatenate([[0.], lens/lens[-1]]) # relative length
        if type(div) == int:
            div = arange(div+1) / float(div)
        z = rlen.searchsorted(div)
        # we need interpolation
        wi = where(z>0)[0]
        zw = z[wi]
        L0 = rlen[zw-1]
        L1 = rlen[zw]
        ai = zw + (div[wi] - L1) / (L1-L0)
        at = zeros(len(div))
        at[wi] = ai
        return at

    def reverse(self):
        return PolyLine(reverseAxis(self.coords,axis=0),closed=self.closed)
        

##############################################################################
#
class Polygon(PolyLine):
    """A Polygon is a closed PolyLine."""

    def __init__(self,coords=[]):
        PolyLine.__init__(self,coords,closed=True)
        



##############################################################################
#

class BezierSpline(Curve):
    """A class representing a Bezier spline curve."""
    coeffs = matrix([[-1.,  3., -3., 1.],
                     [ 3., -6.,  3., 0.],
                     [-3.,  3.,  0., 0.],
                     [ 1.,  0.,  0., 0.]]
                    )

    def __init__(self,coords,deriv=None,curl=0.5,control=None,closed=False):
        """Create a cubic spline curve through the given points.

        The curve is defined by the points and the directions at these points.
        If no directions are specified, the average of the segments ending
        in that point is used, and in the end points of an open curve, the
        direction of the end segment.
        The curl parameter can be set to influence the curliness of the curve.
        curl=0.0 results in straight segments.
        
        The control points can also be specified directly. If they are, they
        override the deriv and curl parameters. Since each segment of the curve
        needs two control points, the control array has shape (npts-1, 2, 3).
        """
        coords = Coords(coords)
        self.coords = coords
        self.nparts = self.coords.shape[0]
        if not closed:
            self.nparts -= 1
            
        if control is None:
            if self.nparts < 2:
                control = self.coords
            else:
                P = PolyLine(coords,closed=closed)
                if deriv is None:
                    deriv = P.avgDirections()
                ampl = P.lengths().reshape(-1,1)
                if not closed:
                    coords = coords[1:-1]
                if not closed:
                    p1 = coords + deriv*curl*ampl[1:]
                    p2 = coords - deriv*curl*ampl[:-1]
                else:
                    p1 = coords + deriv*curl*ampl
                    p2 = coords - deriv*curl*roll(ampl,1,axis=0)
                if not closed:
                    p1 = concatenate([p2[:1],p1],axis=0)
                    p2 = concatenate([p2,p1[-1:]],axis=0)
                else:
                    p2 = roll(p2,-1,axis=0)
                control = concatenate([p1,p2],axis=1)
        control = asarray(control).reshape(-1,2,3)
        self.control = Coords(control)
        if self.control.shape != (self.nparts,2,3):
            print("coords array has shape %s" % str(self.coords.shape))
            print("control array has shape %s" % str(self.control.shape))
            raise ValueError,"Invalid control points for Bezier Spline"
        self.closed = closed


    def sub_points(self,t,j):
        j1 = (j+1) % self.coords.shape[0]
        P = self.coords[[j,j1]]
        D = self.control[j]
        P = concatenate([ P[0],D[0],D[1],P[1] ],axis=0).reshape(-1,3)
        C = self.coeffs * P
        U = column_stack([t**3., t**2., t, ones_like(t)])
        X = dot(U,C)
        return X

##############################################################################
#

class QuadBezierSpline(Curve):
    """A class representing a Bezier spline curve."""
    coeffs = matrix([[ 1., -2.,  1.],
                     [-2.,  2.,  0.],
                     [ 1.,  0.,  0.]]
                    )

    def __init__(self,coords,deriv=None,control=None,closed=False):
        """Create a cubic spline curve through the given points.

        The curve is defined by the points and the directions at these points.
        If no directions are specified, the average of the segments ending
        in that point is used, and in the end points of an open curve, the
        direction of the end segment.
        The curl parameter can be set to influence the curliness of the curve.
        curl=0.0 results in straight segments. curl=1.0 
        The control points can also be specified directly. If they are, they
        override the deriv and curl parameters. Since each segment of the curve
        needs two control points, the control array has shape (npts-1, 2, 3).
        """
        coords = Coords(coords)
        self.coords = coords
        self.nparts = self.coords.shape[0]
        if not closed:
            self.nparts -= 1
            
        if control is None:
            if self.nparts < 2:
                control = self.coords
            else:
                P = PolyLine(coords,closed=closed)
                if deriv is None:
                    deriv = P.avgDirections()
                ampl = P.lengths().reshape(-1,1)
                if not closed:
                    coords = coords[1:-1]
                if not closed:
                    p1 = coords + deriv*curl*ampl[1:]
                    p2 = coords - deriv*curl*ampl[:-1]
                else:
                    p1 = coords + deriv*curl*ampl
                    p2 = coords - deriv*curl*roll(ampl,1,axis=0)
                if not closed:
                    p1 = concatenate([p2[:1],p1],axis=0)
                    p2 = concatenate([p2,p1[-1:]],axis=0)
                else:
                    p2 = roll(p2,-1,axis=0)
                control = concatenate([p1,p2],axis=1)
        control = asarray(control).reshape(-1,3)
        self.control = Coords(control)
        if self.control.shape != (self.nparts,3):
            print("coords array has shape %s" % str(self.coords.shape))
            print("control array has shape %s" % str(self.control.shape))
            raise ValueError,"Invalid control points for Bezier Spline"
        self.closed = closed


    def sub_points(self,t,j):
        j1 = (j+1) % self.coords.shape[0]
        P = self.coords[[j,j1]]
        D = self.control[j]
        P = concatenate([ P[0],D,P[1] ],axis=0).reshape(-1,3)
        C = self.coeffs * P
        U = column_stack([t**2., t, ones_like(t)])
        X = dot(U,C)
        return X

##############################################################################
#

class CardinalSpline(BezierSpline):
    """A class representing a cardinal spline.

    Create a natural spline through the given points.

    The Cardinal Spline with given tension is a Bezier Spline with curl
    :math: `curl = ( 1 - tension) / 3`
    The separate class name is retained for compatibility and convenience. 
    See CardinalSpline2 for a direct implementation (it misses the end
    intervals of the point set).
    """

    def __init__(self,coords,tension=0.0,closed=False):
        """Create a natural spline through the given points."""
        BezierSpline.__init__(self,coords,curl=(1.-tension)/3.,closed=closed)


class CardinalSpline2(BezierSpline):
    """A class representing a cardinal spline."""

    def __init__(self,coords,tension=0.0,closed=False):
        """Create a natural spline through the given points.

        This is a direct implementation of the Cardinal Spline.
        For open curves, it misses the interpolation in the two end
        intervals of the point set.
        It is retained here because the implementation may some day
        replace the BezierSpline implementation.
        """
        coords = Coords(coords)
        self.coords = coords
        self.nparts = self.coords.shape[0]
        if not closed:
            self.nparts -= 3
        self.closed = closed
        self.tension = float(tension)
        self.compute_coefficients()


    def compute_coefficients(self):
        s = (1.-self.tension)/2.
        M = matrix([[-s, 2-s, s-2., s], [2*s, s-3., 3.-2*s, -s], [-s, 0., s, 0.], [0., 1., 0., 0.]])#pag.429 of open GL
        self.coeffs = M


    def sub_points(self,t,j):
        n = self.coords.shape[0]
        i = (j + arange(4)) % n
        P = self.coords[i]
        C = self.coeffs * P
        U = column_stack([t**3., t**2., t, ones_like(t)])
        X = dot(U,C)
        return X  


##############################################################################

class NaturalSpline(Curve):
    """A class representing a natural spline."""

    def __init__(self,coords,endcond=['notaknot','notaknot'],closed=False):
        """Create a natural spline through the given points.

        coords specifies the coordinates of a set of points. A natural spline
        is constructed through this points.
        endcond specifies the end conditions in the first, resp. last point.
        It can be 'notaknot' or 'secder'.
        With 'notaknot', maximal continuity (up to the third derivative)
        is obtained between the first two splines.
        With 'secder', the spline ends with a zero second derivative.
        If closed is True, the spline is closed, and endcond is ignored.
        """
        coords = Coords(coords)
        if closed:
            coords = Coords.concatenate([coords,coords[:1]])
        self.coords = coords
        self.nparts = self.coords.shape[0] - 1
        self.closed = closed
        self.endcond = endcond
        self.compute_coefficients()


    def compute_coefficients(self):
        x, y, z = self.coords.x(),self.coords.y(),self.coords.z()
        n = self.nparts
        M = zeros([4*n, 4*n])
        B = zeros([4*n, 3])
        
        # constant submatrix
        m = array([[0., 0., 0., 1., 0., 0., 0., 0.],
                   [1., 1., 1., 1., 0., 0., 0., 0.],
                   [3., 2., 1., 0., 0., 0.,-1., 0.],
                   [6., 2., 0., 0., 0.,-2., 0., 0.]])

        for i in range(n-1):
            f = 4*i
            M[f:f+4,f:f+8] = m
            B[f:f+2] = self.coords[i:i+2]

        # the last spline passes through the last 2 points
        f = 4*(n-1)
        M[f:f+2, f:f+4] = m[:2,:4]
        B[f:f+2] = self.coords[-2:]

        #add the appropriate end constrains
        if self.closed:
            # first and second derivatives at starting and last point
            # (that are actually the same point) are the same
            M[f+2, f:f+4] = m[2, :4]
            M[f+2, 0:4] = m[2, 4:]
            M[f+3, f:f+4] = m[3, :4]
            M[f+3, 0:4] = m[3, 4:]

        else:
            if self.endcond[0] =='notaknot':
                # third derivative is the same between the first 2 splines
                M[f+2,  [0, 4]] = array([6.,-6.])
            else:
                # second derivatives at start is zero
                M[f+2, 0:4] = m[3, 4:]

            if self.endcond[1] =='notaknot':
                # third derivative is the same between the last 2 splines
                M[f+3, [f-4, f]] = array([6.,-6.])
            else:
                # second derivatives at end is zero
                M[f+3, f:f+4] = m[3, :4]

        #calculate the coefficients
        C = linalg.solve(M,B)
        self.coeffs = array(C).reshape(-1,4,3)


    def sub_points(self,t,j):
        C = self.coeffs[j]
        U = column_stack([t**3., t**2., t, ones_like(t)])
        X = dot(U,C)
        return X

##############################################################################

from plugins.geometry import triangleCircumCircle

def vectorPairAngle(v1,v2):
    """Return the angle between the vectors v1 and v2."""
    v1 = asarray(v1)
    v2 = asarray(v2)
    cosangle = dotpr(v1,v2) / sqrt(dotpr(v1,v1)*dotpr(v2,v2))
    return arccos(cosangle)


class Arc3(Curve):
    """A class representing a circular arc."""

    def __init__(self,coords):
        """Create a circular arc.

        The arc is specified by 3 non-colinear points.
        """
        self.coords = Coords(coords)
        self.nparts = 1
        self.closed = False
        if self.coords.shape != (3,3):
            raise ValueError,"Expected 3 points"
        
        r,C,n = triangleCircumCircle(self.coords.reshape(-1,3,3))
        self.radius,self.center,self.normal = r[0],C[0],n[0]
        self.angles = vectorPairAngle(Coords([1.,0.,0.]),self.coords-self.center)
        print("Radius %s, Center %s, Normal %s" % (self.radius,self.center,self.normal))
        print("ANGLES=%s" % (self.angles))

    def sub_points(self,t,j):
        a = t*(self.angles[-1]-self.angles[0])
        X = Coords(column_stack([cos(a),sin(a),zeros_like(a)]))
        X = X.scale(self.radius).rotate(self.angles[0]/Deg).translate(self.center)
        return X


class Arc(Curve):
    """A class representing a circular arc."""

    def __init__(self,coords):
        """Create a circular arc.

        The arc is specified by the center and begin and end-point.
        """
        self.coords = Coords(coords)
        self.nparts = 1
        self.closed = False
        if self.coords.shape != (3,3):
            raise ValueError,"Expected 3 points"

        self.center = self.coords[1]
        v = self.coords-self.center
        self.radius = length(v[0])
        self.normal = unitVector(cross(v[0],v[2]))
        self.angles = [ vectorPairAngle(Coords([1.,0.,0.]),x-self.center) for x in self.coords[[0,-1]] ]
        print(self.coords)
        print("Radius %s, Center %s, Normal %s" % (self.radius,self.center,self.normal))
        print("ANGLES=%s" % (self.angles))

    def sub_points(self,t,j):
        a = t*(self.angles[-1]-self.angles[0])
        X = Coords(column_stack([cos(a),sin(a),zeros_like(a)]))
        X = X.scale(self.radius).rotate(self.angles[0]/Deg).translate(self.center)
        return X



class Spiral(Curve):
    """A class representing a spiral curve."""

    def __init__(self,turns=2.0,nparts=100,rfunc=None):
        if rfunc == None:
            rfunc = lambda x:x
        self.coords = Coords([0.,0.,0.]).replic(npoints+1).hypercylindrical()
        self.nparts = nparts
        self.closed = False

# End
