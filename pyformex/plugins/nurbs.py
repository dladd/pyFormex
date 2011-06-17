#!/usr/bin/env python
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

"""Using NURBS in pyFormex.

The :mod:`nurbs` module defines functions and classes to manipulate
NURBS curves and surface in pyFormex.
"""
from coords import *
from lib import nurbs
from plugins import curve
from pyformex import options
import olist


###########################################################################
##
##   class Coords4
##
#########################
#
class Coords4(ndarray):
    """A collection of points represented by their homogeneous coordinates.
    
    While most of the pyFormex implementation is based on the 3D Cartesian
    coordinates class :class:`Coords`, some applications may benefit from using
    homogeneous coordinates. The class :class:`Coords4` provides some basic
    functions and conversion to and from cartesian coordinates.
    Through the conversion, all other pyFormex functions, such as
    transformations, are available.
    
    :class:`Coords4` is implemented as a float type :class:`numpy.ndarray`
    whose last axis has a length equal to 4.
    Each set of 4 values (x,y,z,w) along the last axis represents a
    single point in 3D space. The cartesian coordinates of the point
    are obtained by dividing the first three values by the fourth:
    (x/w, y/w, z/w). A zero w-value represents a point at infinity.
    Converting such points to :class:`Coords` will result in Inf or NaN
    values in the resulting object.
    
    The float datatype is only checked at creation time. It is the
    responsibility of the user to keep this consistent throughout the
    lifetime of the object.

    Just like :class:`Coords`, the class :class:`Coords4` is derived from
    :class:`numpy.ndarray`.

    Parameters
        
    `data`: array_like
      If specified, data should evaluate to an array of floats, with the
      length of its last axis not larger than 4. When equal to four, each
      tuple along the last axis represents a ingle point in homogeneous
      coordinates.
      If smaller than four, the last axis will be expanded to four by adding
      values zero in the second and third position and values 1 in the last
      position.
      If no data are given, a single point (0.,0.,0.) will be created.

    `w`: array_like
      If specified, the w values are used to denormalize the homogeneous
       data such that the last component becomes w.

    `dtyp`: data-type
      The datatype to be used. It not specified, the datatype of `data`
      is used, or the default :data:`Float` (which is equivalent to
      :data:`numpy.float32`).

    `copy`: boolean
      If ``True``, the data are copied. By default, the original data are
      used if possible, e.g. if a correctly shaped and typed
      :class:`numpy.ndarray` is specified.
    """
            
    def __new__(cls, data=None, w=None, dtyp=Float, copy=False):
        """Create a new instance of :class:`Coords4`."""
        if data is None:
            # create an empty array
            ar = ndarray((0,4),dtype=dtyp)
        else:
            # turn the data into an array, and copy if requested
            ar = array(data, dtype=dtyp, copy=copy)
            
        if ar.shape[-1] in [3,4]:
            pass
        elif ar.shape[-1] in [1,2]:
            # make last axis length 3, adding 0 values
            ar = growAxis(ar,3-ar.shape[-1],-1)
        elif ar.shape[-1] == 0:
            # allow empty coords objects 
            ar = ar.reshape(0,3)
        else:
            raise ValueError,"Expected a length 1,2,3 or 4 for last array axis"

        # Make sure dtype is a float type
        if ar.dtype.kind != 'f':
            ar = ar.astype(Float)

        # We should now have a float array with last axis 3 or 4
        if ar.shape[-1] == 3:
            # Expand last axis to length 4, adding values 1
            ar = growAxis(ar,1,-1,1.0)

        # Denormalize if w is specified
        if w is not None:
            self.deNormalize(w)
 
        # Transform 'subarr' from an ndarray to our new subclass.
        ar = ar.view(cls)

        return ar


    def normalize(self):
        """Normalize the homogeneous coordinates.

        Two sets of homogeneous coordinates that differ only by a
        multiplicative constant refer to the same points in cartesian space.
        Normalization of the coordinates is a way to make the representation
        of a single point unique. Normalization is done so that the last
        component (w) is equal to 1.

        The normalization of the coordinates is done in place.

        .. warning:: Normalizing points at infinity will result in Inf or
           NaN values.
        """
        self /= self[...,3:]


    def deNormalize(self,w):
        """Denormalizes the homogeneous coordinates.

        This multiplies the homogeneous coordinates with the values w.
        w normally is a constant or an array with shape
        self.shape[:-1] + (1,).
        It then multiplies all 4 coordinates of a point with the same
        value, thus resulting in a denormalization while keeping the
        position of the point unchanged.

        The denormalization of the coordinates is done in place.
        If the Coords4 object was normalized, it will have precisely w as
        its 4-th coordinate value after the call.
        """
        self *= w


    def toCoords(self):
        """Convert homogeneous coordinates to cartesian coordinates.

        Returns a :class:`Coords` object with the cartesian coordinates
        of the points. Points at infinity (w=0) will result in
        Inf or NaN value. If there are no points at infinity, the
        resulting :class:`Coords` point set is equivalent to the
        :class:`Coords4` one.
        """
        return Coords(self[...,:3] / self[...,3:])


    def npoints(self):
        """Return the total number of points."""
        return asarray(self.shape[:-1]).prod()


    ncoords = npoints


    def x(self):
        """Return the x-plane"""
        return self[...,0]
    def y(self):
        """Return the y-plane"""
        return self[...,1]
    def z(self):
        """Return the z-plane"""
        return self[...,2]
    def w(self):
        """Return the w-plane"""
        return self[...,3]

    
    def bbox(self):
        """Return the bounding box of a set of points.

        Returns the bounding box of the cartesian coordinates of
        the object.
        """
        return self.toCoords().bbox()


    def actor(self,**kargs):
        """Graphical representation"""
        return self.toCoords().actor()


class Geometry4(object):
    def scale(self,*args,**kargs):
        self.coords[...,:3] = Coords(self.coords[...,:3]).scale(*args,**kargs)
        return self
    

#######################################################
## NURBS CURVES ##

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

# This should be a valid combination of ntrl/degree
# drawing is only done if degree <= 7


class NurbsCurve(Geometry4):

    """A NURBS curve

    The Nurbs curve is defined by nctrl control points, a degree (>= 1) and
    a knot vector with knots = nctrl+degree+1 parameter values.

    The knots vector should hold nknots values in ascending order. The values
    are only defined upon a multiplicative constant and will be normalized
    to set the last value to 1.
    Sensible default values are constructed automatically by calling
    :func:`knotVector`. 

    If no knots are given and no degree is specified, the degree is set to
    the number of control points - 1 if the curve is blended. If not blended,
    the degree is not set larger than 3.

    """
#    
#    order (2,3,4,...) = degree+1 = min. number of control points
#    ncontrol >= order
#    nknots = order + ncontrol >= 2*order
#
#    convenient solutions:
#    OPEN:
#      nparts = (ncontrol-1) / degree
#      nintern = 
#    
    def __init__(self,control,degree=None,wts=None,knots=None,closed=False,blended=True):
        self.closed = closed
        nctrl = len(control)
        
        if degree is None:
            if knots is None:
                degree = nctrl-1
                if not blended:
                    degree = min(degree,3)
            else:
                degree = len(knots) - nctrl -1
                if degree <= 0:
                    raise ValueError,"Length of knot vector (%s) must be at least number of control points (%s) plus 2" % (len(knots),nctrl)

        order = degree+1
        control = Coords4(control)
        if wts is not None:
            control.deNormalize(wts.reshape(wts.shape[-1],1))

        if closed:
            if knots is None:
                nextra = degree
            else:
                nextra = len(knots) - nctrl - order
            nextra1 = (nextra+1) // 2
            #nextra1 = 0
            nextra2 = nextra-nextra1
            print "extra %s = %s + %s" % (nextra,nextra1,nextra2)
            control = Coords4(concatenate([control[-nextra1:],control,control[:nextra2]],axis=0))

        nctrl = control.shape[0]

        if nctrl < order:
            raise ValueError,"Number of control points (%s) must not be smaller than order (%s)" % (nctrl,order)

        if knots is None:
            knots = knotVector(nctrl,degree,blended=blended,closed=closed)
        else:
            knots = asarray(knots).ravel()
            knots = knots / knots[-1]

        nknots = knots.shape[0]
        
        if nknots != nctrl+order:
            raise ValueError,"Length of knot vector (%s) must be equal to number of control points (%s) plus order (%s)" % (nknots,nctrl,order)

       
        self.coords = control
        self.knots = knots
        self.degree = degree
        self.closed = closed


    def order(self):
        return len(self.knots)-len(self.coords)
        
    def bbox(self):
        """Return the bounding box of the NURBS curve.

        """
        return self.coords.toCoords().bbox()


    def pointsAt(self,u):
        """Return the points on the Nurbs curve at given parametric values.

        Parameters:

        `u`: (nu,) shaped float array: parametric values at which a point
          is to be placed.

        Returns: (nu,3) shaped Coords with nu points at the specified
        parametric values.
        
        """
        ctrl = self.coords.astype(double)
        knots = self.knots.astype(double)
        u = asarray(u).astype(double)

        try:
            pts = nurbs.curvePoints(ctrl,knots,u)
            if isnan(pts).any():
                print "We got a NaN"
                raise RuntimeError
        except:
            raise RuntimeError,"Some error occurred during the evaluation of the Nurbs curve"

        if pts.shape[-1] == 4:
            pts = Coords4(pts).toCoords()
        else:
            pts = Coords(pts)
        return pts

    
    def derivatives(self,at,d=1):
        """Returns the points and derivatives up to d at parameter values at"""
        if type(at) is int:
            u = uniformParamValues(at,self.knots[0],self.knots[-1])
        else:
            u = at
            
        # sanitize arguments for library call
        ctrl = self.coords.astype(double)
        knots = self.knots.astype(double)
        u = asarray(u).astype(double)
        d = int(d)
        
        try:
            pts = nurbs.curveDerivs(ctrl,knots,u,d)
            if isnan(pts).any():
                print "We got a NaN"
                print pts
                raise RuntimeError
        except:
            raise RuntimeError,"Some error occurred during the evaluation of the Nurbs curve"

        if pts.shape[-1] == 4:
            pts = Coords4(pts)
            # When using no weights, ctrl points are Coords4 normalized points,
            # and the derivatives all have w=0: the points represent directions
            # We just strip off the w=0.
            # HOWEVER, if there are weights, not sure what to do.
            # Points themselves could be just normalized and returned.
            pts[0].normalize()
            pts = Coords(pts[...,:3])
        else:
            pts = Coords(pts)
        return pts


    def knotPoints(self):
        """Returns the points at the knot values.

        The multiplicity of the knots is retained in the points set.
        """
        return self.pointsAt(self.knots)


    def insertKnots(self,u):
        """Insert a set of knots in the curve.

        u is a vector with knot parameter values to be inserted into the
        curve. The control points are adapted to keep the curve unchanged.

        Returns a Nurbs curve equivalent with the original but with the
        specified knot values inserted in the knot vector, and the control
        points adapted.
        """
        if self.closed:
            raise ValueError,"insertKnots currently does not work on closed curves"
        newP,newU = nurbs.curveKnotRefine(self.coords,self.knots,u)
        return NurbsCurve(newP,degree=self.degree,knots=newU,closed=self.closed)


    def decompose(self):
        """Decomposes a curve in subsequent Bezier curves.

        Returns an equivalent unblended Nurbs.
        """
        X = nurbs.curveDecompose(self.coords,self.knots)
        return NurbsCurve(X,degree=self.degree,blended=False)

        

    def actor(self,**kargs):
        """Graphical representation"""
        from gui.actors import NurbsActor
        return NurbsActor(self,**kargs)
    

#######################################################
## NURBS Surface ##


class NurbsSurface(Geometry4):

    """A NURBS surface

    The Nurbs surface is defined as a tensor product of NURBS curves in two
    parametrical directions u and v. The control points form a grid of
    (nctrlu,nctrlv) points. The other data are like those for a NURBS curve,
    but need to be specified as a tuple for the (u,v) directions.

    The knot values are only defined upon a multiplicative constant, equal to
    the largest value. Sensible default values are constructed automatically
    by a call to the knotVector() function.

    If no knots are given and no degree is specified, the degree is set to
    the number of control points - 1 if the curve is blended. If not blended,
    the degree is not set larger than 3.

    .. warning:: This is a class under development!

    """
    
    def __init__(self,control,degree=(None,None),wts=None,knots=(None,None),closed=(False,False),blended=(True,True)):

        self.closed = closed

        control = Coords4(control)
        if wts is not None:
            control.deNormalize(wts.reshape(wts.shape[-1],1))

        for d in range(2):
            nctrl = control.shape[1-d] # BEWARE! the order of the nodes
            deg = degree[d]
            kn = knots[d]
            bl = blended[d]
            cl = closed[d]
            
            if deg is None:
                if kn is None:
                    deg = nctrl-1
                    if not bl:
                        deg = min(deg,3)
                else:
                    deg = len(kn) - nctrl -1
                    if deg <= 0:
                        raise ValueError,"Length of knot vector (%s) must be at least number of control points (%s) plus 2" % (len(knots),nctrl)

            order = deg+1

            if nctrl < order:
                raise ValueError,"Number of control points (%s) must not be smaller than order (%s)" % (nctrl,order)

            if kn is None:
                kn = knotVector(nctrl,deg,blended=bl,closed=cl)
            else:
                kn = asarray(kn).ravel()

            nknots = kn.shape[0]

            if nknots != nctrl+order:
                raise ValueError,"Length of knot vector (%s) must be equal to number of control points (%s) plus order (%s)" % (nknots,nctrl,order)

            if d == 0:
                self.uknots = kn
            else:
                self.vknots = kn
                
        self.coords = control
        self.degree = degree
        self.closed = closed


    def order(self):
        return (self.uknots.shape[0]-self.coords.shape[1],
                self.vknots.shape[0]-self.coords.shape[0])
        
    def bbox(self):
        """Return the bounding box of the NURBS surface.

        """
        return self.coords.toCoords().bbox()


    def pointsAt(self,u):
        """Return the points on the Nurbs surface at given parametric values.

        Parameters:

        `u`: (nu,2) shaped float array: `nu` parametric values (u,v) at which
          a point is to be placed.

        Returns: (nu,3) shaped Coords with `nu` points at the specified
        parametric values.
        
        """
        ctrl = self.coords.astype(double)
        U = self.vknots.astype(double)
        V = self.uknots.astype(double)
        u = asarray(u).astype(double)

        try:
            pts = nurbs.surfacePoints(ctrl,U,V,u)
            if isnan(pts).any():
                print "We got a NaN"
                raise RuntimeError
        except:
            raise RuntimeError,"Some error occurred during the evaluation of the Nurbs curve"

        if pts.shape[-1] == 4:
            pts = Coords4(pts).toCoords()
        else:
            pts = Coords(pts)
        return pts
        

    def actor(self,**kargs):
        """Graphical representation"""
        from gui.actors import NurbsActor
        return NurbsActor(self,**kargs)


################################################################


def globalInterpolationCurve(Q,degree=3,strategy=0.5):
    """Create a global interpolation NurbsCurve.

    Given an ordered set of points Q, the globalInterpolationCurve
    is a NURBS curve of the given degree, passing through all the
    points. 

    Returns a NurbsCurve through the given point set. The number of
    control points is the same as the number of input points.

    ..warning: Currently there is the limitation that two consecutive
      points should not coincide. If they do, a warning is shown and
      the double points will be removed.

    The procedure works by computing the control points that will
    produce a NurbsCurve with the given points occurring at predefined
    parameter values. The strategy to set this values uses a parameter
    as exponent. Different values produce (slighly) different curves.
    Typical values are:

    0.0: equally spaced (not recommended)
    0.5: centripetal (default, recommended)
    1.0: chord length (often used)
    """
    from plugins.curve import PolyLine
    # set the knot values at the points
    nc = Q.shape[0]
    n = nc-1

    # chord length
    d = PolyLine(Q).lengths()
    if (d==0.0).any():
        utils.warn("Your point set appears to contain double points. Currently I cannot handle that. I will skip the doubles and try to go ahead.")
        Q = concatenate([Q[d!=0.0],Q[-1:]],axis=0)
        d = PolyLine(Q).lengths()
        if (d==0.0).any():
            raise ValueError,"Double points in the data set are not allowed"
    # apply strategy
    d = d ** strategy
    d = d.cumsum()
    d /= d[-1]
    u = concatenate([[0.], d])
    #print "u = ",u
    U,A = nurbs.curveGlobalInterpolationMatrix(Q,u,degree)
    #print "U = ",U
    #print "A = ",A
    P = linalg.solve(A,Q)
    #print "P = ",P
    return NurbsCurve(P,knots=U,degree=degree)
    

def uniformParamValues(n,umin=0.0,umax=1.0):
    """Create a set of uniformly distributed parameter values in a range.

    Parameters:

    `n`: int: number of intervals in which the range should be divided.
      The number of values returned is ``n+1``.
    `umin`,`umax`: float: start and end value of the interval. Default
      interval is [0.0..1.0].

    Returns: a float array with n+1 equidistant values in the range umin..umax.
      For n > 0, both of the endpoints are included. For n=0, a single
      value at the center of the interval will be returned. For n<0, an
      empty array is returned.
    
    Example:
    
    >>> uniformParamValues(4).tolist()
    [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> uniformParamValues(0).tolist()
    [0.5]
    >>> uniformParamValues(-1).tolist()
    []
    >>> uniformParamValues(2,1.5,2.5).tolist()
    [1.5, 2.0, 2.5]
    """
    if n == 0:
        return array([0.5*(umax+umin)])
    else:
        return umin + arange(n+1) * (umax-umin) / n


def knotVector(nctrl,degree,blended=True,closed=False):
    """Compute sensible knot vector for a Nurbs curve.

    A knot vector is a sequence of non-decreasing parametric values. These
    values define the `knots`, i.e. the points where the analytical expression
    of the Nurbs curve may change. The knot values are only meaningful upon a
    multiplicative constant, and they are usually normalized to the range
    [0.0..1.0].

    A Nurbs curve with ``nctrl`` points and of given ``degree`` needs a knot
    vector with ``nknots = nctrl+degree+1`` values. A ``degree`` curve needs
    at least ``nctrl = degree+1`` control points, and thus at least
    ``nknots = 2*(degree+1)`` knot values.

    To make an open curve start and end in its end points, it needs knots with
    multiplicity ``degree+1`` at its ends. Thus, for an open blended curve, the
    default policy is to set the knot values at the ends to 0.0, resp. 1.0,
    both with multiplicity ``degree+1``, and to spread the remaining
    ``nctrl - degree - 1`` values equally over the interval.

    For a closed (blended) curve, the knots are equally spread over the
    interval, all having a multiplicity 1 for maximum continuity of the curve.

    For an open unblended curve, all internal knots get multiplicity ``degree``.
    This results in a curve that is only one time continuously derivable at
    the knots, thus the curve is smooth, but the curvature may be discontinuous.
    There is an extra requirement in this case: ``nctrl`` sohuld be a multiple
    of ``degree`` plus 1.
    
    Example:

    >>> print knotVector(7,3)
    [ 0.    0.    0.    0.    0.25  0.5   0.75  1.    1.    1.    1.  ]
    >>> print knotVector(7,3,closed=True)
    [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]
    >>> print knotVector(7,3,blended=False)
    [ 0.  0.  0.  0.  1.  1.  1.  2.  2.  2.  2.]
    """
    nknots = nctrl+degree+1
    if closed:
        knots = uniformParamValues(nknots-1)
    else:
        if blended:
            npts = nknots - 2*degree
            knots = [0.]*degree +  uniformParamValues(npts-1).tolist() + [1.]*degree
        else:
            nparts = (nctrl-1) / degree
            if nparts*degree+1 != nctrl:
                raise ValueError,"Discrete knot vectors can only be used if the number of control points is a multiple of the degree, plus one."
            knots = [0.] + [ [float(i)]*degree for i in range(nparts+1) ] + [float(nparts)]
            knots = olist.flatten(knots)

    return asarray(knots)


def toCoords4(x):
    """Convert cartesian coordinates to homogeneous

    `x`: :class:`Coords`
      Array with cartesian coordinates.
       
    Returns a Coords4 object corresponding to the input cartesian coordinates.
    """
    return Coords4(x)

Coords.toCoords4 = toCoords4


def pointsOnBezierCurve(P,u):
    """Compute points on a Bezier curve

    Parameters:
    P is an array with n+1 points defining a Bezier curve of degree n.
    u is a vector with nu parameter values between 0 and 1.

    Returns:
    An array with the nu points of the Bezier curve corresponding with the
    specified parametric values.
    ERROR: currently u is a single paramtric value!

    See also:
    examples BezierCurve, Casteljou
    """
    u = asarray(u).ravel()
    n = P.shape[0]-1
    return Coords.concatenate([
        (nurbs.allBernstein(n,ui).reshape(1,-1,1) * P).sum(axis=1)
        for ui in u ],axis=0)


def deCasteljou(P,u):
    """Compute points on a Bezier curve using deCasteljou algorithm

    Parameters:
    P is an array with n+1 points defining a Bezier curve of degree n.
    u is a single parameter value between 0 and 1.

    Returns:
    A list with point sets obtained in the subsequent deCasteljou
    approximations. The first one is the set of control points, the last one
    is the point on the Bezier curve.

    This function works with Coords as well as Coords4 points. 
    """
    n = P.shape[0]-1
    C = [P]
    for k in range(n):
        Q = C[-1]
        Q = (1.-u) * Q[:-1] + u * Q[1:]
        C.append(Q)
    return C


def curveToNurbs(B):
    """Convert a BezierSpline to NurbsCurve"""
    return NurbsCurve(B.coords,degree=B.degree,closed=B.closed,blended=False)

curve.BezierSpline.toNurbs = curveToNurbs
    

def frenet(d1,d2,d3=None):
    """Returns the 3 Frenet vectors and the curvature.

    d1,d2 are the first and second derivatives at points of a curve.
    d3, if given, is the third derivative. 
    Returns 3 normalized vectors along tangent, normal, binormal
    plus the curvature.
    if d3 is give, also returns the torsion of the curve.
    """
    l = length(d1)
    # What to do when l is 0? same as with k?
    if l.min() == 0.0:
        print "l is zero at %s" % where(l==0.0)[0]
    e1 = d1 / l.reshape(-1,1)
    e2 = d2 - dotpr(d2,e1).reshape(-1,1)*e1
    k = length(e2)
    if k.min() == 0.0:
        print "k is zero at %s" % where(k==0.0)[0]
    w = where(k==0.0)[0]
    # where k = 0: set e2 to mean of previous and following
    e2 /= k.reshape(-1,1)
    #e3 = normalize(ddd - dotpr(ddd,e1)*e1 - dotpr(ddd,e2)*e2)
    e3 = cross(e1,e2)
    m = dotpr(cross(d1,d2),e3)
    #print "m",m
    k = m / l**3
    if d3 is None:
        return e1,e2,e3,k
    # compute torsion    
    t = dotpr(d1,cross(d2,d3)) / dotpr(d1,d2)
    return e1,e2,e3,k,t


### End
