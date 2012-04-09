# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

import pyformex as pf
from coords import *
from geometry import Geometry
from formex import Formex,connect
from mesh import Mesh
from geomtools import triangleCircumCircle,intersectionTimesLWP,intersectionPointsSWP,anyPerpendicularVector
import utils

##############################################################################

class Curve(Geometry):
    """Base class for curve type classes.

    This is a virtual class intended to be subclassed.
    It defines the common definitions for all curve types.
    The subclasses should at least define the following attributes and methods
    or override them if the defaults are not suitable.

    Attributes:

    :coords: coordinates of points defining the curve
    :parts:  number of parts (e.g. straight segments of a polyline)
    :closed: is the curve closed or not
    :range: [min,max] : range of the parameter: default 0..1
    
    Methods:
    
    :sub_points(t,j): returns points at parameter value t,j
    :sub_directions(t,j): returns direction at parameter value t,j
    :pointsOn(): the defining points placed on the curve
    :pointsOff(): the defining points placeded off the curve (control points)
    :parts(j,k)    
    :approx(ndiv,ntot)

    Furthermore it may define, for efficiency reasons, the following methods:
    :sub_points_2:
    :sub_directions_2:
    
    
    """

    N_approx = 10

    def __init__(self):
        Geometry.__init__(self)
        self.prop = None

    def pointsOn(self):
        return self.coords

    def pointsOff(self):
        return Coords()

    def ncoords(self):
        return self.coords.shape[0]

    def npoints(self):
        return self.pointsOn().shape[0]

    def endPoints(self):
        """Return start and end points of the curve.

        Returns a Coords with two points, or None if the curve is closed.
        """
        if self.closed:
            #return self.coords[[0,0]]
            return None
        else:
            return self.coords[[0,-1]]
    
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
    
    def sub_directions(self,t,j):
        """Return the directions at values t in part j

        t can be an array of parameter values, j is a single segment number.
        """
        raise NotImplementedError

    def sub_directions_2(self,t,j):
        """Return the directions at values,parts given by zip(t,j)

        t and j can both be arrays, but should have the same length.
        """
        raise NotImplementedError

    def lengths(self):
        raise NotImplementedError


    def pointsAt(self,t):
        """Return the points at parameter values t.

        Parameter values are floating point values. Their integer part
        is interpreted as the curve segment number, and the decimal part
        goes from 0 to 1 over the segment.
        """
        # Do not use asarray here! We change it!
        t = array(t).ravel()
        ti = floor(t).clip(min=0,max=self.nparts-1)
        t -= ti
        i = ti.astype(Int)
        try:
            allX = self.sub_points_2(t,i)
        except:
            allX = concatenate([ self.sub_points(tj,ij) for tj,ij in zip(t,i)])
        return Coords(allX)


    def directionsAt(self,t):
        """Return the directions at parameter values t.

        Parameter values are floating point values. Their integer part
        is interpreted as the curve segment number, and the decimal part
        goes from 0 to 1 over the segment.
        """
        t = array(t).ravel()
        ti = floor(t).clip(min=0,max=self.nparts-1)
        t -= ti
        i = ti.astype(Int)
        try:
            allX = self.sub_directions_2(t,i)
        except:
            allX = concatenate([ self.sub_directions(tj,ij) for tj,ij in zip(t,i)])
        return Coords(allX)


    def subPoints(self,div=10,extend=[0., 0.]):
        """Return a sequence of points on the Curve.

        - `div`: int or a list of floats (usually in the range [0.,1.])
          If `div` is an integer, a list of floats is constructed by dividing
          the range [0.,1.] into `div` equal parts.
          The list of floats then specifies a set of parameter values for which
          points at in each part are returned. The points are returned in a
          single Coords in order of the parts.

        
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


    def split(self,split=None):
        """Split a curve into a list of partial curves

        split is a list of integer values specifying the node numbers
        where the curve is to be split. As a convenience, a single int may
        be given if the curve is to be split at a single node, or None
        to split all all nodes.

        Returns a list of open curves of the same type as the original.
        """
        if split is None:
            split = range(1,self.nparts)
        elif type(split) is int:
            split = [split]
        start = [0] + split
        end = split + [self.nparts]
        return [ self.parts(j,k) for j,k in zip(start,end) ]


    def length(self):
        """Return the total length of the curve.

        This is only available for curves that implement the 'lengths'
        method.
        """
        return self.lengths().sum()


    def approx(self,ndiv=None,ntot=None):
        """Return a PolyLine approximation of the curve

        If no `ntot` is given, the curve is approximated by `ndiv`
        straight segments over each part of the curve.
        If `ntot` is given, the curve is approximated by `ntot`
        straight segments over the total curve. This is based on a
        first approximation with ndiv segments over each part.
        """
        if ndiv is None:
            ndiv = self.N_approx
        PL = PolyLine(self.subPoints(ndiv),closed=self.closed)
        if ntot is not None:
            at = PL.atLength(ntot)
            X = PL.pointsAt(at)
            PL = PolyLine(X,closed=PL.closed)
        return PL.setProp(self.prop)


    def toFormex(self,*args,**kargs):
        """Convert a curve to a Formex.

        This creates a polyline approximation as a plex-2 Formex.
        This is mainly used for drawing curves that do not implement
        their own drawing routines.

        The method can be passed the same arguments as the `approx` method.
        """
        return self.approx(*args,**kargs).toFormex()


    def setProp(self,p=None):
        """Create or destroy the property array for the Formex.

        A property array is a rank-1 integer array with dimension equal
        to the number of elements in the Formex (first dimension of data).
        You can specify a single value or a list/array of integer values.
        If the number of passed values is less than the number of elements,
        they wil be repeated. If you give more, they will be ignored.
        
        If a value None is given, the properties are removed from the Formex.
        """
        try:
            self.prop = int(p)
        except:
            self.prop = None
        return self
  

##############################################################################
#
#  Curves that can be transformed by Coords Transforms
#
##############################################################################
#
class PolyLine(Curve):
    """A class representing a series of straight line segments.

    coords is a (npts,3) shaped array of coordinates of the subsequent
    vertices of the polyline (or a compatible data object).
    If closed == True, the polyline is closed by connecting the last
    point to the first. This does not change the vertex data.

    The `control` parameter has the same meaning as `coords` and is added
    for symmetry with other Curve classes. If specified, it will override
    the `coords` argument.
    """

    def __init__(self,coords=[],control=None,closed=False):
        """Initialize a PolyLine from a coordinate array."""
        Curve.__init__(self)
        
        if control is not None:
            coords = control
        if isinstance(coords,Formex):
            if coords.nplex() == 1:
                coords = coords.coords
            elif coords.nplex() == 2:
                coords = Coords.concatenate([coords.coords[:,0,:],coords.coords[-1,1,:]])
            else:
                raise ValueError,"Only Formices with plexitude 1 or 2 can be converted to PolyLine"

        else:
            coords = Coords(coords)
            
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 2:
            raise ValueError,"Expected an (npoints,3) shaped coordinate array with npoints >= 2, got shape " + str(coords.shape)
        self.coords = coords
        self.nparts = self.coords.shape[0]
        if not closed:
            self.nparts -= 1
        self.closed = closed


    def close(self):
        """Close a PolyLine.

        If the PolyLine is already closed, it is returned unchanged.
        Else it is closed by adding a segment from the last to the first
        point (even if these are coincident).

        ..warning :: This method changes the PolyLine inplace.
        """
        if not self.closed:
            self.closed = True
            self.parts += 1


    def open(self):
        """Open a closed PolyLine.

        If the PolyLine is closed, it is opened by removing the last segment.
        Else, it is returned unchanged.

        ..warning :: This method changes the PolyLine inplace.

        Use :meth:`split` if you want to open the PolyLine without losing
        a segment.
        """
        if self.closed:
            self.closed = False
            self.parts -= 1


    def nelems(self):
        return self.nparts
    

    def toFormex(self):
        """Return the polyline as a Formex."""
        x = self.coords
        F = connect([x,x],bias=[0,1],loop=self.closed)
        return F.setProp(self.prop)

    
    def toMesh(self):
        """Convert the polyLine to a plex-2 Mesh.

        The returned Mesh is equivalent with the PolyLine.
        """
        e1 = arange(self.ncoords())
        elems = column_stack([e1,roll(e1,-1)])
        if not self.closed:
            elems = elems[:-1]
        return Mesh(self.coords,elems,eltype='line2').setProp(self.prop)


    def sub_points(self,t,j):
        """Return the points at values t in part j"""
        j = int(j)
        t = asarray(t).reshape(-1,1)
        n = self.coords.shape[0]
        X0 = self.coords[j % n]
        X1 = self.coords[(j+1) % n]
        X = (1.-t) * X0 + t * X1
        return X


    def sub_points_2(self,t,j):
        """Return the points at value,part pairs (t,j)"""
        j = int(j)
        t = asarray(t).reshape(-1,1)
        n = self.coords.shape[0]
        X0 = self.coords[j % n]
        X1 = self.coords[(j+1) % n]
        X = (1.-t) * X0 + t * X1
        return X


    def sub_directions(self,t,j):
        """Return the unit direction vectors at values t in part j."""
        j = int(j)
        t = asarray(t).reshape(-1,1)
        return self.directions()[j].reshape(len(t),3)


    def vectors(self):
        """Return the vectors of each point to the next one.

        The vectors are returned as a Coords object.
        If the curve is not closed, the number of vectors returned is
        one less than the number of points.
        """
        x = self.coords
        if self.closed:
            x1 = x
            x2 = roll(x,-1,axis=0) # NEXT POINT
        else:
            x1 = x[:-1]
            x2 = x[1:]
        return x2-x1
        

    def directions(self,return_doubles=False):
        """Returns unit vectors in the direction of the next point.

        This directions are returned as a Coords object with the same
        number of elements as the point set.
        
        If two subsequent points are identical, the first one gets
        the direction of the previous segment. If more than two subsequent
        points are equal, an invalid direction (NaN) will result.

        If the curve is not closed, the last direction is set equal to the
        penultimate.

        If return_doubles is True, the return value is a tuple of the direction
        and an index of the points that are identical with their follower.
        """
        d = normalize(self.vectors())
        w = where(isnan(d).any(axis=-1))[0]
        d[w] = d[w-1]  
        if not self.closed:
            d = concatenate([d,d[-1:]],axis=0)
        if return_doubles:
            return d,w
        else:
            return d
    

    def avgDirections(self,return_doubles=False):
        """Returns the average directions at points.

        For each point the returned direction is the average of the direction
        from the preceding point to the current, and the direction from the
        current to the next point.
        
        If the curve is open, the first and last direction are equal to the
        direction of the first, resp. last segment.

        Where two subsequent points are identical, the average directions
        are set equal to those of the segment ending in the first and the
        segment starting from the last.
        """
        d,w = self.directions(True)
        d1 = d
        d2 = roll(d,1,axis=0) # PREVIOUS DIRECTION
        w = concatenate([w,w+1])
        if not self.closed:
            w = concatenate([[0,self.npoints()-1],w])
        w = setdiff1d(arange(self.npoints()),w)
        d[w] = 0.5 * (d1[w]+d2[w])
        if return_doubles:
            return d,w
        else:
            return d
       
        

    def lengths(self):
        """Return the length of the parts of the curve."""
        return length(self.vectors())


    def atLength(self, div):
        """Returns the parameter values for relative curve lengths div.
        
        ``div`` is a list of relative curve lengths (from 0.0 to 1.0).
        As a convenience, a single integer value may be specified,
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
        """Return the same curve with the parameter direction reversed."""
        return PolyLine(reverseAxis(self.coords,axis=0),closed=self.closed)


    def parts(self,j,k):
        """Return a PolyLine containing only segments j to k (k not included).

        The resulting PolyLine is always open.
        """
        start = j
        end = k + 1
        return PolyLine(control=self.coords[start:end],closed=False)


    def cutWithPlane(self,p,n,side=''):
        """Return the parts of the polyline at one or both sides of a plane.

        If side is '+' or '-', return a list of PolyLines with the parts at
        the positive or negative side of the plane.

        For any other value, returns a tuple of two lists of PolyLines,
        the first one being the parts at the positive side.

        p is a point specified by 3 coordinates.
        n is the normal vector to a plane, specified by 3 components.
        """
        n = asarray(n)
        p = asarray(p)

        d = self.coords.distanceFromPlane(p,n)
        t = d > 0.0
        cut = t != roll(t,-1)
        if not self.closed:
            cut = cut[:-1]
        w = where(cut)[0]

        res = [[],[]]
        i = 0
        if t[0]:
            sid = 0
        else:
            sid = 1
        Q = Coords()

        for j in w:
            #print "%s -- %s" % (i,j)
            P = intersectionPointsSWP(self.coords[j:j+2],p,n,mode='pair')[0]
            #print "%s -- %s cuts at %s" % (j,j+1,P)
            x = Coords.concatenate([Q,self.coords[i:j+1],P])
            #print "%s + %s + %s = %s" % (Q.shape[0],j-i,P.shape[0],x.shape[0])
            res[sid].append(PolyLine(x))
            sid = 1-sid
            i = j+1
            Q = P

        x = Coords.concatenate([Q,self.coords[i:]])
        #print "%s + %s = %s" % (Q.shape[0],j-i,x.shape[0])
        res[sid].append(PolyLine(x))
        if self.closed:
            if len(res[sid]) > 1:
                x = Coords.concatenate([res[sid][-1].coords,res[sid][0].coords])
                res[sid] = res[sid][1:-1]
                res[sid].append(PolyLine(x))
            #print [len(r) for r in res]
            if len(res[sid]) == 1 and len(res[1-sid]) == 0:
                res[sid][0].closed = True

        # Do not use side in '+-', because '' in '+-' returns True
        if side in ['+','-']:
            return res['+-'.index(side)]
        else:
            return res


    def append(self,PL,fuse=True,**kargs):
        """Append another PolyLine to this one.

        Returns the concatenation of two open PolyLines. Closed PolyLines
        cannot be concatenated.
        """
        if self.closed or PL.closed:
            raise RuntimeError,"Closed PolyLines cannot be concatenated."
        X = PL.coords
        if fuse:
            x = Coords.concatenate([self.coords[-1],X[0]])
            x,e = x.fuse(ppb=3) # !!! YES ! > 2 !!!
            if e[0] == e[1]:
                X = X[1:]
        return PolyLine(Coords.concatenate([self.coords,X]))


    # allow syntax PL1 + PL2
    __add__ = append


    


    # BV: I'm not sure what this does and if it belongs here
    
    def distanceOfPoints(self,p,n,return_points=False):
        """_Find the distances of points p, perpendicular to the vectors n.
        
        p is a (np,3) shaped array of points.
        n is a (np,3) shaped array of vectors.
    
        The return value is a tuple OKpid,OKdist,OKpoints where:
        - OKpid is an array with the point numbers having a distance;
        - OKdist is an array with the distances for these points;
        - OKpoints is an array with the footpoints for these points
        and is only returned if return_points = True.
        """
        q = self.pointsOn()
        if not self.closed:
            q = q[:-1]
        m = self.vectors()
        t = intersectionTimesLWP(q,m,p,n)
        t = t.transpose((1,0))
        x = q[newaxis,:] + t[:,:,newaxis] * m[newaxis,:]
        inside = (t >= 0.) * (t <= 1.)
        pid = where(inside)[0]
        p = p[pid]
        x = x[inside]
        dist = length(x-p)
        OKpid = unique(pid)
        OKdist = array([ dist[pid == i].min() for i in OKpid ])
        if return_points:
            minid = array([ dist[pid == i].argmin() for i in OKpid ])
            OKpoints = array([ x[pid == i][j] for i,j in zip(OKpid,minid) ]).reshape(-1,3)
            return OKpid,OKdist,OKpoints
        return OKpid,OKdist

    # BV: same remark: what is this distance?
    
    def distanceOfPolyLine(self,PL,ds,return_points=False):
        """_Find the distances of the PolyLine PL.
        
        PL is first discretised by calculating a set of points p and direction
        vectors n at equal distance of approximately ds along PL. The
        distance of PL is then calculated as the distances of the set (p,n).
        
        If return_points = True, two extra values are returned: an array
        with the points p and an array with the footpoints matching p.        
        """
        ntot = int(ceil(PL.length()/ds))
        t = PL.atLength(ntot)
        p = PL.pointsAt(t)
        n = PL.directionsAt(t)
        res = self.distanceOfPoints(p,n,return_points)
        if return_points:
            return res[1],p[res[0]],res[2]
        return res[1]



##############################################################################
#
class Line(PolyLine):
    """A Line is a PolyLine with exactly two points.

    Parameters:

    - `coords`: compatible with (2,3) shaped float array

    """

    def __init__(self,coords):
        """Initialize the Line."""
        PolyLine.__init__(self,coords)
        if self.coords.shape[0] != 2:
            raise ValueError, "Expected exactly two points, got %s" % coords.shape[0]


    def dxftext(self):
        return "Line(%s,%s,%s,%s,%s,%s)" % tuple(self.coords.ravel().tolist())


##############################################################################
#
class BezierSpline(Curve):
    """A class representing a Bezier spline curve of degree 1, 2 or 3.

    A Bezier spline of degree `d` is a continuous curve consisting of `nparts`
    successive parts, where each part is a Bezier curve of the same degree.
    Currently pyFormex can model linear, quadratic and cubic BezierSplines.
    A linear BezierSpline is equivalent to a PolyLine, which has more
    specialized methods than the BezierSpline, so it might be more
    sensible to use a PolyLine instead of the linear BezierSpline.

    A Bezier curve of degree `d` is determined by `d+1` control points,
    of which the first and the last are on the curve, while the intermediate
    `d-1` points are not.
    Since the end point of one part is the begin point of the next part,
    a BezierSpline is described by `ncontrol=d*nparts+1` control points if the
    curve is open, or `ncontrol=d*nparts` if the curve is closed.

    The constructor provides different ways to initialize the full set of
    control points. In many cases the off-curve control points can be
    generated automatically.

    Parameters:
    
    - `coords` : array_like (npoints,3)
      The points that are on the curve. For an open curve, npoints=nparts+1,
      for a closed curve, npoints = nparts.
      If not specified, the on-curve points should be included in the
      `control` argument.
    - `deriv` : array_like (npoints,3) or (2,3)
      If specified, it gives the direction of the curve at all points or at
      the endpoints only for a shape (2,3) array.
      For points where the direction is left unspecified or where the
      specified direction contains a `NaN` value, the direction
      is calculated as the average direction of the two
      line segments ending in the point. This will also be used
      for points where the specified direction contains a value `NaN`.
      In the two endpoints of an open curve however, this average
      direction can not be calculated: the two control points in these
      parts are set coincident.
    - `curl` : float         
      The curl parameter can be set to influence the curliness of the curve
      in between two subsequent points. A value curl=0.0 results in
      straight segments. The higher the value, the more the curve becomes
      curled.
    - `control` : array(nparts,2,3) or array(ncontrol,3)
      If `coords` was specified, this should be a (nparts,2,3) array with
      the intermediate control points, two for each part.
      
      If `coords` was not specified, this should be the full array of
      `ncontrol` control points for the curve. The number of points should
      be a multiple of 3 plus 1. If the curve is closed, the last point is
      equal to the first and does not need to a multiple of 3 is
      also allowed, in which case the first point will be appended as last.

      If not specified, the control points are generated automatically from
      the `coords`, `deriv` and `curl` arguments.
      If specified, they override these parameters.
    - `closed` : boolean
      If True, the curve will be continued from the last point back
      to the first to create a closed curve.

    - `degree`: int (1, 2 or 3)
      Specfies the degree of the curve. Default is 3.

    - `endzerocurv` : boolean or tuple of two booleans.
      Specifies the end conditions for an open curve.
      If True, the end curvature will be forced to zero. The default is
      to use maximal continuity of the curvature.
      The value may be set to a tuple of two values to specify different
      conditions for both ends.
      This argument is ignored for a closed curve.

    """
    coeffs = {
        1: matrix([
            [ 1.,  0.],
            [-1.,  1.],
            ]),
        2: matrix([
            [ 1.,  0.,  0.],
            [-2.,  2.,  0.],
            [ 1., -2.,  1.],
            ]),
        3: matrix([
            [ 1.,  0.,  0.,  0.],
            [-3.,  3.,  0.,  0.],
            [ 3., -6.,  3.,  0.],
            [-1.,  3., -3.,  1.],
            ]),
        4: matrix([
            [ 1.,  0.,  0.,  0.,  0.],
            [-4.,  4.,  0.,  0.,  0.],
            [ 6.,-12.,  6.,  0.,  0.],
            [-4., 12.,-12.,  4.,  0.],
            [ 1., -4.,  6., -4.,  1.],
            ]),
        }

    def __init__(self,coords=None,deriv=None,curl=1./3.,control=None,closed=False,degree=3,endzerocurv=False):
        """Create a BezierSpline curve."""
        Curve.__init__(self)

        if not degree > 0:
            raise ValueError,"Degree of BezierSpline should be >= 0!"

        if endzerocurv in [False,True]:
            endzerocurv = (endzerocurv,endzerocurv)

        if coords is None:
            # All control points are given, in a single array
            control = Coords(control)
            if len(control.shape) != 2 or control.shape[-1] != 3:
                raise ValueError,"If no coords argument given, the control parameter should have shape (ncontrol,3), but got %s" % str(control.shape)

            if closed:
                control = Coords.concatenate([control,control[:1]])
                
            ncontrol = control.shape[0]
            nextra = (ncontrol-1) % degree
            if nextra != 0:
                nextra = degree - nextra
                control = Coords.concatenate([control,]+[control[:1]]*nextra)
                ncontrol = control.shape[0]
            
            nparts = (ncontrol-1) // degree
                
        else:
            # Oncurve points are specified separately

            if degree > 3:
                raise ValueError,"BezierSpline of degree > 3 can only be specified by a full set of control points"

            coords = Coords(coords)
            ncoords = nparts = coords.shape[0]
            if ncoords < 2:
                raise ValueError,"Need at least two points to define a curve"
            if not closed:
                nparts -= 1

            if control is None:

                if degree == 1:
                    control = coords[:nparts]
                
                elif degree == 2:
                    if ncoords < 3:
                        control = 0.5*(coords[:1] + coords[-1:])
                        if closed:
                            control =  Coords.concatenate([control,control])
                    else:
                        if closed:
                            P0 = 0.5 * (roll(coords,1,axis=0) + roll(coords,-1,axis=0))
                            P1 = 2*coords - P0
                            Q0 = 0.5*(roll(coords,1,axis=0) + P1)
                            Q1 = 0.5*(roll(coords,-1,axis=0) + P1)
                            Q = 0.5*(roll(Q0,-1,axis=0)+Q1)
                            control = Q
                        else:
                            P0 = 0.5 * (coords[:-2] + coords[2:])
                            P1 = 2*coords[1:-1] - P0
                            Q0 = 0.5*(coords[:-2] + P1)
                            Q1 = 0.5*(coords[2:] + P1)
                            Q = 0.5*(Q0[1:]+Q1[:-1])
                            control = Coords.concatenate([Q0[:1],Q,Q1[-1:]],axis=0)

                elif degree == 3:
                    P = PolyLine(coords,closed=closed)
                    ampl = P.lengths().reshape(-1,1)
                    if deriv is None:
                        deriv = array([[nan,nan,nan]]*ncoords)
                    else:
                        deriv = Coords(deriv)
                        nderiv = deriv.shape[0]
                        if nderiv < ncoords:
                            if nderiv != 2:
                                raise ValueError,"Either all or 2 directions expected (got %s)" % nderiv
                            deriv = concatenate([
                                deriv[:1],
                                [[nan,nan,nan]]*(ncoords-2),
                                deriv[-1:]])

                    undefined = isnan(deriv).any(axis=-1)
                    if undefined.any():
                        deriv[undefined] = P.avgDirections()[undefined]

                    if closed:
                        p1 = coords + deriv*curl*ampl
                        p2 = coords - deriv*curl*roll(ampl,1,axis=0)
                        p2 = roll(p2,-1,axis=0)
                    else:
                        # Fix the first and last derivs if they were autoset
                        if undefined[0]:
                            if endzerocurv[0]:
                                # option curvature 0:
                                deriv[0] =  [nan,nan,nan]
                            else:
                                # option max. continuity
                                deriv[0] = 2.*deriv[0] - deriv[1]
                        if undefined[-1]:
                            if endzerocurv[1]:
                                # option curvature 0:
                                deriv[-1] =  [nan,nan,nan]
                            else:
                                # option max. continuity
                                deriv[-1] = 2.*deriv[-1] - deriv[-2]
                        
                        p1 = coords[:-1] + deriv[:-1]*curl*ampl
                        p2 = coords[1:] - deriv[1:]*curl*ampl
                        if isnan(p1[0]).any():
                            p1[0] = p2[0]
                        if isnan(p2[-1]).any():
                            p2[-1] = p1[-1]
                    control = concatenate([p1,p2],axis=1)

            if control is not None and degree > 1:
                try:
                    control = asarray(control).reshape(nparts,degree-1,3)
                    control = Coords(control)
                except:
                    print("coords array has shape %s" % str(coords.shape))
                    print("control array has shape %s" % str(control.shape))
                    raise ValueError,"Invalid control points for BezierSpline of degree %s" % degree

                # Join the coords and controls in a single array
                control = Coords.concatenate([coords[:nparts,newaxis,:],control],axis=1).reshape(-1,3)

            if control is not None:
                # We now have a multiple of degree coordinates, add the last:
                if closed:
                    last = 0
                else:
                    last = -1
                control = Coords.concatenate([control,coords[last]])

        self.degree = degree
        self.coeffs = BezierSpline.coeffs[degree]
        self.coords = control
        self.nparts = nparts
        self.closed = closed


    def report(self):
        return """BezierSpline: degree=%s; nparts=%s, ncoords=%s
  Control points:
%s
""" % (self.degree,self.nparts,self.coords.shape[0],self.coords)


    __repr__ = report
    __str__ = report

    def pointsOn(self):
        """Return the points on the curve.

        This returns a Coords object of shape [nparts+1]. For a closed curve,
        the last point will be equal to the first.
        """ 
        return self.coords[::self.degree]


    def pointsOff(self):
        """Return the points off the curve (the control points)

        This returns a Coords object of shape [nparts,ndegree-1], or
        an empty Coords if degree <= 1.
        """
        if self.degree > 1:
            return self.coords[:-1].reshape(-1,self.degree,3)[:,1:]
        else:
            return Coords()


    def part(self,j):
        """Returns the points defining part j of the curve."""
        start = self.degree * j
        end = start + self.degree + 1
        return self.coords[start:end]


    def sub_points(self,t,j):
        """Return the points at values t in part j."""
        P = self.part(j)
        C = self.coeffs * P        
        U = [ t**d for d in range(0,self.degree+1) ]
        U = column_stack(U)
        X = dot(U,C)
        return X


    def sub_directions(self,t,j):
        """Return the unit direction vectors at values t in part j."""
        P = self.part(j)
        C = self.coeffs * P
        U = [ d*(t**(d-1)) if d >= 1 else zeros_like(t) for d in range(0,self.degree+1) ]
        U = column_stack(U)
        T = dot(U,C)
        T = normalize(T)
        return T


    def sub_curvature(self,t,j):
        """Return the curvature at values t in part j."""
        P = self.part(j)
        C = self.coeffs * P
        U1 = [ d*(t**(d-1)) if d >= 1 else zeros_like(t) for d in range(0,self.degree+1) ]
        U1 = column_stack(U1)
        T1 = dot(U1,C)
        U2 = [ d*(d-1)*(t**(d-2)) if d >=2 else zeros_like(t) for d in range(0,self.degree+1) ]
        U2 = column_stack(U2)
        T2 = dot(U2,C)
        K = length(cross(T1,T2))/(length(T1)**3)
        return K


    def length_intgrnd(self,t,j):
        """Return the arc length integrand at value t in part j."""
        P = self.part(j)
        C = self.coeffs * P
        U = [ d*(t**(d-1)) if d >= 1 else 0. for d in range(0,self.degree+1) ]
        U = array(U)
        T = dot(U,C)
        T = array(T).reshape(3)
        return length(T)


    def lengths(self):
        """Return the length of the parts of the curve."""
        try:
            from scipy.integrate import quad
        except:
            pf.warning("""..
        
**The **lengths** function is not available.
Most likely because 'python-scipy' is not installed on your system.""")
            return
        L = [ quad(self.length_intgrnd,0.,1.,args=(j,))[0] for j in range(self.nparts) ]
        return array(L)


    def parts(self,j,k):
        """Return a curve containing only parts j to k (k not included).

        The resulting curve is always open.
        """
        start = self.degree * j
        end = self.degree * k + 1
        return BezierSpline(control=self.coords[start:end],degree=self.degree,closed=False)

    
    def toMesh(self):
        """Convert the BezierSpline to a Mesh.

        For degrees 1 or 2, the returned Mesh is equivalent with the
        BezierSpline, and will have element type 'line1', resp. 'line2'.

        For degree 3, the returned Mesh will currently be a quadratic
        approximation with element type 'line2'.
        """
        if self.degree == 1:
            return self.approx(ndiv=1).toMesh()
        else:
            coords = self.subPoints(2)
            e1 = 2*arange(len(coords)/2)
            elems = column_stack([e1,e1+1,e1+2])
            if self.closed:
                elems = elems[-1][-1] = 0
            return Mesh(coords,elems,eltype='line3')
  

        # This is not activated (yet) because it would be called for
        # drawing curves.
    ## def toFormex(self):
    ##     """Convert the BezierSpline to a Formex.

    ##     This is notational convenience for::
    ##       self.toMesh().toFormex()
    ##     """
    ##     return self.toMesh().toFormex()


    # BV: This should go as a specialization in the approx() method
    def approx_by_subdivision(self,tol=1.e-3):
        """Return a PolyLine approximation of the curve.

        tol is a tolerance value for the flatness of the curve.
        The flatness of each part is calculated as the maximum
        orthogonal distance of its intermediate control points
        from the straight segment through its end points.
        
        Parts for which the distance is larger than tol are subdivided
        using de Casteljau's algorithm. The subdivision stops
        if all parts are sufficiently flat. The return value is a PolyLine
        connecting the end points of all parts.
        """
        # get the control points (nparts,degree+1,3)
        P = array([ self.part(j) for j in range(self.nparts) ])
        T = resize(True,self.nparts)
        while T.any():
            EP = P[T][:,[0,-1]] # end points (...,2,3)
            IP = P[T][:,1:-1] # intermediate points (...,degree-1,3)
            # compute maximum orthogonal distance of IP from line EP
            q,m = EP[:,0].reshape(-1,1,3),(EP[:,1]-EP[:,0]).reshape(-1,1,3)
            t = (dotpr(IP,m) - dotpr(q,m)) / dotpr(m,m)
            X = q + t[:,:,newaxis] * m
            d = length(X-IP).max(axis=1)
            # subdivide parts for which distance is larger than tol
            T[T] *= d > tol
            if T.any():
                # apply de Casteljau's algorithm for t = 0.5
                M = [ P[T] ]
                for i in range(self.degree):
                    M.append( (M[-1][:,:-1]+M[-1][:,1:])/2 )
                # compute new control points
                Q1 = stack([ Mi[:,0] for Mi in M ],axis=1)
                Q2 = stack([ Mi[:,-1] for Mi in M[::-1] ],axis=1)
                # concatenate the parts
                P = P[~T]
                indQ = T.cumsum()-1
                indP = (1-T).cumsum()-1
                P = [ [Q1[i],Q2[i]] if Ti else [P[j]] for i,j,Ti in zip(indQ,indP,T) ]
                P = Coords(concatenate(P,axis=0))
                T = [ [Ti,Ti] if Ti else [Ti] for Ti in T ]
                T = concatenate(T,axis=0)
        # create PolyLine through end points
        P = Coords.concatenate([P[:,0],P[-1,-1]],axis=0)
        PL = PolyLine(P,closed=self.closed)
        return PL


    def extend(self,extend=[1.,1.]):
        """Extend the curve beyond its endpoints.
    
        This function will add a Bezier curve before the first part and/or
        after the last part by applying de Casteljau's algorithm on this part.
        """
        if self.closed:
            return
        if extend[0] > 0.:
            # get the control points
            P = self.part(0)
            # apply de Casteljau's algorithm
            t = -extend[0]
            M = [ P ]
            for i in range(self.degree):
                M.append( (1.-t)*M[-1][:-1] + t*M[-1][1:] )
            # compute control points
            Q = stack([ Mi[0] for Mi in M[::-1] ],axis=0)
            self.coords = Coords.concatenate([Q[:-1],self.coords])
            self.nparts += 1
        if extend[1] > 0.:
            # get the control points
            P = self.part(self.nparts-1)
            # apply de Casteljau's algorithm
            t = 1.+extend[1]
            M = [ P ]
            for i in range(self.degree):
                M.append( (1.-t)*M[-1][:-1] + t*M[-1][1:] )
            # compute control points
            Q = stack([ Mi[-1] for Mi in M ],axis=0)
            self.coords = Coords.concatenate([self.coords,Q[1:]])
            self.nparts += 1


    def reverse(self):
        """Return the same curve with the parameter direction reversed."""
        control = reverseAxis(self.coords,axis=0)
        return BezierSpline(control=control,closed=self.closed,degree=self.degree)


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

    def __init__(self,coords,tension=0.0,closed=False,endzerocurv=False):
        """Create a natural spline through the given points."""
        curl = (1.-tension)/3.
        BezierSpline.__init__(self,coords,curl=(1.-tension)/3.,closed=closed,endzerocurv=endzerocurv)


##############################################################################
#
class CardinalSpline2(Curve):
    """A class representing a cardinal spline."""

    def __init__(self,coords,tension=0.0,closed=False):
        """Create a natural spline through the given points.

        This is a direct implementation of the Cardinal Spline.
        For open curves, it misses the interpolation in the two end
        intervals of the point set.
        """
        Curve.__init__(self)
        coords = Coords(coords)
        self.coords = coords
        self.nparts = self.coords.shape[0]
        if not closed:
            self.nparts -= 3
        self.closed = closed
        self.tension = float(tension)
        s = (1.-self.tension)/2.
        self.coeffs = matrix([[-s, 2-s, s-2., s], [2*s, s-3., 3.-2*s, -s], [-s, 0., s, 0.], [0., 1., 0., 0.]])#pag.429 of open GL


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

    def __init__(self,coords,closed=False,endzerocurv=False):
        """Create a natural spline through the given points.

        coords specifies the coordinates of a set of points. A natural spline
        is constructed through this points.

        closed specifies whether the curve is closed or not.

        endzerocurv specifies the end conditions for an open curve.
        If True, the end curvature will forced to be zero. The default is
        to use maximal continuity (up to the third derivative) between
        the first two splines. The value may be set to a tuple of two
        values to specify different end conditions for both ends.
        This argument is ignored for a closed curve.
        """
        Curve.__init__(self)
        coords = Coords(coords)
        if closed:
            coords = Coords.concatenate([coords,coords[:1]])
        self.nparts = coords.shape[0] - 1
        self.closed = closed
        if not closed:
            if endzerocurv in [False,True]:
                self.endzerocurv = (endzerocurv,endzerocurv)
            else:
                self.endzerocurv = endzerocurv
        self.coords = coords
        self.compute_coefficients()


    def _set_coords(self,coords):
        C = self.copy()
        C._set_coords_inplace(coords)
        C.compute_coefficients()
        return C


    def compute_coefficients(self):
       # x, y, z = self.coords.x(),self.coords.y(),self.coords.z()
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
            if self.endzerocurv[0]:
                # second derivatives at start is zero
                M[f+2, 0:4] = m[3, 4:]
            else:
                # third derivative is the same between the first 2 splines
                M[f+2,  [0, 4]] = array([6.,-6.])
 
            if self.endzerocurv[1]:
                # second derivatives at end is zero
                M[f+3, f:f+4] = m[3, :4]
            else:
                # third derivative is the same between the last 2 splines
                M[f+3, [f-4, f]] = array([6.,-6.])
 
        #calculate the coefficients
        C = linalg.solve(M,B)
        self.coeffs = array(C).reshape(-1,4,3)


    def sub_points(self,t,j):
        C = self.coeffs[j]
        U = column_stack([t**3., t**2., t, ones_like(t)])
        X = dot(U,C)
        return X

##############################################################################


def circle():
    """Create a spline approximation of a circle.

    The returned circle lies in the x,y plane, has its center at (0,0,0)
    and has a radius 1.
    
    In the current implementation it is approximated by a bezier spline
    with curl 0.375058 through 8 points.
    """
    pts = Formex([1.0,0.0,0.0]).rosette(8,45.).coords.reshape(-1,3)
    return BezierSpline(pts,curl=0.375058,closed=True)


class Arc3(Curve):
    """A class representing a circular arc."""

    def __init__(self,coords):
        """Create a circular arc.

        The arc is specified by 3 non-colinear points.
        """
        Curve.__init__(self)
        self.nparts = 1
        self.closed = False
        self._set_coords(Coords(coords))


    def _set_coords(self,coords):
        C = self.copy()
        C._set_coords_inplace(coords)
        if self.coords.shape != (3,3):
            raise ValueError,"Expected 3 points"
        
        r,C,n = triangleCircumCircle(self.coords.reshape(-1,3,3))
        self.radius,self.center,self.normal = r[0],C[0],n[0]
        self.angles = vectorPairAngle(Coords([1.,0.,0.]),self.coords-self.center)
        print("Radius %s, Center %s, Normal %s" % (self.radius,self.center,self.normal))
        print("ANGLES=%s" % (self.angles))
        return C


    def sub_points(self,t,j):
        a = t*(self.angles[-1]-self.angles[0])
        X = Coords(column_stack([cos(a),sin(a),zeros_like(a)]))
        X = X.scale(self.radius).rotate(self.angles[0]/Deg).translate(self.center)
        return X


class Arc(Curve):
    """A class representing a circular arc.

    The arc can be specified by 3 points (begin, center, end)
    or by center, radius and two angles. In the latter case, the arc
    lies in a plane parallel to the x,y plane.
    If specified by 3 colinear points, the plane of the circle will be
    parallel to the x,y plane if the points are in such plane, else the
    plane will be parallel to the z-axis.
    """
    
    def __init__(self,coords=None,center=None,radius=None,angles=None,angle_spec=Deg):
        """Create a circular arc."""
        # Internally, we store the coords 
        Curve.__init__(self)
        self.nparts = 1
        self.closed = False
        if coords is not None:
            self.coords = Coords(coords)
            if self.coords.shape != (3,3):
                raise ValueError,"Expected 3 points"

            self._center = self.coords[1]
            v = self.coords-self._center
            self.radius = length(self.coords[0]-self.coords[1])
            try:
                self.normal = unitVector(cross(v[0],v[2]))
            except:
                pf.warning("The three points defining the Arc seem to be colinear: I will use a random orientation.")
                self.normal = anyPerpendicularVector(v[0])
            self._angles = [ vectorPairAngle(Coords([1.,0.,0.]),x-self._center) for x in self.coords[[0,2]] ]

        else:
            if center is None:
                center = [0.,0.,0.]
            if radius is None:
                radius = 1.0
            if angles is None:
                angles = (0.,360.)
            try:
                self._center = center
                self.radius = radius
                self.normal = [0.,0.,1.]
                self._angles = [ a * angle_spec for a in angles ]
                while self._angles[1] < self._angles[0]:
                    self._angles[1] += 2*pi
                while self._angles[1] > self._angles[0] + 2*pi:
                    self._angles[1] -= 2*pi
                begin,end = self.sub_points(array([0.0,1.0]),0)
                self.coords = Coords([begin,self._center,end])
            except:
                print "Invalid data for Arc"
                raise

    def getCenter(self):
        return self._center

    def getAngles(self,angle_spec=Deg):
        return (self._angles[0]/angle_spec,self._angles[1]/angle_spec)

    def getAngleRange(self,angle_spec=Deg):
        return ((self._angles[1]-self._angles[0])/angle_spec)
               


    def _set_coords(self,coords):
        """Replace the current coords with new ones.

        Returns a Mesh or subclass exactly like the current except
        for the position of the coordinates.
        """
        if isinstance(coords,Coords) and coords.shape == self.coords.shape:
            return self.__class__(coords)
        else:
            raise ValueError,"Invalid reinitialization of %s coords" % self.__class__


    def __str__(self):
        return """ARC
  Center %s, Radius %s, Normal %s
  Angles=%s
  Pt0=%s; Pt1=%s; Pt2=%s
"""  % ( self._center,self.radius,self.normal,
         self.getAngles(),
         self.coords[0],self.coords[1],self.coords[2]
       )


    def dxftext(self):
        return "Arc(%s,%s,%s,%s,%s,%s)" % tuple(list(self._center)+[self.radius]+list(self.getAngles()))


    def sub_points(self,t,j):
        a = t*(self._angles[-1]-self._angles[0])
        X = Coords(column_stack([cos(a),sin(a),zeros_like(a)]))
        X = X.scale(self.radius).rotate(self._angles[0]/Deg).translate(self._center)
        return X


    def sub_directions(self,t,j):
        a = t*(self._angles[-1]-self._angles[0])
        X = Coords(column_stack([-sin(a),cos(a),zeros_like(a)]))
        return X


    def approx(self,ndiv=None,chordal=0.001):
        """Return a PolyLine approximation of the Arc.

        Approximates the Arc by a sequence of inscribed straight line
        segments.
        
        If `ndiv` is specified, the arc is divided in pecisely `ndiv`
        segments.

        If `ndiv` is not given, the number of segments is determined
        from the chordal distance tolerance. It will guarantee that the
        distance of any point of the arc to the chordal approximation
        is less or equal than `chordal` times the radius of the arc.
        """
        if ndiv is None:
            phi = 2.*arccos(1.-chordal)
            rng = abs(self._angles[1] - self._angles[0])
            ndiv = int(ceil(rng/phi))
        return Curve.approx(self,ndiv)



class Spiral(Curve):
    """A class representing a spiral curve."""

    def __init__(self,turns=2.0,nparts=100,rfunc=None):
        Curve.__init__(self)
        if rfunc == None:
            rfunc = lambda x:x
        self.coords = Coords([0.,0.,0.]).replic(npoints+1).hypercylindrical()
        self.nparts = nparts
        self.closed = False


##############################################################################
# Other functions

def convertFormexToCurve(self,closed=False):
    """Convert a Formex to a Curve.

    The following Formices can be converted to a Curve:
    - plex 2 : to PolyLine
    - plex 3 : to BezierSpline with degree=2
    - plex 4 : to BezierSpline
    """
    points = Coords.concatenate([self.coords[:,0,:],self.coords[-1:,-1,:]],axis=0)
    if self.nplex() == 2:
        curve = PolyLine(points,closed=closed)
    elif self.nplex() == 3:
        control = self.coords[:,1,:]
        curve = BezierSpline(points,control=control,closed=closed,degree=2)
    elif self.nplex() == 4:
        control = self.coords[:,1:3,:]
        curve = BezierSpline(points,control=control,closed=closed)
    else:
        raise ValueError,"Can not convert %s-plex Formex to a Curve" % self.nplex()
    return curve

Formex.toCurve = convertFormexToCurve


##############################################################################
#
# DEPRECATED
#
class Polygon(PolyLine):
    @utils.deprecation('depr_polygon')
    def __init__(self,coords=[]):
        PolyLine.__init__(self,coords,closed=True)

    def area(self,project=None):
        from geomtools import polygonArea
        return polygonArea(self.coords,project)


class QuadBezierSpline(BezierSpline):
    @utils.deprecation('depr_quadbezier')
    def __init__(self,coords,**kargs):
        """Create a natural spline through the given points."""
        kargs['degree'] = 2
        BezierSpline.__init__(self,coords,**kargs)

# End
