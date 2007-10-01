#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.5 Release Fri Aug 10 12:04:07 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Coordinates of points in 3D space"""

from numpy import *


# default float and int types
Float = float32
Int = int32

def istype(a,c):
    return asarray(a).dtype.kind == c


if 'roll' not in dir():
    def roll(a, shift, axis=None): 
        """Roll the elements in the array by 'shift' positions along 
        the given axis.

        A positive shift moves elements to the 'right' in a 1D array.
        """ 
        a = asarray(a) 
        if axis is None: 
            n = a.size 
            reshape=1 
        else: 
            n = a.shape[axis] 
            reshape=0 
        shift %= n 
        indexes = concatenate((arange(n-shift,n),arange(n-shift))) 
        res = a.take(indexes, axis) 
        if reshape: 
            return res.reshape(a.shape) 
        else: 
            return res


###########################################################################
##
##   some math functions
##
#########################

# pi is defined in numpy
# rad is a multiplier to transform degrees to radians
rad = pi/180.

# Convenience functions: trigonometric functions with argument in degrees
# Should we keep this in ???

def sind(arg):
    """Return the sin of an angle in degrees."""
    return sin(arg*rad)

def cosd(arg):
    """Return the cos of an angle in degrees."""
    return cos(arg*rad)

def tand(arg):
    """Return the tan of an angle in degrees."""
    return tan(arg*rad)

def length(arg):
    """Return the quadratic norm of a vector with all elements of arg."""
    a = asarray(arg).flat
    return sqrt(inner(a,a))   # a*a doesn't work here

def norm(v,n=2):
    """Return a norm of the vector v.

    Default is the quadratic norm (vector length)
    n == 1 returns the sum
    n <= 0 returns the max absolute value
    """
    a = asarray(v).flat
    if n == 2:
        return sqrt((a*a).sum())
    if n > 2:
        return (a**n).sum()**(1./n)
    if n == 1:
        return a.sum()
    if n <= 0:
        return abs(a).max()
    return

def inside(p,mi,ma):
    """Return true if point p is inside bbox defined by points mi and ma"""
    return p[0] >= mi[0] and p[1] >= mi[1] and p[2] >= mi[2] and \
           p[0] <= ma[0] and p[1] <= ma[1] and p[2] <= ma[2]


def isClose(values,target,rtol=1.e-5,atol=1.e-8):
    """Returns an array flagging the elements close to target.

    values is a float array, target is a float value.
    values and target should be broadcastable to the same shape.
    
    The return value is a boolean array with shape of values flagging
    where the values are close to target.
    Two values a and b  are considered close if
        | a - b | < atol + rtol * | b |
    """
    values = asarray(values)
    target = asarray(target) 
    return abs(values - target) < atol + rtol * abs(target) 

def translationVector(dir,dist):
    """Return a translation vector in direction dir over distance dist"""
    f = [0.,0.,0.]
    f[dir] = dist
    return f

def rotationMatrix(angle,axis=None):
    """Return a rotation matrix over angle, optionally around axis.

    The angle is specified in degrees.
    If axis==None (default), a 2x2 rotation matrix is returned.
    Else, axis should specifying the rotation axis in a 3D world. It is either
    one of 0,1,2, specifying a global axis, or a vector with 3 components
    specifying an axis through the origin.
    In either case a 3x3 rotation matrix is returned.
    Note that:
      rotationMatrix(angle,[1,0,0]) == rotationMatrix(angle,0) 
      rotationMatrix(angle,[0,1,0]) == rotationMatrix(angle,1) 
      rotationMatrix(angle,[0,0,1]) == rotationMatrix(angle,2)
    but the latter functions calls are more efficient.
    The result is returned as an array.
    """
    a = angle*rad
    c = cos(a)
    s = sin(a)
    if axis==None:
        f = [[c,s],[-s,c]]
    elif type(axis) == int:
        f = [[0.0 for i in range(3)] for j in range(3)]
        axes = range(3)
        i,j,k = axes[axis:]+axes[:axis]
        f[i][i] = 1.0
        f[j][j] = c
        f[j][k] = s
        f[k][j] = -s
        f[k][k] = c
    else:
        t = 1-c
        X,Y,Z = axis
        f = [ [ t*X*X + c  , t*X*Y + s*Z, t*X*Z - s*Y ],
              [ t*Y*X - s*Z, t*Y*Y + c  , t*Y*Z + s*X ],
              [ t*Z*X + s*Y, t*Z*Y - s*X, t*Z*Z + c   ] ]
        
    return array(f)


###########################################################################
##
##   class Coords
##
#########################
#
class Coords(ndarray):
    """A Coords object is a collection of 3D coordinates of points.

    Coords is implemented as a Numerical Python array with a length of its
    last axis equal to 3.
    The datatype should be a float type; default is Float.
    !! These restrictions are currently only check at creation time.
    !! It is the responsibility of the user to keep consistency. 
    Each set of 3 values along the last axis represents a single point in 3D.
    """

    # !! DO WE NEED AN EMPTY Coords  OBJECT?
    # I guess not, so we made the default constructor generate a single
    # point [0.,0.,0.]

            
    def __new__(cls, data=None, dtyp=None, copy=False):
        """Create a new instance of class Coords.

        If specified, data should evaluate to an (...,3) shaped array of floats.
        If copy==True, the data are copied.
        If no dtype is given that of data are used, or float32 by default.
        """
        if data is None:
            data = zeros((3,))

        # Turn the data into an array, and copy if requested
        ar = array(data, dtype=dtyp, copy=copy)
        if ar.shape[-1] != 3:
            raise ValueError,"Expected a length = 3 for last array axis"


        # Make sure dtype is a float type
        if ar.dtype.kind != 'f':
            ar = ar.astype(Float)
 
        # Transform 'subarr' from an ndarray to our new subclass.
        ar = ar.view(cls)

        return ar


##    def __array_finalize__(self,obj):
##         #Make sure array shape is (n,3) float
##         print "SHAPE = %s" % str(self.shape)
##         print "DTYPE = %s" % str(self.dtype)
##        if self.shape[-1] != 3:
##            print 'Expected shape (n,3)'
##         if self.dtype.kind != 'f':
##             raise ValueError,"Expected a floating point type."
##         if len(self.shape) != 2:
##             print self.size
##             self.shape = (self.size // 3,3)

        
###########################################################################
    #
    #   Methods that return information about a Coords object or other
    #   views on the object data, without changing the object itself.


    # General

    def simple(self):
        """Return the data as a simple set of points.

        This reshapes the array to a 2-dimensional array, flattening
        the structure of the points.
        """
        return self.reshape((-1,3))
    
    def pshape(self):
        """Return shape of the points array.

        This is the shape of the Coords array with last axis removed.
        """
        return self.shape[:-1]

    def npoints(self):
        """Return the total number of points."""
        return asarray(self.shape[:-1]).prod()

    def x(self):
        """Return the x-plane"""
        return self[...,0]
    def y(self):
        """Return the y-plane"""
        return self[...,1]
    def z(self):
        """Return the z-plane"""
        return self[...,2]


    # Size
    
    def bbox(self):
        """Return the bounding box of a set of points.

        The bounding box is the smallest rectangular volume in global
        coordinates, such at no points are outside the box.
        It is returned as a Coords object with shape (2,3): the first row
        holds the minimal coordinates and the second row the maximal.
        """
        s = self.simple()
        return row_stack([ s.min(axis=0), s.max(axis=0) ])


    def center(self):
        """Return the center of the Coords.

        The center of a Coords is the center of its bbox().
        The return value is a (3,) shaped Coords object.
        """
        min,max = self.bbox()
        return 0.5 * (max+min)


    def centroid(self):
        """Return the centroid of the Coords.

        The centroid of a Coords is the point whose coordinates
        are the mean values of all points.
        The return value is a (3,) shaped Coords object.
        """
        return self.simple().mean(axis=0)


    def sizes(self):
        """Return the sizes of the Coords.

        Return an array with the length of the bbox along the 3 axes.
        """
        min,max = self.bbox()
        return max-min


    def diagonal(self):
        """Return the size of the Coords.

        The size is the length of the diagonal of the bbox()."""
        min,max = self.bbox()
        return length(max-min)

    
    def bsphere(self):
        """Return the diameter of the bounding sphere of the Coords.

        The bounding sphere is the smallest sphere with center in the
        center() of the Coords, and such that no points of the Coords
        are lying outside the sphere.
        """
        return self.distanceFromPoint(self.center()).max()


    #  Distance

    def distanceFromPlane(self,p,n):
        """Return the distance of points f from the plane (p,n).

        p is a point specified by 3 coordinates.
        n is the normal vector to a plane, specified by 3 components.

        The return value is a [...] shaped array with the distance of
        each point to the plane through p and having normal n.
        Distance values are positive if the point is on the side of the
        plane indicated by the positive normal.
        """
        p = asarray(p).reshape((3))
        n = asarray(n).reshape((3))
        n /= length(n)
        d = inner(self,n) - inner(p,n)
        return d


    def distanceFromLine(self,p,n):
        """Return the distance of points f from the line (p,n).

        p is a point on the line specified by 3 coordinates.
        n is a vector specifying the direction of the line through p.

        The return value is a [...] shaped array with the distance of
        each point to the line through p with direction n.
        All distance values are positive or zero.
        """
        p = asarray(p).reshape((3))
        n = asarray(n).reshape((3))
        t = cross(n,p-self)
        d = sqrt(sum(t*t,-1)) / length(n)
        return d


    def distanceFromPoint(self,p):
        """Return the distance of points f from the point p.

        p is a point specified by 3 coordinates.

        The return value is a [...] shaped array with the distance of
        each point to point p.
        All distance values are positive or zero.
        """
        p = asarray(p).reshape((3))
        d = self-p
        d = sum(d*d,-1)
        d = sqrt(d)
        return d


    # Test position

    def test(self,dir=0,min=None,max=None):
        """Flag points having coordinates between min and max.

        This function is very convenient in clipping a Coords in a specified
        direction. It returns a 1D integer array flagging (with a value 1 or
        True) the elements having nodal coordinates in the required range.
        Use where(result) to get a list of element numbers passing the test.
        Or directly use clip() or cclip() to create the clipped Coords.
        
        The test plane can be define in two ways depending on the value of dir.
        If dir == 0, 1 or 2, it specifies a global axis and min and max are
        the minimum and maximum values for the coordinates along that axis.
        Default is the 0 (or x) direction.

        Else, dir should be compaitble with a (3,) shaped array and specifies
        the direction of the normal on the planes. In this case, min and max
        are points and should also evaluate to (3,) shaped arrays.
        
        Nodes specifies which nodes are taken into account in the comparisons.
        It should be one of the following:
        - a single (integer) node number (< the number of nodes)
        - a list of node numbers
        - one of the special strings: 'all', 'any', 'none'
        The default ('all') will flag all the elements that have all their
        nodes between the planes x=min and x=max, i.e. the elements that
        fall completely between these planes. One of the two clipping planes
        may be left unspecified.
        """
        if min is None and max is None:
            raise ValueError,"At least one of min or max have to be specified."

        if type(dir) == int:
            if not min is None:
                T1 = self[...,dir] > min
            if not max is None:
                T2 = self[...,dir] < max
        else:
            if not min is None:
                T1 = self.distanceFromPlane(min,dir) > 0
            if not max is None:
                T2 = self.distanceFromPlane(max,dir) < 0

        if min is None:
            T = T2
        elif max is None:
            T = T1
        else:
            T = T1 * T2
        return T



    def fprint(self,fmt="%10.3e %10.3e %10.3e"):
        """Formatted printing of a Coords.

        The supplied format should contain 3 formatting sequences for the
        three coordinates of a point.
        """
        for p in self.simple():
            print fmt % tuple(p)


##############################################################################

    def set(self,f):
        """Set the coordinates from those in the given array."""
        self[...] = f      # do not be tempted to use self = f !

##############################################################################
    #
    #   Transformations that preserve the topology (but change coordinates)
    #
    #   A. Affine transformations
    #
    #      Scaling
    #      Translation
    #      Central Dilatation = Scaling + Translation
    #      Rotation
    #      Shear
    #      Reflection
    #      Affine
    #
    #  The following methods return transformed coordinates, but by default
    #  they do not change the original data. If the optional argument inplace
    #  is set True, however, the coordinates are changed inplace. 

   
    def scale(self,scale,inplace=False):
        """Return a copy scaled with scale[i] in direction i.

        The scale should be a list of 3 numbers, or a single number.
        In the latter case, the scaling is homothetic."""
        if inplace:
            out = self
        else:
            out = self.copy()
        out *= scale
        return out
    

    def translate(self,dir,distance=None,inplace=False):
        """Return a copy translated over distance in direction dir.

        dir is either an axis number (0,1,2) or a direction vector.

        If a distance is given, the translation is over the specified
        distance in the specified direction.
        If no distance is given, and dir is specified as an axis number,
        translation is over a distance 1.
        If no distance is given, and dir is specified as a vector, translation
        is over the specified vector.
        Thus, the following are all equivalent:
          F.translate(1)
          F.translate(1,1)
          F.translate([0,1,0])
          F.translate([0,2,0],1)
        """
        if inplace:
            out = self
        else:
            out = self.copy()
        if type(dir) is int:
            if distance is None:
                distance = 1.0
            out[...,dir] += distance
        else:
            if len(dir) == 2:
                dir.append(0.0)
            if distance is not None:
                dir *= distance / length(dir)
            out += dir
        return out
    

    def rotate(self,angle,axis=2,around=None,inplace=False):
        """Return a copy rotated over angle around axis.

        The angle is specified in degrees.
        The axis is either one of (0,1,2) designating the global axes,
        or a vector specifying an axis through the origin.
        If no axis is specified, rotation is around the 2(z)-axis. This is
        convenient for working on 2D-structures.

        As a convenience, the user may also specify a 3x3 rotation matrix,
        in which case the function rotate(mat) is equivalent to affine(mat).

        All rotations are performed around the point [0,0,0], unless a
        rotation origin is specified in the argument 'around'. 
        """
        if inplace:
            out = self
        else:
            out = self.copy()
        if not isinstance(angle,ndarray):
            angle = rotationMatrix(angle,axis)
        if around is not None:
            around = asarray(around)
            out = out.translate(-around,inplace=inplace)
        out = out.affine(angle,around,inplace=inplace)
        return out
    

    def shear(self,dir,dir1,skew,inplace=False):
        """Return a copy skewed in the direction dir of plane (dir,dir1).

        The coordinate dir is replaced with (dir + skew * dir1).
        """
        if inplace:
            out = self
        else:
            out = self.copy()
        out[...,dir] += skew * out[...,dir1]
        return out


    def reflect(self,dir=2,pos=0,inplace=False):
        """Mirror the coordinates in direction dir against plane at pos.

        Default position of the plane is through the origin.
        Default mirror direction is the z-direction.
        """
        if inplace:
            out = self
        else:
            out = self.copy()
        out[...,dir] = 2*pos - out[...,dir]
        return out


    def affine(self,mat,vec=None,inplace=False):
        """Return a general affine transform of the Coords.

        The returned Coords has coordinates given by xorig * mat + vec,
        where mat is a 3x3 matrix and vec a length 3 list.
        """
        if inplace:
            out = self
        else:
            out = self.copy()
        out = dot(out,mat)
        if vec is not None:
            out += vec
        return out
#
#
#   B. Non-Affine transformations.
#
#      These always return copies !
#
#        Cylindrical, Spherical, Isoparametric
#

    def cylindrical(self,dir=[0,1,2],scale=[1.,1.,1.]):
        """Converts from cylindrical to cartesian after scaling.

        dir specifies which coordinates are interpreted as resp.
        distance(r), angle(theta) and height(z). Default order is [r,theta,z].
        scale will scale the coordinate values prior to the transformation.
        (scale is given in order r,theta,z).
        The resulting angle is interpreted in degrees.
        """
        # We put in a optional scaling, because doing this together with the
        # transforming is cheaper than first scaling and then transforming.
        f = zeros_like(self)
        r = scale[0] * self[...,dir[0]]
        theta = (scale[1]*rad) * self[...,dir[1]]
        f[...,0] = r*cos(theta)
        f[...,1] = r*sin(theta)
        f[...,2] = scale[2] * self[...,dir[2]]  
        return f


    def toCylindrical(self,dir=[0,1,2]):
        """Converts from cartesian to cylindrical coordinates.

        dir specifies which coordinates axes are parallel to respectively the
        cylindrical axes distance(r), angle(theta) and height(z). Default
        order is [x,y,z].
        The angle value is given in degrees.
        """
        f = zeros_like(self)
        x,y,z = [ self[...,i] for i in dir ]
        f[...,0] = sqrt(x*x+y*y)
        f[...,1] = arctan2(y,x) / rad
        f[...,2] = z
        return f

    
    def spherical(self,dir=[0,1,2],scale=[1.,1.,1.],colat=False):
        """Converts from spherical to cartesian after scaling.

        <dir> specifies which coordinates are interpreted as resp.
        longitude(theta), latitude(phi) and distance(r).
        <scale> will scale the coordinate values prior to the transformation.
        Angles are then interpreted in degrees.
        Latitude, i.e. the elevation angle, is measured from equator in
        direction of north pole(90). South pole is -90.
        If colat=True, the third coordinate is the colatitude (90-lat) instead.
        """
        f = self.reshape((-1,3))
        theta = (scale[0]*rad) * f[:,dir[0]]
        phi = (scale[1]*rad) * f[:,dir[1]]
        r = scale[2] * f[:,dir[2]]
        if colat:
            phi = 90.0*rad - phi
        rc = r*cos(phi)
        f = column_stack([rc*cos(theta),rc*sin(theta),r*sin(phi)])
        return f.reshape(self.shape)


    def toSpherical(self,dir=[0,1,2]):
        """Converts from cartesian to spherical coordinates.

        dir specifies which coordinates axes are parallel to respectively
        the spherical axes distance(r), longitude(theta) and latitude(phi).
        Latitude is the elevation angle measured from equator in direction
        of north pole(90). South pole is -90.
        Default order is [0,1,2], thus the equator plane is the (x,y)-plane.
        The returned angle values are given in degrees.
        """
        v = self[...,dir].reshape((-1,3))
        dist = sqrt(sum(v*v,-1))
        long = arctan2(v[:,0],v[:,2]) / rad
        lat = where(dist <= 0.0,0.0,arcsin(v[:,1]/dist) / rad)
        f = column_stack([long,lat,dist])
        return f.reshape(self.shape)


    def bump1(self,dir,a,func,dist):
        """Return a Coords with a one-dimensional bump.

        dir specifies the axis of the modified coordinates;
        a is the point that forces the bumping;
        dist specifies the direction in which the distance is measured;
        func is a function that calculates the bump intensity from distance
        !! func(0) should be different from 0.
        """
        f = self.copy()
        d = f[...,dist] - a[dist]
        f[...,dir] += func(d)*a[dir]/func(0)
        return f

    
    def bump2(self,dir,a,func):
        """Return a Coords with a two-dimensional bump.

        dir specifies the axis of the modified coordinates;
        a is the point that forces the bumping;
        func is a function that calculates the bump intensity from distance
        !! func(0) should be different from 0.
        """
        f = self.copy()
        dist = [0,1,2]
        dist.remove(dir)
        d1 = f[...,dist[0]] - a[dist[0]]
        d2 = f[...,dist[1]] - a[dist[1]]
        d = sqrt(d1*d1+d2*d2)
        f[...,dir] += func(d)*a[dir]/func(0)
        return f

    
    # This is a generalization of both the bump1 and bump2 methods.
    # If it proves to be useful, it might replace them one day

    # An interesting modification might be to have a point for definiing
    # the distance and a point for defining the intensity (3-D) of the
    # modification
    def bump(self,dir,a,func,dist=None):
        """Return a Coords with a bump.

        A bump is a modification of a set of coordinates by a non-matching
        point. It can produce various effects, but one of the most common
        uses is to force a surface to be indented by some point.
        
        dir specifies the axis of the modified coordinates;
        a is the point that forces the bumping;
        func is a function that calculates the bump intensity from distance
        (!! func(0) should be different from 0)
        dist is the direction in which the distance is measured : this can
        be one of the axes, or a list of one or more axes.
        If only 1 axis is specified, the effect is like function bump1
        If 2 axes are specified, the effect is like bump2
        This function can take 3 axes however.
        Default value is the set of 3 axes minus the direction of modification.
        This function is then equivalent to bump2.
        """
        f = self.copy()
        if dist == None:
            dist = [0,1,2]
            dist.remove(dir)
        try:
            l = len(dist)
        except TypeError:
            l = 1
            dist = [dist]
        d = f[...,dist[0]] - a[dist[0]]
        if l==1:
            d = abs(d)
        else:
            d = d*d
            for i in dist[1:]:
                d1 = f[...,i] - a[i]
                d += d1*d1
            d = sqrt(d)
        f[...,dir] += func(d)*a[dir]/func(0)
        return f


    # NEW implementation flattens coordinate sets to ease use of
    # complicated functions
    def newmap(self,func):
        """Return a Coords mapped by a 3-D function.

        This is one of the versatile mapping functions.
        func is a numerical function which takes three arguments and produces
        a list of three output values. The coordinates [x,y,z] will be
        replaced by func(x,y,z).
        The function must be applicable to arrays, so it should
        only include numerical operations and functions understood by the
        numpy module.
        This method is one of several mapping methods. See also map1 and mapd.
        Example: E.map(lambda x,y,z: [2*x,3*y,4*z])
        is equivalent with E.scale([2,3,4])
        """
        x,y,z = func(self[...,0].flat,self[...,1].flat,self[...,2].flat)
        shape = list(self.shape)
        shape[2] = 1
        #print shape,reshape(x,shape)
        f = concatenate([reshape(x,shape),reshape(y,shape),reshape(z,shape)],2)
        #print f.shape
        return f


    def map(self,func):
        """Return a Coords mapped by a 3-D function.

        This is one of the versatile mapping functions.
        func is a numerical function which takes three arguments and produces
        a list of three output values. The coordinates [x,y,z] will be
        replaced by func(x,y,z).
        The function must be applicable to arrays, so it should
        only include numerical operations and functions understood by the
        numpy module.
        This method is one of several mapping methods. See also map1 and mapd.
        Example: E.map(lambda x,y,z: [2*x,3*y,4*z])
        is equivalent with E.scale([2,3,4])
        """
        f = zeros_like(self)
        f[...,0],f[...,1],f[...,2] = func(self[...,0],self[...,1],self[...,2])
        return f


    def map1(self,dir,func):
        """Return a Coords where coordinate i is mapped by a 1-D function.

        <func> is a numerical function which takes one argument and produces
        one result. The coordinate dir will be replaced by func(coord[dir]).
        The function must be applicable on arrays, so it should only
        include numerical operations and functions understood by the
        numpy module.
        This method is one of several mapping methods. See also map and mapd.
        """
        f = self.copy()
        f[...,dir] = func(self[...,dir])
        return f


    def mapd(self,dir,func,point,dist=None):
        """Maps one coordinate by a function of the distance to a point.

        <func> is a numerical function which takes one argument and produces
        one result. The coordinate dir will be replaced by func(d), where <d>
        is calculated as the distance to <point>.
        The function must be applicable on arrays, so it should only
        include numerical operations and functions understood by the
        numpy module.
        By default, the distance d is calculated in 3-D, but one can specify
        a limited set of axes to calculate a 2-D or 1-D distance.
        This method is one of several mapping methods. See also map3 and map1.
        Example: E.mapd(2,lambda d:sqrt(10**2-d**2),f.center(),[0,1])
        maps E on a sphere with radius 10
        """
        f = self.copy()
        if dist == None:
            dist = [0,1,2]
        try:
            l = len(dist)
        except TypeError:
            l = 1
            dist = [dist]
        d = f[...,dist[0]] - point[dist[0]]
        if l==1:
            d = abs(d)
        else:
            d = d*d
            for i in dist[1:]:
                d1 = f[...,i] - point[i]
                d += d1*d1
            d = sqrt(d)
        f[...,dir] = func(d)
        return f


    def replace(self,i,j,other=None):
        """Replace the coordinates along the axes i by those along j.

        i and j are lists of axis numbers or single axis numbers.
        replace ([0,1,2],[1,2,0]) will roll the axes by 1.
        replace ([0,1],[1,0]) will swap axes 0 and 1.
        An optionally third argument may specify another Coords object to take
        the coordinates from. It should have the same dimensions.
        """
        if other is None:
            other = self
        f = self.copy()
        f[...,i] = other[...,j]
        return f


    def swapAxes(self,i,j):
        """Swap coordinate axes i and j.

        Beware! This is different from numpy's swapaxes() method !
        """
        return self.replace([i,j],[j,i])


    def rollAxes(self,n=1):
        """Roll the axes over the given amount.

        Default is 1, thus axis 0 becomes the new 1 axis, 1 becomes 2 and
        2 becomes 0.
        """
        return roll(self, int(n) % 3,axis=-1)


    def projectOnSphere(self,radius,center=[0.,0.,0.]):
        """Project Coords on a sphere."""
        d = self.distanceFromPoint(center)
        s = radius / d
        f = self - center
        f[...,0] *= s
        f[...,1] *= s
        f[...,2] *= s
        f += center
        return f


##############################################################################

    def unique(self,nodesperbox=1,shift=0.5,rtol=1.e-5,atol=1.e-5):
        """Finds (almost) identical nodes and returns a compressed set.

        This method finds the points that are very close and replaces them
        with a single point. The return value is a tuple of two arrays:
        - the unique points as a Coords object,
        - an integer (nnod) array holding an index in the unique
        coordinates array for each of the original nodes. This index will
        have the same shape als the pshape() of the coords array.

        The procedure works by first dividing the 3D space in a number of
        equally sized boxes, with a mean population of nodesperbox.
        The boxes are numbered in the 3 directions and a unique integer scalar
        is computed, that is then used to sort the nodes.
        Then only nodes inside the same box are compared on almost equal
        coordinates, using the numpy allclose() function. Two coordinates are
        considered close if they are within a relative tolerance rtol or absolute
        tolerance atol. See numpy for detail. The default atol is set larger than
        in numpy, because pyformex typically runs with single precision.
        Close nodes are replaced by a single one.

        Currently the procedure does not guarantee to find all close nodes:
        two close nodes might be in adjacent boxes. The performance hit for
        testing adjacent boxes is rather high, and the probability of separating
        two close nodes with the computed box limits is very small. Nevertheless
        we intend to access this problem by repeating the procedure with the
        boxes shifted in space.
        """
        x = self.simple()
        nnod = x.shape[0]
        # Calculate box size
        lo = array([ x[:,i].min() for i in range(3) ])
        hi = array([ x[:,i].max() for i in range(3) ])
        sz = hi-lo
        esz = sz[sz > 0.0]  # only keep the nonzero dimensions
        vol = esz.prod()
        nboxes = nnod / nodesperbox # ideal total number of boxes
        boxsz = (vol/nboxes) ** (1./esz.shape[0])
        nx = (sz/boxsz).astype(int)
        dx = where(nx>0,sz/nx,boxsz)
        nx = array(nx) + 1
        ox = lo - dx*shift # origin :  0 < shift < 1
        # Create box coordinates for all nodes
        ind = floor((x-ox)/dx).astype(int)
        # Create unique box numbers in smallest direction first
        o = argsort(nx)
        #print "nx",nx
        #print "ind",ind.dtype
        val = ( ind[:,o[2]] * nx[o[2]] + ind[:,o[1]] ) * nx[o[1]] + ind[:,o[0]]
        #print "val",val.dtype,val.shape
        # sort according to box number
        srt = argsort(val)
        # rearrange the data according to the sort order
        val = val[srt]
        x = x[srt]
        # now compact
        flag = ones((nnod,))   # 1 = new, 0 = existing node
        sel = arange(nnod)     # replacement unique node nr
        #print "Start Compacting %s nodes" % nnod
        #nblk = nnod/100
        for i in range(nnod):
            #if i % nblk == 0:
                #print "Blok %s" % (i/nblk)
            j = i-1
            while j>=0 and val[i]==val[j]:
                if allclose(x[i],x[j],rtol=rtol,atol=atol):
                    # node i is same as node j
                    flag[i] = 0
                    sel[i] = sel[j]
                    sel[i+1:nnod] -= 1
                    break
                j = j-1
        #print "Finished Compacting"
        x = x[flag>0]          # extract unique nodes
        s = sel[argsort(srt)]  # and indices for old nodes
        return (x,s.reshape(self.shape[:-1]))


    # Convenient shorter notations
    rot = rotate
    trl = translate


    @classmethod
    def concatenate(cls,L):
        """Concatenate a list of Coords object.

        All Coords object in the list L should have the same shape
        except for the length of the first axis.
        This function is equivalent to the numpy concatenate, but makes
        sure the result is a Cooords object.
        """
        return Coords(concatenate(L))


    @classmethod
    def fromfile(*args):
        """Read a Coords from file.

        This convenience function uses the numpy fromfile function to read
        the coordinates from file.
        You just have to make sure that the coordinates are read in order
        (X,Y,Z) for subsequent points, and that the total number of
        coordinates read is a multiple of 3.
        """
        return Coords(fromfile(*args).reshape((-1,3)))

    
    @classmethod
    def interpolate(clas,F,G,div):
        """Create interpolations between two Coords.

        F and G are two Coords with the same shape.
        v is a list of floating point values.
        The result is the concatenation of the interpolations of F and G at all
        the values in div.
        An interpolation of F and G at value v is a Coords H where each
        coordinate Hijk is obtained from:  Hijk = Fijk + v * (Gijk-Fijk).
        Thus, a Coords interpolate(F,G,[0.,0.5,1.0]) will contain all points of
        F and G and all points with mean coordinates between those of F and G.

        As a convenience, if an integer is specified for div, it is taken as a
        number of divisions for the interval [0..1].
        Thus, interpolate(F,G,n) is equivalent with
        interpolate(F,G,arange(0,n+1)/float(n))

        The resulting Coords array has an extra axis (the first). Its shape is
        (n,) + F.shape, where n is the number of divisions.
        """
        if F.shape != G.shape:
            raise RuntimeError,"Expected Coords objects with equal shape!"
        if type(div) == int:
            div = arange(div+1) / float(div)
        else:
            div = array(div).ravel()
        return F + outer(div,G-F).reshape((-1,)+F.shape)


##############################################################################
#
#  Testing
#
#  Some of the docstrings above hold test examples. They should be careflly 
#  crafted to test the functionality of the Formex class.
#
#  Ad hoc test examples during development can be added to the test() function
#  below.
#
#  python formex.py
#    will execute the docstring examples silently. 
#  python formex.py -v
#    will execute the docstring examples verbosely.
#  In both cases, the ad hoc tests are only run if the docstring tests
#  are passed.
#

if __name__ == "__main__":

    def testX(X):
        """Run some tests on Coords X."""

        def prt(s,v):
            """Print a statement 's = v' and return v"""
            if isinstance(v,ndarray):
                sep = '\n'
            else:
                sep = ' '
            print "%s =%s%s" % (s,sep,v)
            return v

        prt("###################################\nTests for Coords X",X)

        # Info
        prt("simple",X.simple())
        prt("pshape",X.pshape())
        prt("npoints",X.npoints())
        prt("y",X.y())
        prt("bbox",X.bbox())
        prt("center",X.center())
        prt("centroid",X.centroid())
        prt("sizes",X.sizes())
        prt("diag",X.diagonal())
        prt("bsphere",X.bsphere())
        prt("distanceFromPlane",X.distanceFromPlane([0.,0.,1.],[0.,0.,1.]))
        prt("distanceFromLine",X.distanceFromLine([0.,0.,1.],[0.,0.,1.]))
        prt("distanceFromPoint",X.distanceFromPoint([0.,0.,1.]))
        prt("test",X.test(dir=1,min=0.5,max=1.5))
        prt("test2",X.test(dir=[1.,1.,0.],min=[0.,0.5,0.],max=[0.,1.5,0.]))

        # Transforms
        prt("X_scl",X.scale(2,False))
        prt("X",X)
        prt("X_scl",X.scale(2,True))
        prt("X",X)
        prt("X_scl2",X.scale([0.5,1.,0.]))
        prt("X_trl",X.copy().translate(0,6))
        prt("X_trl2",X.translate([10.,100.,1000.]))
        prt("X_rot",X.rotate(90.))
        prt("X_rot2",X.rotate(90.,0))
        return
        X2 = X1.reflect(1,2)
        print "X =",X
        print "X1 =",X1
        print "X2 =",X2
        X3 = X.copy().reflect(1,1.5).translate(1,2)
        print "X =",X
        print "X3 =",X3
        G = Coords.concatenate([X1,X3,X2,X3])
        print "X1+X3+X2+X3 =",G
        print "unique:",G.unique()

        Y = Coords([[[1,0,0],[0,1,0],[0,0,1]],[[2,0,0],[0,2,0],[0,0,2]]])
        print Y
        Y.translate([0.,100.,0.])
        print Y

        Y = Coords([1.0,0.0,0.0])
        print Y
        Y.translate([0.,100.,0.])
        print Y
        
        return
        print "sizes:",G.sizes()
        print G.bbox()
        print G.center(),G.centroid()
        return
        print G.bsphere()

        F.fprint()

if __name__ == "__main__":
    def test():
        """Run some additional examples.

        This is intended for tests during development. This can be changed
        at will.
        """
        testX(Coords([[1,0,0],[0,1,0]]))
        testX(Coords([[[0,0,0],[1,0,0]],[[0,1,0],[1,1,0]]]))
        testX(Coords([1,0,0]))
        testX(Coords())
        return

    f = 0

    #import doctest, formex
    #f,t = doctest.testmod(formex)

    if f == 0:
        test()

### End
