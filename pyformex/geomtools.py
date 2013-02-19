# $Id$
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
"""Basic geometrical operations.

This module defines some basic operations on simple geometrical entities
such as lines, triangles, circles, planes.
"""
from __future__ import print_function

from coords import *

class Plane(object):
    def __init__(self,P,n):
        self.coords = Coords.concatenate([P,normalize(n)])


def areaNormals(x):
    """Compute the area and normal vectors of a collection of triangles.

    x is an (ntri,3,3) array with the coordinates of the vertices of ntri
    triangles.

    Returns a tuple (areas,normals) with the areas and the normals of the
    triangles. The area is always positive. The normal vectors are normalized.
    """
    x = x.reshape(-1,3,3)
    area,normals = vectorPairAreaNormals(x[:,1]-x[:,0],x[:,2]-x[:,1])
    area *= 0.5
    return area,normals


def degenerate(area,normals):
    """Return a list of the degenerate faces according to area and normals.

    area,normals are equal sized arrays with the areas and normals of a
    list of faces, such as the output of the :func:`areaNormals` function.

    A face is degenerate if its area is less or equal to zero or the
    normal has a nan (not-a-number) value.

    Returns a list of the degenerate element numbers as a sorted array.
    """
    return unique(concatenate([where(area<=0)[0],where(isnan(normals))[0]]))


def levelVolumes(x):
    """Compute the level volumes of a collection of elements.

    x is an (nelems,nplex,3) array with the coordinates of the nplex vertices
    of nelems elements, with nplex equal to 2, 3 or 4.

    If nplex == 2, returns the lengths of the straight line segments.
    If nplex == 3, returns the areas of the triangle elements.
    If nplex == 4, returns the signed volumes of the tetraeder elements.
    Positive values result if vertex 3 is at the positive side of the plane
    defined by the vertices (0,1,2). Negative volumes are reported for
    tetraeders having reversed vertex order.

    For any other value of nplex, raises an error.
    If succesful, returns an (nelems,) shaped float array.
    """
    nplex = x.shape[1]
    if nplex == 2:
        return length(x[:,1]-x[:,0])
    elif nplex == 3:
        return vectorPairArea(x[:,1]-x[:,0], x[:,2]-x[:,1]) / 2
    elif nplex == 4:
        return vectorTripleProduct(x[:,1]-x[:,0], x[:,2]-x[:,1], x[:,3]-x[:,0]) / 6
    else:
        raise ValueError,"Plexitude should be one of 2, 3 or 4; got %s" % nplex


def smallestDirection(x,method='inertia',return_size=False):
    """Return the direction of the smallest dimension of a Coords

    - `x`: a Coords-like array
    - `method`: one of 'inertia' or 'random'
    - return_size: if True and `method` is 'inertia', a tuple of a direction
      vector and the size  along that direction and the cross directions;
      else, only return the direction vector.
    """
    x = x.reshape(-1,3)
    if method == 'inertia':
        # The idea is to take the smallest dimension in a coordinate
        # system aligned with the global axes.
        C,r,Ip,I = x.inertia()
        X = x.trl(-C).rot(r)
        sizes = X.sizes()
        i = sizes.argmin()
        # r gives the directions as column vectors!
        # TODO: maybe we should change that
        N = r[:,i]
        if return_size:
            return N,sizes[i]
        else:
            return N
    elif method == 'random':
        # Take the mean of the normals on randomly created triangles
        from plugins.trisurface import TriSurface
        n = x.shape[0]
        m = 3 * (n // 3)
        e = arange(m)
        random.shuffle(e)
        if n > m:
            e = concatenate([e,[0,1,n-1]])
        el = e[-3:]
        S = TriSurface(x,e.reshape(-1,3))
        A,N = S.areaNormals()
        ok = where(isnan(N).sum(axis=1) == 0)[0]
        N = N[ok]
        N = N*N
        N = N.mean(axis=0)
        N = sqrt(N)
        N = normalize(N)
        return N


def closestPoint(P0,P1):
    """Find the smallest distance between any two points from P0 and P1.

    P0 and P1 are Coords arrays. Any point of P0 is compared with any point
    of P1, and the couple with the closest distance is returned.

    Returns a tuple (i,j,d) where i,j are the indices in P0,P1 identifying
    the closest points, and d is the distance between them.
    """
    P0 = P0.reshape(-1,3)
    P1 = P1.reshape(-1,3)
    if P0.shape[0] == 1:
        d = P1.distanceFromPoint(P0[0])
        j = d.argmin()
        return 0,j,d[j]
    else:
        di = [ closestPoint(x,P1)[1:] for x in P0 ]
        i = array([d[1] for d in di]).argmin()
        j,d = di[i]
    return i,j,d


def projectedArea(x,dir):
    """Compute projected area inside a polygon.

    Parameters:

    - `x`: (npoints,3) Coords with the ordered vertices of a
      (possibly nonplanar) polygonal contour.
    - `dir`: either a global axis number (0, 1 or 2) or a direction vector
      consisting of 3 floats, specifying the projection direction.

    Returns a single float value with the area inside the polygon projected
    in the specified direction.

    Note that if the polygon is planar and the specified direction is that
    of the normal on its plane, the returned area is that of the planar
    figure inside the polygon. If the polygon is nonplanar however, the area
    inside the polygon is not defined. The projected area in a specified
    direction is, since the projected polygon is a planar one.
    """
    if x.shape[0] < 3:
        return 0.0
    if type(dir) is int:
        dir = unitVector(dir)
    x1 = roll(x,-1,axis=0)
##    print x.dtype
##    print Coords(dir).dtype
    area = vectorTripleProduct(Coords(dir),x,x1)
##    print area.dtype
##    print area.sum() / 2
    return 0.5 * area.sum()


def polygonNormals(x):
    """Compute normals in all points of polygons in x.

    x is an (nel,nplex,3) coordinate array representing nel (possibly nonplanar)
    polygons.

    The return value is an (nel,nplex,3) array with the unit normals on the
    two edges ending in each point.
    """
    if x.shape[1] < 3:
        #raise ValueError,"Cannot compute normals for plex-2 elements"
        n = zeros_like(x)
        n[:,:,2] = -1.
        return n

    ni = arange(x.shape[1])
    nj = roll(ni,1)
    nk = roll(ni,-1)
    v1 = x-x[:,nj]
    v2 = x[:,nk]-x
    n = vectorPairNormals(v1.reshape(-1,3),v2.reshape(-1,3)).reshape(x.shape)
    #print "NANs: %s" % isnan(n).sum()
    return n

def triangleInCircle(x):
    """Compute the incircles of the triangles x

    The incircle of a triangle is the largest circle that can be inscribed
    in the triangle.

    x is a Coords array with shape (ntri,3,3) representing ntri triangles.

    Returns a tuple r,C,n with the radii, Center and unit normals of the
    incircles.
    """
    checkArray(x,shape=(-1,3,3))
    # Edge vectors
    v = roll(x,-1,axis=1) - x
    v = normalize(v)
    # create bisecting lines in x0 and x1
    b0 = v[:,0]-v[:,2]
    b1 = v[:,1]-v[:,0]
    # find intersection => center point of incircle
    center = lineIntersection(x[:,0],b0,x[:,1],b1)
    # find distance to any side => radius
    radius = center.distanceFromLine(x[:,0],v[:,0])
    # normals
    normal = cross(v[:,0],v[:,1])
    normal /= length(normal).reshape(-1,1)
    return radius,center,normal


def triangleCircumCircle(x,bounding=False):
    """Compute the circumcircles of the triangles x

    x is a Coords array with shape (ntri,3,3) representing ntri triangles.

    Returns a tuple r,C,n with the radii, Center and unit normals of the
    circles going through the vertices of each triangle.

    If bounding=True, this returns the triangle bounding circle.
    """
    checkArray(x,shape=(-1,3,3))
    # Edge vectors
    v = x - roll(x,-1,axis=1)
    vv = dotpr(v,v)
    # Edge lengths
    lv = sqrt(vv)
    n = cross(v[:,0],v[:,1])
    nn = dotpr(n,n)
    # Radius
    N = sqrt(nn)
    r = asarray(lv.prod(axis=-1) / N / 2)
    # Center
    w = -dotpr(roll(v,1,axis=1),roll(v,2,axis=1))
    a = w * vv
    C = a.reshape(-1,3,1) * roll(x,1,axis=1)
    C = C.sum(axis=1) / nn.reshape(-1,1) / 2
    # Unit normals
    n = n / N.reshape(-1,1)
    # Bounding circle
    if bounding:
        # Modify for obtuse triangles
        for i,j,k in [[0,1,2],[1,2,0],[2,0,1]]:
            obt = vv[:,i] >= vv[:,j]+vv[:,k]
            r[obt] = 0.5 * lv[obt,i]
            C[obt] = 0.5 * (x[obt,i] + x[obt,j])

    return r,C,n


def triangleBoundingCircle(x):
    """Compute the bounding circles of the triangles x

    The bounding circle is the smallest circle in the plane of the triangle
    such that all vertices of the triangle are on or inside the circle.
    If the triangle is acute, this is equivalent to the triangle's
    circumcircle. It the triangle is obtuse, the longest edge is the
    diameter of the bounding circle.

    x is a Coords array with shape (ntri,3,3) representing ntri triangles.

    Returns a tuple r,C,n with the radii, Center and unit normals of the
    bounding circles.
    """
    return triangleCircumCircle(x,bounding=True)


def triangleObtuse(x):
    """Checks for obtuse triangles

    x is a Coords array with shape (ntri,3,3) representing ntri triangles.

    Returns an (ntri) array of True/False values indicating whether the
    triangles are obtuse.
    """
    checkArray(x,shape=(-1,3,3))
    # Edge vectors
    v = x - roll(x,-1,axis=1)
    vv = dotpr(v,v)
    return (vv[:,0] > vv[:,1]+vv[:,2]) + (vv[:,1] > vv[:,2]+vv[:,0]) + (vv[:,2] > vv[:,0]+vv[:,1])


def lineIntersection(P1,D1,P2,D2):
    """Finds the intersection of 2 coplanar lines.

    The lines (P1,D1) and (P2,D2) are defined by a point and a direction
    vector. Let a and b be unit vectors along the lines, and c = P2-P1,
    let ld and d be the length and the unit vector of the cross product a*b,
    the intersection point X is then given by X = 0.5(P1+P2+sa*a+sb*b)
    where sa = det([c,b,d])/ld and sb = det([c,a,d])/ld
    """
    P1 = asarray(P1).reshape((-1,3)).astype(float64)
    D1 = asarray(D1).reshape((-1,3)).astype(float64)
    P2 = asarray(P2).reshape((-1,3)).astype(float64)
    D2 = asarray(D2).reshape((-1,3)).astype(float64)
    N = P1.shape[0]
    # a,b,c,d
    la,a = vectorNormalize(D1)
    lb,b = vectorNormalize(D2)
    c = (P2-P1)
    d = cross(a,b)
    ld,d = vectorNormalize(d)
    # sa,sb
    a = a.reshape((-1,1,3))
    b = b.reshape((-1,1,3))
    c = c.reshape((-1,1,3))
    d = d.reshape((-1,1,3))
    m1 = concatenate([c,b,d],axis=-2)
    m2 = concatenate([c,a,d],axis=-2)
    # This may still be optimized
    sa = zeros((N,1))
    sb = zeros((N,1))
    for i in range(P1.shape[0]):
        sa[i] = linalg.det(m1[i]) / ld[i]
        sb[i] = linalg.det(m2[i]) / ld[i]
    # X
    a = a.reshape((-1,3))
    b = b.reshape((-1,3))
    X = 0.5 * ( P1 + sa*a + P2 + sb*b )
    return Coords(X)


def displaceLines(A,N,C,d):
    """Move all lines (A,N) over a distance a in the direction of point C.

    A,N are arrays with points and directions defining the lines.
    C is a point.
    d is a scalar or a list of scalars.
    All line elements of F are translated in the plane (line,C)
    over a distance d in the direction of the point C.
    Returns a new set of lines (A,N).
    """
    l,v = vectorNormalize(N)
    w = C - A
    vw = (v*w).sum(axis=-1).reshape((-1,1))
    Y = A + vw*v
    l,v = vectorNormalize(C-Y)
    return A + d*v, N


def segmentOrientation(vertices,vertices2=None,point=None):
    """Determine the orientation of a set of line segments.

    vertices and vertices2 are matching sets of points.
    point is a single point.
    All arguments are Coords objects.

    Line segments run between corresponding points of vertices and vertices2.
    If vertices2 is None, it is obtained by rolling the vertices one position
    foreward, thus corresponding to a closed polygon through the vertices).
    If point is None, it is taken as the center of vertices.

    The orientation algorithm checks whether the line segments turn
    positively around the point.

    Returns an array with +1/-1 for positive/negative oriented segments.
    """
    if vertices2 is None:
        vertices2 = roll(vertices,-1,axis=0)
    if point is None:
        point = vertices.center()

    w = cross(vertices,vertices2)
    orient = sign(dotpr(point,w)).astype(Int)
    return orient


def rotationAngle(A,B,m=None,angle_spec=DEG):
    """Return rotation angles and vectors for rotations of A to B.

    A and B are (n,3) shaped arrays where each line represents a vector.
    This function computes the rotation from each vector of A to the
    corresponding vector of B.
    If m is None, the return value is a tuple of an (n,) shaped array with
    rotation angles (by default in degrees) and an (n,3) shaped array with
    unit vectors along the rotation axis.
    If m is a (n,3) shaped array with vectors along the rotation axis, the
    return value is a (n,) shaped array with rotation angles.
    Specify angle_spec=RAD to get the angles in radians.
    """
    A = asarray(A).reshape(-1,3)
    B = asarray(B).reshape(-1,3)
    if m is None:
        A = normalize(A)
        B = normalize(B)
        n = cross(A,B) # vectors perpendicular to A and B
        t = length(n) == 0.
        if t.any(): # some vectors A and B are parallel
            n[t] = anyPerpendicularVector(A[t])
        n = normalize(n)
        c = dotpr(A,B)
        angle = arccosd(c.clip(min=-1.,max=1.),angle_spec)
        return angle,n
    else:
        m = asarray(m).reshape(-1,3)
        # project vectors on plane
        A = projectionVOP(A,m)
        B = projectionVOP(B,m)
        angle,n = rotationAngle(A,B,angle_spec=angle_spec)
        # check sign of the angles
        m = normalize(m)
        inv = isClose(dotpr(n,m),[-1.])
        angle[inv] *= -1.
        return angle


def anyPerpendicularVector(A):
    """Return arbitrary vectors perpendicular to vectors of A.

    A is a (n,3) shaped array of vectors.
    The return value is a (n,3) shaped array of perpendicular vectors.

    The returned vector is always a vector in the x,y plane. If the original
    is the z-axis, the result is the x-axis.
    """
    A = asarray(A)
    x,y,z = hsplit(A,[1,2])
    n = zeros(x.shape,dtype=Float)
    i = ones(x.shape,dtype=Float)
    t = (x==0.)*(y==0.)
    B = where(t,column_stack([i,n,n]),column_stack([-y,x,n]))
    # B = where(t,column_stack([-z,n,x]),column_stack([-y,x,n]))
    return B


def perpendicularVector(A,B):
    """Return vectors perpendicular on both A and B."""
    return cross(A,B)


def projectionVOV(A,B):
    """Return the projection of vector of A on vector of B."""
    L = projection(A,B)
    B = normalize(B)
    shape = list(L.shape)
    shape.append(1)
    return L.reshape(shape)*B


def projectionVOP(A,n):
    """Return the projection of vector of A on plane of B."""
    Aperp = projectionVOV(A,n)
    return A-Aperp


################## intersection tools ###############
#
#  IT SHOULD BE CLEARLY DOCUMENTED WHETHER NORMALS ARE REQUIRED
#  TO BE NORMALIZED OR NOT
#  svc: plane normals and line vectors are not required to be normalized
#
#  MAYBE WE SHOULD ADOPT CONVENTION TO USE m,n FOR NORMALIZED
#  VECTORS, AND u,v,w for (possibly) unnormalized
#

def pointsAtLines(q,m,t):
    """Return the points of lines (q,m) at parameter values t.

    Parameters:

    - `q`,`m`: (...,3) shaped arrays of points and vectors, defining
      a single line or a set of lines.
    - `t`: array of parameter values, broadcast compatible with `q` and `m`.

    Returns an array with the points at parameter values t.
    """
    t = t[...,newaxis]
    return q+t*m


def pointsAtSegments(S,t):
    """Return the points of line segments S at parameter values t.

    Parameters:

    - `S`: (...,2,3) shaped array, defining a single line segment or
      a set of line segments.
    - `t`: array of parameter values, broadcast compatible with `S`.

    Returns an array with the points at parameter values t.
    """
    q0 = S[...,0,:]
    q1 = S[...,1,:]
    return pointsAtLines(q0,q1-q0,t)


def intersectionTimesLWL(q1,m1,q2,m2,mode='all'):
    """Return the intersection of lines (q1,m1) and lines (q2,m2)

    with the perpendiculars between them.

    Parameters:

    - `qi`,`mi` (i=1...2): (nqi,3) shaped arrays of points and vectors (`mode=all`)
      or broadcast compatible arrays (`mode=pair`), defining a single line or a
      set of lines.
    - `mode`: `all` to calculate the intersection of each line (q1,m1) with all lines
      (q2,m2) or `pair` for pairwise intersections.

    Returns a tuple of (nq1,nq2) shaped (`mode=all`) arrays of parameter
    values t1 and t2, such that the intersection points are given by
    ``q1+t1*m1`` and ``q2+t2*m2``.
    """
    if mode == 'all':
        q1 = asarray(q1).reshape(-1,1,3)
        m1 = asarray(m1).reshape(-1,1,3)
        q2 = asarray(q2).reshape(1,-1,3)
        m2 = asarray(m2).reshape(1,-1,3)
    dot11 = dotpr(m1,m1)
    dot22 = dotpr(m2,m2)
    dot12 = dotpr(m1,m2)
    denom = (dot12**2-dot11*dot22)
    q12 = q2-q1
    dot11 = dot11[...,newaxis]
    dot22 = dot22[...,newaxis]
    dot12 = dot12[...,newaxis]
    t1 = dotpr(q12,m2*dot12-m1*dot22) / denom
    t2 = dotpr(q12,m2*dot11-m1*dot12) / denom
    return t1,t2


def intersectionPointsLWL(q1,m1,q2,m2,mode='all'):
    """Return the intersection points of lines (q1,m1) and lines (q2,m2)

    with the perpendiculars between them.

    This is like intersectionTimesLWL but returns a tuple of (nq1,nq2,3)
    shaped (`mode=all`) arrays of intersection points instead of the
    parameter values.
    """
    t1,t2 = intersectionTimesLWL(q1,m1,q2,m2,mode)
    if mode == 'all':
        q1 = q1[:,newaxis]
        m1 = m1[:,newaxis]
    return pointsAtLines(q1,m1,t1),pointsAtLines(q2,m2,t2)


def intersectionTimesLWP(q,m,p,n,mode='all'):
    """Return the intersection of lines (q,m) with planes (p,n).

    Parameters:

    - `q`,`m`: (nq,3) shaped arrays of points and vectors (`mode=all`)
      or broadcast compatible arrays (`mode=pair`), defining a single line
      or a set of lines.
    - `p`,`n`: (np,3) shaped arrays of points and normals (`mode=all`)
      or broadcast compatible arrays (`mode=pair`), defining a single plane
      or a set of planes.
    - `mode`: `all` to calculate the intersection of each line (q,m) with
      all planes (p,n) or `pair` for pairwise intersections.

    Returns a (nq,np) shaped (`mode=all`) array of parameter values t,
    such that the intersection points are given by q+t*m.

    Notice that the result will contain an INF value for lines that are
    parallel to the plane.
    """
    if mode == 'all':
        res = (dotpr(p,n) - inner(q,n)) / inner(m,n)
    elif mode == 'pair':
        res = dotpr(n, (p-q)) / dotpr(m,n)
    return res


def intersectionPointsLWP(q,m,p,n,mode='all'):
    """Return the intersection points of lines (q,m) with planes (p,n).

    This is like intersectionTimesLWP but returns a (nq,np,3) shaped
    (`mode=all`) array of intersection points instead of the
    parameter values.
    """
    t = intersectionTimesLWP(q,m,p,n,mode)
    if mode == 'all':
        q = q[:,newaxis]
        m = m[:,newaxis]
    return pointsAtLines(q,m,t)


def intersectionTimesSWP(S,p,n,mode='all'):
    """Return the intersection of line segments S with planes (p,n).

    Parameters:

    - `S`: (nS,2,3) shaped array (`mode=all`) or broadcast compatible array
      (`mode=pair`), defining a single line segment or a set of line segments.
    - `p`,`n`: (np,3) shaped arrays of points and normals (`mode=all`)
      or broadcast compatible arrays (`mode=pair`), defining a single plane
      or a set of planes.
    - `mode`: `all` to calculate the intersection of each line segment S with
      all planes (p,n) or `pair` for pairwise intersections.

    Returns a (nS,np) shaped (`mode=all`) array of parameter values t,
    such that the intersection points are given by
    `(1-t)*S[...,0,:] + t*S[...,1,:]`.

    This function is comparable to intersectionTimesLWP, but ensures that
    parameter values 0<=t<=1 are points inside the line segments.
    """
    q0 = S[...,0,:]
    q1 = S[...,1,:]
    return intersectionTimesLWP(q0,q1-q0,p,n,mode)


def intersectionSWP(S,p,n,mode='all',return_all=False):
    """Return the intersection points of line segments S with planes (p,n).

    Parameters:

    - `S`: (nS,2,3) shaped array, defining a single line segment or a set of
      line segments.
    - `p`,`n`: (np,3) shaped arrays of points and normals, defining a single
      plane or a set of planes.
    - `mode`: `all` to calculate the intersection of each line segment S with
      all planes (p,n) or `pair` for pairwise intersections.
    - `return_all`: if True, all intersection points of the lines along the
      segments are returned. Default is to return only the points that lie
      on the segments.

    Return values if `return_all==True`:

    - `t`: (nS,NP) parametric values of the intersection points along the line
      segments.
    - `x`: the intersection points themselves (nS,nP,3).

    Return values if `return_all==False`:

    - `t`: (n,) parametric values of the intersection points along the line
      segments (n <= nS*nP)
    - `x`: the intersection points themselves (n,3).
    - `wl`: (n,) line indices corresponding with the returned intersections.
    - `wp`: (n,) plane indices corresponding with the returned intersections
    """
    S = asanyarray(S).reshape(-1,2,3)
    p = asanyarray(p).reshape(-1,3)
    n = asanyarray(n).reshape(-1,3)
    # Find intersection parameters
    t = intersectionTimesSWP(S,p,n,mode)

    if not return_all:
        # Find points inside segments
        ok = (t >= 0.0) * (t <= 1.0)
        t = t[ok]
        if mode == 'all':
            wl,wt = where(ok)
        elif mode == 'pair':
            wl = wt = where(ok)[0]

    if len(t) > 0:
        if mode == 'all':
            S = S[:,newaxis]
        x = pointsAtSegments(S,t)
        if x.ndim == 1:
            x = x.reshape(1,3)
        if not return_all:
            x = x[ok]
    else:
        # No intersection: return empty Coords
        x = Coords()

    if return_all:
        return t,x
    else:
        return t,x,wl,wt


def intersectionPointsSWP(S,p,n,mode='all',return_all=False):
    """Return the intersection points of line segments S with planes (p,n).

    This is like :func:`intersectionSWP` but does not return the parameter
    values. It is equivalent to::

      intersectionSWP(S,p,n,mode,return_all)[1:]
    """
    res = intersectionSWP(S,p,n,mode,return_all)
    if return_all:
        return res[1]
    else:
        return res[1:]


def intersectionTimesLWT(q,m,F,mode='all'):
    """Return the intersection of lines (q,m) with triangles F.

    Parameters:

    - `q`,`m`: (nq,3) shaped arrays of points and vectors (`mode=all`)
      or broadcast compatible arrays (`mode=pair`), defining a single line
      or a set of lines.
    - `F`: (nF,3,3) shaped array (`mode=all`) or broadcast compatible array
      (`mode=pair`), defining a single triangle or a set of triangles.
    - `mode`: `all` to calculate the intersection of each line (q,m) with
      all triangles F or `pair` for pairwise intersections.

    Returns a (nq,nF) shaped (`mode=all`) array of parameter values t,
      such that the intersection points are given q+tm.
    """
    Fn = cross(F[...,1,:]-F[...,0,:],F[...,2,:]-F[...,1,:])
    return intersectionTimesLWP(q,m,F[...,0,:],Fn,mode)


def intersectionPointsLWT(q,m,F,mode='all',return_all=False):
    """Return the intersection points of lines (q,m) with triangles F.

    Parameters:

    - `q`,`m`: (nq,3) shaped arrays of points and vectors, defining a single
      line or a set of lines.
    - `F`: (nF,3,3) shaped array, defining a single triangle or a set of
      triangles.
    - `mode`: `all` to calculate the intersection points of each line (q,m) with
      all triangles F or `pair` for pairwise intersections.
    - `return_all`: if True, all intersection points are returned. Default is
      to return only the points that lie inside the triangles.

    Returns:

      If `return_all==True`, a (nq,nF,3) shaped (`mode=all`) array of
      intersection points, else, a tuple of intersection points with shape (n,3)
      and line and plane indices with shape (n), where n <= nq*nF.
    """
    q = asanyarray(q).reshape(-1,3)
    m = asanyarray(m).reshape(-1,3)
    F = asanyarray(F).reshape(-1,3,3)
    if not return_all:
        # Find lines passing through the bounding spheres of the triangles
        r,c,n = triangleBoundingCircle(F)
        if mode == 'all':
##            d = distancesPFL(c,q,m,mode).transpose() # this is much slower for large arrays
            mode = 'pair'
            d = row_stack([ distancesPFL(c,q[i],m[i],mode) for i in range(q.shape[0]) ])
            wl,wt = where(d<=r)
        elif mode == 'pair':
            d = distancesPFL(c,q,m,mode)
            wl = wt = where(d<=r)[0]
        if wl.size == 0:
            return empty((0,3,),dtype=float),wl,wt
        q,m,F = q[wl],m[wl],F[wt]
    t = intersectionTimesLWT(q,m,F,mode)
    if mode == 'all':
        q = q[:,newaxis]
        m = m[:,newaxis]
    x = pointsAtLines(q,m,t)
    if not return_all:
        # Find points inside the faces
        ok = insideTriangle(F,x[newaxis]).reshape(-1)
        return x[ok],wl[ok],wt[ok]
    else:
        return x


def intersectionTimesSWT(S,F,mode='all'):
    """Return the intersection of lines segments S with triangles F.

    Parameters:

    - `S`: (nS,2,3) shaped array (`mode=all`) or broadcast compatible array
      (`mode=pair`), defining a single line segment or a set of line segments.
    - `F`: (nF,3,3) shaped array (`mode=all`) or broadcast compatible array
      (`mode=pair`), defining a single triangle or a set of triangles.
    - `mode`: `all` to calculate the intersection of each line segment S with
      all triangles F or `pair` for pairwise intersections.

    Returns a (nS,nF) shaped (`mode=all`) array of parameter values t,
    such that the intersection points are given by
    `(1-t)*S[...,0,:] + t*S[...,1,:]`.
    """
    Fn = cross(F[...,1,:]-F[...,0,:],F[...,2,:]-F[...,1,:])
    return intersectionTimesSWP(S,F[...,0,:],Fn,mode)


def intersectionPointsSWT(S,F,mode='all',return_all=False):
    """Return the intersection points of lines segments S with triangles F.

    Parameters:

    - `S`: (nS,2,3) shaped array, defining a single line segment or a set of
      line segments.
    - `F`: (nF,3,3) shaped array, defining a single triangle or a set of
      triangles.
    - `mode`: `all` to calculate the intersection points of each line segment S
      with all triangles F or `pair` for pairwise intersections.
    - `return_all`: if True, all intersection points are returned. Default is
      to return only the points that lie on the segments and inside the
      triangles.

    Returns:

      If `return_all==True`, a (nS,nF,3) shaped (`mode=all`) array of
      intersection points, else, a tuple of intersection points with shape (n,3)
      and line and plane indices with shape (n), where n <= nS*nF.
    """

    S = asanyarray(S).reshape(-1,2,3)
    F = asanyarray(F).reshape(-1,3,3)
    if not return_all:
        # Find lines passing through the bounding spheres of the triangles
        r,c,n = triangleBoundingCircle(F)
        if mode == 'all':
##            d = distancesPFS(c,S,mode).transpose() # this is much slower for large arrays
            mode = 'pair'
            d = row_stack([ distancesPFS(c,S[i],mode) for i in range(S.shape[0]) ])
            wl,wt = where(d<=r)
        elif mode == 'pair':
            d = distancesPFS(c,S,mode)
            wl = wt = where(d<=r)[0]
        if wl.size == 0:
            return empty((0,3,),dtype=float),wl,wt
        S,F = S[wl],F[wt]
    t = intersectionTimesSWT(S,F,mode)
    if mode == 'all':
        S = S[:,newaxis]
    x = pointsAtSegments(S,t)
    if not return_all:
        # Find points inside the segments and faces
        ok = (t >= 0.0) * (t <= 1.0) * insideTriangle(F,x[newaxis]).reshape(-1)
        return x[ok],wl[ok],wt[ok]
    else:
        return x


def intersectionPointsPWP(p1,n1,p2,n2,p3,n3,mode='all'):
    """Return the intersection points of planes (p1,n1), (p2,n2) and (p3,n3).

    Parameters:

    - `pi`,`ni` (i=1...3): (npi,3) shaped arrays of points and normals
      (`mode=all`)
      or broadcast compatible arrays (`mode=pair`), defining a single plane
      or a set of planes.
    - `mode`: `all` to calculate the intersection of each plane (p1,n1) with
      all planes (p2,n2) and (p3,n3) or `pair` for pairwise intersections.

    Returns a (np1,np2,np3,3) shaped (`mode=all`) array of intersection points.
    """
    if mode == 'all':
        p1 = asanyarray(p1).reshape(-1,1,1,3)
        n1 = asanyarray(n1).reshape(-1,1,1,3)
        p2 = asanyarray(p2).reshape(1,-1,1,3)
        n2 = asanyarray(n2).reshape(1,-1,1,3)
        p3 = asanyarray(p3).reshape(1,1,-1,3)
        n3 = asanyarray(n3).reshape(1,1,-1,3)
    dot1 = dotpr(p1,n1)[...,newaxis]
    dot2 = dotpr(p2,n2)[...,newaxis]
    dot3 = dotpr(p3,n3)[...,newaxis]
    cross23 = cross(n2,n3)
    cross31 = cross(n3,n1)
    cross12 = cross(n1,n2)
    denom = dotpr(n1,cross23)[...,newaxis]
    return (dot1*cross23+dot2*cross31+dot3*cross12)/denom


def intersectionLinesPWP(p1,n1,p2,n2,mode='all'):
    """Return the intersection lines of planes (p1,n1) and (p2,n2).

    Parameters:

    - `pi`,`ni` (i=1...2): (npi,3) shaped arrays of points and normals (`mode=all`)
      or broadcast compatible arrays (`mode=pair`), defining a single plane
      or a set of planes.
    - `mode`: `all` to calculate the intersection of each plane (p1,n1) with
      all planes (p2,n2) or `pair` for pairwise intersections.

    Returns a tuple of (np1,np2,3) shaped (`mode=all`) arrays of intersection
    points q and vectors m, such that the intersection lines are given by
    ``q+t*m``.
    """
    if mode == 'all':
        p1 = asanyarray(p1).reshape(-1,1,3)
        n1 = asanyarray(n1).reshape(-1,1,3)
        p2 = asanyarray(p2).reshape(1,-1,3)
        n2 = asanyarray(n2).reshape(1,-1,3)
    m =  cross(n1,n2)
    q = intersectionPointsPWP(p1,n1,p2,n2,p1,m,mode='pair')
    return q,m


def intersectionTimesPOP(X,p,n,mode='all'):
    """Return the intersection of perpendiculars from points X on planes (p,n).

    Parameters:

    - `X`: a (nX,3) shaped array of points (`mode=all`)
      or broadcast compatible array (`mode=pair`).
    - `p`,`n`: (np,3) shaped arrays of points and normals (`mode=all`)
      or broadcast compatible arrays (`mode=pair`), defining a single plane
      or a set of planes.
    - `mode`: `all` to calculate the intersection for each point X with
      all planes (p,n) or `pair` for pairwise intersections.

    Returns a (nX,np) shaped (`mode=all`) array of parameter values t,
    such that the intersection points are given by X+t*n.
    """
    if mode == 'all':
        return (dotpr(p,n) - inner(X,n)) / dotpr(n,n)
    elif mode == 'pair':
        return (dotpr(p,n) - dotpr(X,n)) / dotpr(n,n)


def intersectionPointsPOP(X,p,n,mode='all'):
    """Return the intersection points of perpendiculars from points X on planes (p,n).

    This is like intersectionTimesPOP but returns a (nX,np,3) shaped (`mode=all`)
    array of intersection points instead of the parameter values.
    """
    t = intersectionTimesPOP(X,p,n,mode)
    if mode == 'all':
        X = X[:,newaxis]
    return pointsAtLines(X,n,t)


def intersectionTimesPOL(X,q,m,mode='all'):
    """Return the intersection of perpendiculars from points X on lines (q,m).

    Parameters:

    - `X`: a (nX,3) shaped array of points (`mode=all`)
      or broadcast compatible array (`mode=pair`).
    - `q`,`m`: (nq,3) shaped arrays of points and vectors (`mode=all`)
      or broadcast compatible arrays (`mode=pair`), defining a single line
      or a set of lines.
    - `mode`: `all` to calculate the intersection for each point X with
      all lines (q,m) or `pair` for pairwise intersections.

    Returns a (nX,nq) shaped (`mode=all`) array of parameter values t,
      such that the intersection points are given by q+t*m.
    """
    if mode == 'all':
        return (inner(X,m) - dotpr(q,m)) / dotpr(m,m)
    elif mode == 'pair':
        return (dotpr(X,m) - dotpr(q,m)) / dotpr(m,m)


def intersectionPointsPOL(X,q,m,mode='all'):
    """Return the intersection points of perpendiculars from points X on lines (q,m).

    This is like intersectionTimesPOL but returns a (nX,nq,3) shaped (`mode=all`)
    array of intersection points instead of the parameter values.
    """
    t = intersectionTimesPOL(X,q,m,mode)
    return pointsAtLines(q,m,t)


def intersectionSphereSphere(R,r,d):
    """Intersection of two spheres (or two circles in the x,y plane).

    Computes the intersection of two spheres with radii R, resp. r, having
    their centres at distance d <= R+r. The intersection is a circle with
    its center on the segment connecting the two sphere centers at a distance
    x from the first sphere, and having a radius y. The return value is a
    tuple x,y.
    """
    if d > R+r:
        raise ValueError,"d (%s) should not be larger than R+r (%s)" % (d,R+r)
    dd = R**2-r**2+d**2
    d2 = 2*d
    x = dd/d2
    y = sqrt(d2**2*R**2 - dd**2) / d2
    return x,y


#################### distance tools ###############

def distancesPFL(X,q,m,mode='all'):
    """Return the distances of points X from lines (q,m).

    Parameters:

    - `X`: a (nX,3) shaped array of points (`mode=all`)
      or broadcast compatible array (`mode=pair`).
    - `q`,`m`: (nq,3) shaped arrays of points and vectors (`mode=all`)
      or broadcast compatible arrays (`mode=pair`), defining a single line
      or a set of lines.
    - `mode`: `all` to calculate the distance of each point X from
      all lines (q,m) or `pair` for pairwise distances.

    Returns a (nX,nq) shaped (`mode=all`) array of distances.
    """
    Y = intersectionPointsPOL(X,q,m,mode)
    if mode == 'all':
        X = asarray(X).reshape(-1,1,3)
    return length(Y-X)


def distancesPFS(X,S,mode='all'):
    """Return the distances of points X from line segments S.

    Parameters:

    - `X`: a (nX,3) shaped array of points (`mode=all`)
      or broadcast compatible array (`mode=pair`).
    - `S`: (nS,2,3) shaped array of line segments (`mode=all`)
      or broadcast compatible array (`mode=pair`), defining a single line
      segment or a set of line segments.
    - `mode`: `all` to calculate the distance of each point X from
      all line segments S or `pair` for pairwise distances.

    Returns a (nX,nS) shaped (`mode=all`) array of distances.
    """
    q0 = S[...,0,:]
    q1 = S[...,1,:]
    return distancesPFL(X,q0,q1-q0,mode)


def insideTriangle(x,P,method='bary'):
    """Checks whether the points P are inside triangles x.

    x is a Coords array with shape (ntri,3,3) representing ntri triangles.
    P is a Coords array with shape (npts,ntri,3) representing npts points
    in each of the ntri planes of the triangles.
    This function checks whether the points of P fall inside the corresponding
    triangles.

    Returns an array with (npts,ntri) bool values.
    """
    if method == 'bary':
        return insideSimplex(baryCoords(x,P))
    else:
        # Older, slower algorithm
        xP = x[newaxis,...] - P[:,:,newaxis,:]
        xx = [ cross(xP[:,:,i],xP[:,:,j]) for (i,j) in ((0,1),(1,2),(2,0)) ]
        xy = (xx[0]*xx[1]).sum(axis=-1)
        yz = (xx[1]*xx[2]).sum(axis=-1)
        d = dstack([xy,yz])
        return (d > 0).all(axis=-1)


def faceDistance(X,Fp,return_points=False):
    """Compute the closest perpendicular distance to a set of triangles.

    X is a (nX,3) shaped array of points.
    Fp is a (nF,3,3) shaped array of triangles.

    Note that some points may not have a normal with footpoint inside any
    of the facets.

    The return value is a tuple OKpid,OKdist,OKpoints where:

    - OKpid is an array with the point numbers having a normal distance;
    - OKdist is an array with the shortest distances for these points;
    - OKpoints is an array with the closest footpoints for these points
      and is only returned if return_points = True.
    """
    if not Fp.shape[1] == 3:
        raise ValueError, "Currently this function only works for triangular faces."
    # Compute normals on the faces
    Fn = cross(Fp[:,1]-Fp[:,0],Fp[:,2]-Fp[:,1])
    # Compute intersection points of perpendiculars from X on facets F
    Y = intersectionPointsPOP(X,Fp[:,0,:],Fn)
    # Find intersection points Y inside the facets
    inside = insideTriangle(Fp,Y)
    pid = where(inside)[0]
    if pid.size == 0:
        if return_points:
            return [],[],[]
        else:
            return [],[]

    # Compute the distances
    X = X[pid]
    Y = Y[inside]
    dist = length(X-Y)
    # Get the shortest distances
    OKpid,OKpos = groupArgmin(dist,pid)
    OKdist = dist[OKpos]
    if return_points:
        # Get the closest footpoints matching OKpid
        OKpoints = Y[OKpos]
        return OKpid,OKdist,OKpoints
    return OKpid,OKdist


def edgeDistance(X,Ep,return_points=False):
    """Compute the closest perpendicular distance of points X to a set of edges.

    X is a (nX,3) shaped array of points.
    Ep is a (nE,2,3) shaped array of edge vertices.

    Note that some points may not have a normal with footpoint inside any
    of the edges.

    The return value is a tuple OKpid,OKdist,OKpoints where:

    - OKpid is an array with the point numbers having a normal distance;
    - OKdist is an array with the shortest distances for these points;
    - OKpoints is an array with the closest footpoints for these points
      and is only returned if return_points = True.
    """
    # Compute vectors along the edges
    En = Ep[:,1] - Ep[:,0]
    # Compute intersection points of perpendiculars from X on edges E
    t = intersectionTimesPOL(X,Ep[:,0],En)
    Y = Ep[:,0] + t[:,:,newaxis] * En
    # Find intersection points Y inside the edges
    inside = (t >= 0.) * (t <= 1.)
    pid = where(inside)[0]
    if pid.size == 0:
        if return_points:
            return [],[],[]
        else:
            return [],[]

    # Compute the distances
    X = X[pid]
    Y = Y[inside]
    dist = length(X-Y)
    # Get the shortest distances
    OKpid,OKpos = groupArgmin(dist,pid)
    OKdist = dist[OKpos]
    if return_points:
        # Get the closest footpoints matching OKpid
        OKpoints = Y[OKpos]
        return OKpid,OKdist,OKpoints
    return OKpid,OKdist


def vertexDistance(X,Vp,return_points=False):
    """Compute the closest distance of points X to a set of vertices.

    X is a (nX,3) shaped array of points.
    Vp is a (nV,3) shaped array of vertices.

    The return value is a tuple OKdist,OKpoints where:

    - OKdist is an array with the shortest distances for the points;
    - OKpoints is an array with the closest vertices for the points
      and is only returned if return_points = True.
    """
    # Compute the distances
    dist = length(X[:,newaxis]-Vp)
    # Get the shortest distances
    OKdist = dist.min(-1)
    if return_points:
        # Get the closest points matching X
        minid = dist.argmin(-1)
        OKpoints = Vp[minid]
        return OKdist,OKpoints
    return OKdist,


#################### barycentric coordinates ###############

def baryCoords(S,P):
    """Compute the barycentric coordinates of points  P wrt. simplexes S.

    S is a (nel,nplex,3) shaped array of n-simplexes (n=nplex-1):
    - 1-simplex: line segment
    - 2-simplex: triangle
    - 3-simplex: tetrahedron
    P is a (npts,3), (npts,nel,3) or (npts,1,3) shaped array of points.

    The return value is a (nplex,npts,nel) shaped array of barycentric coordinates.
    """
    if S.ndim != 3:
        raise ValueError,"S should be a 3-dim array, got shape %s" % str(S.shape)
    if P.ndim == 2:
        P = P.reshape(-1,1,3)
    elif P.shape[1] != S.shape[0] and P.shape[1] != 1:
        raise ValueError,"Second dimension of P should be first dimension of S or 1."
    S = S.transpose(1,0,2) # (nplex,nel,3)
    vp = P - S[0]
    vs = S[1:] - S[:1]
    A = dotpr(vs[:,newaxis],vs[newaxis]) # (nplex-1,nplex-1,nel)
    b = dotpr(vp[newaxis],vs[:,newaxis]) # (nplex-1,npts,nel)
    #import timer
    #T = timer.Timer()
    t = solveMany(A,b)
    #print "DIRECT SOLVER: %s" % T.seconds()
    #T.reset()
    #tt = solveMany(A,b,False)
    #print "GENERAL SOLVER: %s" % T.seconds()
    #print "RESULTS MATCH: %s" % (tt-t).sum()

    t0 = (1.-t.sum(0))
    t0 = addAxis(t0,0)
    t = row_stack([t0,t])
    return t


def insideSimplex(BC,bound=True):
    """Check if points are in simplexes.

    BC is an array of barycentric coordinates (along the first axis),
    which sum up to one.
    If bound = True, a point lying on the boundary is considered to
    be inside the simplex.
    """
    if bound:
        return (BC >= 0.).all(0)
    else:
        return (BC > 0.).all(0)


# End
