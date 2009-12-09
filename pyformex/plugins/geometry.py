# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Tue Dec  8 12:25:08 2009
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
"""Basic geometrical operations.

This module defines some basic operations on simple geometrical entities
such as lines, triangles, circles.
"""

from formex import *


def triangleCircumCircle(x):
    """Compute the circumcircles of the triangles x

    x is a Coords array with shape (ntri,3,3) representing ntri triangles.
    
    Returns a tuple r,C,n with the radii, Center and unit normals of the
    circles going through the vertices of each triangle.
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
    return r,C,n


def lineIntersection(P1,D1,P2,D2,visual_debug=False):
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
    if visual_debug:
        f = zeros((N,2,3))
        f[:,0,:] = P1
        f[:,1,:] = P1+D1
        F = Formex(f,2)
        f[:,0,:] = P2
        f[:,1,:] = P2+D2
        G = Formex(f,3)
        #clear()
        draw(F)
        draw(G)
        zoomAll()
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
    if visual_debug:
        H = connect([Formex(X),Formex(P1)])
        H.setProp(1)
        draw(H)
        H = connect([Formex(X),Formex(P2)])
        H.setProp(5)
        draw(H)
        zoomAll()
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


# !! We have to merge the following two functions

def rotation(x,r,angle_spec=Deg):
    """Return a rotation as an axis + angle.

    ``x`` and ``r`` are two unit vectors.
    The return value is a tuple of:
    
    - a vector perpendicular to ``x`` and ``r``,
    - the angle between both ``x`` and ``r``.

    angle_spec can be set to ``Deg`` ro ``Rad`` to get the result in
    degrees (default) or radians.
    """
    w = cross(x,r)
    wl = length(w)
    if wl == 0.0:
        return [0.,0.,1.],0.
    else:
        w /= wl
        angle = arccos(dotpr(x,r)) / angle_spec
        return w,angle
    

def rotationAngle(A,B,angle_spec=Deg):
    """Return rotation vector and angle for rotation of A to B.

    A and B are (n,3)-shaped arrays where each line represents a vector.
    This function computes the rotation from each vector of A to the
    corresponding vector of B. Broadcasting is done if one of A or B has
    only one row.
    The return value is a tuple of an (n,) shaped array with rotation angles
    (by default in degrees) and an (n,3)-shaped array with unit vectors
    along the rotation axis.
    Specify angle_spec=Rad to get the angles in radians.
    """
    A = normalize(A)
    B = normalize(B)
    N = cross(A,B)
    L = length(N)
    S = L / (length(A)*length(B))  # !! SEEMS SUPERFLUOUS 
    ANG = arcsin(S.clip(min=-1.0,max=1.0)) / angle_spec
    N = N/column_stack([L])
    return N,ANG

# End
