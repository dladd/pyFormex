#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""Formex algebra in Python

This module is a Python implementation of most functions of Formex algebra.
It forms the basis of pyFormex, but predates it and is fairly independent.
Therefore, it can also be used as a standalone extension module in Python.
"""

from coords import *
from utils import deprecation,deprecated,functionWasRenamed,functionBecameMethod


def vectorNormalize(vec):
    """Normalize a set of vectors.

    vec is a (n,3) shaped arrays holding a collection of vectors.
    The result is a tuple of two arrays:
      length (n) : the length of the vectors vec.
      normal (n,3) : unit-length vectors along vec.
    """
    length = sqrt((vec*vec).sum(axis=-1))
    normal = vec / length.reshape((-1,1))
    return length,normal


def vectorPairAreaNormals(vec1,vec2):
    """Compute area of and normals on parallellograms formed by vec1 and vec2.

    vec1 and vec2 are (n,3) shaped arrays holding collections of vectors.
    The result is a tuple of two arrays:
      area (n) : the area of the parallellogram formed by vec1 and vec2.
      normal (n,3) : (normalized) vectors normal to each couple (vec1,2).
    These are calculated from the cross product of vec1 and vec, which indeed
    gives area * normal.
    """
    normal = cross(vec1,vec2)
    area = sqrt((normal*normal).sum(axis=-1))
    normal /= area.reshape((-1,1))
    return area,normal


def vectorPairArea(vec1,vec2):
    """Compute area of the parallellogram formed by a vector pair vec1,vec2.

    vec1 and vec2 are (n,3) shaped arrays holding collections of vectors.
    The result is an (n) shaped array with the area of the parallellograms
    formed by each pair of vectors (vec1,vec2).
    """
    return vectorPairAreaNormals(vec1,vec2)[0]


def vectorPairNormals(vec1,vec2,normalized=True):
    """Compute vectors normal to vec1 and vec2.

    vec1 and vec2 are (n,3) shaped arrays holding collections of vectors.
    The result is an (n,3) shaped array of vectors normal to each couple
    (edg1,edg2).
    Default is to normalize the vectors to unit length.
    If not essential, this can be switched off to save computing time.
    """
    if normalized:
        return vectorPairAreaNormals(vec1,vec2)[1]
    else:
        return cross(vec1,vec2)


def vectorTripleProduct(vec1,vec2,vec3):
    """Compute triple product vec1 . (vec2 x vec3).

    vec1, vec2, vec3 are (n,3) shaped arrays holding collections of vectors.
    The result is a (n,) shaped array with the triple product of each set
    of corresponding vectors fromvec1,vec2,vec3.
    This is also the square of the volume of the parallellepid formex by
    the 3 vectors.
    """
    return dot(vec1,cross(vec2,vec3))
    

def polygonNormals(x):
    """Compute normals in all points of polygons in x.

    x is an (nel,nplex,3) coordinate array representing a (possibly not plane)
    polygon.
    
    The return value is an (nel,nplex,3) array with the unit normals on the
    two edges ending in each point.
    """
    ni = arange(x.shape[1])
    nj = roll(ni,1)
    nk = roll(ni,-1)
    v1 = x-x[:,nj]
    v2 = x[:,nk]-x
    return vectorPairNormals(v1.reshape(-1,3),v2.reshape(-1,3)).reshape(x.shape)


def pattern(s):
    """Return a line segment pattern created from a string.

    This function creates a list of line segments where all nodes lie on the
    gridpoints of a regular grid with unit step.
    The first point of the list is [0,0,0]. Each character from the given
    string is interpreted as a code specifying how to move to the next node.
    Currently defined are the following codes:
    0 = goto origin [0,0,0]
    1..8 move in the x,y plane
    9 remains at the same place
    When looking at the plane with the x-axis to the right,
    1 = East, 2 = North, 3 = West, 4 = South, 5 = NE, 6 = NW, 7 = SW, 8 = SE.
    Adding 16 to the ordinal of the character causes an extra move of +1 in
    the z-direction. Adding 48 causes an extra move of -1. This means that
    'ABCDEFGHI', resp. 'abcdefghi', correspond with '123456789' with an extra
    z +/-= 1. This gives the following schema:

                 z+=1             z unchanged            z -= 1
            
             F    B    E          6    2    5         f    b    e 
                  |                    |                   |     
                  |                    |                   |     
             C----I----A          3----9----1         c----i----a  
                  |                    |                   |     
                  |                    |                   |     
             G    D    H          7    4    8         g    d    h
             
    The special character '/' can be put before any character to make the
    move without making a connection.
    The effect of any other character is undefined.
    
    The resulting list is directly suited to initialize a Formex.
    """
    x = y = z = 0
    l = []
    connect=True
    for c in s:
        if c == "/":
            connect = False
            continue
        pos = [x,y,z]
        if c == "0":
            x = y = z = 0
        else:
            i = ord(c)
            d = i/16
            if d == 3:
                pass
            elif d == 4:
                z += 1
            elif d == 6:
                z -= 1
            else:
                raise RuntimeError,"Unknown pattern character %c ignored" % c
            i %= 16
            if i == 1:
                x += 1
            elif i == 2:
                y += 1
            elif i == 3:
                x -= 1
            elif i == 4:
                y -= 1
            elif i == 5:
                x += 1
                y += 1
            elif i == 6:
                x -= 1
                y += 1
            elif i == 7:
                x -= 1
                y -= 1
            elif i == 8:
                x += 1
                y -= 1
            elif i == 9:
                pass
            else:
                raise RuntimeError,"Unknown pattern character %c ignored" % c
        if connect:
            l.append([pos,[x,y,z]])
        connect=True
    return l

def mpattern(s):
    """This is like pattern, but allowing lists with more than 2 points.

    Subsequent points are included in the same list until a '-' occurs.
    A '-' character splits lists. Each list starts at the last point of
    the previous list.
    All lists should have equal length if you want to use the resulting
    list to initialize a Formex.
    """
    x = y = z = 0
    li = [[x,y,z]]
    l = []
    connect=True
    for c in s:
        if c == '/':
            connect = False
            continue
        elif c == '-':
            l.append(li)
            li = []
        elif c == '0':
            x = y = z = 0
        else:
            i = ord(c)
            d = i/16
            if d == 3:
                pass
            elif d == 4:
                z += 1
            elif d == 6:
                z -= 1
            else:
                raise RuntimeError,"Unknown pattern character %c ignored" % c
            i %= 16
            if i == 1:
                x += 1
            elif i == 2:
                y += 1
            elif i == 3:
                x -= 1
            elif i == 4:
                y -= 1
            elif i == 5:
                x += 1
                y += 1
            elif i == 6:
                x -= 1
                y += 1
            elif i == 7:
                x -= 1
                y -= 1
            elif i == 8:
                x += 1
                y -= 1
            elif i == 9:
                pass
            else:
                raise RuntimeError,"Unknown pattern character %c ignored" % c
        if connect:
            li.append([x,y,z])
    l.append(li)
    return l

# Intersection functions
#
# !! These functions currently also exist as formex methods.
# !! They only work on plex-2 formices.
# !! Therefore it is not clear if they really belong in the formex class,
# !! or should go to a specialized subclass
# !! It is also not clear what their return value should be.
# !! Until this is decided, we keep them here as global functions.


def intersectionWithPlane(F,p,n):
    """Return the intersection of a Formex F with the plane (p,n).

    The Formex should have plexitude 2.
    p is a point specified by 3 coordinates.
    n is the normal vector to a plane, specified by 3 components.

    The return value is a [n] shaped array of parameter values t,
    such that for each segment L the intersection point is given
    by (1-t)*L[0]+ t*L[1].
    """
    f = F.f
    if f.shape[1] != 2:
        raise RuntimeError,"Formex should have plexitude 2."
    p = asarray(p).reshape((3))
    n = asarray(n).reshape((3))
    n /= length(n)
    t = (inner(p,n) - inner(f[:,0,:],n)) / inner((f[:,1,:]-f[:,0,:]),n)
    return t


def pointsAt(F,t):
    """Return the points of a plex-2 Formex at times t.
    
    F is a plex 2 Formex and t is an array with F.nelems() float values which
    are interpreted as local parameters along the edges of the Formex, such
    that the first node has value 0.0 and the last has vaue 1.0.
    The return value is a Coords array with the points at values t.
    """
    f = F.f
    t = t[:,newaxis]
    return Coords((1.-t) * f[:,0,:] + t * f[:,1,:])


def intersectionPointsWithPlane(F,p,n):
    """Return the intersection points of a Formex with plane p,n.

    The Formex should have plexitude 2.
    p is a point specified by 3 coordinates.
    n is the normal vector to a plane, specified by 3 components.

    The result is a plex-1 Formex with the same number of elements as the
    original. Some of the points may be NaN's.
    """
    f = F.f
    t = intersectionWithPlane(F,p,n).reshape((-1,1))
    return Formex((1.-t) * f[:,0,:] + t * f[:,1,:])


def intersectionLinesWithPlane(F,p,n,atol=1.e-4):
    """Return the intersection lines of a plex-3 Formex with plane (p,n).
    
    F is a Formex of plexitude 3.
    p is a point specified by 3 coordinates.
    n is the normal vector to a plane, specified by 3 components.
    atol is a tolerance factor defining whether an edge is intersected by the plane.
    """
    n = asarray(n)
    p = asarray(p)
    F = F.cclip(F.test('all',n,p)) # remove elements at the negative side
    if F.nelems() == 0:
        return Formex()
    F = F.cclip(F.test('all',-n,p)) # select elements that will be cut by plane
    if F.nelems() == 0:
        return Formex()
    F1 = F21 = F22 = F31 = F32 = F41 = F42= F43 = Formex(empty((0,2,3,),dtype=float))    
    # Create a Formex with the edges
    C = Formex.concatenate([ F.selectNodes(e) for e in [[0,1],[1,2],[2,0]] ])
    t = C.intersectionWithPlane(p,n)
    P = pointsAt(C,t)
    t = t.reshape(3,-1).transpose()
    Pb = P.reshape(3,-1,3).swapaxes(0,1)
    Pf = F.f
    Ps = roll(F.f,-1,axis=1)
    t1 = t >= 0.+atol
    t2 = t <= 1.-atol
    t3 = t >= 0.-atol
    t4 = t <= 1.+atol
    Tb = t1 * t2
    Tf = (1-t1).astype(bool)*t3
    Ts = (1-t2).astype(bool)*t4
    Nb = Tb.sum(axis=-1)
    Nf = Tf.sum(axis=-1)
    Ns = Ts.sum(axis=-1)
    # Get the triangles with 2 edge intersections
    w1 = where(Nb==2)[0]
    if w1.size > 0:
        P = Pb[w1][Tb[w1]].reshape(-1,2,3)
        F1 = Formex(P)
    # Get the triangles with 1 edge intersection and 1 vertex intersection
    w21 = where( (Nb==1) * (Nf==1) * (Ns==0) )[0]
    if w21.size > 0:
        P1 = Pb[w21][Tb[w21]].reshape(-1,1,3)
        P2 = Pf[w21][Tf[w21]].reshape(-1,1,3)
        P = column_stack([P1,P2])
        F21 = Formex(P)
    w22 = where( (Nb==1) * (Nf==0) * (Ns==1) )[0]
    if w22.size > 0:
        P1 = Pb[w22][Tb[w22]].reshape(-1,1,3)
        P2 = Ps[w22][Ts[w22]].reshape(-1,1,3)
        P = column_stack([P1,P2])
        F22 = Formex(P)
    # Get the triangles with 1 edge intersection and 2 vertex intersections
    w3 = where( (Nb==1) * (Nf==1) * (Ns==1) )[0]
    if w3.size > 0:
        Tb3 = Tb[w3]
        Tf3 = Tf[w3]
        Ts3 = Ts[w3]        
        Pb3 = Pb[w3]
        Pf3 = Pf[w3]
        Ps3 = Ps[w3]
        i = where(Ts3)[1] - where(Tf3)[1]
        w31 = where((i == 1)+(i==-2))[0] # different vertices
        if w31.size > 0:
            P1 = Pf3[w31][Tf3[w31]].reshape(-1,1,3)
            P2 = Ps3[w31][Ts3[w31]].reshape(-1,1,3)            
            P = column_stack([P1,P2])
            F32 = Formex(P)
        w32 = where((i == -1)+(i==2))[0] # equal vertices
        if w32.size > 0:   
            P1 = Pb3[w32][Tb3[w32]].reshape(-1,1,3)
            P2 = Pf3[w32][Tf3[w32]].reshape(-1,1,3)
            P = column_stack([P1,P2])
            F31 = Formex(P)
    # Get the triangles with 0 edge intersections and 2 or 3 vertex intersections
    w41 = where( (Nb==0) * (Nf==2) )[0]
    if w41.size > 0:
        P = Pf[w41][Tf[w41]].reshape(-1,2,3)
        F41 = Formex(P)
    w42 = where( (Nb==0) * (Ns==2) )[0]
    if w42.size > 0:
        P = Ps[w42][Ts[w42]].reshape(-1,2,3)
        F42 = Formex(P)
    w43 = where( (Nb==0) * (Nf==1) * (Ns==1) )[0]
    if w43.size > 0:
        Tf43 = Tf[w43]
        Ts43= Ts[w43]
        Pf43 = Pf[w43]
        Ps43 = Ps[w43]
        i = where(Ts43)[1] - where(Tf43)[1]
        w43 = where((i == 1)+(i==-2))[0] # different vertices
        if w43.size > 0:
            P1 = Pf43[w43][Tf43[w43]].reshape(-1,1,3)
            P2 = Ps43[w43][Ts43[w43]].reshape(-1,1,3)            
            P = column_stack([P1,P2])
            F43 = Formex(P)
    # join all the pieces
    Ft = F1 + F21 + F22 + F31 + F32 + F41 + F42+ F43
    return Ft


def _sane_side(side):
    # allow some old variants of arguments
    if type(side) == str:
        if side.startswith('pos'):
            side = '+'
        if side.startswith('neg'):
            side = '-'
    if not (side == '+' or side == '-'):
        side = ''
    return side

def _select_side(side,alist):
    # return selected parts dependent on side
    if side == '+':
        return alist[0]
    elif side == '-':
        return alist[1]
    else:
        return alist


def cut2AtPlane(F,p,n,side='',atol=None,newprops=None):
    """Returns all elements of the Formex cut at plane.

    F is a Formex of plexitude 2.
    p is a point specified by 3 coordinates.
    n is the normal vector to a plane, specified by 3 components.

    The return value is:
    - with side = '+' or '-' or 'positive'or 'negative' :
    a Formex of the same plexitude with all elements
    located completely at the positive/negative side of the plane(s) (p,n)
    retained, all elements lying completely at the negative/positive side
    removed and the elements intersecting the plane(s) replaced by new
    elements filling up the parts at the positive/negative side.
    - with side = '': two Formices of the same plexitude, one representing
    the positive side and one representing the negative side.

    To avoid roundoff errors and creation of very small elements,
    a tolerance can be specified. Points lying within the tolerance
    distance will be considered lying in the plane, and no cutting near
    these points.
    """
    side = _sane_side(side)
    dist = F.distanceFromPlane(p,n)
    if atol is None:
        atol = 1.e-5*dist.max()
    above = sum(dist>atol,-1)
    below = sum(dist<-atol,-1)
    A = F.clip(below==0)
    B = F.clip(above==0)
    cutting = (above>0)*(below>0)
    if newprops:
       A.setProp(newprops[0]) 
       B.setProp(newprops[1])
    ## print "Elements in F: %s" % F.nelems()
    ## print "Elements in A: %s" % A.nelems()
    ## print "Elements in B: %s" % B.nelems()

    if cutting.any():
        G = F.clip(cutting)
        H = G.copy()
        g = intersectionPointsWithPlane(G,p,n)
        dist = dist[cutting]
        i0 = dist[:,0] < 0.
        i1 = dist[:,1] < 0.
        G[i0,0,:] = H[i0,1,:] = g[i0].reshape(-1,3)
        G[i1,1,:] = H[i1,0,:] = g[i1].reshape(-1,3)
        if newprops:
           G.setProp(newprops[2])
           H.setProp(newprops[3])
        A += G
        B += H

        ## print "Elements in G: %s" % G.nelems()
        ## print "Elements in A: %s" % A.nelems()
        ## print "Elements in B: %s" % B.nelems()
    return _select_side(side,[ A,B ])


def cut3AtPlane(F,p,n,side='',atol=None,newprops=None):
    """Returns all elements of the Formex cut at plane(s).

    F is a Formex of plexitude 3.
    p is a point or a list of points.
    n is the normal vector to a plane or a list of normal vectors.
    Both p and n have shape (3) or (npoints,3).
    
    The return value is:
    - with side = '+' or '-' or 'positive'or 'negative' :
    a Formex of the same plexitude with all elements
    located completely at the positive/negative side of the plane(s) (p,n)
    retained, all elements lying completely at the negative/positive side
    removed and the elements intersecting the plane(s) replaced by new
    elements filling up the parts at the positive/negative side.
    - with side = '': two Formices of the same plexitude, one representing
    the positive side and one representing the negative side.
    
    The elements located completely at the positive/negative side of a plane
    have three vertices for which the |distance| to the plane > atol.
    The elements intersecting a plane can have one or more vertices for which
    |distance| < atol. These vertices are projected on the plane so that their
    distance is zero.
    
    If the Formex has a property set, the new elements will get the property
    numbers defined in newprops. This is a list of 7 property numbers flagging
    elements with following properties:
    0) no vertices with |distance| < atol, triangle after cut
    1) no vertices with |distance| < atol, triangle 1 from quad after cut
    2) no vertices with |distance| < atol, triangle 2 from quad after cut
    3) one vertex with |distance| < atol, two vertices at pos. or neg. side
    4) one vertex with |distance| < atol, one vertex at pos. side, one at neg.
    5) two vertices with |distance| < atol, one vertex at pos. or neg. side
    6) three vertices with |distance| < atol
    """
    if atol is None:
        atol = 1.e-5*F.dsize()
    # make sure we have sane newprops
    if newprops is None:
        newprops = [None,]*7
    else:
        try:
            newprops = newprops[:7]
            for prop in newprops:
                if not (prop is None or type(prop) is int):
                    raise
        except:
            newprops = range(7)
    side = _sane_side(side)
       
    p = asarray(p).reshape(-1,3)
    n = asarray(n).reshape(-1,3)
    nplanes = len(p)
    test = [F.test('any',n[i], p[i],atol=atol) for i in range(nplanes)] # elements at positive side of plane i
    Test= asarray(test).prod(0) # elements at positive side of all planes
    F_pos = F.clip(Test) # save elements at positive side of all planes
    if side in '-': # Dirty trick: this also includes side='' !
        F_neg = F.cclip(Test) # save elements completely at negative side of one of the planes
    else:
        F_neg = None
    if F_pos.nelems() != 0:
        test = [F_pos.test('all',n[i],p[i],atol=-atol) for i in range(nplanes)] # elements completely at positive side of plane i
        Test = asarray(test).prod(0) # elements completely at positive side of all planes
        F_cut = F_pos.cclip(Test) # save elements that will be cut by one of the planes
        F_pos = F_pos.clip(Test)  # save elements completely at positive side of all planes
        if F_cut.nelems() != 0:
            if nplanes == 1:
                if side == '+':
                    F_pos += cutElements3AtPlane(F_cut,p[i],n[i],newprops,side,atol)
                elif side == '-':
                    F_neg += cutElements3AtPlane(F_cut,p[i],n[i],newprops,side,atol)
                elif side == '':
                    cut_pos, cut_neg = cutElements3AtPlane(F_cut,p[i],n[i],newprops,side,atol)
                    F_pos += cut_pos
                    F_neg += cut_neg
            elif nplanes > 1:
                S = F_cut
                for i in range(nplanes):
                    if i > 0:
                        # due to the projection of vertices with |distance| < atol on plane i-1, some elements can be completely at negative side of plane i instead of cut by plane i
                        t = S.test('any',n[i],p[i],atol=atol)
                        if side in '-':
                            F_neg += S.cclip(t) # save elements completely at negative side of plane i
                        S = S.clip(t) # save elements at positive side of plane i
                    t = S.test('all',n[i],p[i],atol=-atol)
                    R = S.clip(t) # save elements completely at positive side of plane i
                    S = S.cclip(t) # save elements that will be cut by plane i
                    if side == '+':
                        cut_pos = cutElements3AtPlane(S,p[i],n[i],newprops,'+',atol)
                    elif side in '-':
                        cut_pos, cut_neg = cutElements3AtPlane(S,p[i],n[i],newprops,'',atol)
                        F_neg += cut_neg
                    S = R + cut_pos
                F_pos += S
                
    return _select_side(side,[ F_pos, F_neg ])


def cutElements3AtPlane(F,p,n,newprops=None,side='',atol=0.):
    """This function needs documentation.

    Should it be called by the user? or only via cut3AtPlane?
    For now, lets suppose the last, so no need to check arguments here.
    
    newprops should be a list of 7 values: each an integer or None
    side is either '+', '-' or ''
    """
    if atol is None:
        atol = 1.e-5*F.dsize()
        
    def get_new_prop(p,ind,newp):
        """Determines the value of the new props for a subset.

        p are the original props (possibly None)
        ind is the list of elements to treat
        newp is the new property value.

        The return value is determined as follows:
        - If p is None: return None (no property set)
        - If p is set, but newp is None: return p[ind] : keep original
        - if p is set, and newp is set: return newp (single value)
        """
        if p is None:
            return None
        elif newp is None:
            return p[ind]
        else:
            return newp
    # get_new_prop

    C = [connect([F,F],nodid=ax) for ax in [[0,1],[1,2],[2,0]]]
    t = column_stack([Ci.intersectionWithPlane(p,n) for Ci in C])
    P = column_stack([Ci.intersectionPointsWithPlane(p,n).f for Ci in C])    
    T = (t >= 0.)*(t <= 1.)
    d = F.f.distanceFromPlane(p,n)
    U = abs(d) < atol
    V = U.sum(axis=-1) # number of vertices with |distance| < atol
    F1_pos = F2_pos = F3_pos = F4_pos = F5_pos = F6_pos = F7_pos = F1_neg = F2_neg = F3_neg = F4_neg = F5_neg = F6_neg = F7_neg = Formex()
    # No vertices with |distance| < atol => triangles with 2 intersections
    w1 = where(V==0)[0]
    if w1.size > 0:
        T1 = T[w1]
        P1 = P[w1][T1].reshape(-1,2,3)
        F1 = F[w1]
        d1 = d[w1]
        if F.p is None:
            p1 = None
        else:
            p1 = F.p[w1]
        # split problem in two cases
        w11 = where(d1[:,0]*d1[:,1]*d1[:,2] > 0.)[0] # case 1: triangle at positive side after cut
        w12 = where(d1[:,0]*d1[:,1]*d1[:,2] < 0.)[0] # case 2: quadrilateral at positive side after cut
        # case 1: triangle at positive side after cut
        if w11.size > 0:
            T11 = T1[w11]
            P11 = P1[w11]
            F11 = F1[w11]
            if side in '+':
                v1 = where(T11[:,0]*T11[:,2] == 1,0,where(T11[:,0]*T11[:,1] == 1,1,2))
                K1 = asarray([F11[j,v1[j]] for j in range(shape(F11)[0])]).reshape(-1,1,3)
                E1_pos = column_stack([P11,K1])
                F1_pos = Formex(E1_pos,get_new_prop(p1,w11,newprops[0]))
            if side in '-': #quadrilateral at negative side after cut
                v2 = where(T11[:,0]*T11[:,2] == 1,2,where(T11[:,0]*T11[:,1] == 1,2,0))
                v3 = where(T11[:,0]*T11[:,2] == 1,1,where(T11[:,0]*T11[:,1] == 1,0,1))
                K2 = asarray([F11[j,v2[j]] for j in range(shape(F11)[0])]).reshape(-1,1,3)
                K3 = asarray([F11[j,v3[j]] for j in range(shape(F11)[0])]).reshape(-1,1,3)
                E2_neg = column_stack([P11,K2])
                F2_neg = Formex(E2_neg,get_new_prop(p1,w11,newprops[1]))
                E3_neg = column_stack([P11[:,0].reshape(-1,1,3),K2,K3])
                F3_neg = Formex(E3_neg,get_new_prop(p1,w11,newprops[2]))
        # case 2: quadrilateral at positive side after cut
        if w12.size > 0:
            T12 = T1[w12]
            P12 = P1[w12]
            F12 = F1[w12]
            if side in '+':
                v2 = where(T12[:,0]*T12[:,2] == 1,2,where(T12[:,0]*T12[:,1] == 1,2,0))
                v3 = where(T12[:,0]*T12[:,2] == 1,1,where(T12[:,0]*T12[:,1] == 1,0,1))
                K2 = asarray([F12[j,v2[j]] for j in range(shape(F12)[0])]).reshape(-1,1,3)
                K3 = asarray([F12[j,v3[j]] for j in range(shape(F12)[0])]).reshape(-1,1,3)
                E2_pos = column_stack([P12,K2])
                F2_pos = Formex(E2_pos,get_new_prop(p1,w12,newprops[1]))
                E3_pos = column_stack([P12[:,0].reshape(-1,1,3),K2,K3])
                F3_pos = Formex(E3_pos,get_new_prop(p1,w12,newprops[2]))
            if side in '-': # triangle at negative side after cut
                v1 = where(T12[:,0]*T12[:,2] == 1,0,where(T12[:,0]*T12[:,1] == 1,1,2))
                K1 = asarray([F12[j,v1[j]] for j in range(shape(F12)[0])]).reshape(-1,1,3)
                E1_neg = column_stack([P12,K1])
                F1_neg = Formex(E1_neg,get_new_prop(p1,w12,newprops[0]))
    # One vertex with |distance| < atol
    w2 = where(V==1)[0]
    if w2.size > 0:
        F2 = F[w2]
        d2 = d[w2]
        U2 = U[w2]
        if F.p is None:
            p2 = None
        else:
            p2 = F.p[w2]
        # split problem in three cases
        W = (d2 > atol).sum(axis=-1)
        w21 = where(W == 2)[0] # case 1: two vertices at positive side
        w22 = where(W == 1)[0] # case 2: one vertex at positive side
        w23 = where(W == 0)[0] # case 3: no vertices at positive side 
        # case 1: two vertices at positive side
        if w21.size > 0 and side in '+':
            F21 = F2[w21]
            U21 = U2[w21]
            K1 = F21[U21] # vertices with |distance| < atol
            n /= length(n)
            K1 = (K1 - n*d2[w21][U21].reshape(-1,1)).reshape(-1,1,3) # project vertices on plane (p,n)
            K2 = F21[d2[w21]>atol].reshape(-1,2,3) # vertices with distance > atol
            E4_pos = column_stack([K1,K2])
            F4_pos = Formex(E4_pos,get_new_prop(p2,w21,newprops[3]))
        # case 2: one vertex at positive side
        if w22.size > 0:
            F22 = F2[w22]
            U22 = U2[w22]
            K1 = F22[U22] # vertices with |distance| < atol
            K1 = (K1 - n*d2[w22][U22].reshape(-1,1)).reshape(-1,1,3) # project vertices on plane (p,n)
            P22 = P[w2][w22][roll(U22,1,axis=-1)].reshape(-1,1,3) # intersection points
            if side in '+':
                K2 = F22[d2[w22]>atol].reshape(-1,1,3) # vertices with distance > atol
                E5_pos = column_stack([P22,K1,K2])
                F5_pos = Formex(E5_pos,get_new_prop(p2,w22,newprops[4]))
            if side in '-':
                K3 = F22[d2[w22]<-atol].reshape(-1,1,3) # vertices with distance < - atol
                E5_neg = column_stack([P22,K1,K3])
                F5_neg = Formex(E5_neg,get_new_prop(p2,w22,newprops[4]))
        # case 3: no vertices at positive side
        if w23.size > 0 and side in '-':
            F23 = F2[w23]
            U23 = U2[w23]
            K1 = F23[U23] # vertices with |distance| < atol
            K1 = (K1 - n*d2[w23][U23].reshape(-1,1)).reshape(-1,1,3) # project vertices on plane (p,n)
            K2 = F23[d2[w23]<-atol].reshape(-1,2,3) # vertices with distance < - atol
            E4_neg = column_stack([K1,K2])
            F4_neg = Formex(E4_neg,get_new_prop(p2,w23,newprops[3]))
    # Two vertices with |distance| < atol
    w3 = where(V==2)[0]
    if w3.size > 0:
        F3 = F[w3]
        d3 = d[w3]
        U3 = U[w3]
        # split problem in two cases
        W = (d3 > atol).sum(axis=-1)
        w31 = where(W == 1)[0] # case 1: one vertex at positive side
        w32 = where(W == 0)[0] # case 2: no vertices at positive side
        # case 1: one vertex at positive side
        if w31.size > 0 and side in '+':
            F31 = F3[w31]
            U31 = U3[w31]
            K1 = F31[U31] # vertices with |distance| < atol
            K1 = (K1 - n*d3[w31][U31].reshape(-1,1)).reshape(-1,2,3) # project vertices on plane (p,n)
            K2 = F31[d3[w31]>atol].reshape(-1,1,3) # vertices with distance > atol
            E6_pos = column_stack([K1,K2])
            F6_pos = Formex(E6_pos,get_new_prop(F.p,w31,newprops[5]))
        # case 2: no vertices at positive side
        if w32.size > 0 and side in '-':
            F32 = F3[w32]
            U32 = U3[w32]
            K1 = F32[U32] # vertices with |distance| < atol
            K1 = (K1 - n*d3[w32][U32].reshape(-1,1)).reshape(-1,2,3) # project vertices on plane (p,n)
            K2 = F32[d3[w32]<-atol].reshape(-1,1,3) # vertices with distance < - atol
            E6_neg = column_stack([K1,K2])
            F6_neg = Formex(E6_neg,get_new_prop(F.p,w32,newprops[5]))
    # Three vertices with |distance| < atol
    w4 = where(V==3)[0]
    if w4.size > 0:
        F4 = F[w4]
        d4 = d[w4]
        U4 = U[w4]
        if side in '+':
            K1 = F4[U4] # vertices with |distance| < atol
            K1 = (K1 - n*d4[U4].reshape(-1,1)).reshape(-1,3,3) # project vertices on plane (p,n)
            E7_pos = K1
            F7_pos = Formex(E7_pos,get_new_prop(F.p,w4,newprops[6]))
        if side in '-':
            E7_neg = K1
            F7_neg = Formex(E7_neg,get_new_prop(F.p,w4,newprops[6]))
    # join all the pieces
    if side in '+':
        cut_pos = F1_pos+F2_pos+F3_pos+F4_pos+F5_pos+F6_pos+F7_pos
    if side in '-':
        cut_neg = F1_neg+F2_neg+F3_neg+F4_neg+F5_neg+F6_neg+F7_neg

    if side == '+':
        return cut_pos
    elif side == '-':
        return cut_neg
    else:
        return [ cut_pos, cut_neg ]


###########################################################################
##
##   Formex class
##
#########################
#

def coords_transformation(f):
    """Define a Formex transformation as the equivalent Coords transformation.

    This decorator replaces a Formex method with the equally named
    Coords method applied on the Formex coordinates attribute (.f).
    The return value is a Formex with changed coordinates but unchanged
    properties.
    """
    def newf(self,*args,**kargs):
        repl = getattr(Coords,f.__name__)
        return Formex(repl(self.f,*args,**kargs),self.p,self.eltype)
        newf.__name__ = f.__name__
        newf.__doc__ = repl.__doc__
    return newf


class Formex(object):
    """A Formex is a numpy array of order 3 (axes 0,1,2) and type Float.
    A scalar element represents a coordinate (F:uniple).

    A row along the axis 2 is a set of coordinates and represents a point
    (node, vertex, F: signet).
    For simplicity's sake, the current implementation only deals with points
    in a 3-dimensional space. This means that the length of axis 2 is always 3.
    The user can create formices (plural of Formex) in a 2-D space, but
    internally these will be stored with 3 coordinates, by adding a third
    value 0. All operations work with 3-D coordinate sets. However, a method
    exists to extract only a limited set of coordinates from the results,
    permitting to return to a 2-D environment

    A plane along the axes 2 and 1 is a set of points (F: cantle). This can be
    thought of as a geometrical shape (2 points form a line segment, 3 points
    make a triangle, ...) or as an element in FE terms. But it really is up to
    the user as to how this set of points is to be interpreted.

    Finally, the whole Formex represents a set of such elements.

    Additionally, a Formex may have a property set, which is an 1-D array of
    integers. The length of the array is equal to the length of axis 0 of the
    Formex data (i.e. the number of elements in the Formex). Thus, a single
    integer value may be attributed to each element. It is up to the user to
    define the use of this integer (e.g. it could be an index in a table of
    element property records).
    If a property set is defined, it will be copied together with the Formex
    data whenever copies of the Formex (or parts thereof) are made.
    Properties can be specified at creation time, and they can be set,
    modified or deleted at any time. Of course, the properties that are
    copied in an operation are those that exist at the time of performing
    the operation.   
    """
            

###########################################################################
#
#   Create a new Formex
#

    def __init__(self,data=[],prop=None,eltype=None):
        """Create a new Formex.

        The Formex data can be initialized by another Formex,
        by a 2D or 3D coordinate list, or by a string to be used in the
        pattern function to create a coordinate list.
        If 2D coordinates are given, a 3-rd coordinate 0.0 is added.
        Internally, Formices always work with 3D coordinates.
        Thus
          F = Formex([[[1,0],[0,1]],[[0,1],[1,2]]])
        Creates a Formex with two elements, each having 2 points in the
        global z-plane.

        If a prop argument is specified, the setProp() function will be
        called to assign the properties.
        """
        if isinstance(data,Formex):
            data = data.f
        else:
            if type(data) == str:
                data = pattern(data)
            data = asarray(data).astype(Float)
            
            if data.size == 0:   ### MAYBE THIS SHOULD BE CHANGED ?????
                if len(data.shape) == 3:
                    nplex = data.shape[1]
                elif len(data.shape) == 2:
                    nplex = 1
                else:
                    nplex = 0
                data = data.reshape(0,nplex,3) # An empty Formex
            else:
                # check dimensions of data
                if not len(data.shape) in [1,2,3]:
                    raise RuntimeError,"Formex init: needs a 1-, 2- or 3-dim. data array, got shape %s" % str(data.shape)
                if len(data.shape) == 1:
                    data = data.reshape(1,1,data.shape[0])
                elif len(data.shape) == 2:
                    data = data.reshape(data.shape[0],1,data.shape[1])
                if not data.shape[-1] in [1,2,3]:
                    raise RuntimeError,"Formex init: last axis dimension of data array should be 1, 2 or 3, got shape %s" % str(data.shape)
                # add 3-rd dimension if data are 1-d or 2-d
                # this will be done by Coords
                pass
                if data.shape[-1] == 2:
                    z = zeros((data.shape[0],data.shape[1],1),dtype=Float)
                    data = concatenate([data,z],axis=-1)
        # data should be OK now
        self.f = Coords(data)    # make sure coordinates are a Coords object 
        self.setProp(prop)
        try:
            self.eltype = eltype.lower()
        except:
            self.eltype = None

    def __getitem__(self,i):
        """Return element i of the Formex.

        This allows addressing element i of Formex F as F[i].
        """
        return self.f[i]

    def __setitem__(self,i,val):
        """Change element i of the Formex.

        This allows writing expressions as F[i] = [[1,2,3]].
        """
        self.f[i] = val

    def element(self,i):
        """Return element i of the Formex"""
        return self.f[i]

    def point(self,i,j):
        """Return point j of element i"""
        return self.f[i,j]

    def coord(self,i,j,k):
        """Return coord k of point j of element i"""
        return self.f[i,j,k]

###########################################################################
#
#   Return information about a Formex
#
    def nelems(self):
        """Return the number of elements in the formex."""
        return self.f.shape[0]


    def nplex(self):
        """Return the number of points per element.

        Examples:
        1: unconnected points,
        2: straight line elements,
        3: triangles or quadratic line elements,
        4: tetraeders or quadrilaterals or cubic line elements.
        """
        return self.f.shape[1]
    
    def ndim(self):
        """Return the number of dimensions.

        This is the number of coordinates for each point. In the
        current implementation this is always 3, though you can
        define 2D Formices by given only two coordinates: the third
        will automatically be set to zero.
        """
        return self.f.shape[2]
    
    def npoints(self):
        """Return the number of points in the formex.

        This is the product of the number of elements in the formex
        with the number of nodes per element.
        """
        return self.f.shape[0]*self.f.shape[1]
    
    def shape(self):
        """Return the shape of the Formex.

        The shape of a Formex is the shape of its data array,
        i.e. a tuple (nelems, nplex, ndim).
        """
        return self.f.shape


    # Coordinates
    def view(self):
        """Return the Formex coordinates as a numpy array (ndarray).

        Since the ndarray object has a method view() returning a view on
        the ndarray, this method allows writing code that works with both
        Formex and ndarray instances. The results is always an ndarray.
        """
        return self.f.view()


    # Properties
    def prop(self):
        """Return the properties as a numpy array (ndarray)"""
        return self.p

    def maxprop(self):
        """Return the highest property value used, or None"""
        if self.p is None:
            return None
        else:
            return self.p.max()

    def propSet(self):
        """Return a list with unique property values on this Formex."""
        if self.p is None:
            return None
        else:
            return unique(self.p)

    # The following functions get the corresponding information from
    # the underlying Coords object

    def x(self):
        return self.f.x()
    def y(self):
        return self.f.y()
    def z(self):
        return self.f.z()

    def bbox(self):
        return self.f.bbox()

    def center(self):
        return self.f.center()

    def centroid(self):
        return self.f.centroid()

    def sizes(self):
        return self.f.sizes()

    def dsize(self):
        return self.f.dsize()

    def bsphere(self):
        return self.f.bsphere()


    def centroids(self):
        """Return the centroids of all elements of the Formex.

        The centroid of an element is the point whose coordinates
        are the mean values of all points of the element.
        The return value is a Coords object with nelems points.
        """
        return self.f.mean(axis=1)

    #  Distance

    def distanceFromPlane(self,*args,**kargs):
        return self.f.distanceFromPlane(*args,**kargs)

    def distanceFromLine(self,*args,**kargs):
        return self.f.distanceFromLine(*args,**kargs)


    def distanceFromPoint(self,*args,**kargs):
        return self.f.distanceFromPoint(*args,**kargs)
 

    # Data conversion
    
    def fuse(self,nodesperbox=1,repeat=True,rtol=1.e-5,atol=None):
        """Return a tuple of nodal coordinates and element connectivity.

        A tuple of two arrays is returned. The first is float array with
        the coordinates of the unique nodes of the Formex. The second is
        an integer array with the node numbers connected by each element.
        The elements come in the same order as they are in the Formex, but
        the order of the nodes is unspecified.
        By the way, the reverse operation of
           coords,elems = feModel(F)
        is accomplished by
           F = Formex(coords[elems])

        There is a (very small) probability that two very close nodes are
        not equivalenced  by this procedure. Use it multiple times with
        different parameters to check.
        You can also set the rtol /atol parameters to influence the
        equivalence checking of two points.
        The default settting for atol is rtol * self.dsize()
        """
        if atol is None:
            atol = rtol * self.dsize()
        f = reshape(self.f,(self.nnodes(),3))
        f,s = f.fuse(nodesperbox,0.5,rtol=rtol,atol=atol)
        if repeat:
            f,t = f.fuse(nodesperbox,0.75,rtol=rtol,atol=atol)
            s = t[s]
        e = reshape(s,self.f.shape[:2])
        return f,e


    def toMesh(self):
        from plugins.mesh import Mesh
        x,e = self.fuse()
        return Mesh(x,e,eltype=self.eltype)

    # retained for compatibility
    feModel = fuse


##############################################################################
# Create string representations of a Formex
#

    @classmethod
    def point2str(clas,point):
        """Return a string representation of a point"""
        s = ""
        if len(point)>0:
            s += str(point[0])
            if len(point) > 1:
                for i in point[1:]:
                    s += "," + str(i)
        return s

    @classmethod
    def element2str(clas,elem):
        """Return a string representation of an element"""
        s = "["
        if len(elem) > 0:
            s += clas.point2str(elem[0])
            if len(elem) > 1:
                for i in elem[1:]:
                    s += "; " + clas.point2str(i) 
        return s+"]"
    
    def asFormex(self):
        """Return string representation of a Formex as in Formian.

        Coordinates are separated by commas, points are separated
        by semicolons and grouped between brackets, elements are
        separated by commas and grouped between braces.
        >>> F = Formex([[[1,0],[0,1]],[[0,1],[1,2]]])
        >>> print F
        {[1.0,0.0,0.0; 0.0,1.0,0.0], [0.0,1.0,0.0; 1.0,2.0,0.0]}
        """
        s = "{"
        if len(self.f) > 0:
            s += self.element2str(self.f[0])
            if len(self.f) > 1:
                for i in self.f[1:]:
                    s += ", " + self.element2str(i)
        return s+"}"

    def asFormexWithProp(self):
        """Return string representation as Formex with properties.

        The string representation as done by asFormex() is followed by
        the words "with prop" and a list of the properties.
        """
        s = self.asFormex()
        if isinstance(self.p,ndarray):
            s += " with prop " + self.p.__str__()
        else:
            s += " no prop "
        return s
                
    def asArray(self):
        """Return string representation as a numpy array."""
        return self.f.__str__()

    #default print function
    __str__ = asFormex

    @classmethod
    def setPrintFunction (clas,func):
        """Choose the default formatting for printing formices.

        This sets how formices will be formatted by a print statement.
        Currently there are two available functions: asFormex, asArray.
        The user may create its own formatting method.
        This is a class method. It should be used asfollows:
        Formex.setPrintFunction(Formex.asArray).
        """
        clas.__str__ = func


    def fprint(self,*args,**kargs):
        self.f.fprint(*args,**kargs)
           

##############################################################################
#
#  These are the only functions that change a Formex !
#
##############################################################################

    def setProp(self,p=None):
        """Create or destroy the property array for the Formex.

        A property array is a rank-1 integer array with dimension equal
        to the number of elements in the Formex (first dimension of data).
        You can specify a single value or a list/array of integer values.
        If the number of passed values is less than the number of elements,
        they wil be repeated. If you give more, they will be ignored.
        
        If a value None is given, the properties are removed from the Formex.
        """
        if p is None:
            self.p = None
        else:
            p = array(p).astype(Int)
            self.p = resize(p,self.f.shape[:1])
        return self


    def append(self,F):
        """Append the members of Formex F to this one.

        This function changes the original one! Use __add__ if you want to
        get a copy with the sum. 
        >>> F = Formex([[[1.0,1.0,1.0]]])
        >>> G = F.append(F)
        >>> print F
        {[1.0,1.0,1.0], [1.0,1.0,1.0]}
        """
        if F.f.size == 0:
            return self
        if self.f.size == 0:
            self.f = F.f
            self.p = F.p
            return self

        self.f = Coords(concatenate((self.f,F.f)))
        ## What to do if one of the formices has properties, the other one not?
        ## The current policy is to use zero property values for the Formex
        ## without props
        if self.p is not None or F.p is not None:
            if self.p is None:
                self.p = zeros(shape=self.f.shape[:1],dtype=Int)
            if F.p is None:
                p = zeros(shape=F.f.shape[:1],dtype=Int)
            else:
                p = F.p
            self.p = concatenate((self.p,p))
        return self


##############################################################################
## 
## All the following functions leave the original Formex unchanged and return
## a new Formex instead.
## This is a design decision intended so that the user can write statements as 
##   G = F.op1().op2().op3()
## without having an impact on F. If the user wishes, he can always change an
## existing Formex by a statement such as
##   F = F.op()
## While this may seem to create a lot of intermediate array data, Python and
## numpy are clever enough to release the memory that is no longer used.
##
##############################################################################
#
# Create copies, concatenations, subtractions, connections, ...
#
 
    def sort(self):
        """Sorts the elements of a Formex.

        Sorting is done according to the bbox of the elements.
        !! NOT FULLY IMPLEMENTED: CURRENTLY ONLY SORTS ACCORDING TO
        !! THE 0-direction OF NODE 0
        """
        sel = argsort(self.x()[:,0])
        f = self.f[sel]
        if self.p:
            p = self.p[sel]
        return Formex(f,p,self.eltype)
       
    def copy(self):
        """Return a deep copy of itself."""
        return Formex(self.f,self.p,self.eltype)
        ## IS THIS CORRECT? Shouldn't this be self.f.copy() ???
        ## In all examples it works, I think because the operations on
        ## the array data cause a copy to be made. Need to explore this.


    def __add__(self,other):
        """Return the sum of two formices.

        This returns a Formex with all elements of self and other.
        It allows us to write simple expressions as F+G to concatenate
        the Formices F and G.
        """
        return self.copy().append(other)


    @classmethod
    def concatenate(clas,Flist):
        """Concatenate all formices in Flist.

        This is a class method, not an instance method!
        >>> F = Formex([[[1,2,3]]],1)
        >>> print Formex.concatenate([F,F,F])
        {[1.0,2.0,3.0], [1.0,2.0,3.0], [1.0,2.0,3.0]}
        
        Formex.concatenate([F,G,H]) is functionally equivalent with F+G+H.
        The latter is simpler to write for a list with a few elements.
        If the list becomes large, or the number of items in the list
        is not fixed, the concatenate method is easier (and faster).
        We made it a class method and not a global function, because that
        would interfere with NumPy's own concatenate function.
        """
        f = concatenate([ F.f for F in Flist ])
        plist = [ F.p for F in Flist ]
        hasp = [ p is not None for p in plist ]
        nhasp = sum(hasp)
        if nhasp == 0:
            p = None # No Formices have properties
        else:
            if nhasp < len(Flist):  # Add zero properties where missing
                for i in range(len(Flist)):
                    if plist[i] is None:
                        plist[i] = zeros(shape=(Flist[i].nelems(),),dtype=Int)
            p = concatenate(plist)
        return Formex(f,p,Flist[0].eltype)

      
    def select(self,idx):
        """Return a Formex which holds only element with numbers in ids.

        idx can be a single element number or a list of numbers or
        any other index mechanism accepted by numpy's ndarray
        """
        if self.p is None:
            return Formex(self.f[idx],eltype=self.eltype)
        else:
            idx = asarray(idx)
            return Formex(self.f[idx],self.p[idx],self.eltype)

      
    def selectNodes(self,idx):
        """Return a Formex which holds only some nodes of the parent.

        idx is a list of node numbers to select.
        Thus, if F is a plex 3 Formex representing triangles, the sides of
        the triangles are given by
        F.selectNodes([0,1]) + F.selectNodes([1,2]) + F.selectNodes([2,0])
        The returned Formex inherits the property of its parent.
        """
        return Formex(self.f[:,idx,:],self.p,self.eltype)


    def points(self):
        """Return a Formex containing only the points.

        This is obviously a Formex with plexitude 1. It holds the same data
        as the original Formex, but in another shape: the number of points
        per element is 1, and the number of elements is equal to the total
        number of points.
        The properties are not copied over, since they will usually not make
        any sense.
        
        The vertices() method returns the same data, but as a Coords object.
        """
        return Formex(self.f.reshape((-1,1,3)))


    def vertices(self):
        """Return the points of a Formex as a 2dim Coords object.

        The return value holds the same coordinate data as the input Formex,
        but in another shape: (npoints,3).
        
        The points() method returns the same data, but as a Formex.
        """
        return self.f.reshape((-1,3))


    def remove(self,F):
        """Return a Formex where the elements in F have been removed.

        This is also the subtraction of the current Formex with F.
        Elements are only removed if they have the same nodes in the same
        order. This is a slow operation: for large structures, you should
        avoid it where possible.
        """
        flag = ones((self.f.shape[0],))
        for i in range(self.f.shape[0]):
            for j in range(F.f.shape[0]):
                if allclose(self.f[i],F.f[j]):
                    # element i is same as element j of F
                    flag[i] = 0
                    break
        if self.p is None:
            p = None
        else:
            p = self.p[flag>0]
        return Formex(self.f[flag>0],p,self.eltype)

    
    def whereProp(self,val):
        """Return the numbers of the elements with property val.

        val is either a single integer, or a list/array of integers.
        The return value is an array holding all the numbers of all the
        elements that have the property val, resp. one of the values in val.
        
        If the Formex has no properties, a empty array is returned.
        """
        if self.p is not None:
            if type(val) == int:
                return where(self.p==val)[0]
            else:
                return unique1d(concatenate([where(self.p==v)[0] for v in val]))
        return array([],dtype=Int)

    
    def withProp(self,val):
        """Return a Formex which holds only the elements with property val.

        val is either a single integer, or a list/array of integers.
        The return value is a Formex holding all the elements that
        have the property val, resp. one of the values in val.
        The returned Formex inherits the matching properties.
        
        If the Formex has no properties, a copy with all elements is returned.
        """
        if self.p is None:
            return Formex(self.f,eltype=self.eltype)
        elif type(val) == int:
            return Formex(self.f[self.p==val],val,self.eltype)
        else:
            t = zeros(self.p.shape,dtype=bool)
            for v in asarray(val).flat:
                t += (self.p == v)
            return Formex(self.f[t],self.p[t],self.eltype)
            

    def splitProp(self):
        """Partition a Formex according to its prop values.

        Returns a dict with the prop values as keys and the corresponding
        partitions as values. Each value is a Formex instance.
        It the Formex has no props, an empty dict is returned.
        """
        if self.p is None:
            return {}
        else:
            return dict([(p,self.withProp(p)) for p in self.propSet()])


    def elbbox(self):
        """Return a Formex where each element is replaced by its bbox.

        The returned Formex has two points for each element: two corners
        of the bbox.
        """
        ## Obviously, in the case of plexitude 1 and 2,
        ## there are shorter ways to perform this
        return Formex( [ [
            [ self.f[j,:,i].min() for i in range(self.f.shape[2])],
            [ self.f[j,:,i].max() for i in range(self.f.shape[2])] ]
                        for j in range(self.f.shape[0]) ] )


        
    def unique(self,rtol=1.e-4,atol=1.e-6):
        """Return a Formex which holds only the unique elements.

        Two elements are considered equal when all its nodal coordinates
        are close. Two values are close if they are both small compared to atol
        or their difference divided by the second value is small compared to
        rtol.
        Two elements are not considered equal if one's points are a
        permutation of the other's.
        """
        ##
        ##  THIS IS SLOW!! IT NEEDS TO BE REIMPLEMENTED BASED ON THE
        ##  feModel, and should probably be moved to a dedicated class
        ##
        flag = ones((self.f.shape[0],))
        for i in range(self.f.shape[0]):
            for j in range(i):
                if allclose(self.f[i],self.f[j],rtol=rtol,atol=atol):
                    # i is a duplicate node
                    flag[i] = 0
                    break
        if self.p is None:
            p = None
        else:
            p = self.p[flag>0]
        return Formex(self.f[flag>0],p,self.eltype)

      
    def nonzero(self):
        """Return a Formex which holds only the nonzero elements.

        A zero element is an element where all nodes are equal."""
        # NOT IMPLEMENTED YET !!! FOR NOW, RETURNS A COPY
        return Formex(self.f,self.p,self.eltype)


    def reverse(self):
        """Return a Formex where all elements have been reversed.

        Reversing an element means reversing the order of its points.
        """
        return Formex(self.f[:,range(self.f.shape[1]-1,-1,-1),:],self.p,self.eltype)


#############################
# Test and clipping functions

    def test(self,nodes='all',dir=0,min=None,max=None,atol=0.):
        """Flag elements having nodal coordinates between min and max.

        This function is very convenient in clipping a Formex in a specified
        direction. It returns a 1D integer array flagging (with a value 1 or
        True) the elements having nodal coordinates in the required range.
        Use where(result) to get a list of element numbers passing the test.
        Or directly use clip() or cclip() to create the clipped Formex.
        
        The test plane can be defined in two ways, depending on the value of dir.
        If dir == 0, 1 or 2, it specifies a global axis and min and max are
        the minimum and maximum values for the coordinates along that axis.
        Default is the 0 (or x) direction.

        Else, dir should be compaitble with a (3,) shaped array and specifies
        the direction of the normal on the planes. In this case, min and max
        are points and should also evaluate to (3,) shaped arrays.
        
        nodes specifies which nodes are taken into account in the comparisons.
        It should be one of the following:
        - a single (integer) point number (< the number of points in the Formex)
        - a list of point numbers
        - one of the special strings: 'all', 'any', 'none'
        The default ('all') will flag all the elements that have all their
        nodes between the planes x=min and x=max, i.e. the elements that
        fall completely between these planes. One of the two clipping planes
        may be left unspecified.
        """
        if min is None and max is None:
            raise ValueError,"At least one of min or max have to be specified."
        f = self.f
        if type(nodes)==str:
            nod = range(f.shape[1])
        else:
            nod = nodes

        if type(dir) == int:
            if min is not None:
                T1 = f[:,nod,dir] > (min - atol)
            if max is not None:
                T2 = f[:,nod,dir] < (max + atol)
        else:
            if min is not None:
                T1 = f.distanceFromPlane(min,dir) > (-atol)
            if max is not None:
                T2 = f.distanceFromPlane(max,dir) < (atol)

        if min is None:
            T = T2
        elif max is None:
            T = T1
        else:
            T = T1 * T2
        if len(T.shape) == 1:
            return T
        if nodes == 'any':
            T = T.any(1)
        elif nodes == 'none':
            T = (1-T.any(1)).astype(bool)
        else:
            T = T.all(1)
        return T


    def clip(self,t):
        """Return a Formex with all the elements where t>0.

        t should be a 1-D integer array with length equal to the number
        of elements of the formex.
        The resulting Formex will contain all elements where t > 0.
        This is a convenience function for the user, equivalent to
        F.select(t>0).
        """
        return self.select(t>0)


    def cclip(self,t):
        """This is the complement of clip, returning a Formex where t<=0.
        """
        return self.select(t<=0)


##############################################################################
#
#   Transformations that preserve the topology (but change coordinates)
#
#   These functions are the equivalents of the corresponding Coords methods.
#   They do not change the original Formex, but create a copy! If you want to
#   change the coordinates of a Formex inplace, directly operate on the
#   coords attribute.

 
    @coords_transformation
    def scale(self,*args,**kargs):
        pass
    @coords_transformation
    def translate(self,*args,**kargs):
        pass
    @coords_transformation
    def rotate(self,*args,**kargs):
        pass
    @coords_transformation
    def shear(self,*args,**kargs):
        pass
    @coords_transformation
    def reflect(self,*args,**kargs):
        pass
    @coords_transformation
    def affine(self,*args,**kargs):
        pass

    def mirror(self,dir=2,pos=0,keep_orig=False):
        if keep_orig:
            return self+self.reflect(dir,pos)
        else:
            return self.reflect(dir,pos)

    def centered(self):
        """Return a centered copy of the Formex."""
        return self.trl(-self.center())

    def resized(self,size=1.,tol=1.e-5):
        """Return a scaled copy of the Formex with given size in all directions.

        If a direction has zero size, it is not rescaled.
        """
        s = self.sizes()
        s[s<tol*s.max()] = size
        return self.scale(size/s)


    @coords_transformation
    def cylindrical(self,*args,**kargs):
        pass
    @coords_transformation
    def toCylindrical(self,*args,**kargs):
        pass
    @coords_transformation
    def spherical(self,*args,**kargs):
        pass
    @coords_transformation
    def superSpherical(self,*args,**kargs):
        pass
    @coords_transformation
    def toSpherical(self,*args,**kargs):
        pass
    @coords_transformation

    def bump(self,*args,**kargs):
        pass
    @coords_transformation
    def bump1(self,*args,**kargs):
        pass
    @coords_transformation
    def bump2(self,*args,**kargs):
        pass
    @coords_transformation

    def map(self,*args,**kargs):
        pass
    @coords_transformation
    def map1(self,*args,**kargs):
        pass
    @coords_transformation
    def mapd(self,*args,**kargs):
        pass
    @coords_transformation
    def newmap(self,*args,**kargs):
        pass

    @coords_transformation
    def replace(self,*args,**kargs):
        pass
    @coords_transformation
    def swapAxes(self,*args,**kargs):
        pass
    @coords_transformation
    def rollAxes(self,*args,**kargs):
        pass

    @coords_transformation
    def projectOnSphere(self,*args,**kargs):
        pass
    @coords_transformation
    def projectOnCylinder(self,*args,**kargs):
        pass

    def circulize(self,angle):
        """Transform a linear sector into a circular one.

        A sector of the (0,1) plane with given angle, starting from the 0 axis,
        is transformed as follows: points on the sector borders remain in
        place. Points inside the sector are projected from the center on the
        circle through the intersection points of the sector border axes and
        the line through the point and perpendicular to the bisector of the
        angle. See Diamatic example."""
        e = tand(0.5*angle)
        return self.map(lambda x,y,z:[where(y==0,x,(x*x+x*y*e)/sqrt(x*x+y*y)),where(x==0,y,(x*y+y*y*e)/sqrt(x*x+y*y)),0])


    def circulize1(self):
        """Transforms the first octant of the 0-1 plane into 1/6 of a circle.

        Points on the 0-axis keep their position. Lines parallel to the 1-axis
        are transformed into circular arcs. The bisector of the first quadrant
        is transformed in a straight line at an angle Pi/6.
        This function is especially suited to create circular domains where
        all bars have nearly same length. See the Diamatic example.
        """
        return self.map(lambda x,y,z:[where(x>0,x-y*y/(x+x),0),where(x>0,y*sqrt(4*x*x-y*y)/(x+x),y),0])


    def shrink(self,factor):
        """Shrinks each element with respect to its own center.

        Each element is scaled with the given factor in a local coordinate
        system with origin at the element center. The element center is the
        mean of all its nodes.
        The shrink operation is typically used (with a factor around 0.9) in
        wireframe draw mode to show all elements disconnected. A factor above
        1.0 will grow the elements.
        """
        c = self.f.mean(1).reshape((self.f.shape[0],1,self.f.shape[2]))
        return Formex(factor*(self.f-c)+c,self.p,self.eltype)


##############################################################################
#
#   Transformations that change the topology
#        

    def replic(self,n,step,dir=0):
        """Return a Formex with n replications in direction dir with step.

        The original Formex is the first of the n replicas.
        """
        f = array( [ self.f for i in range(n) ] )
        for i in range(1,n):
            f[i,:,:,dir] += i*step
        f.shape = (f.shape[0]*f.shape[1],f.shape[2],f.shape[3])
        ## the replication of the properties is automatic!
        return Formex(f,self.p,self.eltype)

    
    def replic2(self,n1,n2,t1=1.0,t2=1.0,d1=0,d2=1,bias=0,taper=0):
        """Replicate in two directions.

        n1,n2 number of replications with steps t1,t2 in directions d1,d2
        bias, taper : extra step and extra number of generations in direction
        d1 for each generation in direction d2
        """
        P = [ self.translatem((d1,i*bias),(d2,i*t2)).replic(n1+i*taper,t1,d1)
              for i in range(n2) ]
        ## We should replace the Formex concatenation here by
        ## separate data and prop concatenations, because we are
        ## guaranteed that either none or all formices in P have props.
        return Formex.concatenate(P)

 
    def rosette(self,n,angle,axis=2,point=[0.,0.,0.]):
        """Return a Formex with n rotational replications with angular
        step angle around an axis parallel with one of the coordinate axes
        going through the given point. axis is the number of the axis (0,1,2).
        point must be given as a list (or array) of three coordinates.
        The original Formex is the first of the n replicas.
        """
        f = self.f - point
        f = array( [ f for i in range(n) ] )
        for i in range(1,n):
            m = array(rotationMatrix(i*angle,axis))
            f[i] = dot(f[i],m)
        f.shape = (f.shape[0]*f.shape[1],f.shape[2],f.shape[3])
        return Formex(f + point,self.p,self.eltype)


    def translatem(self,*args,**kargs):
        """Multiple subsequent translations in axis directions.

        The argument list is a sequence of tuples (axis, step). 
        Thus translatem((0,x),(2,z),(1,y)) is equivalent to
        translate([x,y,z]). This function is especially conveniant
        to translate in calculated directions.
        """
        tr = [0.,0.,0.]
        for d,t in args:
            tr[d] += t
        return self.translate(tr)


##############################################################################
#
#   Transformations that change the plexitude
#        

    def extrude(self,n,step=1.,dir=0,autofix=True):
        """Extrude a Formex in one of the axis directions.

        Returns a Formex with doubled plexitude.

        First the original Formex is translated over n steps of length step in
        direction dir. Then each pair of subsequent Formices is connected to
        form a higher plexitude structure.

        Currently, this function correctly transforms: point1 to line2,
        line2 to quad4, tri3 to wedge6, quad4 to hex8.

        See the 'connect' function for a more versatile tool.
        """
        return self.toMesh().extrude(n,step,dir,autofix=autofix).toFormex()


##############################################################################
#
#   Transformations that work only for some plexitudes
#        
# !! It is not clear if they really belong here, or should go to a subclass


    def divide(self,div):
        """Divide a plex-2 Formex at the values in div.

        Replaces each member of the Formex by a sequence of members obtained
        by dividing the Formex at the relative values specified in div.
        The values should normally range from 0.0 to 1.0.

        As a convenience, if an integer is specified for div, it is taken as a
        number of divisions for the interval [0..1].

        This function only works on plex-2 Formices (line segments).
        """
        if self.nplex() != 2:
            raise RuntimeError,"Can only divide plex-2 Formices"
        if type(div) == int:
            div = arange(div+1) / float(div)
        else:
            div = array(div).ravel()
        A = interpolate(self.selectNodes([0]),self.selectNodes([1]),div[:-1],swap=True)
        B = interpolate(self.selectNodes([0]),self.selectNodes([1]),div[1:],swap=True)
        return connect([A,B])


    def intersectionWithPlane(self,p,n):
        """Return the intersection of a plex-2 Formex with the plane (p,n).
    
        This is equivalent with the function intersectionWithPlane(F,p,n).
        """
        return intersectionWithPlane(self,p,n)
    
    
    def intersectionPointsWithPlane(self,p,n):
        """Return the intersection points of a plex-2 Formex with plane (p,n).
    
        This is equivalent with the function intersectionWithPlane(F,p,n),
        but returns a Formex instead of an array of points.
        """
        return Formex(intersectionPointsWithPlane(self,p,n))


    def intersectionLinesWithPlane(self,p,n):
        """Returns the intersection lines of a plex-3 Formex with plane (p,n).

        This is equivalent with the function intersectionLinesWithPlane(F,p,n).
        """
        return Formex(intersectionLinesWithPlane(self,p,n))


    def cutWithPlane(self,p,n,side='',atol=None,newprops=None):
        """Returns all elements of the Formex cut at plane(s).

        This method currently only works for plexitude 2 or 3!
        
        F is a Formex of plexitude 2 or 3.
        p,n are a point and normal vector defining the cutting plane.
        In case of plexitude 3, p and n can be sequences of points and vector,
        allowing to cut with multiple planes.
        Both p and n have shape (3) or (npoints,3).

        The default return value is a tuple of two Formices of the same
        plexitude as the input: (Fpos,Fneg), where Fpos is the part of the
        Formex at the positive side of the plane (as defined by the normal
        vector), and Fneg is the part at the negative side.
        Elements of the input Formex that are lying completely on one side
        of the plane will return unaltered. Elements that are crossing the
        plane will be cut and split up into multiple parts.
        
        When with side = '+' or '-' (or 'positive'or 'negative'), only one
        of the sides is returned.

        The other arguments (atol,newprops) are currently specific for the
        plexitude. See the cut2AtPlane and cut3AtPlane, which hold the
        actual implementation of this method.
        """
        if atol is None:
            atol = 1.e-5*self.dsize()
        if self.nplex() == 2:
            return cut2AtPlane(self,p,n,side,atol,newprops)
        if self.nplex() == 3:
            return cut3AtPlane(self,p,n,side,atol,newprops)
        # OTHER PLEXITUDES NEED TO BE IMPLEMENTED
        raise ValueError,"Formex should be plex-2 or plex-3"


    @deprecation("\nUse cutWithPlane() instead. Check arguments ")
    def cutAtPlane(self,p,n,newprops=None,side='+',atol=0.):
        return self.cutWithPlane(p,n,side=side,atol=atol,newprops=newprops)


#################### Misc Operations #########################################

    def split(self,n=1):
        """Split a Formex in subFormices containing n elements.

        The number of elements in the Formex should be a multiple of n. 
        Returns a list of Formices each comprising n elements.
        """
        if self.nelems() % n != 0:
            raise RuntimeError,"Number of elements should be integer multiple of n"
        m = self.nelems()/n
        if self.p is None:
            return [ Formex(self.f[n*i:n*(i+1)],self.eltype) for i in range(m) ]
        else:
            return [ Formex(self.f[n*i:n*(i+1)],self.p[n*i:n*(i+1)],self.eltype) for i in range(m) ]


#################### Read/Write Formex File ##################################


    def write(self,fil,sep=' '):
        """Write a Formex to file.

        If fil is a string, a file with that name is opened. Else fil should
        be an open file.
        The Formex is then written to that file in a native format.
        If fil is a string, the file is closed prior to returning.
        """
        isname = type(fil) == str
        if isname:
            fil = file(fil,'w')
        fil.write("# Formex File Format 1.0 (http://pyformex.org)\n")
        nelems,nplex = self.f.shape[:2]
        hasp = self.p is not None
        fil.write("# nelems=%r; nplex=%r; props=%r; eltype=%r\n" % (nelems,nplex,hasp,self.eltype))
        self.f.tofile(fil,sep)
        fil.write('\n')
        if hasp:
            self.p.tofile(fil,sep)
        fil.write('\n')
        if isname:
            fil.close()


    @classmethod
    def read(clas,fil):
        """Read a Formex from file.

        fil is a filename or a file object.
        If the file is in a valid Formex file format, the Formex is read and
        returned. Otherwise, None is returned.
        Valid Formex file formats are described in the manual.
        """
        isname = type(fil) == str
        if isname:
            fil = file(fil,'r')
        s = fil.readline()
        if not s.startswith('# Formex'):
            return None
        eltype = None # for compatibility with older .formex files
        ndim = 3
        while s.startswith('#'):
            s = fil.readline()
            if s.startswith('# nelems'):
                exec(s[1:].strip())
                break
        # read the coordinates
        f = fromfile(file=fil, dtype=Float, count=ndim*nelems*nplex, sep=' ').reshape((nelems,nplex,3))
        if props:
            p = fromfile(file=fil, dtype=Int, count=nelems, sep=' ')
        else:
            p = None
        if isname:
            fil.close()
        return Formex(f,p,eltype)


#########################################################################
    #
    # Obsolete and deprecated functions
    #
    # These functions are retained mainly for compatibility reasons.
    # New users should avoid these functions!
    # They may (will) be removed in future.


    @functionWasRenamed(reverse)
    def reverseElements(self):
        pass

    @deprecated(view)
    def data(self):
        pass

    @deprecated(test)
    def where(self,*args,**kargs):
        pass

    @deprecated("Use Formex.spherical instead")
    def oldspherical(self,dir=[2,0,1],scale=[1.,1.,1.]):
        """Same as spherical, but using colatitude."""
        return self.spherical([dir[1],dir[2],dir[0]],[scale[1],scale[2],scale[0]],colat=True)
    
    nnodel = nplex
    nnodes = npoints
    
    # Formian compatibility functions
    # These will be moved to a separate file in future.
    #
    # About Formian newspeak:
    # The author of formex/formian had an incredible preference for newspeak:
    # for every concept or function, a new name was invented. While this may
    # give formian the aspect of a sophisticated scientific background,
    # it works rather distracting and ennoying for people that are already
    # familiar with the basic ideas of 3D geometry, and are used to using the
    # standardized terms.

    order = nelems
    plexitude = nplex
    grade = ndim

    cantle = element
    signet = point
    uniple = coord
    cop = remove
    
    cantle2str = element2str
    signet2str = point2str
    
    def give(self):
        print self.toFormian()

    def tran(self,dir,dist):
        return self.translate(dir-1,dist)
    
    def ref(self,dir,dist):
        return self.reflect(dir-1,dist)

    def rindle(self,n,dir,step):
        return self.replic(n,step,dir)
    def rin(self,dir,n,dist):
        return self.rindle(n,dir-1,dist)

    

    def lam(self,dir,dist):
        return self+self.reflect(dir-1,dist)

    def ros(self,i,j,x,y,n,angle):
        if (i,j) == (1,2):
            return self.rosette(n,angle,2,[x,y,0])
        elif (i,j) == (2,3):
            return self.rosette(n,angle,0,[0,x,y])
        elif (i,j) == (1,3):
            return self.rosette(n,-angle,1,[x,0,y])

    def tranic(self,*args,**kargs):
        n = len(args)/2
        d = [ i-1 for i in args[:n] ]
        return self.translatem(*zip(d,args[n:]))
    def tranid(self,t1,t2):
        return self.translate([t1,t2,0])
    def tranis(self,t1,t2):
        return self.translate([t1,0,t2])
    def tranit(self,t1,t2):
        return self.translate([0,t1,t2])
    def tranix(self,t1,t2,t3):
        return self.translate([t1,t2,t3])

    def tranad(self,a1,a2,b1,b2,t=None):
        return self.translate([b1-a1,b2-a2,0.],t)
    def tranas(self,a1,a2,b1,b2,t=None):
        return self.translate([b1-a1,0.,b2-a2],t)
    def tranat(self,a1,a2,b1,b2,t=None):
        return self.translate([0.,b1-a1,b2-a2],t)
    def tranax(self,a1,a2,a3,b1,b2,b3,t=None):
        return self.translate([b1-a1,b2-a2,b3-a3],t)
   
    def rinic(self,*args,**kargs):
        n = len(args)/3
        F = self
        for d,m,t in zip(args[:n],args[n:2*n],args[2*n:]):
            F = F.rin(d,m,t)
        return F
    def rinid(self,n1,n2,t1,t2):
        return self.rin(1,n1,t1).rin(2,n2,t2)
    def rinis(self,n1,n2,t1,t2):
        return self.rin(1,n1,t1).rin(3,n2,t2)
    def rinit(self,n1,n2,t1,t2):
        return self.rin(2,n1,t1).rin(3,n2,t2)

    def lamic(self,*args,**kargs):
        n = len(args)/2
        F = self
        for d,p in zip(args[:n],args[n:]):
            F = F.lam(d,p)
        return F
    def lamid(self,t1,t2):
        return self.lam(1,t1).lam(2,t2)
    def lamis(self,t1,t2):
        return self.lam(1,t1).lam(3,t2)
    def lamit(self,t1,t2):
        return self.lam(2,t1).lam(2,t2)
    
    def rosad(self,a,b,n=4,angle=90):
        return self.rosette(n,angle,2,[a,b,0])
    def rosas(self,a,b,n=4,angle=90):
        return self.rosette(n,angle,1,[a,0,b])
    def rosat(self,a,b,n=4,angle=90):
        return self.rosette(n,angle,0,[0,a,b])

    def genid(self,n1,n2,t1,t2,bias=0,taper=0):
        return self.replic2(n1,n2,t1,t2,0,1,bias,taper)
    def genis(self,n1,n2,t1,t2,bias=0,taper=0):
        return self.replic2(n1,n2,t1,t2,0,2,bias,taper)
    def genit(self,n1,n2,t1,t2,bias=0,taper=0):
        return self.replic2(n1,n2,t1,t2,1,2,bias,taper)

    def bb(self,b1,b2):
        return self.scale([b1,b2,1.])

    def bc(self,b1,b2,b3):
        return self.cylindrical(scale=[b1,b2,b3])

    def bp(self,b1,b2):
        return self.cylindrical(scale=[b1,b2,1.])

    def bs(self,b1,b2,b3):
        return self.spherical(scale=[b1,b2,b3],colat=True)

    pex = unique
    def tic(f):
        return int(f)
    def ric(f):
        return int(round(f))

    # Convenience short notations and aliases
    rep = replic
    ros = rosette
    rot = rotate
    trl = translate
    



##############################################################################
#
#    Functions which are not Formex class methods
#
        
def connect(Flist,nodid=None,bias=None,loop=False):
    """Return a Formex which connects the formices in list.

    Flist is a list of formices, nodid is an optional list of nod ids and
    bias is an optional list of element bias values. All lists should have
    the same length.
    The returned Formex has a plexitude equal to the number of
    formices in list. Each element of the Formex consist of a node from
    the corresponding element of each of the formices in list. By default
    this will be the first node of that element, but a nodid list may be
    given to specify the node id to be used for each of the formices.
    Finally, a list of bias values may be given to specify an offset in
    element number for the subsequent formices.
    If loop==False, the order of the Formex will be the minimum order of
    the formices in Flist, each minus its respective bias. By setting
    loop=True however, each Formex will loop around if its end is
    encountered, and the order of the result is the maximum order in Flist.
    """
    try:
        m = len(Flist)
        for i in range(m):
            if isinstance(Flist[i],Formex):
                pass
            elif isinstance(Flist[i],ndarray):
                Flist[i] = Formex(Flist[i])
            else:
                raise TypeError
    except TypeError:
        raise TypeError,'connect(): first argument should be a list of formices'
    
    if not nodid:
        nodid = [ 0 for i in range(m) ]
    if not bias:
        bias = [ 0 for i in range(m) ]
    if loop:
        n = max([ Flist[i].nelems() for i in range(m) ])
    else:
        n = min([ Flist[i].nelems() - bias[i] for i in range(m) ])
    f = zeros((n,m,3),dtype=Float)
    for i,j,k in zip(range(m),nodid,bias):
        v = Flist[i].f[k:k+n,j,:]
        if loop and k > 0:
            v = concatenate([v,Flist[i].f[:k,j,:]])
        f[:,i,:] = resize(v,(n,3))
    return Formex(f)


def interpolate(F,G,div,swap=False):
    """Create interpolations between two formices.

    F and G are two Formices with the same shape.
    v is a list of floating point values.
    The result is the concatenation of the interpolations of F and G at all
    the values in div.
    An interpolation of F and G at value v is a Formex H where each coordinate
    Hijk is obtained from:  Hijk = Fijk + v * (Gijk-Fijk).
    Thus, a Formex interpolate(F,G,[0.,0.5,1.0]) will contain all elements
    of F and G and all elements with mean coordinates between those of F and G.

    As a convenience, if an integer is specified for div, it is taken as a
    number of divisions for the interval [0..1].
    Thus, interpolate(F,G,n) is equivalent with
    interpolate(F,G,arange(0,n+1)/float(n))

    The swap argument sets the order of the elements in the resulting Formex.
    By default, if n interpolations are created of an m-element Formex, the
    element order is in-Formex first (n sequences of m elements).
    If swap==True, the order is swapped and you get m sequences of n
    interpolations.
    """
    r = Coords.interpolate(F.f,G.f,div)
    # r is a 4-dim array
    if swap:
        r = r.swapaxes(0,1)
    return Formex(r.reshape((-1,) + r.shape[-2:]))
    


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

def _test():
    """Run the examples in the docstrings."""
    import doctest, formex
    return doctest.testmod(formex)

if __name__ == "__main__":
    
    def test():
        """Run some additional examples.

        This is intended for tests during development. This can be changed
        at will.
        """
        Formex.setPrintFunction(Formex.asFormexWithProp)
        A = Formex()
        print "An empty formex",A
        F = Formex([[[1,0],[0,1]],[[0,1],[1,2]]])
        print "F =",F
        F1 = F.translate(0,6)
        F1.setProp(5)
        print "F1 =",F1
        F2 = F.ref(1,2)
        print "F2 =",F2
        F3 = F.ref(1,1.5).translate(1,2)
        F3.setProp([1,2])
        G = F1+F3+F2+F3
        print "F1+F3+F2+F3 =",G
        H = Formex.concatenate([F1,F3,F2,F3])
        print "F1+F3+F2+F3 =",H
        print "elbbox:",G.elbbox()
        print "met prop 1:",G.withProp(1)
        print "unique:",G.unique()
        print "vertices:",G.vertices()
        print "points:",G.points()
        print "unique points:",G.points().unique()
        print "diagonal size:",G.dsize()
        F = Formex([[[0,0]],[[1,0]],[[1,1]],[[0,1]]])
        G = connect([F,F],bias=[0,1])
        print G
        G = connect([F,F],bias=[0,1],loop=True)
        print G
        print G[1]
        print G.feModel()
        print F
        print F.bbox()
        print F.center(),F.centroid()
        print F.bsphere()
        F = Formex([[[0,0],[1,0],[0,1]],[[1,0],[1,1],[0,1]]])
        print F
        print F.reverse()
        Formex.setPrintFunction(Formex.asArray)
        print F
        F.fprint()
        F.fprint("%10.3f %10.4f %10.5f")
        F.fprint(fmt="%10.4f %10.5f %10.6f")
        print type(F)

    f = 0
    (f,t) = _test()
    if f == 0:
        test()

### End
