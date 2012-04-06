# $Id$ pyformex
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

"""Polygonal facets.

"""

import pyformex as pf
from formex import *
from geometry import Geometry
from plugins.curve import PolyLine
from plugins.trisurface import TriSurface
import utils

############################################################################

#
# TODO : this should be integrated with CoordinateSystem
#
def projected(X,N):
    """Returns 2-D coordinates of a set of 3D coordinates.

    The returned 2D coordinates are still stored in a 3D Coords object.
    The last coordinate will however (approximately) be zero.
    """
    from geomtools import rotationAngle
    if N is None:
        N = self.normal
    a,A = rotationAngle([0.,0.,1.],N)
    a,A = a[0],A[0]
    X = X.rotate(angle=-a,axis=A)
    C = X.center()
    X = X.translate(-C)
    return X,C,A,a


class Polygon(Geometry):
    """A Polygon is a flat surface bounded by a closed PolyLine.

    The border is specified as a Coords object with shape (nvertex,3)
    specifying the vertex coordinates in order.
    While the Coords are 3d, only the first 2 components are used.
    """

    def __init__(self,border,normal=2,holes=[]):
        """Initialize a Polygon instance"""
        Geometry.__init__(self)
        self.prop = None
        self.coords = border.reshape(-1,3)


    def npoints(self):
        """Return the number of points and edges."""
        return self.coords.shape[0]
    

    def vectors(self):
        """Return the vectors from each point to the next one."""
        x = self.coords
        return roll(x,-1,axis=0) - x


    def angles(self):
        """Return the angles of the line segments with the x-axis."""
        v = self.vectors()
        return arctand2(v[:,1],v[:,0])


    def externalAngles(self):
        """Return the angles between subsequent line segments.

        The returned angles are the change in direction between the segment
        ending at the vertex and the segment leaving.
        The angles are given in degrees, in the range ]-180,180].
        The sum of the external angles is always (a multiple of) 360.
        A convex polygon has all angles of the same sign.
        """
        a = self.angles()
        va =  a - roll(a,1)
        va[va <= -180.] += 360.
        va[va > 180.] -= 360.
        return va


    def isConvex(self):
        """Check if the polygon is convex and turning anticlockwise.

        Returns:

        - +1 if the Polygon is convex and turning anticlockwise,
        - -1 if the Polygon is convex, but turning clockwise,
        - 0 if the Polygon is not convex.
        """
        return int(sign(self.externalAngles()).sum()) / self.npoints()


    def internalAngles(self):
        """Return the internal angles.

        The returned angles are those between the two line segments at
        each vertex.
        The angles are given in degrees, in the range ]-180,180].
        These angles are the complement of the 
        """
        return 180.-self.externalAngles()


    def reverse(self):
        """Return the Polygon with reversed order of vertices."""
        return Polygon(Coords(reverseAxis(self.coords,0)))


    def fill(self):
        """Fill the surface inside the polygon with triangles.

        Returns a TriSurface filling the surface inside the polygon.
        """
        print "AREA(self) %s" % self.area()
        # creating elems array at once (more efficient than appending)
        #from gui.draw import draw,pause,undraw
        from geomtools import insideTriangle
        x = self.coords
        n = x.shape[0]
        tri = -ones((n-2,3),dtype=Int)
        # compute all internal angles
        e = arange(x.shape[0])
        c = self.internalAngles()
        # loop in order of smallest angles
        itri = 0
        while n > 3:
            #print "ANGLES",c
            # try minimal angle
            srt = c.argsort()
            for j in srt:
                #print "ANGLE: %s" % c[j]
                if c[j] > 180.:
                    print "OOPS, I GOT STUCK"
                    #
                    # We could return here also the remaining part
                    #
                    return TriSurface(x,tri[:itri])
                i = (j - 1) % n
                k = (j + 1) % n
                newtri = [ e[i],e[j],e[k]]
                # remove the point j of triangle i,j,k
                # recompute adjacent angles of edge i,k
                ii = (i-1) % n
                kk = (k+1) % n
                iq = e[[ii,i,k,kk]]
                PQ = Polygon(x[iq])
                cn = PQ.internalAngles()
                cnew = cn[1:3]
                reme = roll(e,-j)[2:-1]
                T = x[newtri].reshape(1,3,3)
                P = x[reme].reshape(-1,1,3)
                check = insideTriangle(T,P)
                if not check.any():
                    # Triangle is ok
                    break
            #draw(TriSurface(x,newtri),bbox='last')
            # accept new triangle
            tri[itri] = newtri
            c = roll(concatenate([cnew,roll(c,1-j)[3:]]),j-1)
            e = roll(roll(e,-j)[1:],j)
            n -= 1
            itri += 1
        tri[itri] = e
        return TriSurface(x,tri)


    def area(self):
        """Compute area inside a polygon.

        """
        from plugins.section2d import PlaneSection
        return PlaneSection(Formex(self.coords)).sectionChar()['A']   



if __name__ == 'draw':


    layout(2)
    for i in range(2):
        viewport(i)
        clear()
        smoothwire()

    n = 5
    r = 0.7
    noise = 0.0
    x = randomNoise((n),r*3.,3.)
    y = randomNoise((n),0.,360.)
    y.sort()    # sort
    #y = y[::-1] # reverse
    z = zeros(n)
    X = Coords(column_stack([x,y,z])).cylindrical().addNoise(noise)
    draw(X)
    drawNumbers(X)
    PG = Polygon(X)
    PL = PolyLine(X,closed=True)
    draw(PL)

    v = normalize(PG.vectors())
    drawVectors(PG.coords,v,color=red,linewidth=2)
    
    a = PG.angles()
    ae = PG.externalAngles()
    ai = PG.internalAngles()

    print "Direction angles:", a
    print "External angles:", ae
    print "Internal angles:", ai

    print "Sum of external angles: ",ae.sum()
    print "The polygon is convex: %s" % PG.isConvex()

    # choose one of these
    #B = PL.coords
    B = PL.toMesh()
    #B = PL

    viewport(0)
    clear()
    S = fillBorder(B,method='border')
    draw(S)
    drawNumbers(S)
    drawText(S.check(),100,20)

    
    viewport(1)
    clear()
    S1 = fillBorder(B,method='radial')
    draw(S1,color=red)
    drawNumbers(S1)
    drawText(S1.check(),100,20)
    
# End
