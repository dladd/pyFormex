# $Id$ pyformex

"""Polygonal facets.

"""

import pyformex as pf
from coords import *
from geometry import Geometry
from plugins.curve import PolyLine
import utils

##############################################################################
#
class Polygon(Geometry):
    """A Polygon is a flat surface bounded by a closed PolyLine.

    The border can be specified as:

    - a Coords-like with shape (nvertex,3) specifying the vertex coordinates
      in order
    - an object that has a coords attribute.
    """

    def __init__(self,border,normal=2,holes=[]):
        """Initialize a Polygon instance"""
        Geometry.__init__(self)
        self.prop = None
        if border.__class__ != Coords:
            try:
                border = border.coords
            except:
                raise ValueError,"Invalid border data"
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
       

    

    def fill(self):
        return


    def area(self,project=None):
        """Compute area inside a polygon.

        Parameters:

        - `project`: (3,) Coords array representing a unit direction vector.

        Returns: a single float value with the area inside the polygon. If a
        direction vector is given, the area projected in that direction is
        returned.

        Note that if the polygon is nonplanar and no direction is given,
        the area inside the polygon is not well defined.
        """
        from geomtools import polygonArea
        return polygonArea(self.coords,project)
    


def fillBorder(coords,elems=None,method='radial'):
    """Create a surface inside a closed curve defined by a 2-plex Mesh.

    border is a 2-plex Mesh representing a closed polyline.

    The return value is a TriSurface filling the hole inside the border.

    There are two fill methods:
    
    - 'radial': this method adds a central point and connects all border
      segments with the center to create triangles. It is fast and works
      well if the border is smooth, nearly convex and nearly planar.
    - 'border': this method creates subsequent triangles by connecting the
      endpoints of two consecutive border segments and thus works its way
      inwards until the hole is closed. Triangles are created at the segments
      that form the smallest angle. This method is slower, but works also
      for most complex borders. Because it does not create any new
      points, the returned surface uses the same point coordinate array
      as the input Mesh.
    """
    from plugins.trisurface import TriSurface

    if not isinstance(coords,Coords):
        raise ValueError,"Expected a Coords array as first argument"
    coords = coords.reshape(-1,3)

    if elems is None:
        elems = arange(coords.shape[0])

    n = elems.shape[0]
    if n < 3:
        raise ValueError,"Expected at least 3 points."
    
    if method == 'radial':
        coords = Coords.concatenate([coords,coords.center()])
        elems = column_stack([elems,roll(elems,-1),n*ones(elems.shape[0],dtype=Int)])

    elif method == 'border':
        # creating elems array at once (more efficient than appending)
        tri = -ones((n-2,3),dtype=Int)
        print tri
        # compute all internal angles
        x = coords[elems]
        e = arange(n)
        v = roll(x,-1,axis=0) - x
        v = normalize(v)
        c = vectorPairCosAngle(roll(v,1,axis=0),v)
        # loop in order of smallest angles
        itri = 0
        while n > 3:
            print "c",c,n
            # find minimal angle
            j = c.argmin()
            i = (j - 1) % n
            k = (j + 1) % n
            tri[itri] = [ e[i],e[j],e[k]]
            print "tr",tri
            # remove the point j
            ii = (i-1) % n
            kk = (k+1) % n
            v1 = normalize([ v[e[ii]], x[e[k]] - x[e[i]] ])
            v2 = normalize([ x[e[k]] - x[e[i]], v[e[k]] ])
            #print "v1",v1
            #print "v2",v2
            cnew = vectorPairCosAngle(v1,v2)
            print [ii,i,j,k,kk]
            print "cnew",c[:ii], cnew, c[kk+1:]
            c = roll(concatenate([cnew,roll(c,-i)[3:]]),i)
            e = roll(roll(e,-j)[1:],j)
            print "new c",c
            print "new e",e
            n -= 1
            itri += 1
        print tri.shape
        print e.shape
        print itri
        tri[itri] = e
        print tri
        elems = elems[tri]
        print elems

    else:
        raise ValueError,"Strategy should be either 'radial' or 'border'"
    
    return TriSurface(coords,elems)


def reducePolyline(x,e):
    """Create a triangle within a border.
    
    - coords: (npoints,3) Coords: the ordered vertices of the border.
    Elems is a (nelems,2) shaped array of integers representing
    the border element numbers and must be ordered.
    A list of two objects is returned: the new border elements and the triangle.
    """

if __name__ == 'draw':

    clear()

    n = 5
    x = randomNoise((n),2.,3.)
    y = randomNoise((n),0.,360.)
    y.sort()    # sort
    #y = y[::-1] # reverse
    z = zeros(n)
    X = Coords(column_stack([x,y,z])).cylindrical().addNoise()
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

    M = PL.toMesh()

    layout(2)
    if PG.isConvex():
        S = fillBorder(PL.coords,method='border')
        viewport(0)
        clear()
        draw(S)
        drawNumbers(S)
        viewport(1)
        clear()
        S1 = fillBorder(PL.coords,method='radial')
        draw(S1,color=red)
        drawNumbers(S1)
   
# End
