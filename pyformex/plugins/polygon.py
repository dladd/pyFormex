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
    


def surfaceInsideBorder(border,method='radial'):
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
    if border.nplex() != 2:
        raise ValueError,"Expected Mesh with plexitude 2, got %s" % border.nplex()

    if method == 'radial':
        x = border.getPoints().center()
        n = zeros_like(border.elems[:,:1]) + border.coords.shape[0]
        elems = concatenate([border.elems,n],axis=1)
        coords = Coords.concatenate([border.coords,x])

    elif method == 'border':
        coords = border.coords
        segments = border.elems
        elems = empty((0,3,),dtype=int)
        while len(segments) != 3:
            segments,triangle = _create_border_triangle(coords,segments)
            elems = row_stack([elems,triangle])
        elems = row_stack([elems,segments[:,0]])

    else:
        raise ValueError,"Strategy should be either 'radial' or 'border'"
    
    return TriSurface(coords,elems)


def reducePolyline(X,seq):
    """Create a triangle within a border.
    
    - coords: (npoints,3) Coords: the ordered vertices of the border.
    Elems is a (nelems,2) shaped array of integers representing
    the border element numbers and must be ordered.
    A list of two objects is returned: the new border elements and the triangle.
    """
    P = PolyLine(X[seq],closed=True)
    v = normalize(P.vectors())
    c = vectorPairCosAngle(roll(v,1,axis=0),v)
    print c
    j = c.argmin()
    n = len(seq)
    i = j - 1
    if i < 0:
        i += n
    k = j + 1
    if k >= n:
        k -= n
    tri = [ i,j,k]
    retur
    n = shape(elems)[0]
    if j == n:
        j -= n
    old_edges = take(elems,[i,j],0)
    elems = delete(elems,[i,j],0)
    new_edge = asarray([old_edges[0,0],old_edges[-1,1]])
    if j == 0:
        elems = insert(elems,0,new_edge,0)
    else:
        elems = insert(elems,i,new_edge,0)
    triangle = append(old_edges[:,0],old_edges[-1,1].reshape(1),0)
    return elems,triangle

if __name__ == 'draw':

    clear()

    n = 6
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
    #S = surfaceInsideBorder(M,method='border')
    #draw(S)
    #drawNumbers(S)
    
# End
