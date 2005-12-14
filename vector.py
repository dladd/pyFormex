# A python class for 3D vector operations
# $Id$
#
"""A python class for 3D vector operations.

A 3D vector is a list of three floats.
All operations are implemented in standard Python.
If you need high performance numerical operations on lots of vectors,
you should use scipy instead.
"""

import math

def reverse (v):
    """Return the reverse vector [-x, -y, -z] of v[x,y,z]"""
    return [ -x for x in v ]

def scale (v,a):
    """Return the a * [x, y, z], where a is a scalar"""
    return [ a*x for x in v ]

def square (v):
    """Return the squared components [x^2, y^2, z^2] of v[x,y,z]"""
    return [ x*x for x in v ]

def norm (v):
    """Return the square length [x^2 + y^2 + z^2 of a vector v"""
    return sum(square(v))
 
def length (v):
    """Return the length of the vector v"""
    return math.sqrt(norm(v))

def unitvector (v):
    """Return the normalized (unit length) vector in direction v"""
    return scale(v, 1/length(v))

def add (v,w):
    """Return the sum of the vectors v and w"""
    return [ x+y for x,y in zip(v,w) ]

def diff (v,w):
    """Return the difference vector v-w"""
    return [ x-y for x,y in zip(v,w) ]

def distance (v,w):
    """Return the distance between two points"""
    return length(diff(v,w))

def pointOf (v,w,pos=0.5):
    """Return the point on the line v-w defined by the relative coordinate pos.

    v has pos 0, w has pos 1. For values 0..1 the point lies between v and w.
    If the pos argument is omitted, the midpoint between v and w is returned.
    """
    return add(v, scale(diff(w,v),pos))

def midPoint (v,w):
    """Return the center point of the line v-w.

    This is the same as pointOf(v,w,0.5), but cheaper.
    """
    return scale(add(v,w),0.5)

def centerDiff (v,w):
    """Return the center point and the difference of the line v-w."""
    d = diff(w,v)
    return [ add(v, scale(d,0.5)), d ]

def dotpr (v,w):
    """Return the dot product of vectors v and w"""
    return sum( [ x*y for x,y in zip(v,w) ] )

def cosAngle (v,w):
    """Return the cosine of the angle between two vectors"""
    return dotpr(v,w)/length(v)/length(w)

def projection(v,w):
    """Return the (signed) length of the projection of vector v on vector w."""
    return dotpr(v,w)/length(w)

def parallel(v,w):
    """Returns the part of vector v that is parallel to vector w"""
    return scale(unitvector(w),projection(v,w))

def orthogonal(v,w):
    """Returns the part of vector v that is orthogonal to vector w"""
    return v-parallel(v,w)

def cross (v,w):
    """Return the cross product of two vectors."""
    return [ v[1]*w[2]-v[2]*w[1],  v[2]*w[0]-v[0]*w[2], v[0]*w[1]-v[1]*w[0] ]

def cartesianToCylindrical (v) :
    """Convert cartesian coordinates [x,y,z] to cylindrical [r,theta,z]
    
    The angle is given in degrees: theta: -180..180
    """
    r = math.sqrt(v[0]*v[0]+v[1]*v[1])
    theta = math.degrees( math.atan2(v[1],v[0]) )
    return [ r, theta, v[2] ]

def cylindricalToCartesian (v) :
    """Convert cylindrical coordinates [r,theta,z] to cartesian [x,y,z]
    
    The angle theta must be given in degrees.
    """
    theta = math.radians(v[1])
    return [ v[0]*math.cos(theta),  v[0]*math.sin(theta), v[2] ]

def cartesianToSpherical (v) :
    """Convert cartesian coordinates [x,y,z] to spherical [long,lat,dist]
    
    Angles are given in degrees: lat: -90..90, long:-180..180
    """
    distance = length(v)
    longitude = math.degrees( math.atan2(v[0],v[2]) )
    latitude = math.degrees( math.asin(v[1]/distance) )
    return [ longitude, latitude, distance]

def sphericalToCartesian (v) :
    """Convert spherical coordinates [long,lat,dist] to cartesian [x,y,z]"""
    long = math.radians(v[0])
    lat = math.radians(v[1])
    return scale ([ math.cos(lat)*math.sin(long), math.sin(lat), math.cos(lat)*math.cos(long) ], v[2])

def roll(vector,n):
    """Roll the elements of the vector over n positions forward"""
    return vector[n:] + vector[:n]

def rotationMatrix (axis,angle):
    """Return a rotation matrix over angle(degrees) around axis.

    This is a matrix for postmultiplying a row vector."""
    m = [ [ 0. for i in range(3) ]  for j in range(3) ]
    i,j,k = roll(range(3),axis%3)
    a = math.radians(angle)
    c = math.cos(a)
    s = math.sin(a)
    m[i][i] = 1.
    m[j][j] = c
    m[j][k] = s
    m[k][j] = -s
    m[k][k] = c
    return m

def matrixMultiply (a,b):
    """Multipy matrices a and b."""
    return [ [ sum( [ a[i][k] * b[k][j] for k in range(len(b)) ] ) for j in range(len(b[0])) ] for i in range(len(a)) ]
