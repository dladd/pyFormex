#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""Formex algebra in python"""

from numpy import *

def deprecated(replacement):
    def decorator(func):
        #print "Replacement %s" % replacement.func_name
        def wrapper(*__args,**__kw):
            print "Function %s is deprecated: use %s instead" % (func.func_name,replacement.func_name)
            return replacement(*__args,**__kw)
        return wrapper
    decorator.__doc__ = replacement.__doc__
    return decorator

def istype(a,c):
    return asanarray(a).dtype.kind == c


if 'roll' not in dir():
    def roll(a, shift, axis=None): 
        """Roll the elements in the array by 'shift' positions along 
        the given axis.

        A positive shift moves elements to the 'right' in a 1D array.
        """ 
        a = asanyarray(a) 
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

# default float and int types used in the Formex data
Float = float32
Int = int32


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
    a = arg.flat
    return sqrt(inner(a,a))

def inside(p,mi,ma):
    """Return true if point p is inside bbox defined by points mi and ma"""
    return p[0] >= mi[0] and p[1] >= mi[1] and p[2] >= mi[2] and \
           p[0] <= ma[0] and p[1] <= ma[1] and p[2] <= ma[2]

## OBSOLETE: numpy provides unique

## def unique(a):
##     """Return the unique elements in an integer array."""
##     ## OK, this looks complex. And there might be a simpler way
##     ## to do this in numpy, I just couldn't find any.
##     ## Our algorithm:
##     ## First we sort the values (1-D). Then we create an array
##     ## that flags with a "1" all the elements which are larger
##     ## than their predecessor.
##     ## The first element always gets flagged with a "1".
##     ## Finally we take the flagged elements from the sorted array.
##     b = sort(a.ravel())
##     return b[ concatenate(([1],(b[1:]) > (b[:-1]))) > 0 ]



def isClose(values,target,rtol=1.e-5,atol=1.e-8):
    """Returns an array flagging the elements close to target.

    values is a float array, target is a float value.
    values and target should be broadcastable to the same shape.
    
    The return value is a boolean array with shape of values flagging
    where the values are close to target.
    Two values a and b  are considered close if
        | a - b | < atol + rtol * | b |
    """
    values = array(values)
    target = array(target) 
    return abs(values - target) < atol + rtol * abs(target) 


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
    area = sqrt(sum(normal*normal,axis=-1))
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
             
    The special character '\' can be put before any character to make the
    move without making a connection.
    The effect of any other character is undefined.
    
    The resulting list is directly suited to initialize a Formex.
    """
    x = y = z =0
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


def equivalence(x,nodesperbox=1,shift=0.5,rtol=1.e-5,atol=1.e-5):
    """Finds (almost) identical nodes and returns a compressed list.

    The input x is an (nnod,3) array of nodal coordinates. This functions finds
    the nodes that are very close and replaces them with a single node.
    The return value is a tuple of two arrays: the remaining (nunique,3) nodal
    coordinates, and an integer (nnod) array holding an index in the unique
    coordinates array for each of the original nodes.

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
    if len(x.shape) != 2 or x.shape[1] != 3:
        raise RuntimeError, "equivalence: expects an (nnod,3) coordinate array"
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
    return (x,s)


def distanceFromPlane(f,p,n):
    """Return the distance of points f from the plane (p,n).

    f is an [...,3] array of coordinates.
    p is a point specified by 3 coordinates.
    n is the normal vector to a plane, specified by 3 components.

    The return value is a [...] shaped array with the distance of
    each point to the plane through p and having normal n.
    Distance values are positive if the point is on the side of the
    plane indicated by the positive normal.
    """
#    return (dot(f,n) - dot(p,n)) / sqrt(dot(n,n))
    a = f.reshape((-1,3))
    p = array(p).reshape((3))
    n = array(n).reshape((3))
    d = (inner(f,n) - inner(p,n)) / length(n)
    return d.reshape(f.shape[:-1])


def distanceFromLine(f,p,q):
    """Return the distance of points f from the line (p,q).

    f is an [...,3] array of coordinates.
    p and q are two points specified by 3 coordinates.

    The return value is a [...] shaped array with the distance of
    each point to the line through p and q.
    All distance values are positive or zero.
    """
    a = f.reshape((-1,3))
    p = array(p).reshape((3))
    q = array(q).reshape((3))
    n = q-p
    t = cross(n,p-a)
    d = sqrt(sum(t*t,-1)) / length(n)
    return d.reshape(f.shape[:-1])


def distanceFromPoint(f,p):
    """Return the distance of points f from the point p.

    f is an [...,3] array of coordinates.
    p is a point specified by 3 coordinates.

    The return value is a [...] shaped array with the distance of
    each point to the line through p and q.
    All distance values are positive or zero.
    """
    a = f.reshape((-1,3))
    p = array(p)
    d = a-p
    d = sum(d*d,-1)
    d = sqrt(d)
    return d.reshape(f.shape[:-1])

def boundingBox(a):
    """Return the bounding box of an array of coordinates.

    a is a [...,3] array of coordinates.
    the return value is a (2,3) array, where the first row holds the
    minimum and the second row the maximum coordinates.
    """
    coords = a.reshape((-1,3))
    return row_stack([ coords.min(axis=0), coords.max(axis=0) ])


###########################################################################
##
##   class Formex
##
#########################
#
# About Formex/Formian newspeak:
# The author of formex/formian had an incredible preference for newspeak:
# for every concept or function, a new name was invented. While this may
# give formex/formian the aspect of a sophisticated scientific background,
# it works rather distracting and ennoying for people that are already
# familiar with the basic ideas of 3D geometry, and are used to using the
# standardized terms.
# In our pyFormex we will try to use as much as possible the normal
# terminology, while referring to the formian newspeak in parentheses
# and preceded by a 'F:'. Similar concepts in Finite Element terminology
# are marked with 'FE:'.

# PITFALLS:
# Python by default uses integer math on integer arguments!
# Therefore: always create the array data with a float type!
# (this will be mostly in functions array() and zeros()
#

class Formex:
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

    def __init__(self,data=[[[]]],prop=None):
        """Create a new Formex.

        The Formex can be initialized by a 2D or 3D coordinate list,
        or by a string to be used in the pattern function to create
        a coordinate list.
        If 2D coordinates are given, a 3-rd coordinate 0.0 is added.
        Formices therefore always have 3D coordinates.
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
            data = array(data).astype(Float)

            if data.size == 0:
                data.shape = (0,0,3) # An empty Formex
            else:
                # check dimensions of data
                if not len(data.shape) in [2,3]:
                    raise RuntimeError,"Formex init: needs a rank-2 or rank-3 data array, got shape %s" % str(data.shape)
                if not data.shape[-1] in [2,3]:
                    raise RuntimeError,"Formex init: last axis dimension of data array should be 2 or 3, got shape %s" % str(data.shape)
                # add 3-rd dimension if data are 2-d
                if data.shape[-1] == 2:
                    z = zeros((data.shape[0],1),dtype=Float)
                    data = column_stack([data,z])
        # data should be OK now
        self.f = data
        self.setProp(prop)



    def __getitem__(self,i):
        """Return a formex element i of the Formex.

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
    def data(self):
        """Return the Formex as a numpy array"""
        return self.f
    def x(self):
        """Return the x-plane"""
        return self.f[:,:,0]
    def y(self):
        """Return the y-plane"""
        return self.f[:,:,1]
    def z(self):
        """Return the z-plane"""
        return self.f[:,:,2]

    # Properties
    def prop(self):
        """Return the properties as a numpy array"""
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

    # Size
    def bbox(self):
        """Return the bounding box of the Formex.

        The bounding box is the smallest rectangular volume in global
        coordinates, such at no points of the Formex are outside the
        box.
        It is returned as a [2,3] array: the first row holds the
        minimal coordinates and the second one the maximal.
        """
        return boundingBox(self.f)


    def center(self):
        """Return the center of the Formex.

        The center of the formex is the center of its bbox().
        The return value is a list of 3 coordinates.
        """
        min,max = self.bbox()
        return 0.5 * (max+min)

    def sizes(self):
        """Return the sizes of the Formex.

        Return an array with the length of the bbox along the 3 axes.
        """
        min,max = self.bbox()
        return max-min

    def size(self):
        """Return the size of the Formex.

        The size is the length of the diagonal of the bbox()."""
        min,max = self.bbox()
        return length(max-min)
    
    def bsphere(self):
        """Return the diameter of the bounding sphere of the Formex.

        The bounding sphere is the smallest sphere with center in the
        center() of the Formex, and such that no points of the Formex
        are lying outside the sphere.
        """
        return self.f - array(self.center())


    # Data conversion
    def feModel(self,nodesperbox=1,repeat=True,rtol=1.e-5,atol=1.e-5):
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
        """
        f = reshape(self.f,(self.nnodes(),3))
        f,s = equivalence(f,nodesperbox,0.5,rtol=rtol,atol=atol)
        if repeat:
            f,t = equivalence(f,nodesperbox,0.75,rtol=rtol,atol=atol)
            s = t[s]
        e = reshape(s,self.f.shape[:2])
        return (f,e)


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


    def fprint(self,fmt="%10.3e %10.3e %10.3e"):
        for el in self.data():
            for nod in el:
                print fmt % tuple(nod)
                
            

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
        self.f = concatenate((self.f,F.f))
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
        return Formex(f,p)
       
    def copy(self):
        """Return a deep copy of itself."""
        return Formex(self.f,self.p)
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
        return Formex(f,p)

      
    def select(self,idx):
        """Return a Formex which holds only elements with numbers in ids.

        idx can be a single element number or a list of numbers or
        any other index mechanism accepted by numpy's ndarray
        """
        if self.p is None:
            p = None
        else:
            p = self.p[idx]
        return Formex(self.f[idx],p)

      
    def selectNodes(self,idx):
        """Return a Formex which holds only some nodes of the parent.

        idx is a list of node numbers to select.
        Thus, if F is a grade 3 Formex representing triangles, the sides of
        the triangles are given by
        F.selectNodes([0,1]) + F.selectNodes([1,2]) + F.selectNodes([2,0])
        The returned Formex inherits the property of its parent.
        """
        return Formex(self.f[:,idx,:],self.p)

    def nodes(self):
        """Return a Formex containing only the nodes.

        This is obviously a Formex with plexitude 1. It holds the same data
        as the original Formex, but in another shape: the number of nodes
        per element is 1, and the number of elements is equal to the total
        number of nodes.
        The properties are not copied over, since they will usually not make
        any sense.
        """
        return Formex(reshape(self.f,(-1,1,self.f.shape[2])))


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
        return Formex(self.f[flag>0],p)


    def withProp(self,val):
        """Return a Formex which holds only the elements with property val.

        If the Formex has no properties, a copy is returned.
        The returned Formex inherits the matching properties.
        """
        if self.p is None:
            return Formex(self.f)
        else:
            return Formex(self.f[self.p==val],val)


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
        Two elements are not considered equal if one's elements are a
        permutation of the other's.
        """
        ##
        ##  THIS IS SLOW!! IT NEEDS TO BE REIMPLEMENTED AFTER THE sort
        ##  FUNCTION HAS BEEN DONE
        ##
        ## Maybe we need a variant that tests for equal permutations?
        flag = ones((self.f.shape[0],))
        for i in range(self.f.shape[0]):
            for j in range(i):
                if allclose(self.f[i],self.f[j]):
                    # i is a duplicate node
                    flag[i] = 0
                    break
        if self.p is None:
            p = None
        else:
            p = self.p[flag>0]
        return Formex(self.f[flag>0],p)
      
    def nonzero(self):
        """Return a Formex which holds only the nonzero elements.

        A zero element is an element where all nodes are equal."""
        # NOT IMPLEMENTED YET !!! FOR NOW, RETURNS A COPY
        return Formex(self.f)


    def reverseElements(self):
        """Return a Formex where all elements have been reversed.

        Reversing an element means reversing the order of its points.
        """
        return Formex(self.f[:,range(self.f.shape[1]-1,-1,-1),:],self.p)

# Test and clipping functions

    def test(self,nodes='all',dir=0,min=None,max=None):
        """Flag elements having nodal coordinates between min and max.

        This function is very convenient in clipping a Formex in one of
        the coordinate directions. It returns a 1D integer array flagging
        (with a value 1) the elements having nodal coordinates in the
        required range.
        Use where(result) to get a list of element numbers passing the test.
        Or directly use clip() or cclip() to create the clipped Formex.

        min,max are there minimum and maximum values required for the
        coordinates in direction dir (default is the x or 0 direction).
        nodes specifies which nodes are taken into account in the comparisons.
        It should be one of the following:
        - a single (integer) node number (< the number of nodes)
        - a list of node numbers
        - one of the special strings: 'all', 'any', 'none'
        The default ('all') will flag all the elements that have all their
        nodes between the planes x=min and x=max, i.e. the elements that
        fall completely between these planes. One of the two clipping planes
        may be left unspecified.
        """
        f = self.f
        if type(nodes)==str:
            nod = range(f.shape[1])
        else:
            nod = nodes
        if not min is None:
            T1 = f[:,nod,dir] > min
        if not max is None:
            T2 = f[:,nod,dir] < max
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
            T = (1-T.all(1)).astype(bool)
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
    def scale(self,scale):
        """Return a copy scaled with scale[i] in direction i.

        The scale should be a list of 3 numbers, or a single number.
        In the latter case, the scaling is homothetic."""
        return Formex(self.f*scale,self.p)


    def translate(self,dir,distance=None):
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
        if type(dir) is int:
            if distance is None:
                distance = 1.0
            f = self.f.copy()
            f[:,:,dir] += distance
            return Formex(f,self.p)
        if len(dir) == 2:
            dir.append(0.0)
        if distance is not None:
            dir *= distance / length(dir)
        return Formex(self.f + dir,self.p)


    def rotate(self,angle,axis=2):
        """Return a copy rotated over angle around axis.

        The angle is specified in degrees.
        The axis is either one of (0,1,2) designating the global axes,
        or a vector specifying an axis through the origin.
        If no axis is specified, rotation is around the 2(z)-axis. This is
        convenient for working on 2D-structures.

        As a convenience, the user may also specify a 3x3 rotation matrix,
        in which case the function rotate(mat) is equivalent to affine(mat).
        """
        if not isinstance(angle,ndarray):
            angle = rotationMatrix(angle,axis)
        return self.affine(angle)


    def shear(self,dir,dir1,skew):
        """Return a copy skewed in the direction dir of plane (dir,dir1).

        The coordinate dir is replaced with (dir + skew * dir1).
        """
        f = self.f.copy()
        f[:,:,dir] += skew * f[:,:,dir1]
        return Formex(f,self.p)

    def reflect(self,dir=2,pos=0):
        """Return a Formex mirrored in direction dir against plane at pos.

        Default position of the plane is through the origin.
        Default mirror direction is the z-direction.
        """
        f = self.f.copy()
        f[:,:,dir] = 2*pos - f[:,:,dir]
        return Formex(f,self.p)

    def affine(self,mat,vec=None):
        """Return a general affine transform of the Formex.

        The returned Formex has coordinates given by xorig * mat + vec,
        where mat is a 3x3 matrix and vec a length 3 list.
        """
        f = dot(self.f,mat)
        if vec is not None:
            f += vec
        return Formex(f,self.p)
#
#
#   B. Non-Affine transformations
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
        f = zeros(self.f.shape,dtype=Float)
        r = scale[0] * self.f[:,:,dir[0]]
        theta = (scale[1]*rad) * self.f[:,:,dir[1]]
        f[:,:,0] = r*cos(theta)
        f[:,:,1] = r*sin(theta)
        f[:,:,2] = scale[2] *  self.f[:,:,dir[2]]
        return Formex(f,self.p)

    def toCylindrical(self,dir=[0,1,2]):
        """Converts from cartesian to cylindrical coordinates.

        dir specifies which coordinates axes are parallel to respectively the
        cylindrical axes distance(r), angle(theta) and height(z). Default
        order is [x,y,z].
        The angle value is given in degrees.
        """
        # We can not just leave the z's in place, because there might be
        # permutation of axes.
        f = zeros(self.f.shape,dtype=Float)
        x,y,z = [ self.f[:,:,i] for i in dir ]
        f[:,:,0] = sqrt(x*x+y*y)
        f[:,:,1] = arctan2(y,x) / rad
        f[:,:,2] = z
        return Formex(f,self.p)

    
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
        f = self.f.reshape((-1,3))
        theta = (scale[0]*rad) * f[:,dir[0]]
        phi = (scale[1]*rad) * f[:,dir[1]]
        r = scale[2] * f[:,dir[2]]
        if colat:
            phi = 90.0*rad - phi
        rc = r*cos(phi)
        f = column_stack([rc*cos(theta),rc*sin(theta),r*sin(phi)])
        return Formex(f.reshape(self.f.shape),self.p)


    def cospherical(self,dir=[0,1,2],scale=[1.,1.,1.]):
        """Same as spherical with colat=True."""
        return self.spherical(dir,scale,True)


    def toSpherical(self,dir=[0,1,2]):
        """Converts from cartesian to spherical coordinates.

        dir specifies which coordinates axes are parallel to respectively
        the spherical axes distance(r), longitude(theta) and latitude(phi).
        Latitude is the elevation angle measured from equator in direction
        of north pole(90). South pole is -90.
        Default order is [0,1,2], thus the equator plane is the (x,y)-plane.
        The returned angle values are given in degrees.
        """
        v = self.f[:,:,dir].reshape((-1,3))
        dist = sqrt(sum(v*v,-1))
        long = arctan2(v[:,0],v[:,2]) / rad
        lat = where(dist <= 0.0,0.0,arcsin(v[:,1]/dist) / rad)
        f = column_stack([long,lat,dist])
        return Formex(f.reshape(self.f.shape),self.p)


    def bump1(self,dir,a,func,dist):
        """Return a Formex with a one-dimensional bump.

        dir specifies the axis of the modified coordinates;
        a is the point that forces the bumping;
        dist specifies the direction in which the distance is measured;
        func is a function that calculates the bump intensity from distance
        !! func(0) should be different from 0.
        """
        f = self.f.copy()
        d = f[:,:,dist] - a[dist]
        f[:,:,dir] += func(d)*a[dir]/func(0)
        return Formex(f,self.p)
    
    def bump2(self,dir,a,func):
        """Return a Formex with a two-dimensional bump.

        dir specifies the axis of the modified coordinates;
        a is the point that forces the bumping;
        func is a function that calculates the bump intensity from distance
        !! func(0) should be different from 0.
        """
        f = self.f.copy()
        dist = [0,1,2]
        dist.remove(dir)
        d1 = f[:,:,dist[0]] - a[dist[0]]
        d2 = f[:,:,dist[1]] - a[dist[1]]
        d = sqrt(d1*d1+d2*d2)
        f[:,:,dir] += func(d)*a[dir]/func(0)
        return Formex(f,self.p)

    
    # This is a generalization of both the bump1 and bump2 methods.
    # If it proves to be useful, it might replace them one day

    # An interesting modification might be to have a point for definiing
    # the distance and a point for defining the intensity (3-D) of the
    # modification
    def bump(self,dir,a,func,dist=None):
        """Return a Formex with a bump.

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
        f = self.f.copy()
        if dist == None:
            dist = [0,1,2]
            dist.remove(dir)
        try:
            l = len(dist)
        except TypeError:
            l = 1
            dist = [dist]
        d = f[:,:,dist[0]] - a[dist[0]]
        if l==1:
            d = abs(d)
        else:
            d = d*d
            for i in dist[1:]:
                d1 = f[:,:,i] - a[i]
                d += d1*d1
            d = sqrt(d)
        f[:,:,dir] += func(d)*a[dir]/func(0)
        return Formex(f,self.p)

    # NEW implementation flattens coordinate sets to ease use of
    # complicated functions
    def newmap(self,func):
        """Return a Formex mapped by a 3-D function.

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
        x,y,z = func(self.f[:,:,0].flat,self.f[:,:,1].flat,self.f[:,:,2].flat)
        shape = list(self.f.shape)
        shape[2] = 1
        #print shape,reshape(x,shape)
        f = concatenate([reshape(x,shape),reshape(y,shape),reshape(z,shape)],2)
        #print f.shape
        return Formex(f,self.p)

    def map(self,func):
        """Return a Formex mapped by a 3-D function.

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
        f = zeros(self.f.shape,dtype=Float)
        f[:,:,0],f[:,:,1],f[:,:,2] = func(self.f[:,:,0],self.f[:,:,1],self.f[:,:,2])
        return Formex(f,self.p)

    def map1(self,dir,func):
        """Return a Formex where coordinate i is mapped by a 1-D function.

        <func> is a numerical function which takes one argument and produces
        one result. The coordinate dir will be replaced by func(coord[dir]).
        The function must be applicable on arrays, so it should only
        include numerical operations and functions understood by the
        numpy module.
        This method is one of several mapping methods. See also map and mapd.
        """
        f = self.f.copy()
        f[:,:,dir] = func[i](self.f[:,:,dir])
        return Formex(f,self.p)

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
        f = self.f.copy()
        if dist == None:
            dist = [0,1,2]
        try:
            l = len(dist)
        except TypeError:
            l = 1
            dist = [dist]
        d = f[:,:,dist[0]] - point[dist[0]]
        if l==1:
            d = abs(d)
        else:
            d = d*d
            for i in dist[1:]:
                d1 = f[:,:,i] - point[i]
                d += d1*d1
            d = sqrt(d)
        f[:,:,dir] = func(d)
        return Formex(f,self.p)

    # This could be done by a map, but it is slightly cheaper to do it this way
    def replace(self,i,j,other=None):
        """Replace the coordinates along the axes i by those along j.

        i and j are lists of axis numbers.
        replace ([0,1,2],[1,2,0]) will roll the axes by 1.
        replace ([0,1],[1,0]) will swap axes 0 and 1.
        An optionally third argument may specify another Formex to take
        the coordinates from. It should have the same dimensions.
        """
        ## Is there a way to do this in 1 operation ?
        # if self.shape != other.shape:
        # ERROR
        if other is None:
            other = self
        f = self.f.copy()
        for k in range(len(i)):
            f[:,:,i[k]] = other.f[:,:,j[k]]
        return Formex(f,self.p)


    def swapaxes(self,i,j):
        """Swap coordinate axes i and j"""
        return self.replace([i,j],[j,i])


    def rollaxes(self,n=1):
        """Roll the axes over the given amount.

        Default is 1, thus axis 0 becomes the new 1 axis, 1 becomes 2 and
        2 becomes 0.
        """
        return Formex(roll(self.f, int(n) % 3,axis=-1),self.p)
        

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
        return Formex(factor*(self.f-c)+c,self.p)


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
        return Formex(f,self.p)
    
    def replic2(self,n1,n2,t1,t2,d1=0,d2=1,bias=0,taper=0):
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
        return Formex(f + point,self.p)

    ## A formian compatibility function that we may keep
        
    def translatem(self,*args):
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
        fil.write("# Formex File Format 1.0 (http://pyformex.berlios.de)\n")
        nelems,nplex = self.f.shape[:2]
        hasp = self.p is not None
        fil.write("# nelems=%d; nplex=%d; props=%d\n" % (nelems,nplex,hasp))
        self.f.tofile(fil,sep)
        fil.write('\n')
        if hasp:
            self.p.tofile(fil,sep)
        fil.write('\n')
        if isname:
            fil.close()


    @classmethod
    def read(clas,fil):
        """Read a Formex from file."""
        isname = type(fil) == str
        if isname:
            fil = file(fil,'r')
        s = fil.readline()
        if not s.startswith('# Formex'):
            return None
        while s.startswith('#'):
            s = fil.readline()
            if s.startswith('# nelems'):
                #print s[1:].strip()
                exec(s[1:].strip())
                break
        #print "Read %d elems of plexitude %d" % (nelems,nplex)
        f = fromfile(file=fil, dtype=Float, count=3*nelems*nplex, sep=' ').reshape((nelems,nplex,3))
        if props:
            p = fromfile(file=fil, dtype=Int, count=nelems, sep=' ')
        else:
            p = None
        if isname:
            fil.close()
        return Formex(f,p)


#########################################################################
#
# Obsolete and deprecated functions
#
# These functions are retained mainly for compatibility reasons.
# New users should avoid these functions!
# The may be removed in future.
#

    @deprecated(test)
    def where(self,*args,**kargs):
        pass

    @deprecated(feModel)
    def nodesAndElements(self):
        pass

    @deprecated(translate)
    def translate1(self,dir,distance):
        pass

    @deprecated(setProp)
    def removeProp(self):
        pass

    
    def oldspherical(self,dir=[2,0,1],scale=[1.,1.,1.]):
        """Same as spherical, but using colatitude."""
        return self.spherical([dir[1],dir[2],dir[0]],[scale[1],scale[2],scale[0]],colat=True)
    nnodel = nplex
    nnodes = npoints
    
    # Formian compatibility functions
    # These will be moved to a separate file in future.
    #
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

    def tranic(self,*args):
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
   
    def rinic(self,*args):
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

    def lamic(self,*args):
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
    m = len(Flist)
    for i in range(m):
        if isinstance(Flist[i],Formex):
            pass
        elif isinstance(Flist[i],ndarray):
            Flist[i] = Formex(Flist[i])
        else:
            raise RuntimeError,'connect(): first argument should be a list of formices'
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


def interpolate(F,G,div):
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
    """
    shape = F.shape()
    if G.shape() != shape:
        raise RuntimeError,"Can only interpolate between equal size Formices"
    if type(div) == int:
        div = arange(div+1) / float(div)
    else:
        div = array(div).ravel()
    c = F.f
    d = G.f - F.f
    r = c + outer(div,d).reshape((-1,)+shape)
    return Formex(r.reshape((-1,) + shape[1:]))


def divide(F,div):
    """Divide a plex-2 Formex at the values in div.

    Replaces each member of the Formex F by a sequence of members obtained
    by dividing the Formex at the relative values specified in div. The values
    should normally range from 0.0 to 1.0.
    
    As a convenience, if an integer is specified for div, it is taken as a
    number of divisions for the interval [0..1].

    This function only works on plex-2 Formices (line segments).
    """
    if F.nplex() != 2:
        raise RuntimeError,"Can only divide plex-2 Formices"
    if type(div) == int:
        div = arange(div+1) / float(div)
    else:
        div = array(div).ravel()
    A = interpolate(F.selectNodes([0]),F.selectNodes([1]),div[:-1])
    B = interpolate(F.selectNodes([0]),F.selectNodes([1]),div[1:])
    return connect([A,B])
    

def readfile(file,sep=',',plexitude=1,dimension=3):
    """Read a Formex from file.

    This convenience function uses the numpy fromfile function to read
    the coordinates of a Formex from file. 
    Args:
      file: either an open file object or a string with the file name.
      sep: the separator string between subsequent coordinates.
           There can be extra blanks around the separator, and the separator
           may be omitted at the end of line.
           If an empty string is specified, the file is read in binary mode.
      dimension: the number of coordinates that make up a point (1,2 or 3).
      plexitude: the number of points that make up an element of the Formex.
                 The default is to return a plex-1 Formex (unconnected points).
      closed: If True, an extra point will be added to the
    The total number of coordinates on the file should be a multiple of
    dimension * plexitude.
    """
    return Formex(fromfile(file,sep=sep).reshape((-1,plexitude,dimension)))


def bbox(formexlist):
    """Computes the bounding box of a collection of formices.

    This is like the bbox() method of the Formex class, but the resulting
    box encloses all the Formices in the list.
    """
    return Formex(concatenate([ [f.bbox()] for f in formexlist ])).bbox()



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
        print "nodes:",G.nodes()
        print "unique nodes:",G.nodes().unique()
        print "size:",G.size()
        F = Formex([[[0,0]],[[1,0]],[[1,1]],[[0,1]]])
        G = connect([F,F],bias=[0,1])
        print G
        G = connect([F,F],bias=[0,1],loop=True)
        print G
        print G[1]
        print G.feModel()
        print F
        print F.bbox()
        print F.center()
        print F.bsphere()

        F.fprint()

        F = Formex([[[0,0],[1,0],[0,1]],[[1,0],[1,1],[0,1]]])
        print F
        print F.reverseElements()
        Formex.setPrintFunction(Formex.asArray)
        print F
        F.fprint()
        #F = F.translate1(1,1)

    (f,t) = _test()
    if f == 0:
        test()

### End
