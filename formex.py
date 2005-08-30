#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.2.1 Release Fri Apr  8 23:30:39 2005
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where otherwise stated 
##
"""Formex algebra in python"""

from numarray import *
import math
import vector

###########################################################################
##
##   some math functions
##
#########################

# multiplier to transform degrees to radians
rad = math.pi/180.

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
    a = arg.getflat()
    return math.sqrt(sum(multiply(a,a)))

def inside(p,mi,ma):
    """Return true if point p is inside bbox defined by points mi and ma"""
    return p[0] >= mi[0] and p[1] >= mi[1] and p[2] >= mi[2] and \
           p[0] <= ma[0] and p[1] <= ma[1] and p[2] <= ma[2]

def unique(a):
    """Return the unique elements in an integer array."""
    ## OK, this looks complex. And there might be a simpler way
    ## to do this in numarray, I just couldn't find any.
    ## Our algorithm:
    ## First we sort the values (1-D). Then we create an array that flags
    ## the elements which are larger than their predecessor with a "1".
    ## The first element always gets flagged with a "1".
    ## Finally we take the flagged elements from the sorted array.
    b = sort(a.getflat())
    return b[ concatenate(([1],(b[1:]) > (b[:-1]))) > 0 ]

##
## If p1 and p2 are arrays, this can better be replaced by
## allclose(p1,p2,rtol,atol)
def equal(p1,p2,tol=1.e-6):
    """Return true if two points are considered equal within tolerance."""
    return inside([ p1[i]-p2[i] for i in range(3) ],
                  [ -tol for i in range(3) ], [ +tol for i in range(3) ] )

def pattern(s):
    """Return a line segment pattern created from a string.

    This function creates a list of line segments where all nodes lie on the
    gridpoints of a regular grid with unit step.
    The first point of the list is [0,0]. Each character from the given
    string is interpreted as a code specifying how to move to the next node.
    Currently defined are the following codes:
    0 = goto origin [0,0]
    1 = East, 2 = North, 3 = West, 4 = South, 5 = NE, 6 = NW, 7 = SW, 8 = SE
    / = go to the next point without connecting.
    The resulting list is directly suited to initialize a Formex in the
    (x,y)-plane.
    """
    x = y = 0
    l = []
    connect=True
    for c in s:
        pos = [x,y]
        if c=="0":
            x = y = 0
        elif c=="1":
            x += 1
        elif c == "2":
            y += 1
        elif c == "3":
            x -= 1
        elif c == "4":
            y -= 1
        elif c == "5":
            x += 1
            y += 1
        elif c == "6":
            x -= 1
            y += 1
        elif c == "7":
            x -= 1
            y -= 1
        elif c == "8":
            x += 1
            y -= 1
        elif c == "/":
            connect = False
            continue
        else:
            print "Unknown pattern character %c ignored" % c
            continue
        if connect:
            l.append([pos,[x,y]])
        connect=True
    return l

def translationVector(dir,dist):
    """Returns a translation vector in direction dir over distance dist"""
    f = [0.,0.,0.]
    f[dir] = dist
    return f

def rotationMatrix(angle,axis=None):
    """Returns a rotation matrix over angle, optionally around axis.

    If axis==None (default), a 2x2 rotation matrix is returned.
    Else, axis should be one of [ 0,1,2] and specifies the rotation axis
    in a 3D world. A 3x3 rotation matrix is returned.
    """
    a = angle*rad
    c = math.cos(a)
    s = math.sin(a)
    if axis==None:
        f = [[c,s],[-s,c]]
    else:
        f = [[0 for i in range(3)] for j in range(3)]
        axes = range(3)
        i,j,k = axes[axis:]+axes[:axis]
        f[i][i] = 1.0
        f[j][j] = c
        f[j][k] = s
        f[k][j] = -s
        f[k][k] = c
    return f


###########################################################################
##
##   class Formex
##
#########################
#
# Update 02 Jul 2004
# For simplicity's sake, we work now only with 3-D coordinates.
# The user can create formices in a 2-D space,
# but internally they will be stored with 3 coordinates, adding a z-value 0.
# A special operator formex2D lets you extract a 2-D coordinate list

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
# Therefore: always create the numarray data with type Float32!
# (this will be mostly in functions array() and zeros()
#

class Formex:
    """A Formex is a numarray of order 3 (axes 0,1,2) and type Float.
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
            
    def globals(self):
        """Return the list of globals defined in this module."""
        return globals()
    globals = classmethod(globals)

###########################################################################
#
#   Create a new Formex
#

    def __init__(self,data=[[[]]],prop=None):
        """Create a new Formex.

        The Formex can be initialized by a 3D coordinate list,
        or by a string to be used in the pattern function to create
        a coordinate list.
        """
        if type(data) == str:
            data = pattern(data)
        self.f = array(data,type=Float32)
        self.p = None
        if len(self.f.shape) != 3:
            raise RuntimeError,"Invalid data in creating Formex"
        if self.f.shape[2] == 2:
            f = zeros((self.f.shape[:2]+(3,)),type=Float32)
            f[:,:,:2] = self.f
            self.f = f
        if prop != None:
            self.setProp(prop)

###########################################################################
#
#   Return information about a Formex
#
    def nelems(self):
        """Return the number of elements in the formex."""
        return self.f.shape[0]
    
    def nnodel(self):
        """Return the number of nodes per element.

        Examples:
        1: unconnected nodes,
        2: straight line elements,
        3: triangles or quadratic line elements,
        4: tetraeders or quadrilaterals or cubic line elements.
        """
        return self.f.shape[1]
    
    def ndim(self):
        """Return the number of dimensions.

        This is the number of coordinates for each node. In the
        current implementation this is always 3, though you can
        define 2D Formices by given only two coordinates: the third
        will automatically be set to zero.
        """
        return self.f.shape[2]
    
    def nnodes(self):
        """Return the number of nodes in the formex.

        This is the product of the number of elements in the formex
        with the number of nodes per element.
        """
        return self.f.shape[0]*self.f.shape[1]
    
    def shape(self):
        """Return the shape of the Formex.

        The shape of a Formex is the shape of its data array,
        i.e. a tuple (nelems, nnodel, ndim).
        """
        return self.f.shape

    def data(self):
        """Return the Formex as a numarray"""
        return self.f
    def x(self):
        """Return the x-plane (can be modified)"""
        return self.f[:,:,0]
    def y(self):
        """Return the y-plane (can be modified)"""
        return self.f[:,:,1]
    def z(self):
        """Return the z-plane (can be modified)"""
        return self.f[:,:,2]
    def prop(self):
        """Return the properties as a numarray"""
        return self.p

    def element(self,i):
        """Return element i of the Formex"""
        return self.f[i]

    def point(self,i,j):
        """Return point j of element i"""
        return self.f[i][j]

    def coord(self,i,j,k):
        """Return coord k of point j of element i"""
        return self.f[i][j][k]

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
    
    def bbox(self):
        """Return the bounding box of the Formex.

        The bounding box is the smallest rectangular volume in global
        coordinates, such at no points of the Formex are outside the
        box. It is returned as a list of two points: one with the
        minimal coordinates and one with the maximal."""
        min = [ self.f[:,:,i].min() for i in range(self.f.shape[2]) ]
        max = [ self.f[:,:,i].max() for i in range(self.f.shape[2]) ]
        return [min, max]

    def center(self):
        """Return the center of the Formex.

        The center of the formex is the center of its bbox().
        """
        min,max = self.bbox()
        return vector.midPoint(min,max)

    def size(self):
        """Return the size of the Formex.

        The size is the length of the diagonal of the bbox()."""
        min,max = self.bbox()
        return vector.distance(min,max)
    
    def bsphere(self):
        """Return the diameter of the bounding sphere of the Formex.

        The bounding sphere is the smallest sphere with center in the
        center() of the Formex, and such that no points of the Formex
        are lying outside the sphere.
        """
        return self.f - self.center()

    def propSet(self):
        """Return a list with unique property values on this Formex."""
        return unique(self.p)

    def nodesAndElements(self):
        """Return a tuple of nodal coordinates and element connectivity.

        A tuple of two arrays is returned. The first is float array with
        the coordinates of the unique nodes of the Formex. The second is
        an integer array with the node numbers connected by each element.
        """
        ## THIS IS EXTREMELY SLOW IF PERFORMED ON A LARGE FORMEX
        ## SHOULD BE IMPLEMENTED IN C
        nnod = self.nnodes()
        f = reshape(self.f,(nnod,3))
        e = arange(nnod)
        # first sort
        sel = argsort(f[:,0])
        old = argsort(sel)
        f = f[sel]
        e = old[e]
        # now compact
        flag = ones((nnod,))
        sel = arange(nnod)
        for i in range(nnod):
            j = i-1
            while j>=0 and allclose(f[i][0],f[j][0]):
                if allclose(f[i],f[j]):
                    # node i is same as node j
                    flag[i] = 0
                    sel[i] = sel[j]
                    sel[i+1:nnod] -= 1
                    break
                j = j-1
##        print flag
##        print sel
        f = f[flag>0]
        e = reshape(sel,self.f.shape[:2])
        return (f,e)
        

##############################################################################
# Create string representations of a Formex
#

    def signet2str(self,sig):
        """Returns a string representation of a signet"""
        s = ""
        if len(sig)>0:
            s += str(sig[0])
            if len(sig) > 1:
                for i in sig[1:]:
                    s += "," + str(i)
        return s

    def cantle2str(self,can):
        """Returns a string representation of a cantle"""
        s = "["
        if len(can) > 0:
            s += self.signet2str(can[0])
            if len(can) > 1:
                for i in can[1:]:
                    s += "; " + self.signet2str(i) 
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
            s += self.cantle2str(self.f[0])
            if len(self.f) > 1:
                for i in self.f[1:]:
                    s += ", " + self.cantle2str(i)
        return s+"}"
                
    def asArray(self):
        """Return string representation as a numarray."""
        return self.f.__str__()

    def asFormexWithProp(self):
        """Return string representation as Formex with properties.

        The string representation as done by asFormex() is followed by
        the words "with prop" and a list of the properties.
        """
        s = self.asFormex()
        if self.p != None:
            s += " with prop "+self.p.__str__()
        else:
            s += " no prop "
        return s

    #default print function
    __str__ = asFormex

    def setPrintFunction (clas,func):
        """Choose the default formatting for printing formices.

        This sets how formices will be formatted by a print statement.
        Currently there are two available functions: asFormex, asArray.
        The user may create its own formatting method.
        This is a class function. It should be used asfollows:
        Formex.setPrintFunction(Formex.asArray).
        """
        clas.__str__ = func
    setPrintFunction = classmethod(setPrintFunction)

##############################################################################
#
#  Functions that change the Formex itself
#

    def setProp(self,p=0):
        """Create a property array for the Formex.

        You can specify a single value or a list/array of values.
        If the number of passed values is less than the number of
        elements, they are repeated.
        The default argument will give all elements a property value 0.
        """
        p = asarray(p,type=Int32)
        self.p = resize(p,self.f.shape[:1])
        return self

    def removeProp(self):
        """Remove the properties from a Formex."""
        self.p = None
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
        ## I suggest to use zero property values for the Formex without props
        if self.p != None or F.p != None:
            if self.p == None:
                self.p = zeros(shape=self.f.shape[:1],type=Int32)
            if F.p == None:
                p = zeros(shape=F.f.shape[:1],type=Int32)
            else:
                p = F.p
            self.p = concatenate((self.p,p))
        return self

    def sort(self):
        """Sorts the elements of a Formex.

        Sorting is done according to the bbox of the elements.
        !! NOT FULLY IMPLEMENTED: CURRENTLY ONLY SORTS ACCORDING TO
        !! THE 0-direction OF NODE 0
        """
        sel = argsort(self.x()[:,0])
        self.f = self.f[sel]
        if self.p:
            self.p = self.p[sel]
        return self

## MAYBE WE SHOULD LET MORE FUNCTIONS OPERATE ON THE FORMEX ITSELF
## INSTEAD OF RETURNING A NEW ONE??

## 
## All others functions should operate on and return copies. This in intended
## so that the user can write statements like
##   G = F.op1().op2().op3()
## without changing F.
## While this may seem to create a lot of intermediate data, I think that
## Python is clever enough to free this storage upon garbage collection
##

##############################################################################
#
# Create copies, concatenations, subtractions, connections, ...
#
        
    def copy(self):
        """Returns a deep copy of itself."""
        return Formex(self.f,self.p)
        ## IS THIS CORRECT? Shouldn't this be self.f.copy() ???
        ## In all examples it works, I think because the operations on
        ## the numarray data cause a copy to be made. Need to explore this.

    def __add__(self,other):
        """Return the sum of two formices"""
        return self.copy().append(other)


    ## DO we really need this? Could be written as F+F+F
    ## Find out if there would be performance penalty?
    ## Then maybe move to deprecated compatibility functions
    ## It may come in handy though.

    def concatenate(self,list):
        """Concatenate all formices in list.

        This is a class method, not an instance method!
        >>> F = Formex([[[1,2,3]]])
        >>> print Formex.concatenate([F,F,F])
        {[1.0,2.0,3.0], [1.0,2.0,3.0], [1.0,2.0,3.0]}
        """
        ## return Formex( concatenate([a.f for a in list]) )
        ## This is not so simple anymore because of the handling of properties
        F = list[0]
        for G in list[1:]:
            F += G
        return F
    concatenate = classmethod(concatenate)

    def hasProp(self,val):
        """Return a Formex which holds only the elements with property val.

        If the Formex has no properties, a copy is returned.
        The returned Formex is always without properties.
        """
        if self.p == None:
            return Formex(self.f)
        else:
            return Formex(self.f[self.p==val])

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
        if self.p == None:
            p = None
        else:
            p = self.p[flag>0]
        return Formex(self.f[flag>0],p)
      
    def nonzero(self):
        """Return a Formex which holds only the nonzero elements.

        A zero element is an element where all nodes are equal."""
        # NOT IMPLEMENTED YET !!! FOR NOW, RETURNS A COPY
        return Formex(self.f)
      
    def select(self,idx):
        """Return a Formex which holds only elements with numbers in ids.

        idx can be a single element number or a list of numbers"""
        if self.p == None:
            p = None
        else:
            p = self.p[idx]
        return Formex(self.f[idx],p)

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
        if self.p == None:
            p = None
        else:
            p = self.p[flag>0]
        return Formex(self.f[flag>0],p)
        
    def connect(self,Flist,nodid=None,bias=None,loop=False):
        """Return a Formex which connects the formices in list.

        This is a class method, not an instance method!
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
        ## !! Loop does not work correctly. And I'm not sure if we will
        ## ever fix it: It might be better to let the user extend the
        ## formices where needed
        ## Maybe give a warning?
        m = len(Flist)
        if not nodid:
            nodid = [ 0 for i in range(m) ]
        if not bias:
            bias = [ 0 for i in range(m) ]
        if loop:
            n = max([ Flist[i].order() for i in range(m) ])
        else:
            n = min([ Flist[i].order() - bias[i] for i in range(m) ])
        f = zeros((n,m,3),type=Float32)
        for i,j,k in zip(range(m),nodid,bias):
            f[:,i,:] = resize(Flist[i].f[k:k+n,j,:],(n,3))
        return Formex(f)
    connect = classmethod(connect)


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
        """Returns a copy scaled with scale[i] in direction i.

        The scale should be a list of 3 numbers, or a single number.
        In the latter case, the scaling is homothetic."""
        return Formex(self.f*scale,self.p)

    def translate(self,vector,distance=None):
        """Returns a copy translated over distance in direction of vector.

        If no distance is given, translation is over the specified vector.
        If a distance is given, translation is over the specified distance
        in the direction of the vector."""
        if distance:
            return Formex(self.f + scale(unitvector(vector),distance),self.p)
        else:
            return Formex(self.f + vector,self.p)

    # This could be replaced by a call to translate(), but it is cheaper
    # because we operate on one third of the coordinates only
    def translate1(self,dir,distance):
        """Returns a copy translated in direction dir over distance dist.

        The direction is specified by the axis number (0,1,2).
        """
        f = self.f.copy()
        f[:,:,dir] += distance
        return Formex(f,self.p)

    def rotate(self,angle,axis=2):
        """Returns a copy rotated over angle around coordinate axis."""
        m = rotationMatrix(angle,axis)
        return Formex(matrixmultiply(self.f,m),self.p)

    # This could be made the same function as rotate, but differentiated
    # by means of the value of the second argument
    def rotateAround(self,vector,angle):
        """Returns a copy rotated over angle around vector."""
        print "rotateAround has not been implemented yet!"
        return self

    def shear(self,dir,dir1,skew):
        """Returns a copy skewed in the direction dir of plane (dir,dir1).

        The coordinate dir is replaced with (dir + skew * dir1).
        """
        f = self.f.copy()
        f[:,:,dir] += skew * f[:,:,dir1]
        return Formex(f,self.p)

    def reflect(self,dir,pos=0):
        """Returns a Formex mirrored in direction dir against plane at pos.

        Default position of the plane is through the origin.
        """
        f = self.f.copy()
        f[:,:,dir] = 2*pos - f[:,:,dir]
        return Formex(f,self.p)

    def affine(self,mat,vec=None):
        """Returns a general affine transform of the Formex.

        The returned Formex has coordinates given by mat * xorig + vec,
        where mat is a 3x3 matrix and vec a length 3 list.
        """
        f = matrixmultiply(self.f,mat)
        if not vec==None:
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
        The resulting angle is interpreted in degrees.
        """
        # We put in a optional scaling, because doing this together with the
        # transforming is cheaper than first scaling and then transforming.
        f = zeros(self.f.shape,type=Float32)
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
        f = zeros(self.f.shape,type=Float32)
        x,y,z = [ self.f[:,:,i] for i in dir ]
        #print x
        #print y
        #print z
        f[:,:,0] = sqrt(x*x+y*y)
        f[:,:,1] = arctan2(y,x) / rad
        f[:,:,2] = z
        return Formex(f,self.p)
    
    def spherical(self,dir=[0,1,2],scale=[1.,1.,1.]):
        """Converts from spherical to cartesian after scaling.

        <dir> specifies which coordinates are interpreted as resp.
        distance(r), longitude(theta) and colatitude(phi).
        <scale> will scale the coordinate values prior to the transformation.
        Angles are then interpreted in degrees.
        Colatitude is 90 degrees - latitude, i.e. the elevation angle measured
        from north pole(0) to south pole(180). This choice facilitates the
        creation of spherical domes.
        """
        f = zeros(self.f.shape,type=Float32)
        r = scale[0] * self.f[:,:,dir[0]]
        theta = (scale[1]*rad) * self.f[:,:,dir[1]]
        phi = (scale[2]*rad) * self.f[:,:,dir[2]]
        rc = r*sin(phi)
        f[:,:,0] = rc*cos(theta)
        f[:,:,1] = rc*sin(theta)
        f[:,:,2] = r*cos(phi)
        return Formex(f,self.p)

    def toSpherical(self,dir=[0,1,2]):
        """Converts from cartesian to spherical coordinates.

        dir specifies which coordinates axes are parallel to respectively
        the spherical axes distance(r), longitude(theta) and colatitude(phi).
        Colatitude is 90 degrees - latitude, i.e. the elevation angle measured
        from north pole(0) to south pole(180).
        Default order is [0,1,2], thus the equator plane is the (x,y)-plane.
        The returned angle values are given in degrees.
        """
        f = zeros(self.f.shape,type=Float32)
        x,y,z = [f[:,:,i] for i in dir]
        distance = length(v)
        longitude = arctan2(v[0],v[2]) / rad
        latitude = arcsin(v[1]/distance) / rad
        return [ longitude, latitude, distance ]

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
        distdir is the direction in which the distance is measured : this can
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
        #print d
        #print a[dir]/func(0)
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
        The function must be applicable to numarrays, so it should
        only include numerical operations and functions understood by the
        numarray module.
        This method is one of several mapping methods. See also map1 and mapd.
        Example: E.map(lambda x,y,z: [2*x,3*y,4*z])
        is equivalent with E.scale([2,3,4])
        """
        x,y,z = func(self.f[:,:,0].flat,self.f[:,:,1].flat,self.f[:,:,2].flat)
        shape = list(self.f.shape)
        shape[2] = 1
        print shape,reshape(x,shape)
        f = concatenate([reshape(x,shape),reshape(y,shape),reshape(z,shape)],2)
        print f.shape
        return Formex(f,self.p)

    def map(self,func):
        """Return a Formex mapped by a 3-D function.

        This is one of the versatile mapping functions.
        func is a numerical function which takes three arguments and produces
        a list of three output values. The coordinates [x,y,z] will be
        replaced by func(x,y,z).
        The function must be applicable to numarrays, so it should
        only include numerical operations and functions understood by the
        numarray module.
        This method is one of several mapping methods. See also map1 and mapd.
        Example: E.map(lambda x,y,z: [2*x,3*y,4*z])
        is equivalent with E.scale([2,3,4])
        """
        f = zeros(self.f.shape,type=Float32)
        f[:,:,0],f[:,:,1],f[:,:,2] = func(self.f[:,:,0],self.f[:,:,1],self.f[:,:,2])
        return Formex(f,self.p)

    def map1(self,dir,func):
        """Return a Formex where coordinate i is mapped by a 1-D function.

        <func> is a numerical function which takes one argument and produces
        one result. The coordinate dir will be replaced by func(coord[dir]).
        The function must be applicable on numarrays, so it should only
        include numerical operations and functions understood by the
        numarray module.
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
        The function must be applicable on numarrays, so it should only
        include numerical operations and functions understood by the
        numarray module.
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
        ## IS there a numarray way to do this in 1 operation ?
        # if self.shape != other.shape:
        # ERROR
        if other == None:
            other=self
        f = self.f.copy()
        for k in range(len(i)):
            f[:,:,i[k]] = other.f[:,:,j[k]]
        return Formex(f,self.p)

    def swapaxes(self,i,j):
        """Swap coordinate axes i and j"""
        return self.replace([i,j],[j,i])

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
            m = rotationMatrix(i*angle,axis)
            f[i] = matrixmultiply(f[i],m)
        f.shape = (f.shape[0]*f.shape[1],f.shape[2],f.shape[3])
        return Formex(f + point,self.p)
        

##############################################################################
#
# Compatibility functions # deprecated !
#
# New users should avoid these functions!
# They will be moved to a separate file in future!
#

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


    order = nelems
    plexitude = nnodel
    grade = ndim

    cantle = element
    signet = point
    uniple = coord
    cop = remove
    
    def give():
        print self.toFormian()

    def tran(self,dir,dist):
        return self.translate1(dir-1,dist)
    
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
        return self.spherical(scale=[b1,b2,b3])

    pex = unique
    def tic(f):
        return int(f)
    def ric(f):
        return int(round(f))


            
##############################################################################
#
#  Testing
#
#  Some of the docstrings above hold test examples. They should be carefully 
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
        F1 = F.tran(1,6)
        F1.setProp(5)
        print "F1 =",F1
        F2 = F.ref(1,2)
        print "F2 =",F2
        F3 = F.ref(1,1.5).tran(2,2)
        F3.setProp([1,2])
        G = F1+F3+F2+F3
        print "F1+F3+F2+F3 =",G
        print "elbbox:",G.elbbox()
        print "met prop 1:",G.hasProp(1)
        print "unique:",G.unique()
        print "nodes:",G.nodes()
        print "unique nodes:",G.nodes().unique()
        print "size:",G.size()
        F = Formex([[[0,0]],[[1,0]],[[1,1]],[[0,1]]])
        G = Formex.connect([F,F],bias=[0,1])
        print G
        G = Formex.connect([F,F],bias=[0,1],loop=True)
        print G
        print G[1]
        print G.nodesAndElements()
        print F
        print F.center()
        print F.bsphere()


    (f,t) = _test()
    if f == 0:
        test()
    

#### End
