#!/usr/bin/env python
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
## Distributed under the General Public License, see file COPYING for details
##
"""Formex algebra in python"""

from numarray import *
import math

# Convenience functions: trigonometric functions with argument in degrees
# Should we keep this in ???
# or shall we redefine them as in
#   def sin(arg): return math.sin(math.radians(arg))

def sind(arg):
    """Return the sin of an angle in degrees."""
    return sin(radians(arg))

def cosd(arg):
    """Return the sin of an angle in degrees."""
    return cos(radians(arg))

def tand(arg):
    """Return the sin of an angle in degrees."""
    return tan(radians(arg))

def inside(p,mi,ma):
    """Return true if point p is inside bbox defined by points mi and ma"""
    return p[0] >= mi[0] and p[1] >= mi[1] and p[2] >= mi[2] and \
           p[0] <= ma[0] and p[1] <= ma[1] and p[2] <= ma[2]

def equal(p1,p2,tol=1.e-6):
    """Return true if two points are considered equal within tolerance."""
    return inside([ p1[i]-p2[i] for i in range(3) ],
                  [ -tol for i in range(3) ], [ +tol for i in range(3) ] )

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
# In our pyformex we will try to use as much as possible the normal
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
    The user can create formices (plural of formex) in a 2-D space, but
    internally these will be stored with 3 coordinates, by adding a third
    value 0. All operations work with 3-D coordinate sets. However, a method
    exists to extract only a limited set of coordinates from the results,
    premitting to return to a 2-D environment

    A plane along the axes 2 and 1 is a set of points (F: cantle). This can be
    thought of as a geometrical shape (2 points form a line segment, 3 points
    make a triangle, ...) or as an element in FE terms. But it reaaly is up to
    the user as to how this set of points is to be interpreted.
    The number of nodes of an element is its

    Finally, the whole Formex represents a set of such elements

    """

    def __init__(self,l=[[[]]]):
        self.f = array(l,type=Float32)
        if len(self.f.shape) != 3:
            raise RuntimeError,"Invalid data in creating Formex"
        if self.f.shape[2] == 2:
            f = zeros((self.f.shape[:2]+(3,)),type=Float32)
            f[:,:,:2] = self.f
            self.f = f

    def order(self):
        """Return the order of the Formex

        The order is the number of elements in the Formex.
        """
        return self.f.shape[0]

    def plexitude(self):
        """Return the plexitude of the Formex

        The plexitude is the number of number of nodes in the cantle.
        2 = bar, 3 = triangle, 4= quadrilateral, etc.
        """
        return self.f.shape[1]

    def grade(self):
        """Return the grade of the Formex.

        The grade is the number of dimensions of the signet.
        2 = 2D, 3 = 3D.
        """
        return self.f.shape[2]

    def data(self):
        """Return the formex as a numarray"""
        return self.f
    def x(self):
        """Return the x-plane (can be modified)"""
        return self.f[:,:,0]
    def y(self):
        """Return the x-plane (can be modified)"""
        return self.f[:,:,1]
    def z(self):
        """Return the x-plane (can be modified)"""
        return self.f[:,:,2]

    def cantle(self,i):
        """Return cantle i of the formex"""
        return self.f[i]

    def signet(self,i,j):
        """Return signet j of cantle i"""
        return self.f[i][j]

    def uniple(self,i,j,k):
        """Return uniple k of signet j of canlte i"""
        return self.f[i][j][k]

    def signet2str(self,sig):
        s = ""
        if len(sig)>0:
            s += str(sig[0])
            if len(sig) > 1:
                for i in sig[1:]:
                    s += "," + str(i)
        return s

    def cantle2str(self,can):
        s = "["
        if len(can) > 0:
            s += self.signet2str(can[0])
            if len(can) > 1:
                for i in can[1:]:
                    s += "; " + self.signet2str(i) 
        return s+"]"
    
    def asFormex(self):
        """String representation of a formex as in Formian"""
        s = "{"
        if len(self.f) > 0:
            s += self.cantle2str(self.f[0])
            if len(self.f) > 1:
                for i in self.f[1:]:
                    s += ", " + self.cantle2str(i)
        return s+"}"
                
    def asArray(self):
        return self.f.__str__()

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
        
    def copy(self):
        """Returns a deep copy of itself"""
        return Formex(self.f)

    def append(self,F):
        """Append the members of formex F to this one

        This function changes the original one! Use __add__ if you want to
        get a copy with the sum"""
        self.f = concatenate((self.f,F.f))
        return self

    def __add__(self,other):
        """Return the sum of two formices"""
        return self.copy().append(other)

    def concatenate(self,list):
        """Concatenate all formices in list.

        This is a class method, not an instance method!
        """
        return Formex( concatenate([a.f for a in list]) )


    def bbox(self):
        """Return the boundary box of the Formex"""
        min = [ self.f[:,:,i].min() for i in range(self.f.shape[2]) ]
        max = [ self.f[:,:,i].max() for i in range(self.f.shape[2]) ]
        return array([min, max]) 

    def center(self):
        """Return the center of the Formex"""
        min,max = self.bbox()
        return [ (min[i]+max[i])/2 for i in range(self.grade()) ]

    def translationVector(self,dir,dist):
        """Returns a translation vector in direction dir over distance dist"""
        f = zeros((self.grade()),type=Float32)
        f[dir] = dist
        return f

    def rotationMatrix(self,angle,axis=2):
        """Returns a rotation matrix over angle around axis.

        If grade=2, a 2x2 matrix is returned and axis is always 2.
        If grade is 3, a 3x3 matrix is returned, and default axis is 2.
        """
        n = self.grade()
        a = math.radians(angle)
        c = math.cos(a)
        s = math.sin(a)
        if n == 2:
            f = array([[c,s],[-s,c]],type=Float32)
        elif n == 3:
            axes = range(3)
            i,j,k = axes[axis:]+axes[:axis]
            f = zeros((n,n),type=Float32)
            f[i,i] = 1.0
            f[j,j] = c
            f[j,k] = s
            f[k,j] = -s
            f[k,k] = c
        return f


    # Common tranformations
        

    def translate(self,vector,distance=None):
        """Returns a copy translated over translation vector.

        If no distance is given, translation is over the specified vector.
        If a distance is given, translation is over the specified distance
        in the direction of the vector"""
        if distance:
            return Formex(self.f + scale(unitvector(vector),distance))
        else:
            return Formex(self.f + vector)

    # This could be replaced by a call to translate(), but it is cheaper
    # because we operate on one third of the coordinates only
    def translate1(self,dir,distance):
        """Returns a copy translated in direction dir over distance dist.

        The direction is specified by the axis number (0,1,2).
        """
        f = self.f.copy()
        f[:,:,dir] += distance
        return Formex(f)

    def translatem(self,*args):
        """Multiple subsequent translations in axis directions.

        The argument list is a sequence of tuples (axis number, step). 
        Thus translatem((0,x),(2,z),(1,y)) is equivalent to
        translate([x,y,z]). This function is especially conveniant
        to translate in calculated directions.
        """
        tr = [0.,0.,0.]
        for d,t in args:
            tr[d] += t
        return self.translate(tr)
        

    def rotate(self,angle,axis=2):
        """Returns a copy rotated over distance dist of matching grade."""
        m = self.rotationMatrix(angle,axis)
        return Formex(matrixmultiply(self.f,m))

    def scale(self,scale):
        """Returns a copy scaled with scale[i] in direction i"""
        return Formex(self.f*scale)

    def reflect(self,dir,pos):
        """Returns a formex mirrored in direction dir against plane at pos"""
        f = self.f.copy()
        f[:,:,dir] = 2*pos - f[:,:,dir]
        return Formex(f)

    def reflectAdd(self,dir,pos):
        """Return the sum of original plus reflection"""
        return self + self.reflect(dir,pos)

    def rindle(self,n,dir,step):
        """Returns a formex with n replications in direction dir with step.

        The original formex is the first of the n replicas."""
        f = array( [ self.f for i in range(n) ] )
        for i in range(1,n):
            f[i,:,:,dir] += i*step
        f.shape = (f.shape[0]*f.shape[1],f.shape[2],f.shape[3])
        return Formex(f)
 
    def rosette(self,n,axis,point,angle):
        """Returns a formex with n rotational replications around axis through point with angular step angle.

        axis is the number of the axis (0,1,2).
        point must have same grade as formex.
        The original formex is the first of the n replicas."""
        f = self.f - point
        f = array( [ f for i in range(n) ] )
        for i in range(1,n):
            m = self.rotationMatrix(i*angle,axis)
            f[i] = matrixmultiply(f[i],m)
        f.shape = (f.shape[0]*f.shape[1],f.shape[2],f.shape[3])
        return Formex(f + point)
    
    def generate2(self,n1,n2,d1,d2,t1,t2,bias=0,taper=0):
        """Generate copies in two directions.

        n1,n2 number of replications in direction d1,d2
        t1,t2 step in these directions
        bias, taper : extra step and extra number of generations in direction
        d1 for each generation in direction d2
        """
        P = [ self.translatem((d1,i*bias),(d2,i*t2)).rindle(n1+i*taper,d1,t1)
              for i in range(n2) ]
        return self.concatenate(P)

    def cylindrical(self,dir=[0,1,2],scale=[1.,1.,1.]):
        """Converts from cylindrical to cartesian after scaling.

        dir specifies which coordinates are interpreted as resp.
        distance(r), angle(theta) and height(z).
        scale will scale the coordinate values prior to the transformation.
        angle is then interpreted as degrees.
        """
        f = zeros(self.f.shape,type=Float32)
        r = scale[0] * self.f[:,:,dir[0]]
        theta = math.radians(scale[1]) * self.f[:,:,dir[1]]
        f[:,:,0] = r*cos(theta)
        f[:,:,1] = r*sin(theta)
        f[:,:,2] = scale[2] *  self.f[:,:,dir[2]]
        return Formex(f)
    
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
        theta = math.radians(scale[1]) * self.f[:,:,dir[1]]
        phi = math.radians(scale[2]) * self.f[:,:,dir[2]]
        rc = r*sin(phi)
        f[:,:,0] = rc*cos(theta)
        f[:,:,1] = rc*sin(theta)
        f[:,:,2] = r*cos(phi)
        return Formex(f)
      
    def unique(self):
        """Return a formex which holds only the unique elements."""
        # NOT IMPLEMENTED YET !!! FOR NOW, RETURNS A COPY
        return Formex(self.f)
      
    def nonzero(self):
        """Return a formex which holds only the nonzero elements.

        A zero element is an element where all nodes are equal."""
        # NOT IMPLEMENTED YET !!! FOR NOW, RETURNS A COPY
        return Formex(self.f)


    def bump1(self,dir,a,func,dist):
        """Return a formex with a one-dimensional bump.

        dir specifies the axis of the modified coordinates;
        a is the point that forces the bumping;
        dist specifies the direction in which the distance is measured;
        func is a function that calculates the bump intensity from distance
        !! func(0) should be different from 0.
        """
        f = copy.deepcopy(self.f)
        d = f[:,:,dist] - a[dist]
        f[:,:,dir] += func(d)*a[dir]/func(0)
        return Formex(f)
    
    def bump2(self,dir,a,func):
        """Return a formex with a two-dimensional bump.

        dir specifies the axis of the modified coordinates;
        a is the point that forces the bumping;
        func is a function that calculates the bump intensity from distance
        !! func(0) should be different from 0.
        """
        f = copy.deepcopy(self.f)
        dist = [0,1,2]
        dist.remove(dir)
        d1 = f[:,:,dist[0]] - a[dist[0]]
        d2 = f[:,:,dist[1]] - a[dist[1]]
        d = sqrt(d1*d1+d2*d2)
        f[:,:,dir] += func(d)*a[dir]/func(0)
        return Formex(f)

    
    # This is a generalization of both the bump1 and bump2 methods.
    # If it proves to be useful, it might replace them one day

    # An interesting modification might be to have a point for definiing
    # the distance and a point for defining the intensity (3-D) of the
    # modification
    def bump(self,dir,a,func,dist=None):
        """Return a formex with a bump.

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
        f = copy.deepcopy(self.f)
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
        return Formex(f)

    def map(self,func):
        """Return a Formex where point is mapped by a 3-D function.

        This is one of the versatile mapping functions.
        func is a numerical function which takes three arguments and produces
        a list of three output values. The coordinates [x,y,z] will be
        replaced by func(x,y,z).
        The function must be applicable on numarrays, so it should
        only include numerical operations and functions understood by the
        numarray module.
        This method is one of several mapping methods. See also map1 and mapd.
        Example: E.map(lambda x,y,z: [2*x,3*y,4*z])
        is equivalent with E.scale([2,3,4])
        """
        f = zeros(self.f.shape,type=Float32)
        f[:,:,0],f[:,:,1],f[:,:,2] = func(self.f[:,:,0],self.f[:,:,1],self.f[:,:,2])
        return Formex(f)

    def replace(self,i,j):
        """Replace the coordinates along the axes i by those along j.

        i and j are lists of axis numbers.
        replace ([0,1,2],[1,2,0]) will roll the axes by 1.
        replace ([0,1],[1,0]) will swap axes 0 and 1.
        """
        f = copy.deepcopy(self.f.shape)
        for k in range(len(i)):
            f[:,:,i[k]] = self.f[:,:,j[k]]

    def map1(self,func):
        """Return a Formex where each coordinate is mapped by a 1-D function.

        This is one of the versatile mapping functions.
        func is a list of three numerical functions [f,g,h], each of which
        takes one argument and produces one value. The coordinates [x,y,z]
        will be replaced by [ f(x), g(y), h(z) ].
        The functions f,g,h must be applicable on numarrays, so they should
        only include numerical operations and functions understood by the
        numarray module.
        This method is one of several mapping methods. See also map and mapd.
        """
        f = zeros(self.f.shape,type=Float32)
        for i in range(3):
            f[:,:,i] = func[i](self.f[:,:,i])
        return Formex(f)
        

    # Compatibility functions # deprecated !
        
    def give():
        print self.toFormian()

    def tran(self,dir,dist):
        return self.translate1(dir-1,dist)
    
    def ref(self,dir,dist):
        return self.reflect(dir-1,dist)

    def rin(self,dir,n,dist):
        return self.rindle(n,dir-1,dist)

    def lam(self,dir,dist):
        return self.reflectAdd(dir-1,dist)

    def ros(self,i,j,x,y,n,angle):
        if (i,j) == (1,2):
            return self.rosette(n,2,[x,y,0],angle)
        elif (i,j) == (2,3):
            return self.rosette(n,0,[0,x,y],angle)
        elif (i,j) == (1,3):
            return self.rosette(n,1,[x,0,y],-angle)

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
        return self.rosette(n,2,[a,b,0],angle)
    def rosas(self,a,b,n=4,angle=90):
        return self.rosette(n,1,[a,0,b],angle)
    def rosat(self,a,b,n=4,angle=90):
        return self.rosette(n,0,[0,a,b],angle)

    def genid(self,n1,n2,t1,t2,bias=0,taper=0):
        return self.generate2(n1,n2,0,1,t1,t2,bias,taper)
    def genis(self,n1,n2,t1,t2,bias=0,taper=0):
        return self.generate2(n1,n2,0,2,t1,t2,bias,taper)
    def genit(self,n1,n2,t1,t2,bias=0,taper=0):
        return self.generate2(n1,n2,1,2,t1,t2,bias,taper)

    def bb(self,b1,b2):
        return self.scale([b1,b2,1.])

    def bc(self,b1,b2,b3):
        return self.cylindrical(scale=[b1,b2,b3])

    def bp(self,b1,b2):
        return self.cylindrical(scale=[b1,b2,1.])

    def bs(self,b1,b2,b3):
        return self.spherical(scale=[b1,b2,b3])

    pex = unique
    tic = int
    def ric(f):
        return int(round(f))

    def globals(self):
        return globals()

    globals = classmethod(globals)


#### Test
if __name__ == "__main__":
    
    def test():
        print "This is a test of formex algebra"
##        F = Formex([[[1,0],[0,1]],[[0,1],[1,2]]])
##        print "F =",F
##        F1 = F.tran(1,6)
##        print "F1 =",F1
##        F2 = F.ref(1,2)
##        print "F2 =",F2
##        F3 = F.ref(1,1.5).tran(2,2)
##        print "F3 =",F3
##        H = F.rin(1,4,2)
##        print "H =",H
##        R = F.lam(1,1)
##        print "R =",R
##        G = F.lam(1,1).lam(2,1).rin(1,10,2).rin(2,6,2)
##        print "G =",G
        F = Formex([[[1,0,0],[0,1,1]]])
        print F
        G = F.translatem((1,4),(2,10))
        print G
        H = F.tranic(2,3,4,10)
        print H
    test()

#### End
