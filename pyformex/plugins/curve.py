#!/usr/bin/env pyformex --gui
# $Id$

"""Definition of curves in pyFormex.

(C) 2008 Benedict Verhegghe (bverheg at users.berlios.de)
I wrote this software in my free time, for my joy, not as a commissioned task.
Any copyright claims made by my employer should therefore be considered void.
Acknowledgements: Gianluca De Santis

This module defines classes and functions specialized for handling
one-dimensional geometry in pyFormex. These may be straight lines, polylines,
higher order curves and collections thereof. In general, the curves are 3D,
but special cases may be created for handling plane curves.
"""


from formex import *


##############################################################################
#
class PolyLine(Coords):
    """A class representing a series of straight line segments."""

    def __init__(self,coords=[],closed=False):
        """Initialize a PolyLine from a coordinate array.

        coords is a (npts,3) shaped array of coordinates of the subsequent
        vertices of the polyline (or a compatible data object).
        If closed == True, the polyline is closed by connecting the last
        point to the first. This does not change the vertex data.
        """
        coords = Coords(coords)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError,"Expected an (npoints,3) coordinate array"
        self.coords = coords
        self.closed = closed
        print "Polyline has %s segments" % self.nlines()


    def nlines(self):
        """Return the number of straight segments in the polyline."""
        n = self.coords.shape[0]
        if not self.closed:
            n -= 1
        return n
    

    def asFormex(self):
        """Return the polyline as a Formex."""
        x = self.coords
        return connect([x,x],bias=[0,1],loop=self.closed)


    def length(self):

        """Return the total length of the polyline."""
        x = self.coords
        y = roll(x,-1,axis=0)
        if not self.closed:
            n = self.coords.shape[0] - 1
            x = x[:n]
            y = y[:n]
        return length(y-x)#.sum()



##############################################################################
#
class NaturalSpline(object):
    """A class representing a natural spline."""

    def __init__(self,pts,endcond=['notaknot','notaknot'],closed=False):
        """Create a natural spline through the given points.

        pts specifies the coordinates of a set of points. A natural spline
        is constructed through this points.
        endcond specifies the end conditions in the first, resp. last point.
        It can be 'notaknot' or 'secder'.
        With 'notaknot', maximal continuity (up to the third derivative)
        is obtained between the first two splines.
        With 'secder', the spline ends with a zero second derivative.
        If closed is True, the spline is closed, and endcond is ignored.
        """
        pts = Coords(pts)
        if closed:
            pts = Coords.concatenate([pts,pts[:1]])
        self.coords = pts
        self.nsplines = self.coords.shape[0] - 1
        self.closed = closed
        self.endcond = endcond
        self.compute_coefficients()


    def compute_coefficients(self):
        x, y, z = self.coords.x(),self.coords.y(),self.coords.z()
        n = self.nsplines
        M = zeros([4*n, 4*n])
        
        # constant submatrix
        m = array([[0., 0., 0., 1., 0., 0., 0., 0.],
                   [1., 1., 1., 1., 0., 0., 0., 0.],
                   [3., 2., 1., 0., 0., 0.,-1., 0.],
                   [6., 2., 0., 0., 0.,-2., 0., 0.]])

        bx, by, bz=zeros([4*n, 1]), zeros([4*n, 1]), zeros([4*n, 1])
        for i in range(n-1):
            f=4*i
            M[f,  [f ,f+1, f+2, f+3, f+4, f+5, f+6, f+7]]=m[0]
            M[f+1,  [f ,f+1, f+2, f+3, f+4, f+5, f+6, f+7]]=m[1]
            M[f+2, [f ,f+1, f+2, f+3, f+4, f+5, f+6, f+7]]=m[2]           
            M[f+3, [f ,f+1, f+2, f+3, f+4, f+5, f+6, f+7]]=m[3]          
            bx[f], bx[f+1], bx[f+2], bx[f+3]=x[i], x[i+1], 0., 0.
            by[f], by[f+1], by[f+2], by[f+3]=y[i], y[i+1], 0., 0.
            bz[f], bz[f+1], bz[f+2], bz[f+3]=z[i], z[i+1], 0., 0.

        #the last spline passes trough the last 2 points
        f=4*(n-1)
        M[f,  [f ,f+1, f+2, f+3]]= m[0, :4]
        M[f+1,  [f ,f+1, f+2, f+3]]=m[1, :4]
        bx[f] , bx[f+1] =x[-2] , x[-1]
        by[f] , by[f+1] =y[-2] , y[-1]
        bz[f] , bz[f+1] =z[-2] , z[-1]

        #add the appropriate end constrains
        if self.closed:
            #first and second derivatives at starting and last point
            # (that are actually the same point) are the same
            M[f+2, [f+0 ,f+1,f+ 2, f+3]] = m[2, :4]
            M[f+2, [0 ,1, 2, 3]] = m[2, 4:]    
            M[f+3, [f+0 ,f+1,f+ 2,f+ 3]] = m[3, :4]
            M[f+3, [0 ,1, 2, 3]] = m[3, 4:]

        else:
            if self.endcond[0] =='notaknot':
                # third derivative is the same between the first 2 splines
                M[f+2,  [0, 4]] = array([6.,-6.])
            else:
                # second derivatives at start is zero
                M[f+2,  [0 ,1, 2, 3]] = m[3, :4]

            if self.endcond[1] =='notaknot':
                # third derivative is the same between the last 2 splines
                M[f+3,  [f-4, f]] = array([6.,-6.])
            else:
                # second derivatives at end is zero
                M[f+3,  [f ,f+1, f+2, f+3]] = m[3, :4]

        
        bx[f+2] , bx[f+3] ,  by[f+2],  by[f+3], bz[f+2],  bz[f+3] =0. ,  0., 0., 0., 0., 0.
        M , bx , by, bz  =asmatrix(M) ,  asmatrix(bx) ,  asmatrix(by) , asmatrix(bz)

        #calculate the coefficients
        B = column_stack([bx,by,bz])
        C = linalg.solve(M,B)
        self.coeffs = array(C).reshape(-1,4,3)


    def subPoints(self,t,j):
        """Compute the points at values t in subspline j"""
        C = self.coeffs[j]
        U = column_stack([t**3., t**2., t, ones_like(t)])
        X = dot(U,C)
        return X


    def points(self,ndiv=10,extend=[0., 0.]):
        """Return a series of points on the spline.

        The parameter space of each subspline is divided into ndiv parts.
        The coordinates of the points at these parameter values are returned
        as a Coords object.
        The extend parameter allows to extend the curve beyond the endpoints.
        The normal parameter space of each part is [0.0 .. 1.0]. The extend
        parameter will add a curve with parameter space [-extend[0] .. 0.0]
        for the first spline, and a curve with parameter space
        [1.0 .. 1 + extend[0]] for the last spline.
        Currently, the values are not exact, but rounded to a multiple of
        the normal parameter step (1./ndiv).
        If the NaturalSpline is closed, extend is disregarded. 
        """
        # Subspline parts do not include closing point
        u = arange(ndiv) / float(ndiv)
        parts = [ self.subPoints(u,j) for j in range(self.nsplines) ]

        if not self.closed:
            nstart,nend = ceil(asarray(extend)*ndiv).astype(Int)

            # Extend at start
            if nstart > 0:
                u = arange(-nstart, 0) * extend[0] / nstart
                parts.insert(0,self.subPoints(u,0))

            # Extend at start
            if nend > 0:
                u = 1. + arange(0, nend+1) * extend[1] / nend
            else:
                # Always extend at end to have last point
                u = array([1.])
            parts.append(self.subPoints(u,self.nsplines-1))

        X = concatenate(parts,axis=0)
        return Coords(X) 


# End
