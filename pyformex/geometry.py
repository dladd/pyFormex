# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
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
"""A generic interface to the Coords transformation methods

This module defines a generic Geometry superclass which adds all the
possibilities of coordinate transformations offered by the
Coords class to the derived classes.
"""
from __future__ import print_function

from coords import Coords


class Geometry(object):
    """A generic geometry object allowing transformation of coords sets.

    The Geometry class is a generic parent class for all geometric classes,
    intended to make the Coords transformations available without explicit
    declaration. This class is not intended to be used directly, only
    through derived classes. Examples of derived classes are :class:`Formex`,
    :class:`Mesh`, :class:`Curve`.

    There is no initialization to be done when constructing a new instance of
    this class. The class just defines a set of methods which operate on
    the attribute `coords`, which should be a Coords object.
    Most of the transformation methods of the Coords class are thus exported
    through the Geometry class to its derived classes, and when called, will
    get executed on the `coords` attribute. 
    The derived class constructor should make sure that the `coords` attribute
    exists, has the proper type and contains the coordinates of all the points
    that should get transformed under a Coords transformation. 

    Derived classes can (and in most cases should) declare a method
    `_set_coords(coords)` returning an object that is identical to the
    original, except for its coords being replaced by new ones with the
    same array shape.
    
    The Geometry class provides two possible default implementations:
    
    - `_set_coords_inplace` sets the coords attribute to the provided new
      coords, thus changing the object itself, and returns itself,
    - `_set_coords_copy` creates a deep copy of the object before setting
      the coords attribute. The original object is unchanged, the returned
      one is the changed copy.

    When using the first method, a statement like ``B = A.scale(0.5)``
    will result in both `A` and `B` pointing to the same scaled object,
    while with the second method, `A` would still be the untransformed
    object. Since the latter is in line with the design philosophy of
    pyFormex, it is set as the default `_set_coords` method.
    Most derived classes that are part of pyFormex however override this
    default and implement a more efficient copy method.

    The following :class:`Geometry` methods return the value of the same
    method applied on the `coords` attribute. Refer to the correponding
    :class:`coords.Coords` method for their precise arguments.

    :meth:`x`,
    :meth:`y`,
    :meth:`z`,
    :meth:`bbox`,
    :meth:`center`,
    :meth:`centroid`,
    :meth:`sizes`,
    :meth:`dsize`,
    :meth:`bsphere`,
    :meth:`inertia`,
    :meth:`distanceFromPlane`,
    :meth:`distanceFromLine`,
    :meth:`distanceFromPoint`,
    :meth:`directionalSize`,
    :meth:`directionalWidth`,
    :meth:`directionalExtremes`,
    :meth:`__str__`.
    

    The following :class:`Coords` transformation methods can be directly applied
    to a :class:`Geometry` object or a derived class object. The return value
    is a new object identical to the original, except for the coordinates,
    which will have been transformed by the specified method.
    Refer to the correponding :class:`coords.Coords` method in for the precise
    arguments of these methods:
    
    :meth:`scale`,
    :meth:`translate`,
    :meth:`centered`,
    :meth:`rotate`,
    :meth:`shear`,
    :meth:`reflect`,
    :meth:`affine`,
    :meth:`position`,
    :meth:`cylindrical`,
    :meth:`hyperCylindrical`,
    :meth:`toCylindrical`,
    :meth:`spherical`,
    :meth:`superSpherical`,
    :meth:`toSpherical`,
    :meth:`bump`,
    :meth:`bump1`,
    :meth:`bump2`,
    :meth:`flare`,
    :meth:`map`,
    :meth:`map1`,
    :meth:`mapd`,
    :meth:`replace`,
    :meth:`swapAxes`,
    :meth:`rollAxes`,
    :meth:`projectOnPlane`,
    :meth:`projectOnSphere`,
    :meth:`projectOnCylinder`,
    :meth:`isopar`,
    :meth:`transformCS`,
    :meth:`addNoise`,
    :meth:`rot`,
    :meth:`trl`.
    """
    
    ########### Change the coords #################

    def _coords_transform(func):
        """Perform a transformation on the .coords attribute of the object.

        This is a decorator function.
        """
        coords_func = getattr(Coords,func.__name__)
        def newf(self,*args,**kargs):
            """Performs the Coords %s transformation on the coords attribute""" 
            return self._set_coords(coords_func(self.coords,*args,**kargs))
        newf.__name__ = func.__name__
        newf.__doc__ ="""Apply '%s' transformation to the Geometry object. 

        See :meth:`coords.Coords.%s` for details.
""" % (func.__name__,func.__name__)
        return newf


    def _set_coords_inplace(self,coords):
        """Replace the current coords with new ones.

        """
        if isinstance(coords,Coords) and coords.shape == self.coords.shape:
            self.coords = coords
            return self
        else:
            raise ValueError,"Invalid reinitialization of Geometry coords"


    def _set_coords_copy(self,coords):
        """Return a copy of the object with new coordinates replacing the old.

        """
        return self.copy()._set_coords_inplace(coords)

    _set_coords = _set_coords_copy

    ########### Return information about the coords #################

    def getCoords(self):
        """Get the coords data.

        Returns the full array of coordinates stored in the Geometry object.
        Note that subclasses may store more points in this array than are used
        to define the geometry.
        """
        return self.coords
    def x(self):
        return self.coords.x()
    def y(self):
        return self.coords.y()
    def z(self):
        return self.coords.z()
    def bbox(self):
        return self.coords.bbox()
    def apt(self,*args,**kargs):
        return self.coords.apt(*args,**kargs)
    def center(self):
        return self.coords.center()
    def centroid(self):
        return self.coords.centroid()
    def sizes(self):
        return self.coords.sizes()
    def dsize(self):
        return self.coords.dsize()
    def bsphere(self):
        return self.coords.bsphere()
    def inertia(self,*args,**kargs):
        return self.coords.inertia(*args,**kargs)

    def info(self):
        return "Geometry: coords shape = %s; level = %s" % (self.coords.shape,self.level())
    def level(self):
        """Return the dimensionality of the Geometry, or -1 if unknown"""
        return -1

    def distanceFromPlane(self,*args,**kargs):
        return self.coords.distanceFromPlane(*args,**kargs)
    def distanceFromLine(self,*args,**kargs):
        return self.coords.distanceFromLine(*args,**kargs)
    def distanceFromPoint(self,*args,**kargs):
        return self.coords.distanceFromPoint(*args,**kargs)
    def directionalSize(self,*args,**kargs):
        return self.coords.directionalSize(*args,**kargs)
    def directionalWidth(self,*args,**kargs):
        return self.coords.directionalWidth(*args,**kargs)
    def directionalExtremes(self,*args,**kargs):
        return self.coords.directionalExtremes(*args,**kargs)

    def __str__(self):
        return self.coords.__str__()

    ########### Return a copy #################

    def copy(self):
        """Return a deep copy of the object."""
        from copy import deepcopy
        return deepcopy(self)


    ########### Coords transformations #################
 
    @_coords_transform
    def scale(self,*args,**kargs):
        pass


    def resized(self,size=1.,tol=1.e-5):
        """Return a copy of the Geometry scaled to the given size.

        size can be a single value or a list of three values for the
        three coordinate directions. If it is a single value, all directions
        are scaled to the same size.
        Directions for which the geometry has a size smaller than tol times
        the maximum size are not rescaled.
        """
        from numpy import resize
        s = self.sizes()
        size = Coords(resize(size,(3,)))
        ignore = s<tol*s.max()
        s[ignore] = size[ignore]
        return self.scale(size/s)


    @_coords_transform
    def translate(self,*args,**kargs):
        pass
    @_coords_transform
    def centered(self,*args,**kargs):
        pass
    @_coords_transform
    def align(self,*args,**kargs):
        pass
    @_coords_transform
    def rotate(self,*args,**kargs):
        pass
    @_coords_transform
    def shear(self,*args,**kargs):
        pass
    @_coords_transform
    def reflect(self,*args,**kargs):
        pass
    @_coords_transform
    def affine(self,*args,**kargs):
        pass
    @_coords_transform
    def position(self,*args,**kargs):
        pass


    @_coords_transform
    def cylindrical(self,*args,**kargs):
        pass
    @_coords_transform
    def hyperCylindrical(self,*args,**kargs):
        pass
    @_coords_transform
    def toCylindrical(self,*args,**kargs):
        pass
    @_coords_transform
    def spherical(self,*args,**kargs):
        pass
    @_coords_transform
    def superSpherical(self,*args,**kargs):
        pass
    @_coords_transform
    def egg(self,*args,**kargs):
        pass
    @_coords_transform
    def toSpherical(self,*args,**kargs):
        pass
    @_coords_transform

    def bump(self,*args,**kargs):
        pass
    @_coords_transform
    def bump1(self,*args,**kargs):
        pass
    @_coords_transform
    def bump2(self,*args,**kargs):
        pass
    @_coords_transform
    def flare(self,*args,**kargs):
        pass
    @_coords_transform
    def map(self,*args,**kargs):
        pass
    @_coords_transform
    def map1(self,*args,**kargs):
        pass
    @_coords_transform
    def mapd(self,*args,**kargs):
        pass

    @_coords_transform
    def replace(self,*args,**kargs):
        pass
    @_coords_transform
    def swapAxes(self,*args,**kargs):
        pass
    @_coords_transform
    def rollAxes(self,*args,**kargs):
        pass

    @_coords_transform
    def projectOnPlane(self,*args,**kargs):
        pass
    @_coords_transform
    def projectOnSphere(self,*args,**kargs):
        pass
    @_coords_transform
    def projectOnCylinder(self,*args,**kargs):
        pass
    @_coords_transform
    def isopar(self,*args,**kargs):
        pass
    @_coords_transform
    def transformCS(self,*args,**kargs):
        pass
    @_coords_transform
    def addNoise(self,*args,**kargs):
        pass

    rot = rotate
    trl = translate

    
    def write(self,fil,sep=' ',mode='w'):
        """Write a Geometry to a .pgf file.

        If fil is a string, a file with that name is opened. Else fil should
        be an open file.
        The Geometry is then written to that file in a native format, using
        sep as separator between the coordinates.
        If fil is a string, the file is closed prior to returning.
        """
        from geomfile import GeometryFile
        f = GeometryFile(fil,mode='w',sep=sep)
        f.write(self)
        if f.isname and mode[0]=='w':
            f.close()



if __name__ == "draw":

    from gui.draw import draw

    def draw(self,*args,**kargs):
        draw(self.coords,*args,**kargs)


    clear()
    F = Formex('4:0123')

    G = Geometry()
    G.coords = F.coords
    G.draw(color='red')

    G1 = G.scale(2).trl([0.2,0.7,0.])
    G1.draw(color='green')

    G2 = G.translate([0.5,0.5,0.5])
    G2.draw(color='blue')

    print(Geometry.scale.__doc__)
    print(Coords.scale.__doc__)

# End
