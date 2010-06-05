#!/usr/bin/env pyformex
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

.. warning:: This is experimental stuff!
"""

from coords import Coords


class Geometry(object):
    """A generic geometry object allowing transformation of coords sets.

    The Geometry class is a generic parent class for all geometric classes,
    intended to make the Coords transformations available without explicit
    declaration. This class is not intended to be used directly, only
    through derived classes.

    There is no initialization to be done when constructing a new instance of
    this class. The class just defines a set of methods which operate on
    the attribute `coords` which is be a Coords object.
    Most of the transformation methods of the Coords class are thus exported
    through the Geometry class to its derived classes, and when called, will
    get executed on the `coords` attribute. 
    The derived class should make sure this attribute exists and contains
    the coordinates of all the points that should get transformed under a
    Coords transformation. 

    Derived classes can (and in most cases should) declare a method
    `setCoords(coords)` returning an object that is identical to the
    original, except for its coords being replaced by new ones with the
    same array shape.
    
    The Geometry class provides two possible default implementations:
    - `setCoords_inplace` sets the coords attribute to the provided new
      coords, thus changing the object itself, and returns itself,
    - `setCoords_copy` creates a deep copy of the object before setting
      the coords attribute. The original object is unchanged, the returned
      one is the changed copy.

    When using the first method, a statement like ```B = A.scale(0.5)```
    will result in both `A` and `B` pointing to the same scaled object,
    while with the second method, `A` would still be the untransformed
    object. Since the latter is in line with the design philosophy of
    pyFormex, it is set as the default `setCoords` method.
    Most derived classes that are part of pyFormex however override this
    default and implement a more efficient copy method.
    
    """

    ########### Return information about the coords #################

    def x(self):
        return self.coords.x()
    def y(self):
        return self.coords.y()
    def z(self):
        return self.coords.z()
    def bbox(self):
        return self.coords.bbox()
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

    def distanceFromPlane(self,*args,**kargs):
        return self.coords.distanceFromPlane(*args,**kargs)
    def distanceFromLine(self,*args,**kargs):
        return self.coords.distanceFromLine(*args,**kargs)
    def distanceFromPoint(self,*args,**kargs):
        return self.coords.distanceFromPoint(*args,**kargs)

    def __str__(self):
        return self.coords.__str__()

    ########### Return a copy #################

    def copy(self):
        """Return a deep copy of the object."""
        from copy import deepcopy
        return deepcopy(self)
    

    ########### Change the coords #################

    def _coords_transform(func):
        """Perform a transformation on the .coords attribute of the object

        """
        coords_func = getattr(Coords,func.__name__)
        def newf(self,*args,**kargs):
            """Performs the Coords %s transformation on the coords attribute""" 
            return self.setCoords(coords_func(self.coords,*args,**kargs))
        newf.__name__ = func.__name__
        newf.__doc__ = coords_func.__doc__
        return newf


    def setCoords_inplace(self,coords):
        """Replace the current coords with new ones.

        """
        if isinstance(coords,Coords) and coords.shape == self.coords.shape:
            self.coords = coords
            return self
        else:
            raise ValueError,"Invalid reinitialization of Geometry coords"


    def setCoords_copy(self,coords):
        """Return a copy of the object with new coordinates replacing the old.

        """
        return self.copy().setCoords_inplace(coords)

    setCoords = setCoords_copy

 
    @_coords_transform
    def scale(self,*args,**kargs):
        pass
    @_coords_transform
    def translate(self,*args,**kargs):
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
    def newmap(self,*args,**kargs):
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
    def projectOnSphere(self,*args,**kargs):
        pass
    @_coords_transform
    def projectOnCylinder(self,*args,**kargs):
        pass

    rot = rotate
    trl = translate



if __name__ == "draw":

    from gui.draw import draw

    def draw(self,*args,**kargs):
        draw(self.coords,*args,**kargs)


    clear()
    F = Formex(mpattern('123'))

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
