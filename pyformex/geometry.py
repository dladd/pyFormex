#!/usr/bin/env pyformex
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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
from formex import Formex


class Geometry(object):
    """A generic geometry object allowing transformation of coords sets.

    The Geometry class is a generic parent class for all geometric classes,
    intended to make the Coords transformations available without explicit
    declaration. This class is not intended to be used directly, only
    through derived classes.

    The derived classes should have an attribute `coords` which is
    a Coords object containing the coordinates of all the points that should
    get transformed under a Coords transformation.
    All the Coords transformation methods are inherited by the Geometry
    class and will be executed on the `coords` attribute.

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
    pyFormex, is is
    
    The default `setCoords` method is `setCoords_copy`.
    Most derived classes that are part of pyFormex however override this
    default and implement a more efficient cop
    Neither of both are good enough for pyFormex classes though
      
    the coords attribute in the object and returns the existing object::
        
        self.coords = coords
        return self

    While the user can stick with this default in his scripts,
    it is not in line with the default scripting language design of
    pyFormex, which is to not change an existing object inplace, but rather
    have the transformation return a changed object.
    * Therefore, derived classes that are part of pyFormex should always
    define their own implementation of setCoords(coords) *.
    
    
    A Geometry contains a single attribute, coords, which is a Coords object.
    """

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


    def copy(self):
        """Return a deep copy of the object."""
        from copy import deepcopy
        return deepcopy(self)
    

    def __str__(self):
        return self.coords.__str__()

 
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
    print "hallo"
    F = Formex(mpattern('123'))

    G = Geometry()
    G.coords = F.coords
    G.draw(color='red')

    G1 = G.scale(2).trl([0.2,0.7,0.])
    G1.draw(color='green')

    G2 = G.translate([0.5,0.5,0.5])
    G2.draw(color='blue')

    print Geometry.scale.__doc__
    print Coords.scale.__doc__
