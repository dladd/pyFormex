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
    declaration.
    A Geometry contains a single attribute, coords, which is a Coords object.
    All the Coords transformation methods are inherited by the Geometry
    class.
    """

    def _coords_transform(func):
        """Perform a transformation on the .coords attribute of the object

        """
        coords_func = getattr(Coords,func.__name__)
        def newf(self,*args,**kargs):
            """Performs the Coords %s transformation on the coords attribute""" 
            self.setCoords(coords_func(self.coords,*args,**kargs))
            return self
        newf.__name__ = func.__name__
        newf.__doc__ = coords_func.__doc__
        return newf

    def setCoords(self,coords):
        """Replace the current coords with new ones.

        The default implementation imposes the restriction that the
        new coordinate array should have the same shape. It also overwrites
        the coords of the current object. Derived classes can change this
        behavior, but should nake sure to keep the data consistent.
        The new coords structure should have the same
        """
        if isinstance(coords,Coords) and coords.shape == self.coords.shape:
            self.coords = coords
        else:
            raise ValueError,"Invalid reinitialization of Geometry coords"
        

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
