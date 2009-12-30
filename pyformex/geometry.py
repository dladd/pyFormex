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

from pyformex.coords import *

def coords_transformation(func):
    """Perform a transformation on the .coords attribute of the object

    """
    repl = getattr(Coords,func.__name__)
    def newf(self,*args,**kargs):
        """Performs the Coords %s transformation on the coords attribute""" 
        self.coords = repl(self.coords,*args,**kargs)
        return self
    newf.__name__ = func.__name__
    newf.__doc__ = repl.__doc__
    return newf


class Geometry(object):

    def transform(self,funcname,*args,**kargs):
        """Transform the coordinates of the object using the function
        func(*args,**kargs), where func is a transformation method of the
        Coords class.
        """
        func = getattr(self.coords,funcname)
        self.coords = func(self.coords,*args,**kargs)

    def __str__(self):
        return self.coords.__str__()

 
    @coords_transformation
    def scale(self,*args,**kargs):
        pass
    @coords_transformation
    def translate(self,*args,**kargs):
        pass

    def draw(self,color='red'):
        print self.coords
        draw(Formex(self.coords),color=color)


if __name__ == "draw":

    F = Formex(mpattern('123'))

    G = Geometry()
    G.coords = F.f
    G.draw()

    G1 = G.scale(2)
    G1.draw(color='green')

    G2 = G.translate([0.5,0.5,0.5])
    G2.draw(color='blue')

    print Geometry.scale.__doc__
    print Coords.scale.__doc__
