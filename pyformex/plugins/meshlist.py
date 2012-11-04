# $Id: geometry.py 2426 2012-09-09 13:31:06Z bverheg $  pyformex
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""MeshList

"""
from __future__ import print_function

from mesh import Mesh

class MeshList(list):
    """
    
    """
    def __init__(self,ML):
        # should check
        #self._list = ML
        list.__init__(self,ML)
        
    def _list_method(func):
        """Perform a method on all items in the list.

        This is a decorator function.
        """
        meth = getattr(Mesh,func.__name__)
        def newf(self,*args,**kargs):
            """Performs the Mesh %s transformation on each of the list items""" 
            return MeshList([ meth(i,*args,**kargs) for i in self ])
        newf.__name__ = func.__name__
        newf.__doc__ ="""Apply '%s' to all the Meshes in the list. 

        See :meth:`mesh.Mesh.%s` for details.
""" % (func.__name__,func.__name__)
        return newf

    ########### Return information about the coords #################

    def copy(self):
        """Return a deep copy of the object."""
        from copy import deepcopy
        return deepcopy(self)


    ########### Coords transformations #################
 
    @_list_method
    def scale(self,*args,**kargs):
        pass
    @_list_method
    def resized(self,*args,**kargs):
        pass
    @_list_method
    def translate(self,*args,**kargs):
        pass
    @_list_method
    def centered(self,*args,**kargs):
        pass
    @_list_method
    def align(self,*args,**kargs):
        pass
    @_list_method
    def rotate(self,*args,**kargs):
        pass
    @_list_method
    def shear(self,*args,**kargs):
        pass
    @_list_method
    def reflect(self,*args,**kargs):
        pass
    @_list_method
    def affine(self,*args,**kargs):
        pass
    @_list_method
    def position(self,*args,**kargs):
        pass
    @_list_method
    def cylindrical(self,*args,**kargs):
        pass
    @_list_method
    def hyperCylindrical(self,*args,**kargs):
        pass
    @_list_method
    def toCylindrical(self,*args,**kargs):
        pass
    @_list_method
    def spherical(self,*args,**kargs):
        pass
    @_list_method
    def superSpherical(self,*args,**kargs):
        pass
    @_list_method
    def egg(self,*args,**kargs):
        pass
    @_list_method
    def toSpherical(self,*args,**kargs):
        pass
    @_list_method
    def bump(self,*args,**kargs):
        pass
    @_list_method
    def bump1(self,*args,**kargs):
        pass
    @_list_method
    def bump2(self,*args,**kargs):
        pass
    @_list_method
    def flare(self,*args,**kargs):
        pass
    @_list_method
    def map(self,*args,**kargs):
        pass
    @_list_method
    def map1(self,*args,**kargs):
        pass
    @_list_method
    def mapd(self,*args,**kargs):
        pass

    @_list_method
    def replace(self,*args,**kargs):
        pass
    @_list_method
    def swapAxes(self,*args,**kargs):
        pass
    @_list_method
    def rollAxes(self,*args,**kargs):
        pass

    @_list_method
    def projectOnPlane(self,*args,**kargs):
        pass
    @_list_method
    def projectOnSphere(self,*args,**kargs):
        pass
    @_list_method
    def projectOnCylinder(self,*args,**kargs):
        pass
    @_list_method
    def isopar(self,*args,**kargs):
        pass
    @_list_method
    def transformCS(self,*args,**kargs):
        pass
    @_list_method
    def addNoise(self,*args,**kargs):
        pass

    rot = rotate
    trl = translate
    
    @_list_method
    def write(self,*args,**kargs):
         pass


if __name__ == "draw":

    from gui.draw import draw

    clear()
    M0 = Formex('4:0123').toMesh().setProp(1)
    M1 = M0.convert('tri3').trl([1.,0.,0.]).setProp(3)
    ML = MeshList([M0,M1])
    draw(ML.rotate(45))
    draw(ML[1].rotate(90),color=green)
    zoomAll()

# End
