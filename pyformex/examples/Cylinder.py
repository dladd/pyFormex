# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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

"""Cylinder

level = 'beginner'
topics = ['geometry', 'surface', 'cylinder']
techniques = ['import']

.. Description

Cylinder
--------
This example illustrates the use of simple.sector() and simple.cylinder()
to create a parametric cylindrical surface.

"""
import simple
from plugins.trisurface import TriSurface

n=12
h=5.
A = simple.sector(1.,360.,1,n,diag='u')
B = simple.cylinder(2.,h,n,4,diag='u').reverse()
C = A.reverse()+B+A.trl(2,h)
S = TriSurface(C)
export({'surface':S})

smoothwire()
view('iso')
draw(S,color=red)

# End
