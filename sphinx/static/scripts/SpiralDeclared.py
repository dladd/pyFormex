#!/usr/bin/env pyformex
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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
"""Spiral"""

#This is a small function that is only defined in this script. It clears the 
#screen and draws the Formex at the same time. 
def drawit(F,view='front'):
    clear()
    draw(F,view)

#These are the parameters. They can easily be changed, and a whole new 
#spiral will be created without any extra effort.
m = 36 # number of cells along torus big circle
n = 10 # number of cells along torus small circle

#The first step is to create a basic Formex. In this case, it's a triangle which 
#has a different property number for every edge.    
F = Formex(pattern("164"),[1,2,3]); drawit(F)

#This basic Formex is copied 'm' times in the 0-direction with a translation 
#step of '1' (the length of an edge of the triangle). After that, the new 
#Formex is copied 'n' times in the 1-direction with a translation step of '1'. 
#Because of the recursive definition (F=F.replic), the original Formex F is 
#overwritten by the transformed one.
F = F.replic(m,1,0); drawit(F)
F = F.replic(n,1,1); drawit(F)

#Now a copy of this last Formex is translated in direction '2' with a 
#translation step of '1'. This necessary for the transformation into a cilinder.
#The result of all previous steps is a rectangular pattern with the desired 
#dimensions, in a plane z=1.
F = F.translate1(2,1); drawit(F,'iso')

#This pattern is rolled up into a cilinder around the 2-axis. 
F = F.cylindrical([2,1,0],[1.,360./n,1.]); drawit(F,'iso')

#This cilinder is copied 5 times in the 2-direction with a translation step of 
#'m' (the lenght of the cilinder). 
F = F.replic(5,m,2); drawit(F,'iso')

#The next step is to rotate this cilinder -10 degrees around the 0-axis. 
#This will determine the pitch angle of the spiral.
F = F.rotate(-10,0); drawit(F,'iso')

#A copy of this last formex is now translated in direction '0' with a 
#translation step of '5'. 
F = F.translate1(0,5); drawit(F,'iso')

#Finally, the Formex is rolled up, but around a different axis then before. 
#Due to the pitch angle, a spiral is created. If the pitch angle would be 0 
#(no rotation of -10 degrees around the 0-axis), the resulting Formex 
#would be a torus. 
F = F.cylindrical([0,2,1],[1.,360./m,1.]); drawit(F,'iso')
drawit(F,'right')
