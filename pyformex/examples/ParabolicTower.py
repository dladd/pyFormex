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
"""Parabolic Tower

level = 'beginner'
topics = ['geometry']
techniques = ['color']

"""

clear()
global a,b,c,d  # constants in lambda need to be made global
h = 25.   # height of tower
h1 = 18.  # height at neck of tower
r = 10.   # radius at base of tower
r1 = 5.   # radius at neck of tower
m = 10    # number of sides at the base   
n = 8     # number of levels
a = (r-r1)/h1**2; b = -2*a*h1; c = r; d = h/n
g = lambda i: a*(d*i)**2 + b*d*i + c
f = concatenate([  [[[g(i),i,i], [g(i+1),i-1,i+1]],
             [[g(i),i,i], [g(i+1),i+1,i+1]],
             [[g(i+1),i-1,i+1], [g(i+1),i+1,i+1]]] for i in range(n) ])
F = Formex(f,[3,0,1]).replic(m,2,dir=1)
T = F.cylindrical(scale=[1,360./(2*m),d])
draw(T,view='bottom')
