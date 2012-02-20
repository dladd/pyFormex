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
"""Schwedler Dome

A Schwedler Dome is a dome-shaped frame structure in which the main
structural elements are organized along the longitude and latitude
circles of a sphere. Furthermore the construction usually has diagonal
stiffeners.

The example defines a function to create a Schwedler dome with hardwired
parameters.
This function is called creating a dome with diagonals (g).
The 'withProp method' is then used to create a copy without the diagonals (h).
Both are drawn at the same time, juxtaposed by the align function.
"""
_status = 'checked'
_level = 'normal'
_topics = ['geometry','domes']
_techniques = ['color','align','withprop']

from gui.draw import *


def schwedler():
    nx=16   # number of modules in circumferential direction
    ny=8    # number of modules in meridional direction
    rd=100  # radius of the sphere cap
    base=50 # slope of the dome at its base (= half angle of the sphere cap)
    top=5   # slope of the dome at its top opening (0 = no opening) 
    a = ny*float(top)/(base-top)
    e1 = Formex('l:54',[1,3]) # diagonals and meridionals
    e2 = Formex('l:1',0)      # horizontals
    f1 = e1.replic2(nx,ny,1,1)
    f2 = e2.replic2(nx,ny+1,1,1)
    g = (f1+f2).translate([0,a,1]).spherical(scale=[360./nx,base/(ny+a),rd],colat=True)
    return g


def run():
    clear()
    wireframe()
    g = schwedler()
    h = g.withProp([0,3]) # only horizontals and meridionals
    draw(align([g,h],'|0-'))

if __name__ == 'draw':
    run()
# End
