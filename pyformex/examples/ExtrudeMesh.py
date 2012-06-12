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

"""ExtrudeMesh

This example illustrates Mesh extrusion using quadratic elements.

First, a Mesh is created consisting of a single point.
The point is extruded in the x-direction, resulting in a line.
The line is further extrude in y-direction to yield a quadratic surface.
A final extrusion in the z-direction delivers a quadratic volume.
"""
_status = 'checked'
_level = 'beginner'
_topics = ['mesh']
_techniques = ['extrude', 'quadratic']

from gui.draw import *

def run():
    clear()

    nx,ny,nz = 5,3,2
    #nx,ny,nz = 1,1,1
    degree = 2           # create quadratic extrusions, change to 1 for linear
    serendipity = False
    show3Dbyborder = False
    noise = 0.0          # set nonzero to add some noise to the coordinates 
    sleep = 2

    smoothwire()
    view('iso')
    delay(0)

    a = Formex([0.,0.,0.]).toMesh()   # a point at the origin
    print a.eltype
    draw(a,color='black')

    delay(sleep)

    b = a.extrude(nx,1.,0,degree=degree)  # point extruded to quadratic line 
    print b.eltype
    draw(b.coords,wait=False)
    draw(b,color='red')

    c = b.extrude(ny,1.,1,degree=degree)  # line extruded to quadratic surface
    if serendipity:
        c = c.convert('quad8')#.compact()
    print c.eltype
    draw(c.coords,wait=False)
    draw(c,color='blue')

    #c1 = c.trl(2,1.)
    #d = c.connect(c1,degree=2)

    #d = d.convert('hex20')
    d = c.extrude(nz,1.,2,degree=degree)  # surface extruded to quadratic volume
    d = d.compact()
    print d.eltype
    #d = d.reverse()
    if show3Dbyborder:
        d = d.getBorderMesh()
    print "Shown as %s" % d.eltype
    clear()
    draw(d.coords,wait=False)
    #drawNumbers(d.coords)
    print d.elems
    draw(d,color='yellow',bkcolor='black')

    if noise:
        e = d.addNoise(noise)
        draw(e.coords,wait=False,clear=True)
        draw(e,color=cyan)
    
if __name__ == 'draw':
    run()
# End
