# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
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

"""WedgeHex

This example illustrates the creation of geometry by a revolution around
an axis and the automatic reduction of the resulting degenerate elements
to lower plexitude.

First a 2D rectangular quad mesh is created. It is then revolved around an axis
cutting the rectangle. The result is a fan shaped volume of hexahedrons,
of which some elements are degenerate (those touching the axis). The
splitDegenerate method is then used to split the mesh in nondegenerat meshes
of Wedge6 (magenta) and Hex8 (cyan) type.
"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['mesh']
_techniques = ['revolve','degenerate'] 

from gui.draw import *
import simple

delay(1)

def run():
    clear()
    smoothwire()
    view('iso')

    # create a 2D xy mesh
    nx,ny = 6,2
    G = simple.rectangle(1,1,1.,1.).replic2(nx,ny)
    M = G.toMesh()
    draw(M, color='red')

    # create a 3D axial-symmetric mesh by REVOLVING
    n,a = 8,45.
    R = M.revolve(n,angle=a,axis=1,around=[1.,0.,0.])
    draw(R,color='yellow')

    # reduce the degenerate elements to WEDGE6
    clear()
    ML = R.fuse().splitDegenerate()
    # keep only the non-empty meshes
    ML = [ m for m in ML if m.nelems() > 0 ]
    print("After splitting: %s meshes:" % len(ML))
    for m in ML:
        print("  %s elements of type %s" % (m.nelems(),m.eltype))
    ML = [ Mi.setProp(i+4) for i,Mi in enumerate(ML) ]
    draw(ML)

if __name__ == 'draw':
    run()
# End
