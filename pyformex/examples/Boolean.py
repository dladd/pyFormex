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
"""Boolean

Perform boolean operations on surfaces
"""
_status = 'checked'
_level = 'normal'
_topics = ['surface']
_techniques = ['boolean','partition']

from gui.draw import *
from simple import cylinder
from connectivity import connectedLineElems


def run():
    global F,G
    clear()
    smooth()
    F = cylinder(L=10.,D=2.,nt=36,nl=20,diag='u').centered()
    draw(F)
    F = TriSurface(F).setProp(3).close(method='planar').fixNormals()
    G = F.rotate(90.,0).trl(0,1.).setProp(1)
    export({'F':F,'G':G})
    draw([F,G])
    return

    I = F.boolean(G,'*')
    clear()
    draw(I)
    I.setProp(I.partitionByAngle())
    clear()
    draw(I)
    S = I.splitProp()
    K,L = S[0], S[1] # there may be some cruft in the border area
    clear()
    draw([K,L.trl(0,0.1*L.dsize())])

    b = K.getBorderMesh().setProp(1)
    clear()
    draw(b)
    drawNumbers(b)
    drawNumbers(b.coords,color=red)

    # fuse
    b = b.fuse().compact()
    clear()
    draw(b)
    drawNumbers(b)
    drawNumbers(b.coords,color=red)

    # chain the segments to continuous curve
    b = Mesh(b.coords,connectedLineElems(b.elems)[0])
    clear()
    draw(b)
    drawNumbers(b)
    drawNumbers(b.coords,color=red)

    c = b.toFormex().toCurve()
    print c.getAngles()
    return

    # remove cruft
    b = b.notConnectedTo(30)
    clear()
    draw(b)
    drawNumbers(b)
    drawNumbers(b.coords,color=red)
    return

    b = b.fuse().compact()
    draw(b)
    drawNumbers(b.coords)
    print b.nelems()
    return

if __name__ == 'draw':
    run()
# End
