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

"""MeshSmoothing

This example illustrates the use of the mesh smoothing algorithm.

The smoothing is applied to a hexahedral, tetrahedral, quadrilateral,
and triangular mesh.
"""
_status = 'checked'
_level = 'normal'
_topics = ['geometry', 'mesh','illustration']
_techniques = ['dialog','smooth']

from gui.draw import *
from simple import cuboid


def run():
    clear()
    n = 12      #   Number of elements in each direction
    tol = 4     #   Amount of noise added to the coordinates
    iter = 10   #   Number of smoothing iterations 

    res = askItems(items=[
        _I('n', n, text='Number of elements', itemtype='slider', min=2, max=24),
        _I('tol', tol, text='Noise', itemtype='slider', min=0, max=10),
        _I('iter', iter, text='Smoothing iterations', itemtype='slider', min=1, max=20),  
    ])

    if not res:
        return
    
    globals().update(res)

    tol /= 10.
    cube = cuboid().replic2(n, n, 1., 1.).rep(n, 1., 2).toMesh()
    cubeTet = cuboid().toMesh().convert('tet4')
    cubeTet += cubeTet.reflect()            #Reflecting is needed to align the edges of adjacent tetrahedrons
    cubeTet += cubeTet.reflect(dir=1)
    cubeTet += cubeTet.reflect(dir=2)
    cubeTet = cubeTet.trl([1., 1., 1.])
    cubeTet = cubeTet.toFormex().replic2(n/2, n/2, 2., 2.).rep(n/2., 2., 2).toMesh().convert('tet4').trl(1, -2.*n)
    surf = Formex(xpattern('0123', nplex=4)).replic2(2*n, 2*n, 1., 1.).scale(0.5).bump(2, [n/2., n/2., 1.], lambda x: 1.-x**2/n).toMesh().trl(1, -4.*n)
    surfTri = surf.convert('tri3').trl(1, -2.*n)

    for part in [cube, cubeTet, surf, surfTri]:
        noise = tol * random.random(part.coords.shape) - tol/2.
        noisy = Mesh(part.coords + noise, part.elems, 1).trl(0, 2.*n)
        smoothed = noisy.smooth(iterations=iter).trl(0, 2.*n)
        smoothed.setProp(2)
        draw([part, noisy, smoothed])

    zoomAll()

if __name__ == 'draw':
    run()
# End
