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

"""MeshSmoothing

This example illustrates the use of the mesh smoothing algorithm.

Three versions of a mesh are shown: the original regular mesh (in black),
the mesh after some noise has been added (in red), the noised mesh after
smoothing (in blue).

The user can choose the element type (quadrilateral, triangular, hexahedral
or tetrahedral), the number of elements in the regular grid, the amount of
noise to be added, and the number of smoothing iterations
"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['geometry','mesh']
_techniques = ['dialog','smooth','noise','convert']

from gui.draw import *

eltype = 'quad4'
n = 6            # Number of elements in each direction (should be even)
noise = 0.05     # Amount of noise added to the coordinates
niter = 5        # Number of smoothing iterations 


def createMesh(eltype,n):
    """Create a mesh of the given type with n cells in each direction.

    eltype should be one of 'quad4','tri3','hex8','tet4'.
    """
    if eltype == 'tet4':   # Tet conversions produces many elements, reduce n
        n /= 2
    M = Formex('4:0123').rep([n,n]).toMesh()
    if eltype == 'tri3':
        M = M.convert('tri3')
    elif eltype in ['hex8','tet4']:
        M = M.extrude(n,dir=2).convert(eltype)
    return M


def noiseSmooth(M,noise,niter):
    """Draw 3 versions of a mesh: original, with noise, smoothed noise

    M is any mesh. A version with added noise is created. Then that version
    is smoothed. The three versions are displayed.
    """
    draw(M)
    M1 = M.addNoise(noise).trl(0,M.dsize()).setProp(1)
    draw(M1)
    M2 = M1.smooth(niter).trl(0,M.dsize()).setProp(3)
    draw([M,M1,M2])

    
def run():
    clear()

    res = askItems(items=[
        _I('eltype',eltype,text='Element type',itemtype='radio',choices=['quad4','tri3','hex8','tet4']),
        _I('n',n,text='Grid size',itemtype='slider',min=2,max=24),
        _I('noise',noise,text='Noise',itemtype='fslider',min=0,max=100,scale=0.01),
        _I('niter',niter,text='Smoothing iterations',itemtype='slider',min=1,max=20),
    ])

    if res:
        globals().update(res)
        M = createMesh(eltype,n)
        noiseSmooth(M,noise,niter)


if __name__ == 'draw':
    run()
# End
