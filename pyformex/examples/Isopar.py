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
"""Isopar

level = 'normal'
topics = ['geometry']
techniques = ['dialog', 'color','isopar']

"""
from gui.draw import *

from plugins import isopar
import simple
import elements

def run():
    wireframe()

    ttype = ask("Select type of transformation",['Cancel','1D','2D','3D'])
    if not ttype or ttype ==  'Cancel':
        exit()

    tdim = int(ttype[0])

    # create a unit quadratic grid in tdim dimensions
    x = Coords(simple.regularGrid([0.]*tdim, [1.]*tdim, [2]*tdim)).reshape(-1,3)
    x1 = Formex(x)
    x2 = x1.copy()

    # move a few points
    if tdim == 1:
        eltype = 'line3'
        x2[1] = x2[1].rot(-22.5)
        x2[2] = x2[2].rot(22.5)
    elif tdim == 2:
        eltype = 'quad9'
        x2[5] = x2[2].rot(-22.5)
        x2[8] = x2[2].rot(-45.)
        x2[7] = x2[2].rot(-67.5)
        x2[4] = x2[8] * 0.6
    else:
        eltype = 'hex27'
        tol = 0.01
        d = x2.distanceFromPoint(x2[0])
        w = where((d > 0.5+tol) * (d < 1.0 - tol))[0]
        # avoid error messages during projection 
        errh = seterr(all='ignore')
        x2[w] = x2.projectOnSphere(0.5)[w]
        w = where(d > 1.+tol)[0]
        x2[w] = x2.projectOnSphere(1.)[w]
        seterr(**errh)

    clear()
    message('This is the set of nodes in natural coordinates')
    draw(x1,color=blue)
    message('This is the set of nodes in cartesian coordinates')
    draw(x2,color=red)
    drawNumbers(x2,color=red)
    drawNumbers(x1)

    n = 8
    stype = ask("Select type of structure",['Cancel','1D','2D','3D'])
    if stype == 'Cancel':
        exit()

    sdim = int(stype[0])
    if sdim == 1:
        F = simple.line([0.,0.,0.],[1.,1.,0.],10)
    elif sdim == 2:
        F = simple.rectangle(1,1,1.,1.)
    else:
        ## v = array(elements.Hex8.vertices)
        ## f = array(elements.Hex8.faces[1])
        ## F = Formex(v[f])
        F = elements.Hex8.toFormex()

    if sdim > 1:
        for i in range(sdim):
            F = F.replic(n,1.,dir=i)

    if sdim < tdim:
        F = F.trl(2,0.5)
    clear()
    message('This is the initial Formex')
    FA=draw(F)
    sz = F.sizes()


    if sdim < tdim:
        sz[sdim:tdim] = 2.
    x1 = x1.scale(sz)
    x2 = x2.scale(sz)

    G=F.isopar(eltype,x2.points(),x1.points())
    G.setProp(1)

    message('This is the transformed Formex')
    draw(G)

    pause()
    undraw(FA)

if __name__ == 'draw':
    run()
# End
