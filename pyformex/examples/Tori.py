# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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
#
"""Torus variants

"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['geometry']
_techniques = ['programming','widgets','globals']

from gui.draw import *

def torus(m,n,surface=True):
    """Create a torus with m cells along big circle and n cells along small."""
    if surface:
        C = Formex([[[0,0,0],[1,0,0],[0,1,0]],[[1,0,0],[1,1,0],[0,1,0]]],[1,3])
    else:
        C = Formex('l:164',[1,2,3])
    F = C.replic2(m,n,1,1)
    G = F.translate(2,1).cylindrical([2,1,0],[1.,360./n,1.])
    H = G.translate(0,5).cylindrical([0,2,1],[1.,360./m,1.])
    return H


def series():
    view='iso'
    for n in [3,4,6,8,12]:
        for m in [3,4,6,12,36]:
            clear()
            draw(torus(m,n),view)
            view=None

def drawTorus(m,n):
    clear()
    print(m,n)
    draw(torus(m,n),None)

def nice():
    drawTorus(72,36)


def run():
    global m,n
    m = 20
    n = 10
    while not dialogTimedOut():
        res = askItems([
            _I('m',m,itemtype='slider',text='Number of elements along large circle',min=3,max=72),
            _I('n',n,itemtype='slider',text='Number of elements along small circle',min=3,max=36)
            ])
        if not res:
            break

        globals().update(res)
        drawTorus(m,n)


if __name__ == 'draw':
    run()
# End
