# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
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
"""Helix

This is basically the same example as Torus.py, but it constructs the
resulting geometry step by step, making it suited for a first time
initiation.
"""
from __future__ import print_function
_status = 'checked'
_level = 'beginner'
_topics = ['geometry']
_techniques = ['color']

from gui.draw import *

def run():
    m = 36 # number of cells along helix
    n = 10 # number of cells along circular cross section
    reset()
    setDrawOptions({'clear':True})
    bgcolor(white)
    F = Formex('l:164',[1,2,3]); draw(F)
    F = F.replic(m,1.,0); draw(F)
    F = F.replic(n,1.,1); draw(F)
    F = F.translate(2,1.); draw(F,view='iso')
    F = F.cylindrical([2,1,0],[1.,360./n,1.]); draw(F)
    F = F.replic(5,m*1.,2); draw(F)
    F = F.rotate(-10.,0); draw(F)
    F = F.translate(0,5.); draw(F)
    F = F.cylindrical([0,2,1],[1.,360./m,1.]); draw(F)
    draw(F,view='right')

if __name__ == 'draw':
    run()
# End
