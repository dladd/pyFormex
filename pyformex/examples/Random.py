#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Random

level = 'beginner'
topics = ['surface']
techniques = ['color']

Creates random points, bars, triangles, quads, ...
"""

reset()
setDrawOptions(dict(clear=True))
npoints = 30
p = arange(120)
P = Formex(random.random((npoints,1,3)),p)
draw(P,alpha=0.5)
for n in range(2,5):
    F = connect([P for i in range(n)],bias=[i*(n-1) for i in range(n)],loop=True)
    F.setProp(p)
    draw(F,alpha=0.5)

# End
