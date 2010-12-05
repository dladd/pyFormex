#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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
"""HorseTorse

level = 'advanced'
topics = ['geometry','surface','mesh']
techniques = ['intersection']

"""
from plugins.trisurface import TriSurface
from plugins.mesh import Mesh

reset()
smooth()
lights(True)

S = TriSurface.read(getcfg('datadir')+'/horse.off')
SA = draw(S)

res = askItems([
    ('direction',[1.,0.,0.]),
    ('number of sections',20),
    ('color','red'),
    ]) 
if not res:
    exit()

d = res['direction']
n = res['number of sections']
c = res['color']

slices = S.slice(dir=d,nplanes=n)
linewidth(2)
draw(slices,color=c,view=None,bbox='last',flat=True)


#undraw(SA)
zoomAll()
