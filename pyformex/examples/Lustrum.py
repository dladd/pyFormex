#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
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
"""Lustrum

level = 'normal'
topics = ['curve','drawing','illustration']
techniques = ['color','persistence','lima','import']

"""
reset()
from pyformex.examples.Lima import *
from project import Project
linewidth(2)
fgcolor(blue)
grow('Plant1',ngen=7,clearing=False,text=False)
data = readGeomFile(os.path.join(pf.cfg['datadir'],'blippo.pgf'))
curve = data['blippo_000']
bb = curve.coords.bbox()
ctr = bb.center()
siz = bb.sizes()
curve.coords = curve.coords.trl(0,-ctr[0]).scale(50./siz[0])
draw(curve,color=pyformex_pink,linewidth=5)

# End
