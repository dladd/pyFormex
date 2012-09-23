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
"""HorseTorse

This example illustrates working with triangulated surfaces in pyFormex.
A triangulated surface (TriSurface) is read from the file 'horse.off'
in the pyformex data directory. The surface is drawn.

Next the surface is intersected with a series of parallel plan. A dialog
dialog pops up where the user can set the parameters to define this planes.
Finally, the intersection curves are drawn on the surface. The 'ontop' option
will draw the curves fully visible (like if the surface were transparent).
The 'remove surface' option removes the surface, leaving only the curves.
"""
_status = 'checked'
_level = 'advanced'
_topics = ['geometry','surface','mesh']
_techniques = ['intersection','dialog']

from gui.draw import *
from plugins.trisurface import TriSurface

def run():
    reset()
    smooth()
    lights(True)

    S = TriSurface.read(getcfg('datadir')+'/horse.off')
    SA = draw(S)

    res = askItems([
        ('direction',[1.,0.,0.]),
        ('number of sections',20),
        ('color','red'),
        ('ontop',False),
        ('remove surface',False),
        ]) 
    if not res:
        return

    d = res['direction']
    n = res['number of sections']
    c = res['color']

    slices = S.slice(dir=d,nplanes=n)
    linewidth(2)
    draw(slices,color=c,view=None,bbox='last',nolight=True,ontop=res['ontop'])

    if res['remove surface']:
        undraw(SA)
        
    zoomAll()

if __name__ == 'draw':
    run()
# End
