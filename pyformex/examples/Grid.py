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
"""Grid

level = 'beginner'
topics = ['geometry']
techniques = ['dialog']

"""

import gui.actors


def base(type,m,n=None):
    """A regular pattern for type.

    type = 'tri' or 'quad' or 'triline' or 'quadline'
    m = number of cells in direction 0
    n = number of cells in direction 1
    """
    n = n or m
    if type == 'triline':
        return Formex('l:164').replic2(m,n,1,1,0,1,0,-1)
    elif type == 'quadline':
        return Formex('l:2').replic2(m+1,n,1,1) + \
               Formex('l:1').replic2(m,n+1,1,1)
    elif type == 'tri':
        return Formex('m:12-34').replic2(m,n)
    elif type == 'quad':
        return Formex('m:123').replic2(m,n)
    else:
        raise ValueError,"Unknown type '%s'" % str(type)

# This is the old input format, but relies on auto-conversion 
res = askItems([('nx',4),('ny',3),('nz',2),('Grid type','','select',{'choices':['Box','Plane']}),('alpha',0.3)])

if not res:
    exit()
    
nx = (res['nx'],res['ny'],res['nz'])
gridtype = res['Grid type']
alpha = res['alpha']

if gridtype == 'Box':
    GA = actors.GridActor(nx=nx,linewidth=0.2,alpha=alpha)
else:
    GA = actors.CoordPlaneActor(nx=nx,linewidth=0.2,alpha=alpha)

smooth()
pf.canvas.addActor(GA)
pf.canvas.setBbox(GA.bbox())
zoomAll()
pf.canvas.update()

