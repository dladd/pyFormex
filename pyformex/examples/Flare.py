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
"""Torus

level = 'beginner'
topics = ['geometry']
techniques = ['dialog','transform','function']
"""

def addFlares(F,dir=[0,2]):
    """Adds flares at both ends of the structure.

    The flare parameters are hardcoded, a real-life example would
    make them adjustable.
    Returns the flared structure.
    """
    F = F.flare(m/4.,-1.,dir,0,0.5)
    F = F.flare(m/4.,1.5,dir,1,2.)
    return F
    
view('iso')
data = [
    ('m',36,{'text':'number of cells in longest grid direction'}),
    ('n',12,{'text':'number of cells in shortes grid direction'}),
    ('f0',True,{'text':'add flares on rectangle'}),
    ('f1',False,{'text':'add flares on cylinder'}),
    ('f2',False,{'text':'add flares on torus'}),
    ('geom','cylinder','radio',['rectangle','cylinder','torus'],{'text':'geometry'}),
    ]
res = askItems(data)
if not res:
    exit()
globals().update(res)
F = Formex(mpattern("12-34"),[1,3]).replic2(m,n,1,1)
if f0:
    F = addFlares(F)

if geom != 'rectangle':
    F = F.translate(2,1).cylindrical([2,1,0],[1.,360./n,1.])
    if f1:
        F = addFlares(F,dir=[2,0])
    if geom == 'torus':
        F = F.translate(0,5).cylindrical([0,2,1],[1.,360./m,1.])
        if f2:
            F = addFlares(F)

clear()
draw(F)

# End
