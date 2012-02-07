# $Id$ *** pyformex app ***
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
"""Barrel Vault

Create a parametric barrel vault.

level = 'beginner'
topics = ['frame']
techniques = ['dialog']

Description
===========
This is a pyFormex application. It is equivalent to the Barrelvault example
script. An application requires a little different syntax though:

- Everything should be imported explicitely.
- The outer code only initializes the application and is executed only when
  the application is loaded.
- If the application defines a 'run' method, it can be executed from the GUI.
"""
from gui.draw import *

def run():
    reset()
    wireframe()

    res = askItems([
        dict(name='m',value=10,text='number of modules in axial direction'),
        dict(name='n',value=8,text='number of modules in tangential direction'),
        dict(name='r',value=10.,text='barrel radius'),
        dict(name='a',value=180.,text='barrel opening angle'),
        dict(name='l',value=30.,text='barrel length'),
        ])
    if not res:
        return

    globals().update(res)

    # Diagonals
    d = Formex('l:5',1).rosette(4,90).translate([1,1,0]).replic2(m,n,2,2)
    # Longitudinals
    h = Formex('l:1',3).replic2(2*m,2*n+1,1,1)
    # End bars
    e = Formex('l:2',0).replic2(2,2*n,2*m,1)
    # Create barrel
    barrel = (d+h+e).rotate(90,1).translate(0,r).scale([1.,a/(2*n),l/(2*m)]).cylindrical()

    draw(barrel)


# End
