#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""Barrel Vault

level = 'beginner'
topics = ['geometry']
techniques = ['dialog']

"""

reset()

res = askItems([('number of modules in axial direction',10),
                ('number of modules in tangential direction',8),
                ('barrel radius',10.),
                ('barrel opening angle',180.),
                ('barrel length',30.),
                ],
               )
if not res:
    exit()
    
m = res['number of modules in axial direction']
n = res['number of modules in tangential direction']
r = res['barrel radius']
a = res['barrel opening angle']
l = res['barrel length']

# Diagonals
d = Formex(pattern("5"),1).rosette(4,90).translate([1,1,0]).replic2(m,n,2,2)

# Longitudinals
h = Formex(pattern("1"),3).replic2(2*m,2*n+1,1,1)

# End bars
e = Formex(pattern("2"),0).replic2(2,2*n,2*m,1)

# Create barrel
barrel = (d+h+e).rotate(90,1).translate(0,r).scale([1.,a/(2*n),l/(2*m)]).cylindrical()

draw(barrel)
