#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Barrel Vault"""

reset()

res = askItems([('number of modules in axial direction',10),
                ('number of modules in tangential direction',8),
                ('barrel radius',10.),
                ('barrel opening angle',180.),
                ('barrel length',30.),
                ])
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
