#!/usr/bin/env pyformex --gui
# $Id: Sphere2.py 154 2006-11-03 19:08:25Z bverheg $
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Sphere2"""

from simple import sphere2,sphere3

reset()

nx = 4
ny = 4
m = 1.6
ns = 6

smooth()
setView('front')
for i in range(ns):
    b = sphere2(nx,ny,bot=-90,top=90).translate(0,-1.0)
    s = sphere3(nx,ny,bot=-90,top=90)
    s = s.translate(0,1.0)
    s.setProp(3)
    clear()
    bb = bbox([b,s])
    draw(b,bbox=bb,wait=False)
    draw(s,bbox=bb)#,color='random')
    nx = int(m*nx)
    ny = int(m*ny)

