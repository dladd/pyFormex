#!/usr/bin/env python pyformex.py
# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

"""Cone

level = 'beginner'
topics = ['geometry','surface']
techniques = ['dialog', 'colors']

"""

import simple

smooth()
r=3.
h=15.
n=64

F = simple.sector(r,360.,n,n,h=h,diag=None)
F.setProp(0)
draw(F,view='bottom')
zoomall()
zoom(1.5)


ans = ask('How many balls do you want?',['3','2','1','0'])

try:
    nb = int(ans)
except:
    nb = 3
    
if nb > 0:
    B = simple.sphere3(n,n,r=0.9*r,bot=-90,top=90)
    B1 = B.translate([0.,0.,0.95*h])
    B1.setProp(1)
    draw(B1,bbox=None)
    zoomall()
    zoom(1.5)

if nb > 1:
    B2 = B.translate([0.2*r,0.,1.15*h])
    B2.setProp(2)
    draw(B2,bbox=None)
    zoomall()
    zoom(1.5)

if nb > 2:
    B3 = B.translate([-0.2*r,0.1*r,1.25*h])
    B3.setProp(6)
    draw(B3,bbox=None)
    zoomall()
    zoom(1.5)
