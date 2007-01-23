#!/usr/bin/env pyformex
# $Id: Sphere2.py 154 2006-11-03 19:08:25Z bverheg $
#
"""Sphere2"""

from simple import Sphere2,Sphere3

nx = 4
ny = 4
m = 1.6
ns = 6
smooth()
setView('front')
for i in range(ns):
    b = Sphere2(nx,ny,bot=-90,top=90).translate(0,-1.0)
    s = Sphere3(nx,ny,bot=-90,top=90).translate(0,1.0)
    s.setProp(3)
    clear()
    bb = bbox([b,s])
    draw(b,bbox=bb,wait=False)
    draw(s,bbox=bb)
    nx = int(m*nx)
    ny = int(m*ny)
