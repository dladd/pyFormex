#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
"""Geodesic Dome"""
clear()
m=5; n=5; v=0.5*sqrt(3.)
a = Formex([[[0,0],[1,0],[0.5,v]]],1)
aa = Formex([[[1,0],[1.5,v],[0.5,v]]],2)
draw(a+aa)

d = a.genid(m,n,1,v,0.5,-1)
dd = aa.genid(m-1,n-1,1,v,0.5,-1)
clear()
draw(d+dd)

e = (d+dd).rosad(m*0.5,n*v,6,60)
clear()
draw(e)

f = e.mapd(2,lambda d:sqrt(5.1**2-d**2),e.center(),[0,1])
clear()
draw(f)
