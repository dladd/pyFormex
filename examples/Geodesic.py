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
wireframe()

m=n=5
data = askItems({'m':m,'n':n})
m = int(data['m'])
n = int(data['n'])

v=0.5*sqrt(3.)
a = Formex([[[0,0],[1,0],[0.5,v]]],1)
aa = Formex([[[1,0],[1.5,v],[0.5,v]]],2)
draw(a+aa)

#d = a.genid(m,n,1,v,0.5,-1)
#dd = aa.genid(m-1,n-1,1,v,0.5,-1)
d = a.replic2(m,min(m,n),1.,v,bias=0.5,taper=-1)
dd = aa.replic2(m-1,min(m-1,n),1.,v,bias=0.5,taper=-1)
clear()
draw(d+dd)

#e = (d+dd).rosad(m*0.5,m*v,6,60)
e = (d+dd).rosette(6,60,point=[m*0.5,m*v,0])
draw(e)

f = e.mapd(2,lambda d:0.8*sqrt((m+1)**2-d**2),e.center(),[0,1])
clear()
draw(f)

clear()
draw(f.shrink(0.85))

flat()
draw(f)

clear()
draw(f.shrink(0.85))

f.setProp(3)
clear()
smooth()
draw(f)

clear()
draw(f.shrink(0.85))
