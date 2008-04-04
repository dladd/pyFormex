#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7 Release Fri Apr  4 18:41:11 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Geodesic Dome"""
clear()
wireframe()

m=n=5
data = askItems({'m':m,'n':n})
if not data.has_key('m'):
    exit()

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
