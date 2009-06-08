#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8 Release Mon Jun  8 11:56:55 2009
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
"""Geodesic Dome

level = 'normal'
topics = ['geometry','surface','domes']
techniques = ['dialog', 'colors']

"""

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
