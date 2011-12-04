#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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
"""Geodesic Dome

level = 'beginner'
topics = ['geometry','domes']
techniques = ['dialog', 'color']

"""

clear()
view('front')

m=n=6
f=0.8
res = askItems([('m',m),('n',n),('f',f)])
if not res:
    exit()

m = res['m']
n = res['n']
f = res['f']

v=0.5*sqrt(3.)
a = Formex([[[0,0],[1,0],[0.5,v]]],1)
aa = Formex([[[1,0],[1.5,v],[0.5,v]]],2)
draw(a+aa)
#pause()

d = a.replic2(m,min(m,n),1.,v,bias=0.5,taper=-1)
dd = aa.replic2(m-1,min(m-1,n),1.,v,bias=0.5,taper=-1)
clear()
draw(d+dd)
#pause()

e = (d+dd).rosette(6,60,point=[m*0.5,m*v,0])
draw(e)
#pause()

g = e.mapd(2,lambda d:f*sqrt((m+1)**2-d**2),e.center(),[0,1])
clear()
draw(g)

# End
