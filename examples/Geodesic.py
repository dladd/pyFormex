#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
## Distributed under the General Public License, see file COPYING for details
##
#
"""Geodesic Dome"""
clear()
m=5; n=5; v=0.5*sqrt(3.)
a = Formex([[[0,0,0],[1,0,0],[0.5,v,0]]])
aa = Formex([[[1,0,0],[1,0,0],[0.5,v,0]]])
draw(a+aa)
sleep(3)
d = a.genid(m,n,1,v,0.5,-1)
dd = aa.genid(m-1,n-1,1,v,0.5,-1)
clear()
draw(d+dd)
sleep(3)
e = (d+dd).rosad(m*0.5,n*v,6,60)
clear()
draw(e)
sleep(3)
f = e.mapd(2,lambda d:sqrt(5.1**2-d**2),e.center(),[0,1])
clear()
draw(f)
