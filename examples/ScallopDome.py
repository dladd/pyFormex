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
"""Scallop Dome"""
clear()
a=0; g=60; m=1; p=0;
f1 = Formex([[[0,0,0],[0,1,0]],[[0,1,0],[1,1,0]]]).genid(1,8,1,1,0,1) + Formex([[[0,1,0],[1,2,0]]]).genid(1,7,1,1,0,1)
f1 = f1.remove(Formex([[[0,1,0],[1,1,0]]]).genid(1,4,0,2,1,0))
draw(f1)
glisid=lambda x,y,z:[where(y>0,x*sqrt(4*y*y-x*x)/(y+y),x),where(y>0,y-x*x/(y+y),0),0]
f2 = f1.map(glisid)
f2.setProp(1)
drawProp(f2)
f3 = f2.mapd(2,lambda d: -d*d/40, [0,0,0],[0,1])
f3.setProp(3)
f3 = f3.rosette(6,2,[0,0,0],60)
drawProp(f3)

