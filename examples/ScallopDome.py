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
pc=2; # pijl in het centrum van de koepel
pr=4; # pijl aan de rand
a=0; g=60; m=1; p=0;
f1 = Formex([[[0,0,0],[0,1,0]],[[0,1,0],[1,1,0]]]).genid(1,8,1,1,0,1) + Formex([[[0,1,0],[1,2,0]]]).genid(1,7,1,1,0,1)
f1 = f1.remove(Formex([[[0,1,0],[1,1,0]]]).genid(1,4,0,2,1,0))
glisid=lambda x,y,z:[where(y>0,x*sqrt(4*y*y-x*x)/(y+y),x),where(y>0,y-x*x/(y+y),0),0]
f2 = f1.map(glisid)
f1.setProp(1)
f2.setProp(3)
drawProp(f1+f2)
sleep()
func1 = lambda x,y,z: [x,y,pc*(1.-x*x/64.)+pr*x*x/64.*4*(1.-y)*y]
func2 = lambda x,y,z: [x,y,pc*(1.-x*x/64.)+pr*x*x/64.*4*pow((1.-y)*y,2)]
func3 = lambda x,y,z: [x,y,pc*(1.-x*x/64.)+pr*x*x/64.*4*pow((1.-y)*y,2)]
def show(n,f,c,r):
    global pc,pr
    pc = c
    pr = r
    a=360./n
    f3 = f2.toCylindrical([1,0,2]).scale([1.,1./60.,1.])
    f4 = f3.map(f).cylindrical([0,1,2],[1.,a,1.]).rosette(n,2,[0,0,0],a)
    clear()
    drawProp(f4,0)
    sleep()
for r in [-2,0,2,4,6]:
    show(6,func1,2,r)
for c in [0,2,4]:
    show(6,func1,c,2)
for r in [-4,0,4,8]:
    show(6,func2,4,r)
for r in [-2,0,2]:
    show(12,func1,2,r)
