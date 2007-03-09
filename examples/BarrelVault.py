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
print """Barrel Vault"""
clear()
m=10; n=10; r=10
e1 = Formex([[[r,0,0],[r,1,1]]],1).lamic(2,3,1,1).rinic(2,3,m,n,2,2)
e2 = Formex([[[r,0,0],[r,1,0]]]).rin(2,2*m,1).lam(3,n)
e3 = Formex([[[r,0,0],[r,0,2]]],3).rinic(2,3,2*m+1,n,1,2)
ee = (e1+e2+e3).bc(1,90/m,1)
draw(ee)
