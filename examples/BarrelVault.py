#!/usr/bin/env pyformex
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
## Distributed under the General Public License, see file COPYING for details
##
#
"""Barrel Vault"""
clear()
m=10; n=10; r=10
e1 = Formex([[[r,0,0],[r,1,1]]])
e2 = e1.lamic(2,3,1,1).rinic(2,3,m,n,2,2)
e3 = Formex([[[r,0,0],[r,1,0]]]).rin(2,2*m,1).lam(3,n)
e4 = Formex([[[r,0,0],[r,0,2]]]).rinic(2,3,2*m+1,n,1,2)
ee = e1+e2+e3+e4
out = ee.bc(1,90/m,1)
draw(out)


