#!/usr/bin/env pyformex
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
## Distributed under the General Public License, see file COPYING for details
##
#
ny=12; nz=8; rd=100; a=2; t=50;
e1 = Formex([[[1,0,a],[1,1,1+a]]])
e2 = Formex([[[1,0,a],[1,2,a]]])
f1 = e1.rosat(1,1+a,4,90).rinit(ny,nz,2,2).bs(rd,180/ny,t/(2*nz+a))
f2 = e2.rinit(ny,2,2,2*nz).bs(rd,180/ny,t/(2*nz+a))
out = f1+f2
draw(out)
