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
global out
n=10; a=2./3.; d=1./n;
e1 = Formex([[[0,0,d],[2,0,d]],[[2,0,d],[1,1,d]],[[1,1,d],[0,0,d]]])
e2 = Formex([[[0,0,d],[1,1-a,0]],[[2,0,d],[1,1-a,0]],[[1,1,d],[1,1-a,0]]])
e4 = e1.genid(n,n,2,1,1,-1).bb(1./(2*n),1./(2*n)/tan(radians(30)))
e5 = e1.genid(n-1,n-1,2,1,1,-1).translate([1,1-a,-d]).bb(1./(2*n),1./(2*n)/tan(radians(30)))
e6 = e2.genid(n,n,2,1,1,-1).bb(1./(2*n),1./(2*n)/tan(radians(30)))
out = (e4+e5+e6).tran(3,-d)
draw(out)
