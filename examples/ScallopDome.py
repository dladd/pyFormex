#!/usr/bin/env pyformex
##
## This file is part of pyformex 0.1.2 Release Fri Jul  9 14:48:57 2004
## pyformex is a python implementation of Formex algebra
## (c) 2004 Benedict Verhegghe (email: benedict.verhegghe@ugent.be)
## Releases can be found at ftp://mecatrix.ugent.be/pub/pyformex/
## Distributed under the General Public License, see file COPYING for details
##
#
global out
a=0; g=60; m=1; p=0;
f1 = Formex([[[10,0,0],[10,0,1]],[[10,0,1],[10,1,1]]]).genit(1,8,1,1,0,1) + Formex([[[10,0,1],[10,1,2]]]).genit(1,7,1,1,0,1)
f2 = Formex([[[15,0,1],[15,1,1]]]).genit(1,4,0,2,1,0)
out = f1+f2
draw(out)

