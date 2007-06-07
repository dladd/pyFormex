#!/usr/bin/env python pyformex.py
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
"""DoubleLayer"""
from math import *
Formex.setPrintFunction(Formex.asFormexWithProp)
clear()
n=10; a=2./3.; d=1./n;
e1 = Formex([[[0,0,d],[2,0,d]],[[2,0,d],[1,1,d]],[[1,1,d],[0,0,d]]],prop=1)
e2 = Formex([[[0,0,d],[1,1-a,0]],[[2,0,d],[1,1-a,0]],[[1,1,d],[1,1-a,0]]],prop=3)
# top and bottom layers
e4 = e1.genid(n,n,2,1,1,-1).bb(1./(2*n),1./(2*n)/tan(radians(30)))
e5 = e1.genid(n-1,n-1,2,1,1,-1).translate([1,1-a,-d]).bb(1./(2*n),1./(2*n)/tan(radians(30)))
# diagonals
e6 = e2.genid(n,n,2,1,1,-1).bb(1./(2*n),1./(2*n)/tan(radians(30)))
e5.setProp(2)
out = (e4+e5+e6).tran(3,-d)
message("The structure has %s nodes and %s elements" % (out.nnodes(),out.nelems()))
draw(out)
