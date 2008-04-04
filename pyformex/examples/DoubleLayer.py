#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7 Release Fri Apr  4 18:41:11 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""DoubleLayer"""

Formex.setPrintFunction(Formex.asFormexWithProp)
clear()
n=10; a=2./3.; d=1./n;
e1 = Formex([[[0,0,d],[2,0,d]],[[2,0,d],[1,1,d]],[[1,1,d],[0,0,d]]],prop=1)
e2 = Formex([[[0,0,d],[1,1-a,0]],[[2,0,d],[1,1-a,0]],[[1,1,d],[1,1-a,0]]],prop=3)
# top and bottom layers
e4 = e1.genid(n,n,2,1,1,-1).bb(1./(2*n),1./(2*n)/tand(30))
e5 = e1.genid(n-1,n-1,2,1,1,-1).translate([1,1-a,-d]).bb(1./(2*n),1./(2*n)/tand(30))
# diagonals
e6 = e2.genid(n,n,2,1,1,-1).bb(1./(2*n),1./(2*n)/tand(30))
e5.setProp(2)
out = (e4+e5+e6).tran(3,-d)
message("The structure has %s nodes and %s elements" % (out.nnodes(),out.nelems()))
draw(out)
