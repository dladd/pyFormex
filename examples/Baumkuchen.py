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
"""Baumkuchen Vault"""
clear()
m = 12 # number of cells in direction 0
n = 36 # number of cells in direction 1
a1 = Formex([[[0,0,0],[0,1,0]]]).rinid(m+1,n,1,1) + Formex([[[0,0,0],[1,0,0]]]).rinid(m,n+1,1,1)
p = a1.center()
p[2] = 24
f = lambda x:1-(x/18)**2/2
a2 = a1.bump(2,p,f,1)
p[2] = 4
a3 = a2.bump(2,p,lambda x:1-(x/6)**2/2,0)
out = a3.rin(1,5,12)
draw(out)
