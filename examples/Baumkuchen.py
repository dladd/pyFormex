#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.2 Release Mon Jan  3 14:54:38 2005
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Copyright (C) 2004 Benedict Verhegghe (benedict.verhegghe@ugent.be)
## Copyright (C) 2004 Bart Desloovere (bart.desloovere@telenet.be)
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
draw(out,'bottom')
