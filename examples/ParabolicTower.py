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
"""Parabolic Tower"""
clear()
global a,b,c,d  # constants in lambda need to be made global
h = 25.   # height of tower
h1 = 18.  # height at neck of tower
r = 10.   # radius at base of tower
r1 = 5.   # radius at neck of tower
m = 10    # number of sides at the base   
n = 8     # number of levels
a = (r-r1)/h1**2; b = -2*a*h1; c = r; d = h/n
g = lambda i: a*(d*i)**2 + b*d*i + c
f = concatenate([  [[[g(i),i,i], [g(i+1),i-1,i+1]],
             [[g(i),i,i], [g(i+1),i+1,i+1]],
             [[g(i+1),i-1,i+1], [g(i+1),i+1,i+1]]] for i in range(n) ])
F = Formex(f,[3,0,1]).rin(2,m,2)
T = F.bc(1,360./(2*m),d)
drawProp(T,side="bottom")
