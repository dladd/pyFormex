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
draw(T,view='bottom')
