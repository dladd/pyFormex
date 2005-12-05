#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.2.1 Release Fri Apr  8 23:30:39 2005
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where otherwise stated 
##
#
"""Baumkuchen Vault"""
clear()
m = 12 # number of cells in direction 0
n = 36 # number of cells in direction 1
k = 7  # number of vaults in direction 0
e1 = 30 # elevation of the major arcs
e2= 5  # elevation of the minor arcs
# Using beam elements
a1 = Formex(pattern('2')).replic2(m+1,n,1,1,0,1) + \
     Formex(pattern('1')).replic2(m,n+1,1,1,0,1)
draw(a1,'front')
p = a1.center()
p[2] = e1
f = lambda x:1-(x/18)**2/2
a2 = a1.bump(2,p,f,1)
draw(a2,'bottom')
p[2] = e2
a3 = a2.bump(2,p,lambda x:1-(x/6)**2/2,0)
draw(a3,'bottom')
a4 = a3.replic(k,m,0)
draw(a4,'bottom')
clear()
draw(a4,'bottom')
