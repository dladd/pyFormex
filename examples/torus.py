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
"""Torus"""
clear()
F1 = Formex([[[0,0],[1,0]]],1)
F2 = Formex([[[0,0],[0,1]]],3)
F3 = Formex([[[0,1],[1,0]]],2)
F = (F1+F2+F3).replicate(36,0,1).replicate(12,1,1)
drawProp(F)
sleep()
G = F.translate1(2,1).cylindrical([2,1,0],[1.,30.,1.])
clear()
drawProp(G)
sleep()
H = G.translate1(0,5).cylindrical([0,2,1],[1.,10.,1.])
clear()
drawProp(H)
