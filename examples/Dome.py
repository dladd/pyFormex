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
"""Dome"""
clear()
ny=12; nz=8; rd=100; a=2; t=50;
e1 = Formex([[[1,0,a],[1,1,1+a]]])
e2 = Formex([[[1,0,a],[1,2,a]]])
f1 = e1.rosat(1,1+a,4,90).rinit(ny,nz,2,2).bs(rd,180/ny,t/(2*nz+a))
f2 = e2.rinit(ny,2,2,2*nz).bs(rd,180/ny,t/(2*nz+a))
draw(f1+f2)

