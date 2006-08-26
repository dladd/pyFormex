#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
"""Novation"""
clear()
n = 40

baseGeom = ask("Create a surface model with",
                   ['Triangles','Quadrilaterals'])
if baseGeom == 0:
    # The base are two triangles
    e = Formex([[[0,0,0],[1,0,0],[0,1,0]],[[1,0,0],[1,1,0],[0,1,0]]],1).rinid(n,n,1,1)
else:
    # The base is one quadrilateral
    e = Formex([[[0,0,0],[1,0,0],[1,1,0],[0,1,0]]],1).rinid(n,n,1,1)

# These are lines forming quadrilaterals
#e = Formex([[[0,0,0],[1,0,0]]]).rosad(.5,.5).rinid(n,n,1,1)

# Novation (Spots)
m = 4
h = 12
r = n/m
s = n/r
a = [ [r*i,r*j,h]  for j in range(1,s) for i in range(1,s) ]

#create a bottom
b = e.reverseElements()

for p in a:
    e = e.bump(2,p, lambda x:exp(-0.5*x),[0,1])

draw (e,color=red)
#draw (b,color=blue)


if baseGeom == 0 and ack('Export to .stl?'):
    import stl
    f = file('novation.stl','w')
    F = e # + b
    # Create triangles
    G = F.selectNodes([0,1,2])
    # If polygones, add more triangles
    for i in range(3,F.nnodel()):
        G += F.selectNodes([0,i-1,i])
    stl.write_ascii(G.f,f)
    f.close()
