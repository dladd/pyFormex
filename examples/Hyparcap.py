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
"""Hyparcap"""
clear()
from math import *
a = 3 # verdeelparameter
x = -((1-sqrt(5))/2) # gulden getal
s = 30. # overspanning
m = 5; b = 360./m # pentacap
k1 = 0.035 # steilte
hoek = (90.-b)/2
d = 2. # laagdikte
c = (x*s+k1*s*s/2*sin(radians(2*hoek)))/(k1*s*cos(radians(hoek))+k1*s*sin(radians(hoek))) # pentacapvoorwaarde
# compret van 1 blad
T = Formex([[[-a,0,d],[-a+2,0,d]],[[-a,0,d],[1-a,3,d]],[[1-a,3,d],[2-a,0,d]]])
B = Formex([[[1-a,-1,0],[3-a,-1,0]],[[1-a,-1,0],[2-a,2,0]],[[2-a,2,0],[3-a,-1,0]]])
W1 = Formex([[[2-a,2,0],[1-a,3,d]],[[2-a,2,0],[3-a,3,d]],[[2-a,2,0],[2-a,0,d]]])
W2 = Formex([[[1-a,-1,0],[-a,0,d]],[[1-a,-1,0],[2-a,0,d]],[[1-a,-1,0],[1-a,-3,d]]])
W3 = Formex([[[0,3*a,d],[0,3*(a-1)-1,0]]])
top = T.genid(a,a,2,3,1,-1).lam(2,0).unique()
bot = B.genid(a-1,a-1,2,3,1,-1).lam(2,-1).unique()
web = W1.genid(a-1,a-1,2,3,1,-1) + W2.genid(a,a,2,-3,1,-1) + W3
blad = (top+bot+web).scale([1.,1./3,1.]).tranid(0,a)
vlakblad = blad.scale([s*sin(radians(b/2))/a,s*cos(radians(b/2)),1.])

nod = blad.nodes().unique()
nod.setProp(4)
print nod
draw(blad)
drawProp(nod)

