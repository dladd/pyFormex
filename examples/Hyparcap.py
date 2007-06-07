#!/usr/bin/env python pyformex.py
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
# This example created (C) by Bart Desloovere (bart.desloovere@telenet.be)
#
"""Hyparcap"""
clear()
from math import *
a = 5 # verdeelparameter
x = -((1-sqrt(5))/2) # gulden getal
s = 30. # overspanning
m = 5; b = 360./m # pentacap (script nog vervolledigen zodat m andere waarden kan aannemen)
k1 = 0.035 # steilte
hoek = (90.-b)/2
d = 2. # laagdikte
c = (x*s+k1*s*s/2*sin(radians(2*hoek)))/(k1*s*cos(radians(hoek))+k1*s*sin(radians(hoek))) # pentacapvoorwaarde

# compret van 1 blad
T = Formex([[[-a,0,d],[-a+2,0,d]],[[-a,0,d],[1-a,3,d]],[[1-a,3,d],[2-a,0,d]]],1)
B = Formex([[[1-a,-1,0],[3-a,-1,0]],[[1-a,-1,0],[2-a,2,0]],[[2-a,2,0],[3-a,-1,0]]],2)
W1 = Formex([[[2-a,2,0],[1-a,3,d]],[[2-a,2,0],[3-a,3,d]],[[2-a,2,0],[2-a,0,d]]])
W2 = Formex([[[1-a,-1,0],[-a,0,d]],[[1-a,-1,0],[2-a,0,d]],[[1-a,-1,0],[1-a,-3,d]]])
W3 = Formex([[[0,3*a,d],[0,3*(a-1)-1,0]]])
top = T.genid(a,a,2,3,1,-1).lam(2,0).unique()
bot = B.genid(a-1,a-1,2,3,1,-1).lam(2,-1).unique()
web = W1.genid(a-1,a-1,2,3,1,-1) + W2.genid(a,a,2,-3,1,-1) + W3
blad = (top+bot+web).scale([1.,1./3,1.]).tranid(0,a)
# herschalen
vlakblad = blad.scale([s*sin(radians(b/2))/a,s*cos(radians(b/2))/a,1.]).rotate(-45.)
# transleren en mappen op hyperbolische paraboloide (z=k1*x*y)
vlakblad2=vlakblad.translate([-c,-c,0])
j=vlakblad2.map(lambda x,y,z:[x,y,k1*x*y])
#overige bladen genereren
hyparcap=j.translate([c,c,0]).rosette(m,360/m,2,[0.,0.,0.])
draw(hyparcap)


