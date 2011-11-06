#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
# This example created (C) by Bart Desloovere (bart.desloovere@telenet.be)
#
"""Hyparcap

level = 'beginner'
topics = ['geometry']
techniques = ['color']

"""

clear()
wireframe()

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
top = T.replic2(a,a,2,3,bias=1,taper=-1).reflect(1,0,True).unique()
bot = B.replic2(a-1,a-1,2,3,bias=1,taper=-1).reflect(1,-1,True).unique()
web = W1.replic2(a-1,a-1,2,3,bias=1,taper=-1) + W2.replic2(a,a,2,-3,bias=1,taper=-1) + W3
blad = (top+bot+web).scale([1.,1./3,1.]).translate([0,a,0])
# herschalen
vlakblad = blad.scale([s*sin(radians(b/2))/a,s*cos(radians(b/2))/a,1.]).rotate(-45.)
# transleren en mappen op hyperbolische paraboloide (z=k1*x*y)
vlakblad2=vlakblad.translate([-c,-c,0])
j=vlakblad2.map(lambda x,y,z:[x,y,k1*x*y])
#overige bladen genereren
hyparcap=j.translate([c,c,0]).rosette(m,360/m,2,[0.,0.,0.])
draw(hyparcap)


