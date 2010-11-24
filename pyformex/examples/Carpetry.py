#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
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

"""Carpetry

level = 'normal'
topics = ['mesh','illustration','surface']
techniques = ['color','random']

.. Description

Carpetry
--------
This example illustrates the use of the Mesh conversion techniques and the
creation of colored value plots on surfaces.

"""
from plugins import mesh,trisurface,surface_menu

def atExit():
    pf.GUI.setBusy(False)



def drawMesh(M):
    clear()
    draw(M)
    drawText("%s %s elements" % (M.nelems(),M.eltype),20,20,size=20)

pf.GUI.setBusy()
    
clear()
view('front')
smoothwire()
transparent()


nx,ny = 4,2
M = Formex(origin()).extrude(nx,1.,0).extrude(ny,1.,1).toMesh().setProp(1)


conversions = []
#drawMesh(M)

maxconv = 10
minconv = 5
minelems = 10000
maxelems = 100000

V = surface_menu.SelectableStatsValues
possible_keys = [ k for k in V.keys() if not V[k][1] ][:-1]
print possible_keys
nkeys = len(possible_keys)

nconv = random.randint(minconv,maxconv)

while (len(conversions) < nconv and M.nelems() < maxelems) or (M.nelems() < minelems):
    possible_conversions = mesh._conversions_[M.eltype].keys()
    i = random.randint(len(possible_conversions))
    conv = possible_conversions[i]
    conversions.append(conv)
    M = M.convert(conv)
    #drawMesh(M)
    
if M.eltype != 'tri3':
    M = M.convert('tri3')
print "%s patches" % M.nelems()
print "conversions: %s" % conversions
key = possible_keys[random.randint(nkeys)]
print "colored by %s" % key
S = trisurface.TriSurface(M)
export({'surface':S})
surface_menu.selection.set(['surface'])
surface_menu.showStatistics(key=key)
pf.canvas.removeDecorations()
       
# q4 - q9 - t3-d - q4 - t3-x
# smallest altitude
