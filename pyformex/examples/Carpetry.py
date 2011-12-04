#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 15:52:41 CET 2011)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
from plugins import trisurface,surface_menu
from elements import *

def atExit():
    pf.cfg['gui/autozoomfactor'] = saved_autozoomfactor
    pf.GUI.setBusy(False)


def drawMesh(M):
    clear()
    draw(M)
    drawText("%s %s elements" % (M.nelems(),M.eltype),20,20,size=20)

# make sure this is a good aspect ratio if you want a movie
nx,ny = 4,3

saved_autozoomfactor = pf.cfg['gui/autozoomfactor']

pf.GUI.setBusy()
pf.cfg['gui/autozoomfactor'] = 2.0

clear()
view('front')
smoothwire()
transparent()

M = Formex(origin()).extrude(nx,1.,0).extrude(ny,1.,1).toMesh().setProp(1)

V = surface_menu.SelectableStatsValues
possible_keys = [ k for k in V.keys() if not V[k][1] ][:-1]
nkeys = len(possible_keys)

maxconv = 10
minconv = 5
minelems = 10000
maxelems = 50000

save = False

def carpet(M):
    conversions = []
    nconv = random.randint(minconv,maxconv)

    while (len(conversions) < nconv and M.nelems() < maxelems) or M.nelems() < minelems:
        possible_conversions = M.eltype.conversions.keys()
        i = random.randint(len(possible_conversions))
        conv = possible_conversions[i]
        conversions.append(conv)
        #clear()
        #draw(M)
        #print "%s -> %s" % (M.eltype,conv)
        M = M.convert(conv)
        #print "type %s, plex %s" % (M.eltype,M.nplex())

    if M.eltype != Tri3:
        M = M.convert('tri3')
        conversions.append('tri3')

    print "%s patches" % M.nelems()
    print "conversions: %s" % conversions

    # Coloring
    key = possible_keys[random.randint(nkeys)]
    print "colored by %s" % key
    func = V[key][0]
    S = trisurface.TriSurface(M)
    val = func(S)
    export({'surface':S})
    surface_menu.selection.set(['surface'])
    surface_menu.showSurfaceValue(S,str(conversions),val,False)
    pf.canvas.removeDecorations()
    

clear()
flatwire()
lights(True)
transparent(False)

if pf.interactive:
    canvasSize(nx*200,ny*200)
    #canvasSize(720,576)
    print "running interactively"
    n = 1#ask("How many?",['0','1000','100','10','1'])
    n = int(n)
    save = False#ack("Save images?")
    if save:
        image.save(filename='Carpetry-000.jpg',window=False,multi=True,hotkey=False,autosave=False,border=False,rootcrop=False,format=None,quality=95,verbose=False)

    A = None
    for i in range(n):
        carpet(M)
        B = pf.canvas.actors[-1:]
        if A:
            undraw(A)
        A = B
        if save:
            image.saveNext()

else:
    import sys
    print sys.argv
    print argv
    canvasSize(nx*200,ny*200)
    print "just saving image"
    from gui import image,guimain
    carpet(M)
    image.save('testje2.png')
    #exit(all=True)
    guimain.quitGUI()


## print "ATEXIT"
## from PyQt4 import QtGui
## print "current:%s" % pf.GUI.viewports.current.size()
## print "max:%s" % pf.GUI.viewports.current.maximumSize()
## pf.GUI.viewports.current.setMaximumSize(1000,2000)
## pf.GUI.central.setSizePolicy(QtGui.QSizePolicy.Maximum,QtGui.QSizePolicy.Maximum)
## pf.GUI.central.resize(pf.GUI.central.size().width()+0,pf.GUI.central.size().height()+0)
## pf.GUI.viewports.activate()
## pf.GUI.resize(pf.GUI.size().width()+0,pf.GUI.size().height()+0)
## pf.GUI.update()

# End
