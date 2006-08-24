#!/usr/bin/env python
# $Id$
"""Functions from the Camera menu."""

import globaldata as GD
         

def zoomIn():
    global canvas
    GD.canvas.zoom(1./GD.cfg.gui['zoomfactor'])
    GD.canvas.update()
def zoomOut():
    global canvas
    GD.canvas.zoom(GD.cfg.gui['zoomfactor'])
    GD.canvas.update()
##def panRight():
##    global canvas,config
##    canvas.camera.pan(+5)
##    canvas.update()   
##def panLeft():
##    global canvas,config
##    canvas.camera.pan(-5)
##    canvas.update()   
##def panUp():
##    global canvas,config
##    canvas.camera.pan(+5,0)
##    canvas.update()   
##def panDown():
##    global canvas,config
##    canvas.camera.pan(-5,0)
##    canvas.update()   
def rotRight():
    global canvas
    GD.canvas.camera.rotate(+GD.cfg.gui['rotfactor'],0,1,0)
    GD.canvas.update()   
def rotLeft():
    global canvas
    GD.canvas.camera.rotate(-GD.cfg.gui['rotfactor'],0,1,0)
    GD.canvas.update()   
def rotUp():
    global canvas
    GD.canvas.camera.rotate(-GD.cfg.gui['rotfactor'],1,0,0)
    GD.canvas.update()   
def rotDown():
    global canvas
    GD.canvas.camera.rotate(+GD.cfg.gui['rotfactor'],1,0,0)
    GD.canvas.update()   
def twistLeft():
    global canvas
    GD.canvas.camera.rotate(+GD.cfg.gui['rotfactor'],0,0,1)
    GD.canvas.update()   
def twistRight():
    global canvas
    GD.canvas.camera.rotate(-GD.cfg.gui['rotfactor'],0,0,1)
    GD.canvas.update()   
def transLeft():
    global canvas
    GD.canvas.camera.translate(-GD.cfg.gui['panfactor'],0,0,GD.cfg.gui['localaxes'])
    GD.canvas.update()   
def transRight():
    global canvas
    GD.canvas.camera.translate(GD.cfg.gui['panfactor'],0,0,GD.cfg.gui['localaxes'])
    GD.canvas.update()   
def transDown():
    global canvas
    GD.canvas.camera.translate(0,-GD.cfg.gui['panfactor'],0,GD.cfg.gui['localaxes'])
    GD.canvas.update()   
def transUp():
    global canvas
    GD.canvas.camera.translate(0,GD.cfg.gui['panfactor'],0,GD.cfg.gui['localaxes'])
    GD.canvas.update()   
def dollyIn():
    global canvas
    GD.canvas.camera.dolly(1./GD.cfg.gui['zoomfactor'])
    GD.canvas.update()   
def dollyOut():
    global canvas
    GD.canvas.camera.dolly(GD.cfg.gui['zoomfactor'])
    GD.canvas.update()   
