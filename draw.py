#!/usr/bin/env python
# $Id$
"""Functions for drawing and for executing scripts."""

import globaldata as GD
from formex import *
from canvas import *
import pyfotemp
import threading

def clear():
    """Remove all actors from the canvas"""
    global out
    GD.canvas.removeAllActors()
    GD.canvas.clear()
    out = None
def wireframe():
    GD.canvas.glinit("wireframe")
    GD.canvas.redrawAll()
def smooth():
    GD.canvas.glinit("render")
    GD.canvas.redrawAll()
def redraw():
    GD.canvas.redrawAll()

def message(s):
    """Show a message in the status bar"""
    if GD.gui:
        GD.gui.showMessage(s)
    else:
        print s
    
# A timed lock to slow down drawing processes

allowwait = True
drawwait = False
drawtimeout = GD.config.get('drawwait',2)

def setdrawwait(n):
    """Set the waiting time for the draw function."""
    global drawtimeout
    drawtimeout = n

def drawlock():
    """Lock the drawing function.

    This locks the drawing function for the next drawtimeout seconds.
    """
    global drawwait, drawtimeout, drawtimer
    while drawwait:
        GD.app.processEvents()
        continue
    drawwait = True
    drawtimer = threading.Timer(drawtimeout,drawrelease)
    drawtimer.start()

def drawrelease():
    """Release the drawing function."""
    global drawwait, drawtimer
    drawwait = False
    if drawtimer:
        drawtimer.cancel()


def draw(F,side='front',color="prop"):
    """Draw a Formex on the canvas.

    This draws an actor on the canvas, and directs the camera to it from
    the specified side. Default is looking in the -z direction.
    Specifying side=None leaves the camera settings unchanged.
    If other actors are on the scene, they may be visible as well.
    Clear the canvas before drawing if you only want one actor!

    If the Formex has properties and a color list is specified, then the
    the properties will be used as an index in the color list and each member
    will be drawn with the resulting color.
    If color is one color value, the whole Formex will be drawn with
    that color.
    Finally, if color=None is specified, the whole Formex is drawn in black.
    Each draw action activates a 
    """
    global allowwait
    if allowwait:
        print "waits allowed"
        drawlock()
    lastdrawn = F
    if F.p == None or color==None:
        # use the Formex directly as actor
        GD.canvas.addActor(FormexActor(F))
    else:
        # use the prop as entry in a color table
        colorset=GD.config['propcolors']
        GD.canvas.addActor(CFormexActor(F,colorset))
    if side:
        GD.canvas.setView(F.bbox(),side)
    # If side == None we still should calculate the bbox and zoom accordingly
    GD.canvas.update()

scriptDisabled = False
scriptRunning = False
 
def playScript(scr,name="unnamed"):
    """Play a pyformex script scr. scr should be a valid Python text.

    If a second parameter is given, it will be displayed on the status line.
    There is a lock to prevent multiple scripts from being executed at the
    same time.
    """
    global scriptRunning, scriptDisabled, allowwait
    # (We only allow one script executing at a time!)
    if scriptRunning or scriptDisabled :
        return
    scriptRunning = True
    message("Running script (%s)" % name)
    GD.canvas.update()
    allowwait = True
    GD.gui.actions['Step'].setEnabled(True)
    GD.gui.actions['Continue'].setEnabled(True)
    # We need to pass formex globals to the script
    # This would be done automatically if we put this function
    # in the formex.py file. But hen we need to pass other globals
    # from this file (like draw,...)
    # We might create a module with all operations accepted in
    # scripts.
    g = globals()
    g.update(Formex.globals())
    try:
        exec scr in g
    finally:
        scriptRunning = False # release the lock in case of an error
    message("Finished script")
    GD.gui.actions['Step'].setEnabled(False)
    GD.gui.actions['Continue'].setEnabled(False)


def playFile(fn,name=None):
    """Play a formex script from file fn."""
    playScript(file(fn,'r'),fn)

def step():
    drawrelease()

def fforward():
    global allowwait
    print "Blocking further waits"
    allowwait = False
    drawrelease()
    

def exit():
    if GD.app and GD.app_started:
        if scriptRunning:
            scriptDisabled = True
            fforward()
        GD.app.quit()  # exit on success (no script running)
    else: # the gui didn't even start
        sys.exit(0)


def sleep(n=None):
    print "The sleep() function is deprecated and ignored!"
