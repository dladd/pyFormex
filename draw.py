#!/usr/bin/env python
# $Id$
"""Functions for drawing and for executing scripts."""

import globaldata as GD
from formex import *
from canvas import *
from colors import *
import pyfotemp

import gui
import threading

class Exit(Exception):
    """Exception raised to exit from a running script."""
    pass    

class ExitAll(Exception):
    """Exception raised to exit pyFormex from a script."""
    pass    

def renderMode(mode):
    global allowwait
    if allowwait:
        drawwait()
    GD.canvas.glinit(mode)
    GD.canvas.redrawAll()
    
def wireframe():
    renderMode("wireframe")
    
def flat():
    renderMode("flat")
    
def smooth():
    renderMode("smooth")

def message(s):
    """Show a message to the user.

    Currently, messages are shown in the status bar."""
    if GD.gui:
        GD.gui.showMessage(s)
    else:
        print s

def warning(s):
    if GD.gui:
        GD.gui.showWarning(s)
    else:
        print s

def ack(s):
    return GD.gui.yesNo(s)

# A timed lock to slow down drawing processes

allowwait = True
drawlocked = False
drawtimeout = GD.cfg.gui.get('drawwait',2)
# set = 0 to disable wait
# what if we want an indefinite wait (until step pressed)
drawtimer = None

def drawwait():
    """Wait for the drawing lock to be released.

    While we are waiting, events are processed.
    """
    global drawlocked, drawtimeout, drawtimer
    while drawlocked:
        GD.app.processEvents()

def drawlock():
    """Lock the drawing function for the next drawtimeout seconds."""
    global drawlocked, drawtimeout, drawtimer
    if not drawlocked and drawtimeout > 0:
        drawlocked = True
        drawtimer = threading.Timer(drawtimeout,drawrelease)
        drawtimer.start()

def drawblock():
    """Lock the drawing function indefinitely."""
    global drawlocked, drawtimer
    if drawtimer:
        drawtimer.cancel()
    if not drawlocked:
        drawlocked = True

def drawrelease():
    """Release the drawing function.

    If a timer is running, cancel it.
    """
    global drawlocked, drawtimer
    drawlocked = False
    if drawtimer:
        drawtimer.cancel()

currentView = 'front'

def draw(F,view='__last__',color='prop',wait=True,eltype=None):
    """Draw a Formex on the canvas.

    Draws an actor on the canvas, and directs the camera to it from
    the specified view. Named views are either predefined or can be added by
    the user.
    If view=None is specified, the camera settings remain unchanged.
    This may make the drawn object out of view!
    A special name '__last__' may be used to keep the same camera angles
    as in the last draw operation. The camera will be zoomed on the newly
    drawn object.
    The initial default view is 'front' (looking in the -z direction).

    If other actors are on the scene, they may or may not be visible with the
    new camera settings. Clear the canvas before drawing if you only want
    a single actor!

    If the Formex has properties and a color list is specified, then the
    the properties will be used as an index in the color list and each member
    will be drawn with the resulting color.
    If color is one color value, the whole Formex will be drawn with
    that color.
    Finally, if color=None is specified, the whole Formex is drawn in black.
    
    Each draw action activates a locking mechanism for the next draw action,
    which will only be allowed after drawtimeout seconds have elapsed. This
    makes it easier to see subsequent images and is far more elegant that an
    explicit sleep() operation, because all script processing will continue
    up to the next drawing instruction.
    The user can disable the wait cycle for the next draw operation by
    specifying wait=False. Setting drawtimeout=0 will disable the waiting
    mechanism for all subsequent draw statements (until set >0 again).
    """
    global allowwait, currentView
    if not isinstance(F,Formex):
        raise RuntimeError,"draw() can only draw Formex instances"
    if allowwait:
        drawwait()
    lastdrawn = F
    if type(color) == str:
        if color == 'prop':
            if type(F.p) == type(None):
                color = GD.cfg.get('defaultcolor',[black]) 
            else: # use the prop as entry in a color table
                color = GD.cfg.get('propcolors',[black])
        elif color == 'random':
            color = random.random((F.nelems(),3))
        else:
            color = GLColor(color)
    GD.canvas.addActor(FormexActor(F,color,GD.cfg.get('linewidth',1),eltype=eltype))
    if view:
        if view == '__last__':
            view = currentView
        GD.canvas.useView(F.bbox(),view)
        currentView = view
        # calculate the bbox and zoom accordingly
        GD.canvas.update()
    if allowwait and wait:
        drawlock()

def drawTriade():
    """Show the global axes."""
    GD.canvas.addActor(TriadeActor(1.0))
    GD.canvas.update()


def drawtext(text,x,y,font='9x15'):
    """Show a text at position x,y using font."""
    decorate(TextActor(text,x,y,font))

def decorate(actor):
    """Draw a decoration."""
    GD.canvas.addDecoration(actor)
    GD.canvas.update()


def view(v,wait=False):
    """Show a named view, either a builtin or a user defined."""
    global allowwait,currentView
    if allowwait:
        drawwait()
    if GD.canvas.views.has_key(v):
        GD.canvas.useView(None,v)
        currentView = v
        GD.canvas.update()
        if allowwait and wait:
            drawlock()
    else:
        warning("A view named '%s' has not been created yet" % v)

def bgcolor(color):
    """Change the background color (and redraw)."""
    color = GLColor(color)
    GD.canvas.bgcolor = color
    GD.canvas.display()
    GD.canvas.update()

def linewidth(wid):
    """Set the linewidth to be used in line drawings."""
    #GD.canvas.setLinewidth(float(wid))
    GD.cfg.linewidth = wid

def clear():
    """Clear the canvas"""
    global allowwait
    if allowwait:
        drawwait()
    GD.canvas.removeAll()
    GD.canvas.clear()

def redraw():
    GD.canvas.redrawAll()

def setview(name,angles):
    """Declare a new named view (or redefine an old).

    The angles are (longitude, latitude, twist).
    If the view name is new, and there is a views toolbar,
    a view button will be added to it."""
    gui.addView(name,angles)
    

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
    # and scripts are non-reentrant
    if scriptRunning or scriptDisabled :
        return
    scriptRunning = True
    GD.scriptName = name
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
    #print "Voor",g.get('__name__','Geen')
    g.update(Formex.globals())
    #print "Na",g.get('__name__','Geen')
    exitall = False
    try:
        try:
            exec scr in g
        except Exit:
            pass
        except ExitAll:
            exitall = True
    finally:
        scriptRunning = False # release the lock in case of an error
        message("Finished script")
        GD.gui.actions['Step'].setEnabled(False)
        GD.gui.actions['Continue'].setEnabled(False)
    if exitall:
        exit()

def playFile(fn,name=None):
    """Play a formex script from file fn."""
    global currentView,drawtimeout 
    drawtimeout = GD.cfg.gui.get('drawwait',2)
    currentView = 'front'
    playScript(file(fn,'r'),fn)


def pause():
    drawblock()

def step():
    drawrelease()

def fforward():
    global allowwait
    allowwait = False
    drawrelease()
    

def exit(all=False):
    if scriptRunning:
        if all:
            raise ExitAll # exit from pyformex
        else:
            raise Exit # exit from script only
    else:
        gui.exit() # exit from pyformex
        
wakeupMode=0
def sleep(timeout=None):
    """Sleep until key/mouse press in the canvas or until timeout"""
    global sleeping,wakeupMode,timer
    if wakeupMode > 0:  # don't bother : sleeps inactivated
        return
    # prepare for getting wakeup event 
    qt.QObject.connect(GD.canvas,qt.PYSIGNAL("wakeup"),wakeup)
    # create a Timer to wakeup after timeout
    if timeout and timeout > 0:
        timer = threading.Timer(timeout,wakeup)
        timer.start()
    else:
        timer = None
    # go into sleep mode
    sleeping = True
    ## while sleeping, we have to process events
    ## (we could start another thread for this)
    while sleeping:
        GD.app.processEvents()
        #time.sleep(0.1)
    # ignore further wakeup events
    qt.QObject.disconnect(GD.canvas,qt.PYSIGNAL("wakeup"),wakeup)
        
def wakeup(mode=0):
    """Wake up from the sleep function.

    This is the only way to exit the sleep() function.
    Default is to wake up from the current sleep. A mode > 0
    forces wakeup for longer period.
    """
    global timer,sleeping,wakeupMode
    if timer:
        timer.cancel()
    sleeping = False
    wakeupMode = mode


def listall():
    """List all Formices in globals()"""
    print "Formices currently in globals():"
    for n,t in globals().items():
        if isinstance(t,Formex):
            print "%s, " % n

##def printit():
##    global out
##    print out
##def printbbox():
##    global out
##    if out:
##        print "bbox of displayed Formex",out.bbox()
def printglobals():
    print globals()


def save(filename,fmt):
    GD.canvas.save(filename,fmt)

def system(*args):
    pyfotemp.system(*args)
