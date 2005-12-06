#!/usr/bin/env python
# $Id$
"""Functions for drawing and for executing scripts."""

import globaldata as GD
from formex import *
from canvas import *
import gui
#import pyfotemp as PT
import threading

def wireframe():
    global allowwait
    GD.canvas.glinit("wireframe")
    GD.canvas.redrawAll()

def smooth():
    global allowwait
    if allowwait:
        drawwait()
    GD.canvas.glinit("render")
    GD.canvas.redrawAll()

def message(s):
    """Show a message to the user.

    Currently, messages are shown in the status bar."""
    if GD.gui:
        GD.gui.showMessage(s)
    else:
        print s
    
# A timed lock to slow down drawing processes

allowwait = True
drawlocked = False
drawtimeout = GD.config.get('drawwait',2)
drawtimer = None

def drawwait():
    """Wait for the drawing lock to be released.

    While we are waiting, events are processed.
    """
    global drawlocked, drawtimeout, drawtimer
    while drawlocked:
        GD.app.processEvents()

def drawlock():
    """Lock the drawing function.

    This locks the drawing function for the next drawtimeout seconds.
    """
    global drawlocked, drawtimeout, drawtimer
    if not drawlocked and drawtimeout > 0:
        drawlocked = True
        drawtimer = threading.Timer(drawtimeout,drawrelease)
        drawtimer.start()

def drawrelease():
    """Release the drawing function.

    If a timer is running, cancel it.
    """
    global drawlocked, drawtimer
    drawlocked = False
    if drawtimer:
        drawtimer.cancel()


def draw(F,side='front',color='prop',wait=True):
    """Draw a Formex on the canvas.

    Draws an actor on the canvas, and directs the camera to it from
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
    
    Each draw action activates a locking mechanism for the next draw action,
    which will only be allowed after drawtimeout seconds have elapsed. This
    makes it easier to see subsequent images and is far more elegant that an
    explicit sleep() operation, because all script processing will continue
    up to the next drawing instruction.
    The user can disable the wait cycle for the next draw operation by
    specifying wait=False. Setting drawtimeout=0 will disable the waiting
    mechanism for all subsequent draw statements (until set >0 again).
    """
    global allowwait
    if allowwait:
        drawwait()
    lastdrawn = F
    # Maybe we should move some of this color handling to the FormexActor
    if F.p == None or color==None:
        color = black
    if type(color) == str and color == 'prop':
        # use the prop as entry in a color table
        color=GD.config['propcolors']
    if len(color) == 3 and type(color[0]) == float and \
           type(color[1]) == float and type(color[2]) == float:
        # it is a single color
        GD.canvas.addActor(FormexActor(F,color))
    else:
        # assume color is a colorset
        GD.canvas.addActor(CFormexActor(F,color))
    if side:
        GD.canvas.useView(F.bbox(),side)
    # If side == None we still should calculate the bbox and zoom accordingly
    GD.canvas.update()
    if allowwait and wait:
        drawlock()

def bgcolor(color):
    """Change the background color (and redraw)."""
    if type(color) == str:
        color = eval(color)
    GD.canvas.bgcolor = color
    GD.canvas.display()
    GD.canvas.update()

def clear():
    """Remove all actors from the canvas"""
    global allowwait
    if allowwait:
        drawwait()
    GD.canvas.removeAllActors()
    GD.canvas.clear()

def redraw():
    GD.canvas.redrawAll()

def view(v,wait=False):
    """Show a named view, either a builtin or a user defined."""
    global allowwait
    if allowwait:
        drawwait()
    if GD.canvas.views.has_key(v):
        GD.canvas.useView(None,v)
        GD.canvas.update()
        if allowwait and wait:
            drawlock()
    else:
        warning("A view named '%s' has not been created yet" % v)

def addview(name,angles):
    dir(gui)
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
