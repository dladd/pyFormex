# $Id$
##
## This file is part of pyFormex 0.6 Release Sun Sep 30 14:33:15 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Functions for drawing and for executing pyFormex scripts."""

import globaldata as GD
import threading,os,sys,types,copy,commands,time

from PyQt4 import QtCore, QtGui  # needed for events, signals

import numpy
import utils
import widgets
import toolbar
import actors
import decors
import marks
import image
import canvas
import colors
import formex
from script import *
from plugins import surface,tools

# import some functions for scripts:
from toolbar import setPerspective as perspective, setTransparency as transparency, timeout


############################# Globals for scripts ############################


def Globals():
    """Return the globals that are pased to the scripts on execution.

    This basically contains the globals defined in draw.py, colors.py,
    and formex.py, as well as the globals from numpy.
    It also contains the definitions put into the globaldata.PF, by
    preference using the export() function.
    During execution of the script, the global variable __name__ will be
    set to either 'draw' or 'script' depending on whether the script
    was executed in the 'draw' module (--gui option) or the 'script'
    module (--nogui option).
    """
    g = copy.copy(GD.PF)
    g.update(colors.__dict__)
    g.update(globals())
    g.update(formex.__dict__) 
    g.update({'__name__':'draw'})
    return g

        
#################### Interacting with the user ###############################


def textView(text):
    """Display a text file and wait for user response."""
    w = QtGui.QDialog()
    t = QtGui.QTextEdit()
    t.setReadOnly(True)
    t.setPlainText(text)
    b = QtGui.QPushButton('Close')
    QtCore.QObject.connect(b,QtCore.SIGNAL("clicked()"),w,QtCore.SLOT("accept()"))
    l = QtGui.QVBoxLayout()
    l.addWidget(t)
    l.addWidget(b)
    w.setLayout(l)
    w.resize(800,400)
    return w.exec_()
   

def ask(question,choices=None,default=None,timeout=None):
    """Ask a question and present possible answers.

    Return answer.
    """
    if choices:
        return widgets.messageBox(question,'question',choices)

    if choices is None:
        if default is None:
            default = choices[0]
        items = [ [question, default] ]
    else:
        items = [ [question, choices, 'combo', default] ]

    #print "INPUT TIMEOUT %s" % timeout
    res,accept = widgets.InputDialog(items,'Ask Question').getResult(timeout)
    GD.gui.update()
    if accept:
        return res[question]
    else:
        return default


def ack(question):
    """Show a Yes/No question and return True/False depending on answer."""
    return ask(question,['Yes','No']) == 'Yes'
    
def error(message,actions=['OK']):
    """Show an error message and wait for user acknowledgement."""
    widgets.messageBox(message,'error',actions)
    
def warning(message,actions=['OK']):
    """Show a warning message and wait for user acknowledgement."""
    widgets.messageBox(message,'warning',actions)

def info(message,actions=['OK']):
    """Show a neutral message and wait for user acknowledgement."""
    widgets.messageBox(message,'info',actions)


def askItems(items,caption=None,timeout=None):
    """Ask the value of some items to the user.

    Create an interactive widget to let the user set the value of some items.
    Input is a list of input items (basically [key,value] pairs).
    See the widgets.InputDialog class for complete description of the
    available input items.
    A timeout (in seconds) can be specified to have the input dialog
    interrupted automatically.

    Return a dictionary with the results: for each input item there is a
    (key,value) pair.
    
    If the user exited with a cancel or a timeout has occurred, the output
    values will be equal to the input.
    """
    if type(items) == dict:
        items = items.items()
    w = widgets.InputDialog(items,caption)
    res,status = w.getResult(timeout)
    return res


def askFilename(cur,filter="All files (*.*)",file=None,exist=False,multi=False):
    """Ask for a file name or multiple file names."""
    w = widgets.FileSelection(cur,filter,exist,multi)
    if file:
        w.selectFile(file)
    fn = w.getFilename()
    if fn:
        if multi:
            chdir(fn[0])
        else:
            chdir(fn)
    GD.gui.update()
    GD.canvas.update()
    GD.app.processEvents()
    return fn


def askDirname(cur):
    """Ask for an existing directory name."""
    fn = widgets.FileSelection(cur,'*',dir=True).getFilename()
    if fn:
        chdir(fn)
    GD.gui.update()
    GD.canvas.update()
    GD.app.processEvents()
    return fn


def chdir(fn):
    """Change the current pyFormex working directory.

    If fn is a directory name, the current directory is set to fn.
    If fn is a file name, the current directory is set to the directory
    holding fn.
    In either case, the current dirctory is stored in GD.cfg['workdir']
    for persistence between pyFormex invocations.
    
    If fn does not exist, nothing is done.
    """
    if os.path.exists:
        if not os.path.isdir(fn):
            fn = os.path.dirname(fn)
        os.chdir(fn)
        GD.cfg['workdir'] = fn
        GD.message("Your current workdir is %s" % os.getcwd())


def log(s):
    """Display a message in the cmdlog window."""
    GD.gui.board.write(str(s))
    GD.gui.update()
    GD.app.processEvents()

# message is the preferred function to send text info to the user.
# The default message handler is set here.
# Best candidates are log/info
message = log


########################### PLAYING SCRIPTS ##############################

scriptDisabled = False
scriptRunning = False
stepmode = False
exitrequested = False
starttime = 0.0


def playScript(scr,name=None):
    """Play a pyformex script scr. scr should be a valid Python text.

    There is a lock to prevent multiple scripts from being executed at the
    same time.
    If a name is specified, sets the global variable GD.scriptName if and
    when the script is started.
    
    If step==True, an indefinite pause will be started after each line of
    the script that starts with 'draw'. Also (in this case), each line
    (including comments) is echoed to the message board.
    """
    global scriptDisabled,scriptRunning,stepmode,exitrequested, \
           allowwait,exportNames,starttime
    # (We only allow one script executing at a time!)
    # and scripts are non-reentrant
    GD.debug('SCRIPT MODE %s,%s,%s'% (scriptRunning, scriptDisabled, stepmode))
    if scriptRunning or scriptDisabled :
        return
    scriptRunning = True
    exitrequested = False
    allowwait = True
    if GD.canvas:
        GD.canvas.update()
    if GD.gui:
        GD.gui.actions['Play'].setEnabled(False)
        #GD.gui.actions['Step'].setEnabled(True)
        GD.gui.actions['Continue'].setEnabled(True)
        GD.gui.actions['Stop'].setEnabled(True)
       
        GD.app.processEvents()
    
    # Get the globals
    g = Globals()
    exportNames = []
    GD.scriptName = name
    exitall = False

    starttime = time.clock()
    GD.debug('STARTING SCRIPT (%s)' % starttime)
    try:
        try:
            if stepmode:
                step_script(scr,g,True)
            else:
                exec scr in g
            if GD.cfg['autoglobals']:
                exportNames.extend(listAll(clas=formex.Formex,dic=g))
            GD.PF.update([(k,g[k]) for k in exportNames])
        except Exit:
            pass
        except ExitAll:
            exitall = True
    finally:
        scriptRunning = False # release the lock in case of an error
        stepmode = False
        elapsed = time.clock() - starttime
        GD.debug('SCRIPT RUNTIME : %s seconds' % elapsed)
        if GD.gui:
            GD.gui.actions['Play'].setEnabled(True)
            #GD.gui.actions['Step'].setEnabled(False)
            GD.gui.actions['Continue'].setEnabled(False)
            GD.gui.actions['Stop'].setEnabled(False)

    if exitall:
        GD.DEBUG("Calling exit() from playscript")
        exit()


def breakpt(msg=None):
    """Set a breakpoint where the script can be halted on pressing a button.

    If an argument is specified, it will be written to the message board.
    """
    if exitrequested:
        if msg is not None:
            GD.message(msg)
        raise Exit


def stopatbreakpt():
    """Set the exitrequested flag."""
    global exitrequested
    exitrequested = True


def force_finish():
    global scriptRunning,stepmode
    scriptRunning = False # release the lock in case of an error
    stepmode = False


def step_script(s,glob,paus=True):
    buf = ''
    for line in s:
        if buf.endswith('\\'):
            buf[-1:] = line
            break
        else:
            buf += line
        if paus and (line.strip().startswith('draw') or
                     line.find('draw(') >= 0 ):
            drawblock()
            message(buf)
            exec(buf) in glob
    info("Finished stepping through script!")


def play(fn=None,step=False):
    """Play a formex script from file fn or from the current file.

    This function does nothing if no file is passed or no current
    file was set.
    """
    global stepmode
    if not fn:
        if GD.canPlay:
            fn = GD.cfg['curfile']
        else:
            return
    stepmode = step
    reset()
    GD.debug("Current Drawing Options: %s" % DrawOptions)
    message("Running script (%s)" % fn)
    GD.gui.history.add(fn)
    stepmode = step
    playScript(file(fn,'r'),fn)
    message("Script finished")


############################## drawing functions ########################


def renderMode(mode):
    GD.canvas.setRenderMode(mode)
    GD.canvas.update()
    GD.app.processEvents()
    
def wireframe():
    renderMode("wireframe")
    
def flat():
    renderMode("flat")
    
def smooth():
    renderMode("smooth")

def smoothwire():
    renderMode("smoothwire")
    
def flatwire():
    renderMode("flatwire")
    
def flattrans():
    renderMode("flattrans")

    
# A timed lock to slow down drawing processes

allowwait = True
drawlocked = False
drawtimer = None

def drawwait():
    """Wait for the drawing lock to be released.

    While we are waiting, events are processed.
    """
    global drawlocked
    while drawlocked:
        GD.app.processEvents()
        GD.canvas.update()


def drawlock():
    """Lock the drawing function for the next drawdelay seconds."""
    global drawlocked, drawtimer
    drawtimeout = DrawOptions['wait']
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


def reset():
    global DrawOptions
    DrawOptions = dict(
        view = None,       # Keep the current camera angles
        bbox = 'auto',     # Automatically zoom on the drawed object
        clear = False,
        shrink = None,
        wait = GD.cfg['draw/wait']
        )
    GD.canvas.resetDefaults(GD.cfg['canvas'])
    clear()
    view('front')

def resetAll():
    wireframe()
    reset()
    
def setDrawOptions(d):
    global DrawOptions
    DrawOptions.update(d)
    
def showDrawOptions():
    global DrawOptions
    GD.message("Current Drawing Options: %s" % DrawOptions)


def shrink(v):
    setDrawOptions({'shrink':v})
    

def setView(name,angles=None):
    """Set the default view for future drawing operations.

    If no angles are specified, the name should be an existing view, or
    the predefined value '__last__'.
    If angles are specified, this is equivalent to createView(name,angles)
    followed by setView(name).
    """
    global DrawOptions
    if name != '__last__' and angles:
        createView(name,angles)
    DrawOptions['view'] = name


def draw(F, view=None,bbox='auto',
         color='prop',colormap=None,linewidth=None,alpha=0.5,
         shrink=None,eltype=None,marksize=None,color1=None,
         wait=True,clear=None,allviews=False):
    """Draw object(s) with specified settings and direct camera to it.

    The first argument is an object to be drawn. All other arguments are
    settings that influence how  the object is being drawn.

    F is either a Formex or a Surface object, or a name of such object
    (global or exported), or a list thereof.
    If F is a list, the draw() function is called repeatedly with each of
    ithe items of the list as first argument and with the remaining arguments
    unchanged.

    view is either the name of a defined view or 'last' or None.
    Predefined views are 'front','back','top','bottom','left','right','iso'.
    With view=None the camera settings remain unchanged (but might be changed
    interactively through the user interface). This may make the drawn object
    out of view!
    With view='last', the camera angles will be set to the same camera angles
    as in the last draw operation, undoing any interactive changes.
    The initial default view is 'front' (looking in the -z direction).

    bbox specifies the 3D volume at which the camera will be aimed (using the
    angles set by view). The camera position wil be set so that the volume
    comes in view using the current lens (default 45 degrees).
    bbox is a list of two points or compatible (array with shape (2,3)).
    Setting the bbox to a volume not enclosing the object may make the object
    invisible on the canvas.
    The default bbox='auto' will use the bounding box of the objects getting
    drawn (object.bbox()), thus ensuring that the camera will focus on the
    shown object.
    With bbox=None, the camera's target volume remains unchanged.

    color,colormap,linewidth,alpha,eltype,marksize,color1 are passed to the
    creation of the 3D actor.

    shrink is a floating point shrink factor that will be applied to object
    before drawing it.

    If the Formex has properties and a color list is specified, then the
    the properties will be used as an index in the color list and each member
    will be drawn with the resulting color.
    If color is one color value, the whole Formex will be drawn with
    that color.
    Finally, if color=None is specified, the whole Formex is drawn in black.
    
    Each draw action activates a locking mechanism for the next draw action,
    which will only be allowed after drawdelay seconds have elapsed. This
    makes it easier to see subsequent images and is far more elegant that an
    explicit sleep() operation, because all script processing will continue
    up to the next drawing instruction.
    The value of drawdelay is set in the config, or 2 seconds by default.
    The user can disable the wait cycle for the next draw operation by
    specifying wait=False. Setting drawdelay=0 will disable the waiting
    mechanism for all subsequent draw statements (until set >0 again).
    """

    global allowwait

    if type(F) == list:
        actor = []
        nowait = False
        for Fi in F:
            if Fi == F[-1]:
                nowait = wait
            actor.append(draw(Fi,view,bbox,
                              color,colormap,linewidth,alpha,
                              shrink,eltype,marksize,color1,
                              wait,clear,allviews))
            if Fi == F[0]:
                clear = False
                view = None
        return actor

    if type(F) == str:
        F = named(F)
        if F is None:
            return None

    if not (isinstance(F,formex.Formex) or
            isinstance(F,surface.Surface) or
            isinstance(F,tools.Plane)):
        raise RuntimeError,"draw() can not draw objects of type %s" % type(F)

    if allowwait:
        drawwait()

    if clear is None:
        clear = DrawOptions.get('clear',False)
    if clear:
        clear_canvas()

    if view is not None and view != 'last':
        GD.debug("SETTING VIEW to %s" % view)
        setView(view)

    if shrink is None:
        shrink = DrawOptions.get('shrink',None)
        
    # Create the colors
    if color == 'prop':
        if hasattr(F,'p'):
            color = F.p
        else:
            color = colors.black
    elif color == 'random':
        # create random colors
        color = numpy.random.random((F.nelems(),3))

    try:
        marksize = float(marksize)
    except:
        marksize = GD.cfg.get('marksize',0.01)

    GD.gui.setBusy()
    if shrink is not None:
        #GD.debug("DRAWING WITH SHRINK = %s" % shrink)
        F = _shrink(F,shrink)
    try:
        if isinstance(F,formex.Formex):
            if color1 is not None:
                #GD.debug("DRAWING WITH COLOR1\n%s" % str(color1))
                canvas.glSmooth()
            actor = actors.FormexActor(F,color=color,colormap=colormap,linewidth=linewidth,eltype=eltype,marksize=marksize,alpha=alpha,color1=color1)
        elif isinstance(F,surface.Surface):
            actor = actors.SurfaceActor(F,color=color,colormap=colormap,linewidth=linewidth,alpha=alpha)
        elif isinstance(F,tools.Plane):
            actor = actors.PlaneActor(F.point(),F.normal(),linecolor=color,linewidth=linewidth,alpha=alpha)
        GD.canvas.addActor(actor)
        if view is not None or bbox is not None:
            #GD.debug("CHANGING VIEW to %s" % view)
            if view == 'last':
                view = DrawOptions['view']
            if bbox == 'auto':
                bbox = F.bbox()
            #GD.debug("SET CAMERA TO: bbox=%s, view=%s" % (bbox,view))
            GD.canvas.setCamera(bbox,view)
            #setView(view)
        GD.canvas.update()
        GD.app.processEvents()
        GD.debug("AUTOSAVE %s" % image.autoSaveOn())
        if image.autoSaveOn():
            image.saveNext()
        if allowwait and wait:
    ##        if stepmode:
    ##            drawblock()
    ##        else:
            drawlock()
    finally:
        GD.gui.setBusy(False)
    return actor


def _shrink(F,factor):
    """Return a shrinked object.

    A shrinked object is one where each element is shrinked with a factor
    around its own center.
    """
    if isinstance(F,surface.Surface):
        F = F.toFormex()
    return F.shrink(factor)


def drawNumbers(F,color=colors.black):
    """Draw numbers on all elements of F."""
    FC = F.centroids().trl([0.,0.,0.1]) # put labels in front
    M = marks.MarkList(FC,range(FC.shape[0]),color=color)
    GD.canvas.addMark(M)
    GD.canvas.numbers = M
    GD.canvas.update()
    return M


def drawText3D(P,text,color=colors.black,font=None):
    """Draw a text at a 3D point."""
    M = marks.TextMark(P,text,color=color,font=font)
    GD.canvas.addMark(M)
    GD.canvas.update()
    return M


def drawBbox(A):
    """Draw the bbox of the actor A."""
    B = actors.BboxActor(A.bbox())
    drawActor(B)
    return B


def drawActor(A):
    """Draw an actor and update the screen."""
    GD.canvas.addActor(A)
    GD.canvas.update()


def undraw(itemlist):
    """Remove an item or a number of items from the canvas.

    Use the return value from one of the draw... functions to remove
    the item that was drawn from the canvas.
    A single item or a list of items may be specified.
    """
    GD.canvas.remove(itemlist)
    GD.canvas.update()
    GD.app.processEvents()


def focus(object):
    """Move the camera thus that object comes fully into view.

    object can be anything having a bbox() method.

    The camera is moved with fixed axis directions to a place
    where the whole object can be viewed using a 45. degrees lens opening.
    This technique may change in future!
    """
    GD.canvas.setCamera(bbox=object.bbox())
    

def view(v,wait=False):
    """Show a named view, either a builtin or a user defined."""
    global allowwait
    if allowwait and wait:
        drawwait()
    if v != '__last__':
        angles = GD.canvas.view_angles.get(v)
        if not angles:
            warning("A view named '%s' has not been created yet" % v)
            return
        GD.canvas.setCamera(None,angles)
    setView(v)
    GD.canvas.update()
    if allowwait and wait:
        drawlock()

def setTriade(on=None):
    """Toggle the display of the global axes on or off.

    If on is True, the axes triade is displayed, if False it is
    removed. The default (None) toggles between on and off.
    """
    if on is None:
        on = not hasattr(GD.canvas,'triade') or GD.canvas.triade is None
    if on:
        GD.canvas.triade = actors.TriadeActor(1.0)
        GD.canvas.addMark(GD.canvas.triade)
    else:
        GD.canvas.removeMark(GD.canvas.triade)
        GD.canvas.triade = None
    GD.canvas.update()
    GD.app.processEvents()


def drawtext(text,x,y,font='9x15'):
    """Show a text at position x,y using font."""
    TA = decors.Text(text,x,y,font)
    decorate(TA)
    return TA

def decorate(decor):
    """Draw a decoration."""
    GD.canvas.addDecoration(decor)
    GD.canvas.update()

def undecorate(decor):
    GD.canvas.removeDecoration(decor)
    GD.canvas.update()




def frontView():
    view("front")
def backView():
    view("back")
def leftView():
    view("left")
def rightView():
    view("right")
def topView():
    view("top");
def bottomView():
    view("bottom")
def isoView():
    view("iso")

def createView(name,angles):
    """Create a new named view (or redefine an old).

    The angles are (longitude, latitude, twist).
    If the view name is new, and there is a views toolbar,
    a view button will be added to it.
    """
    GD.gui.setViewAngles(name,angles)   
    

def zoomAll():
    GD.canvas.setBbox(formex.bbox(GD.canvas.actors))
    GD.canvas.setCamera()
    GD.canvas.redrawAll()
    GD.canvas.update()


def bgcolor(color):
    """Change the background color (and redraw)."""
    GD.canvas.setBgColor(color)
    GD.canvas.display()
    GD.canvas.update()

def fgcolor(color):
    """Set the default foreground color."""
    GD.canvas.setFgColor(color)

def opacity(alpha):
    """Set the viewports transparency."""
    GD.canvas.alpha = float(alpha)

def linewidth(wid):
    """Set the linewidth to be used in line drawings."""
    GD.canvas.setLineWidth(wid)


def clear_canvas():
    """Clear the canvas.

    This is a low level function not intended for the user.
    """
    GD.canvas.removeAll()
    GD.canvas.clear()


def clear():
    """Clear the canvas"""
    global allowwait
    if allowwait:
        drawwait()
    clear_canvas()
    GD.canvas.update()


def redraw():
    GD.canvas.redrawAll()
    GD.canvas.update()


def pause():
    if allowwait:
        drawblock()    # will need external event to release it
        while (drawlocked):
            sleep(0.5)


def step():
    """Perform one step of a script.

    A step is a set of instructions until the next draw operation.
    If a script is running, this just releases the draw lock.
    Else, it starts the script in step mode.
    """
    if scriptRunning:
        drawrelease()
    else:
        if ack("""
STEP MODE is currently only possible with specially designed,
very well behaving scripts. If you're not sure what you are
doing, you should cancel the operation now.

Are you REALLY SURE you want to run this script in step mode?
"""):
            play(step=True)
        

def fforward():
    global allowwait
    allowwait = False
    drawrelease()


def delay(i):
    """Set the draw delay in seconds."""
    i = int(i)
    if i >= 0:
        GD.cfg['draw/wait'] = i
    

        
wakeupMode=0
def sleep(timeout=None):
    """Sleep until key/mouse press in the canvas or until timeout"""
    global sleeping,wakeupMode,timer
    if wakeupMode > 0:  # don't bother : sleeps inactivated
        return
    # prepare for getting wakeup event 
    QtCore.QObject.connect(GD.canvas,QtCore.SIGNAL("Wakeup"),wakeup)
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
    QtCore.QObject.disconnect(GD.canvas,QtCore.SIGNAL("Wakeup"),wakeup)

        
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


## def drawNamed(name,*args):
##     g = globals()
##     if g.has_key(name):
##         F = g[name]
##         if isinstance(F,formex.Formex):
##             draw(F,*args)

## def drawSelected(*args):
##     name = ask("Which Formex shall I draw ?")
##     drawNamed(name,*args)


def exit(all=False):
    """Exit from the current script or from pyformex if no script running."""
    if scriptRunning:
        if all:
            raise ExitAll # exit from pyformex
        else:
            raise Exit # exit from script only
    if GD.app and GD.app_started: # exit from GUI
        GD.debug("draw.exit called while no script running")
        GD.app.quit() 
    else: # the gui didn't even start
        sys.exit(0)


########################## print information ################################
    


def formatInfo(F):
    """Return formatted information about a Formex."""
    bb = F.bbox()
    return """shape    = %s
bbox[lo] = %s
bbox[hi] = %s
center   = %s
maxprop  = %s
""" % (F.shape(),bb[0],bb[1],F.center(),F.maxprop())
    

def printall():
    """Print all Formices in globals()"""
    print "Formices currently in globals():\n%s" % listAll(clas=formex.Formex)


def printglobals():
    print globals()

def printglobalnames():
    a = globals().keys()
    a.sort()
    print a

def printbbox():
    print GD.canvas.bbox

    
def printconfig():
    print "Reference Configuration: " + str(GD.refcfg)
    print "User Configuration: " + str(GD.cfg)


def printviewportsettings():
    GD.gui.viewports.printSettings()
        

def printdetected():
    print "Detected Python Modules:"
    for k,v in GD.version.items():
        if v:
            print "%s (%s)" % ( k,v)
    print "\nDetected External Programs:"
    for k,v in GD.external.items():
        if v:
            print "%s (%s)" % ( k,v)


def updateGUI():
    """Update the GUI."""
    GD.gui.update()
    GD.canvas.update()
    GD.app.processEvents()


def flyAlong(path,upvector=[0.,1.,0.],sleeptime=None):
    for seg in path:
        GD.debug("Eye: %s; Center: %s" % (seg[0],seg[1]))
        GD.canvas.camera.lookAt(seg[0],seg[1],upvector)
        GD.canvas.display()
        GD.canvas.update()
        image.saveNext()
        if sleeptime is None:
            sleeptime = GD.cfg['draw/flywait']
        sleeptime = float(sleeptime)
        if sleeptime > 0.0:
            sleep(sleeptime)


def pick():
    return GD.canvas.pick() 


def pickDraw():
    K = pick()
    GD.debug("PICKED: %s"%K)
    if len(K) > 0:
        undraw(K)
        GD.debug("DRAWING PICKED: %s"%K)
        draw(K,color='red',bbox=None)
    return K


def pickNumbers(marks=None):
    if marks:
        GD.canvas.numbers = marks
    return GD.canvas.pickNumbers()

################################ saving images ########################
         

def runtime():
    """Return the time elapsed since start of execution of the script."""
    return time.clock() - starttime



#### Change settings

def setLocalAxes(mode=True):
    GD.cfg['draw/localaxes'] = mode 

def setGlobalAxes(mode=True):
    setLocalAxes(not mode)


######### DEPRECATED ############################
    
from utils import deprecated


@deprecated(export)
def Export(dict):
    pass



#### End
