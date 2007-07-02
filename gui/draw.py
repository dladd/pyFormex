#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""Functions for drawing and for executing pyFormex scripts."""

import globaldata as GD
import threading,os,sys,types,copy,commands,time

from PyQt4 import QtCore, QtGui  # needed for events, signals

import numpy
import utils
import widgets
import colors
import actors
import decors
import marks
import formex
from script import *
from cameraMenu import setPerspective,setProjection


#################### Interacting with the user ###############################


## def gui_update():
##     GD.gui.update()


##def textView(text,actions=['OK']):
##    """Display a text file and wait for user response."""
##    w = QtGui.QMessageBox()
##    w.setText(text)
##    w.setIcon(QtGui.QMessageBox.Question)
##    w.show()
##    w.resize(100,40)
##    w.update()
##    return w.exec_()

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
    

def messageBox(message,level='info',actions=['OK']):
    """Display a message box and wait for user response.

    The message box displays a text, an icon depending on the level
    (either 'about', 'info', 'warning' or 'error') and 1-3 buttons
    with the specified action text. The 'about' level has no buttons.

    The function returns the number of the button that was clicked.
    """
    w = QtGui.QMessageBox()
    if level == 'error':
        ans = w.critical(w,GD.Version,message,*actions)
    elif level == 'warning':
        ans = w.warning(w,GD.Version,message,*actions)
    elif level == 'info':
        ans = w.information(w,GD.Version,message,*actions)
    elif level == 'about':
        ans = w.about(w,GD.Version,message)
    GD.gui.update()
    return ans


outtimer = None

def outrelease():
    """Release the outtime."""
    global outtimer
    if outtimer:
        outtimer.release()
        outtimer = None
        
def timeout(secs):
    """Raise a timeout error after secs."""
    global outtimer
    outtimer = threading.Timer(secs,outrelease)
    outtimer.start()
    

def ask(question,choices=None,default='',timeout=-1):
    """Ask a question and present possible answers.

    Returns index of the chosen answer.
    """
    if choices:
        # this currently only supports 3 answers
        return info(question,choices)
    else:
        items = [ [question, default] ]
        res,accept = widgets.InputDialog(items,'Config Dialog').getResult()
        GD.gui.update()
        if accept:
            return res[0][1]
        else:
            return default

def ack(question):
    """Show a Yes/No question and return True/False depending on answer."""
    return ask(question,['Yes','No']) == 0
    
def error(message,actions=['OK']):
    """Show an error message and wait for user acknowledgement."""
    return messageBox(message,'error',actions)
    
def warning(message,actions=['OK']):
    """Show a warning message and wait for user acknowledgement."""
    return messageBox(message,'warning',actions)

def info(message,actions=['OK']):
    """Show a neutral message and wait for user acknowledgement."""
    return messageBox(message,'info',actions)
   
def about(message=GD.Version):
    """Show a informative message and wait for user acknowledgement."""
    messageBox(message,'about')

def askItems(items,caption=None):
    """Ask the value of some items to the user. !! VERY EXPERIMENTAL!!

    Input is a dictionary of items or a list of [key,value] pairs.
    The latter is recommended, because a dictionary does not guarantee
    the order of the items.
    Returns a dictionary (maybe we should just return the list??)
    """
    if type(items) == dict:
        items = items.items()
    if type(caption) is str:
        w = widgets.InputDialog(items,caption)
    else:
        w = widgets.InputDialog(items)
    res,status = w.getResult()
    items = {}
    for r in res:
        items[r[0]] = r[1]
    return items

def askFilename(cur,filter="All files (*.*)",file=None,exist=False,multi=False):
    """Ask for an existing file name or multiple file names."""
    GD.debug("Create widget")
    w = widgets.FileSelection(cur,filter,exist,multi)
    #sleep(5)
    GD.debug("Get filename")
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
    if type(s) != str:
        s = '%s' % s
    GD.gui.board.write(s)
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
    global scriptRunning, scriptDisabled, allowwait, stepmode, exportNames, starttime
    # (We only allow one script executing at a time!)
    # and scripts are non-reentrant
    GD.debug('SCRIPT MODE %s,%s,%s'% (scriptRunning, scriptDisabled, stepmode))
    if scriptRunning or scriptDisabled :
        return
    scriptRunning = True
    allowwait = True
    if GD.canvas:
        GD.canvas.update()
    if GD.gui:
        GD.gui.actions['Step'].setEnabled(True)
        GD.gui.actions['Continue'].setEnabled(True)
       
        GD.app.processEvents()
    # We need to pass formex globals to the script
    # This would be done automatically if we put this function
    # in the formex.py file. But then we need to pass other globals
    # from this file (like draw,...)
    # We might create a module with all operations accepted in
    # scripts.

    # Our solution is to take a copy of the globals in this module,
    # and add the globals from the 'colors' and 'formex' modules
    # !! Taking a copy is needed to avoid changing this module's globals !!
    #g = copy.copy(globals())
    # An alternative is to use the GD.PF and update it with these modules
    # globals
    # OOPS! THis was a big mistake!
    #g = GD.PF
    #g.update(globals())
    g = copy.copy(globals())
    g.update(GD.PF)  # We could do away with PF altogether
    if GD.gui:
        g.update(colors.__dict__)
    g.update(formex.__dict__) # this also imports everything from numpy
    # Finally, we set the name to 'script' or 'draw', so that the user can
    # verify that the script is the main script being excuted (and not merely
    # an import) and also whether the script is executed under the GUI or not.
    if GD.gui:
        modname = 'draw'
    else:
        modname = 'script'
    g.update({'__name__':modname})
#
#  WE COULD also set __file__ to the script name
#   g.update({'__file__':name})
    
    # Now we can execute the script using these collected globals

    exportNames = []
    GD.scriptName = name
    exitall = False
    #print "EXEC globals:"
    #k = g.keys()
    #k.sort()
    starttime = time.clock()
    GD.debug('STARTING SCRIPT (%s)' % starttime)
    try:
        try:
            if stepmode:
                step_script(scr,g,True)
            else:
                exec scr in g
            if GD.cfg['autoglobals']:
                exportNames.extend(listAll(g))
            globals().update([(k,g[k]) for k in exportNames])
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
            #GD.gui.actions['Step'].setEnabled(False)
            GD.gui.actions['Continue'].setEnabled(False)
    #print scriptRunning,stepmode
    if exitall:
        GD.DEBUG("Calling exit() from playscript")
        exit()


def step_script(s,glob,paus=True):
    buf = ''
    for line in s:
        if buf.endswith('\\'):
            buf[-1:] = line
            break
        else:
            buf = line
        if paus and ( buf.strip().startswith('draw') or
                      buf.find('draw(') >= 0 ):
            pause()
        message(buf)
        exec(buf) in glob
    info("Finished stepping through script!")


## def exportNames(names):
##     globals().update(dict)

def export(dict):
    globals().update(dict)

Globals = globals
Export = export

def forget(names):
    g = globals()
    for name in names:
        if g.has_key(name):
            del g[name]

def named(name):
    """Returns the global object named name."""
    GD.debug("name %s" % name)
    if globals().has_key(name):
        GD.debug("Found %s in globals()" % name)
        dict = globals()
    elif GD.PF.has_key(name):
        GD.debug("Found %s in GD.PF" % name)
        dict = GD.PF
    return dict[name]

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
    message("Running script (%s)" % fn)
    reset()
    GD.debug("Current Drawing Options: %s" % DrawOptions)
    stepmode = step
    playScript(file(fn,'r'),fn)
    message("Script finished")


############################## drawing functions ########################

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

def smoothwire():
    renderMode("smoothwire")
    
def flatwire():
    renderMode("flatwire")

    
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
        #sleep(0.5)

def drawlock():
    """Lock the drawing function for the next drawdelay seconds."""
    global drawlocked, drawtimer
    drawtimeout = GD.cfg['draw/wait']
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
        view = '__last__',       # Keep the current camera angles
        bbox = 'auto',           # Automatically zoom on the drawed object
        linewidth = GD.cfg['draw/linewidth'],
        bgcolor = GD.cfg['draw/bgcolor'],
        clear = False,
        )
    #bgcolor(DrawOptions['bgcolor'])
    GD.canvas.reset()
    clear()
    view('front')
    
def setDrawingOptions(d):
    global DrawOptions
    DrawOptions.update(d)
    
def showDrawingOptions():
    global DrawOptions
    GD.message("Current Drawing Options: %s" % DrawOptions)
    
    

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


def draw(F,view=None,bbox='auto',color='prop',wait=True,eltype=None,allviews=False,marksize=None):
    """Draw a Formex or a list of Formices on the canvas.

    If F is a list, all its items are drawn with the same settings.

    If a setting is unspecified, its current values are used.
    
    Draws an actor on the canvas, and directs the camera to it from
    the specified view. Named views are either predefined or can be added by
    the user.
    If view=None is specified, the camera settings remain unchanged.
    This may make the drawn object out of view!
    A special name '__last__' may be used to keep the same camera angles
    as in the last draw operation. The camera will be zoomed on the newly
    drawn object.
    The initial default view is 'front' (looking in the -z direction).

    If bbox == 'auto', the camera will zoom automatically on the shown
    object. A bbox may be specified to have other zoom settings, e.g. to
    keep the previous settings. If bbox == None, the previous bbox will be
    kept.

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
    which will only be allowed after drawdelay seconds have elapsed. This
    makes it easier to see subsequent images and is far more elegant that an
    explicit sleep() operation, because all script processing will continue
    up to the next drawing instruction.
    The value of drawdelay is set in the config, or 2 seconds by default.
    The user can disable the wait cycle for the next draw operation by
    specifying wait=False. Setting drawdelay=0 will disable the waiting
    mechanism for all subsequent draw statements (until set >0 again).
    """
    global allowwait, multisave

    if type(F) == list:
        return [ draw(Fi,view,bbox,color,wait,eltype,allviews) for Fi in F ]

    if type(F) == str:
        F = named(F)
        if F is None:
            return None

    if DrawOptions.get('clear',False):
        clear()

    if view is None:
        view = DrawOptions['view']
        #print "VIEW=%s" % view
    elif view != '__last__':
        setView(view)

    if not isinstance(F,formex.Formex):
        raise RuntimeError,"draw() can only draw Formex instances"
    if allowwait:
        drawwait()
    # Create the colors
    if color == 'prop':
        if F.p is None:
            # No properties defined: draw in defaultcolor or black
            color = colors.GLColor(GD.cfg['draw/fgcolor']) 
        else:
            # use the property as entry in a default color table
            color = GD.cfg['draw/propcolors']
            color = map(colors.GLColor,color)
    elif color == 'random':
        # create random colors
        color = numpy.random.random((F.nelems(),3))
    elif type(color) == str:
        # convert named color to RGB tuple
        color = colors.GLColor(color)
    elif isinstance(color,numpy.ndarray) and color.shape[-1] == 3:
        pass
    elif (type(color) == tuple or type(color) == list) and len(color) == 3:
        pass
    else:
        # The input should be compatible to a list of color compatible items.
        # An array with 3 colums will be fine.
        color = map(colors.GLColor,color)

    try:
        marksize = float(marksize)
    except:
        marksize = GD.cfg.get('marksize',0.01)

    GD.gui.setBusy()
    actor = actors.FormexActor(F,color,linewidth=GD.cfg['draw/linewidth'],eltype=eltype,markscale=marksize)
    GD.canvas.addActor(actor)
    if view:
        if view == '__last__':
            view = DrawOptions['view']
        if bbox == 'auto':
            bbox = F.bbox()
        #print "DRAW: bbox=%s, view=%s" % (bbox,view)
        GD.canvas.setCamera(bbox,view)
        #setView(view)
    GD.canvas.update()
    GD.app.processEvents()
    if multisave and multisave[4]:
        saveNext()
    if allowwait and wait:
##        if stepmode:
##            drawblock()
##        else:
        drawlock()
    GD.gui.setBusy(False)
    return actor


def drawNumbers(F):
    """Draw numbers on all elements of F."""
    FC = F.centroids().trl([0.,0.,0.1])
    M = marks.MarkList(FC.f[:,0,:],range(FC.nelems()))
    GD.canvas.addMark(M)
    GD.canvas.update()
    return M


def undraw(itemlist):
    """Remove an item or a number of items from the canvas.

    Use the return value from one of the draw... functions to remove
    the item that was drawn from the canvas.
    A single item or a list of items may be specified.
    """
    GD.canvas.remove(itemlist)
    GD.canvas.update()
    GD.app.processEvents()
    

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
        
@formex.deprecated(setTriade)
def drawTriade():
    setTriade(True)

@formex.deprecated(setTriade)
def removeTriade():
    setTriade(False)
        
@formex.deprecated(setTriade)
def toggleTriade():
    pass

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
    GD.canvas.current.bgcolor = colors.GLColor(color)
    GD.canvas.display()
    GD.canvas.update()


def fgcolor(color):
    """Set the default foreground color."""
    color = colors.GLColor(color)
    #print color
    GD.canvas.setFgColor(color)


def linewidth(wid):
    """Set the linewidth to be used in line drawings."""
    #GD.canvas.setLinewidth(float(wid))
    GD.cfg['draw/linewidth'] = wid


def clear():
    """Clear the canvas"""
    global allowwait
    if allowwait:
        drawwait()
    #print "CLEAR: %s" % GD.canvas
    GD.canvas.removeAll()
    GD.canvas.clear()
    GD.canvas.update()


def redraw():
    GD.canvas.redrawAll()


def pause():
    drawblock()


def step():
    """Perform one step of a script.

    A step is a set of instructions until the next draw operation.
    If a script is running, this just releases the draw lock.
    Else, it starts the script in step mode.
    """
    if scriptRunning:
        drawrelease()
    else:
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
    

def listAll(dict=None):
    """Return a list of all Formices in dict or by default in globals()"""
    if dict is None:
        dict = Globals()
        dict.update(GD.PF)
    flist = []
    for n,t in dict.items():
        if isinstance(t,formex.Formex) and t.__class__.__name__ == 'Formex':
            flist.append(n)
    return flist


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
    GD.message("Formices currently in globals():\n%s" % listAll())


def printglobals():
    print globals()


def printbbox():
    print GD.canvas.bbox

    
def printconfig():
    print "Reference Configuration",GD.refcfg
    print "User Configuration",GD.cfg


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
        if multisave and multisave[4]:
            saveNext()
        if sleeptime is None:
            sleeptime = GD.cfg['draw/flywait']
        sleeptime = float(sleeptime)
        if sleeptime > 0.0:
            sleep(sleeptime)


def pick():
    return GD.canvas.pick() 


def pickDraw():
    K = pick()
    if len(K) > 0:
        undraw(K)
        draw(K,color='red',bbox=None)
    return K


################################ saving images ########################

def imageFormats():
    """Return a list of the valid image formats.

    image formats are lower case strings as 'png', 'gif', 'ppm', 'eps', etc.
    The available image formats are derived from the installed software.
    """
    return GD.image_formats_qt + \
           GD.image_formats_gl2ps + \
           GD.image_formats_fromeps


def checkImageFormat(fmt,verbose=False):
    """Checks image format; if verbose, warn if it is not.

    Returns the image format, or None if it is not OK.
    """
    GD.debug("Format requested: %s" % fmt)
    GD.debug("Formats available: %s" % imageFormats())
    if fmt in imageFormats():
        if fmt == 'tex' and verbose:
            warning("This will only write a LaTeX fragment to include the 'eps' image\nYou have to create the .eps image file separately.\n")
        return fmt
    else:
        if verbose:
            error("Sorry, can not save in %s format!\n"
                  "I suggest you use 'png' format ;)"%fmt)
        return None


def save_window(filename,format,windowname=None):
    """Save a window as an image file.

    This function needs a filename AND format.
    If a window is specified, the named window is saved.
    Else, the main pyFormex window is saved.
    """
    if windowname is None:
        windowname = GD.gui.windowTitle()
    GD.gui.raise_()
    GD.gui.repaint()
    GD.gui.toolbar.repaint()
    GD.gui.update()
    GD.canvas.makeCurrent()
    GD.canvas.raise_()
    GD.canvas.update()
    GD.app.processEvents()
    cmd = 'import -window "%s" %s:%s' % (windowname,format,filename)
    sta,out = commands.getstatusoutput(cmd)
    return sta


def saveWindow(filename,format,border=False):
    """Save the main window as an image file.

    This function needs a filename AND format.
    If a window is specified, the named window is saved.
    Else, the main pyFormex window is saved.
    """
    GD.gui.repaint()
    GD.gui.toolbar.repaint()
    GD.gui.update()
    GD.canvas.update()
    GD.app.processEvents()
    if border:
        geom = GD.gui.frameGeometry()
    else:
        geom = GD.gui.geometry()
    x,y,w,h = geom.getRect()
    return utils.saveRectangle(x,y,w,h,filename,format)


# global parameters for multisave mode
multisave = None 

def saveImage(filename=None,window=False,multi=False,hotkey=True,autosave=False,format=None,verbose=False):
    """Starts or stops saving a sequence of images.

    With a filename, this starts or changes the multisave mode.
    In multisave mode, each call to saveNext() will save an image to the
    next generated file name.
    Filenames are generated by incrementing a numeric part of the name.
    If the supplied filename (after removing the extension) has a trailing
    numeric part, subsequent images will be numbered continuing from this
    number. Otherwise a numeric part '-000' will be added to the filename.

    Without filename, exits the multisave mode. It is acceptable to have
    two subsequent saveMulti calls with a filename, without an intermediate
    call without filename.

    If window is True, the full pyFormex window is saved.
    If window is False, only the canvas is saved.

    If hotkey is True, a new image will be saved by hitting the 'S' key.
    If autosave is True, a new image will be saved on each execution of
    the 'draw' function.
    If neither hotkey nor autosave are True, images can only be saved by
    executing the saveNext() function from a script.

    If no format is specified, it is derived from the filename extension.
    fmt should be one of the valid formats as returned by imageFormats()
  
    If verbose=True, error/warnings are activated. This is usually done when
    this function is called from the GUI.
    
    """
    global multisave

    # Leave multisave mode
    if filename is None:
        if multisave:
            log("Leave multisave mode")
            QtCore.QObject.disconnect(GD.gui,QtCore.SIGNAL("Save"),saveNext)
        multisave = None
        return

    #chdir(filename)
    name,ext = os.path.splitext(filename)
    # Get/Check format
    if format is None:
        format = checkImageFormat(utils.imageFormatFromExt(ext))
    if not format:
        return

    if multi: # Start multisave mode
        names = utils.FilenameSequence(name,ext)
        log("Start multisave mode to files: %s (%s)" % (names.name,format))
        #print hotkey
        if hotkey:
             QtCore.QObject.connect(GD.gui,QtCore.SIGNAL("Save"),saveNext)
             if verbose:
                 warning("Each time you hit the 'S' key,\nthe image will be saved to the next number.")
        multisave = (names,format,window,hotkey,autosave)
        return multisave is None

    else: # Save the image
        if window:
            sta = saveWindow(filename,format)
        else:
            sta = GD.canvas.save(filename,format)
        if sta:
            GD.debug("Error while saving image %s" % filename)
        else:
            log("Image file %s written" % filename)
        return

    
def saveNext():
    """In multisave mode, saves the next image."""
    if multisave:
        names,format,window,hotkey,autosave = multisave
        name = names.next()
        saveImage(name,window,False,hotkey,autosave,format,False)


def createMovie():
    """Create a movie from a saved sequence of images."""
    if not multisave:
        warning('You need to start multisave mode first!')
        return

    names,format,window,hotkey,autosave = multisave
    glob = names.glob()
    if glob.split('.')[-1] != 'y4m':
        warning("Currently you need to save in 'y4m' format to create movies")
        return
    
    cmd = "mencoder -ovc lavc -fps 5 -o output.avi %s" % names.glob()
    GD.debug(cmd)
    utils.runCommand(cmd)
         

def runtime():
    """Return the time elapsed since start of execution of the script."""
    return time.clock() - starttime
    


#### Change settings

def setLocalAxes(mode=True):
    GD.cfg['draw/localaxes'] = mode 

def setGlobalAxes(mode=True):
    setLocalAxes(not mode)

#### End
