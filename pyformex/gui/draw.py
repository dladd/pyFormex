# $Id$
##
## This file is part of pyFormex 0.5 Release Fri Aug 10 12:04:07 2007
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
import colors
import formex
from script import *

## # import some functions for scripts:

from toolbar import setPerspective as perspective, setTransparency as transparency

## # import a few functions for user scripts
## from image import saveImage,saveNext

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
    Returns a dictionary with the results, equal to the input if the user
    exited with a cancel.
    """
    if type(items) == dict:
        items = items.items()
    w = widgets.InputDialog(items,caption,GD.gui)
    res,status = w.getResult()
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


# We currently have two mechanisms for transferring variables between
# scripts by using global variables.
# - put them into globaldata.PF (GD.PF), see surface_menu plugin
# - export them using the export() function, see formex_menu plugin


def Globals():
    """Return the globals that are pased to the scripts on execution.

    This basically contains the globals defined in draw.py, colors.py,
    and formex.py, as well as the globals from numpy.
    It also contains the definitions put into the globaldata.PF
    Finally, because the scripts are executed in the context of the
    draw module, it will also contain any global definitions made in
    your scripts or explicitely exported by a script.
    During execution of the script, the global variable __name__ will be
    set to either 'draw' or 'script' depending on whether the script
    was executed in the 'draw' module (--gui option) or the 'script'
    module (--nogui option).
    """
    # We need to pass formex globals to the script
    # This would be done automatically if we put this function
    # in the formex.py file. But then we need to pass other globals
    # from this file (like draw,...)
    # We might create a module with all operations accepted in
    # scripts.

    # Our current solution is to take a copy of the globals in this module,
    # and add the globals from the 'colors' and 'formex' modules
    # !! Taking a copy is needed to avoid changing this module's globals !!
    # Also, do not be tempted to take a user dict and update it with this
    # module's globals: you might override many user variables.
    g = copy.copy(globals())
    # Add the user globals (We could do away with PF altogether,
    # if we always store them in this module's globals.
    g.update(GD.PF)
    # Add these last, because too much would go broke if the user
    # overrides them.
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
    #  Would this help the user in debugging?
    #   g.update({'__file__':name})
    return g

 
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



def export(dict):
    globals().update(dict)

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
    stepmode = step
    reset()
    GD.debug("Current Drawing Options: %s" % DrawOptions)
    message("Running script (%s)" % fn)
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
        view = '__last__',       # Keep the current camera angles
        bbox = 'auto',           # Automatically zoom on the drawed object
        clear = False,
        wait = GD.cfg['draw/wait']
        )
    #GD.canvas.reset()
    clear()
    view('front')
    
def setDrawOptions(d):
    global DrawOptions
    DrawOptions.update(d)
    
def showDrawOptions():
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


def draw(F,view=None,bbox='auto',color='prop',colormap=None,wait=True,clear=None,eltype=None,allviews=False,marksize=None,linewidth=None,alpha=0.5):
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

    global allowwait

    if type(F) == list:
        actor = []
        nowait = False
        for Fi in F:
            if Fi == F[-1]:
                nowait = wait
            actor.append(draw(Fi,view,bbox,color,colormap,nowait,clear,eltype,allviews,marksize,linewidth,alpha))
            if Fi == F[0]:
                clear = False
                view = None
        return actor

    if type(F) == str:
        F = named(F)
        if F is None:
            return None

    if not isinstance(F,formex.Formex):
        raise RuntimeError,"draw() can only draw Formex instances"

    if allowwait:
        drawwait()

    if clear is None:
        clear = DrawOptions.get('clear',False)
    if clear:
        clear_canvas()

    if view is None:
        view = DrawOptions['view']
        #print "VIEW=%s" % view
    elif view != '__last__':
        setView(view)
        
    # Create the colors
    if color == 'prop':
        color = F.p
    elif color == 'random':
        # create random colors
        color = numpy.random.random((F.nelems(),3))

    try:
        marksize = float(marksize)
    except:
        marksize = GD.cfg.get('marksize',0.01)

    GD.gui.setBusy()
    actor = actors.FormexActor(F,color=color,colormap=colormap,linewidth=linewidth,eltype=eltype,marksize=marksize,alpha=alpha)
 
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
    if image.autoSaveOn():
        image.saveNext()
    if allowwait and wait:
##        if stepmode:
##            drawblock()
##        else:
        drawlock()
    GD.gui.setBusy(False)
    return actor


def drawNumbers(F,color=colors.black):
    """Draw numbers on all elements of F."""
    FC = F.centroids().trl([0.,0.,0.1])
    M = marks.MarkList(FC.f[:,0,:],range(FC.nelems()),color=color)
    GD.canvas.addMark(M)
    GD.canvas.numbers = M
    GD.canvas.update()
    return M


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
    

def listAll(dict=None):
    """Return a list of all Formices in dict or by default in globals()"""
    if dict is None:
        dict = globals()
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
    print "Formices currently in globals():\n%s" % listAll()


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

@deprecated(export)
def export(dict):
    pass



#### End
