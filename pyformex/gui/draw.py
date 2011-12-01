## $Id$
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
"""Create 3D graphical representations.

The draw module provides the basic user interface to the OpenGL
rendering capabilities of pyFormex. The full contents of this module
is available to scripts running in the pyFormex GUI without the need
to import it.
"""

import pyformex as pf

import threading,os,sys,types,copy,commands,time

import numpy
import utils
import messages
import widgets
Dialog = widgets.InputDialog
_I = widgets.simpleInputItem
_G = widgets.groupInputItem
_T = widgets.tabInputItem
import toolbar
import actors
import decors
import marks
import image
import canvas
import colors

import coords
from mesh import Mesh
from plugins import trisurface,tools,fe

from script import *
from signals import *
from formex import *
        
#################### Interacting with the user ###############################


def closeGui():
    """Close the GUI.

    Calling this function from a script closes the GUI and terminates
    pyFormex.
    """
    pf.debug("Closing the GUI: currently, this will also terminate pyformex.")
    pf.GUI.close()


def closeDialog(name):
    """Close the named dialog.

    Closes the InputDialog with the given name. If multiple dialogs are
    open with the same name, all these dialogs are closed.

    This only works for dialogs owned by the pyFormex GUI.
    """
    pf.GUI.closeDialog(name)


def showMessage(text,actions=['OK'],level='info',modal=True,**kargs):
    """Show a short message widget and wait for user acknowledgement.

    There are three levels of messages: 'info', 'warning' and 'error'.
    They differ only in the icon that is shown next to the test.
    By default, the message widget has a single button with the text 'OK'.
    The dialog is closed if the user clicks a button.
    The return value is the button text. 
    """
    w = widgets.MessageBox(text,level=level,actions=actions,**kargs)
    if modal:
        return w.getResult()
    else:
        w.show()
        return None
        

def showInfo(text,actions=['OK'],modal=True):
    """Show an informational message and wait for user acknowledgement."""
    return showMessage(text,actions,'info',modal)
    
def warning(text,actions=['OK']):
    """Show a warning message and wait for user acknowledgement."""
    return showMessage(text,actions,'warning')
    
def error(text,actions=['OK']):
    """Show an error message and wait for user acknowledgement."""
    return showMessage(text,actions,'error')
    

def ask(question,choices=None,**kargs):
    """Ask a question and present possible answers.

    Return answer if accepted or default if rejected.
    The remaining arguments are passed to the InputDialog getResult method.
    """
    return showMessage(question,choices,'question',**kargs)


def ack(question,**kargs):
    """Show a Yes/No question and return True/False depending on answer."""
    return ask(question,['No','Yes'],**kargs) == 'Yes'


def showText(text,type=None,actions=[('OK',None)],modal=True,mono=False):
    """Display a text and wait for user response.

    This opens a TextBox widget and displays the text in the widget.
    Scrollbars will be added if the text is too large to display at once.

    The text can be plain text format. Some rich text formats will be 
    recognized and rendered appropriately. See widgets.TextBox.

    `mono=True` forces the use of a monospaced font.
    """
    w = widgets.TextBox(text,type,actions,modal=modal,mono=mono)
    if modal:
        return w.getResult()
    else:
        w.show()
        return None


def showFile(filename,mono=False,**kargs):
    """Display a text file.

    This will use the :func:`showText()` function to display a text read
    from a file.
    By default this uses a monospaced font.
    Other arguments may also be passed to ShowText.
    """
    try:
        f = open(filename,'r')
    except IOError:
        return
    showText(f.read(),mono=mono,**kargs)
    f.close()
    
  
def showDescription(filename=None):
    """Show the Description part of the docstring of a pyFormex script.

    If no file name is specified, the current script is used.
    If the script's docstring has no Description part, a default text is
    shown.
    """
    from scriptMenu import getDocString,getDescription
    if pf.GUI.canPlay:
        scriptfile = pf.prefcfg['curfile']
        doc = getDocString(scriptfile)
        des = getDescription(doc)
        if len(des.strip()) == 0:
            des = """.. NoDescription

No help available
=================

The maintainers of this script have not yet added a description
for this example.

You can study the source code, and if anything is unclear,
ask for help on the pyFormex `Support tracker <%s>`_.
""" % pf.cfg['help/support']

        showText(des,modal=False)


# widget and result status of the widget in askItems() function
_dialog_widget = None
_dialog_result = None

def askItems(items,caption=None,timeout=None,**kargs):
    """Ask the value of some items to the user.

    Create an interactive widget to let the user set the value of some items.
    'items' is a list of input items (basically [key,value] pairs).
    See the widgets.InputDialog class for complete description of the
    available input items.

    Two InputDialog classes are defined in gui.widgets.
    The OldInputDialog class is deprecated in favor of InputDialog, which
    has become the default as of pyFormex 0.8.3.
    The two classes differ in how the input is specified.
    In the new format, each input item is either a simpleInputItem, a
    groupInputItem or a tabInputItem.
    
    You can specify 'legacy=False' to indicate that you are using the new
    format, or 'legacy=True' if your data are in the old format.
    The default ('legacy = None'), will make this function try to detect
    the format and convert the input items to the proper new format.
    This conversion will work on most, but not all legacy formats that
    have been used in the past by pyFormex.
    Since the legacy format is scheduled to be withdrawn in future, users
    are encouraged to change their input to the new format.

    The remaining arguments are keyword arguments that are passed to the
    InputDialog.getResult method.
    A timeout (in seconds) can be specified to have the input dialog
    interrupted automatically.

    Return a dictionary with the results: for each input item there is a
    (key,value) pair. Returns an empty dictionary if the dialog was canceled.
    Sets the dialog timeout and accepted status in global variables.
    """
    global _dialog_widget,_dialog_result
    if 'legacy' in kargs:
        warnings.warn("The use of the 'legacy' argument in askitems is deprecated.")
    # convert items, allows for sloppy style
    items = [ widgets.convertInputItem(i) for i in items ]
    w = widgets.InputDialog(items,caption,**kargs)
        
    _dialog_widget = w
    _dialog_result = None
    res = w.getResult(timeout)
    _dialog_widget = None
    _dialog_result = w.result()
    return res


def currentDialog():
    """Returns the current dialog widget.

    This returns the dialog widget created by the askItems() function,
    while the dialog is still active. If no askItems() has been called
    or if the user already closed the dialog, None is returned.
    """
    return _dialog_widget

def dialogAccepted():
    """Returns True if the last askItems() dialog was accepted."""
    return _dialog_result == widgets.ACCEPTED

def dialogRejected():
    """Returns True if the last askItems() dialog was rejected."""
    return _dialog_result == widgets.REJECTED

def dialogTimedOut():
    """Returns True if the last askItems() dialog timed out."""
    return _dialog_result == widgets.TIMEOUT


def askFilename(cur=None,filter="All files (*.*)",exist=True,multi=False,change=True):
    """Ask for a file name or multiple file names using a file dialog.

    cur is a directory or filename. All the files matching the filter in that
    directory (or that file's directory) will be shown.
    If cur is a file, it will be selected as the current filename.

    Unless the user cancels the operation, or the change parameter was set to
    False, the parent directory of the selected file will become the new
    working directory.
    """
    if cur is None:
        cur = pf.cfg['workdir']
    if os.path.isdir(cur):
        fn = ''
    else:
        fn = os.path.basename(cur)
        cur = os.path.dirname(cur)
    w = widgets.FileSelection(cur,filter,exist,multi)
    if fn:
        w.selectFile(fn)
    fn = w.getFilename()
    if fn and change:
        if multi:
            chdir(fn[0])
        else:
            chdir(fn)
    pf.GUI.update()
    pf.app.processEvents()
    return fn


def askNewFilename(cur=None,filter="All files (*.*)"):
    """Ask a single new filename.

    This is a convenience function for calling askFilename with the
    arguments exist=False.
    """
    return askFilename(cur=cur,filter=filter,exist=False,multi=False)


def askDirname(path=None,change=True):
    """Interactively select a directory and change the current workdir.

    The user is asked to select a directory through the standard file
    dialog. Initially, the dialog shows all the subdirectories in the
    specified path, or by default in the current working directory.

    The selected directory becomes the new working directory, unless the
    user canceled the operation, or the change parameter was set to False.
    """
    if path is None:
        path = pf.cfg['workdir']
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    fn = widgets.FileSelection(path,'*',dir=True).getFilename()
    if fn and change:
        chdir(fn)
    pf.GUI.update()
    #pf.canvas.update()
    pf.app.processEvents()
    return fn


def askImageFile(fn=None):
    if not fn:
        fn = pf.cfg['pyformexdir']
    filt = map(utils.fileDescription,['img','all'])
    return askFilename(fn,filter=filt,multi=False,exist=True)


def checkWorkdir():
    """Ask the user to change the current workdir if it is not writable.

    Returns True if the new workdir is writable.
    """
    workdir = os.getcwd()
    ok = os.access(workdir,os.W_OK)
    if not ok:
        warning("Your current working directory (%s) is not writable. Change your working directory to a path where you have write permission." % workdir)
        askDirname()
        ok = os.access(os.getcwd(),os.W_OK)
    return ok
    

logfile = None     # the log file
    

def printMessage(s):
    """Print a message on the message board.

    If a logfile was opened, the message is also written to the log file.
    """
    if logfile is not None:
        logfile.write(str(s)+'\n')
    pf.GUI.board.write(str(s))
    pf.GUI.update()
    pf.app.processEvents()

# message is the preferred function to send text info to the user.
# The default message handler is set here.
message = printMessage


############################## drawing functions ########################

def flatten(objects,recurse=True):
    """Flatten a list of geometric objects.

    Each item in the list should be either:

    - a drawable object,
    - a string with the name of such an object,
    - a list of any of these three.

    This function will flatten the lists and replace the string items with
    the object they point to. The result is a single list of drawable
    objects. This function does not enforce the objects to be drawable.
    That should be done by the caller.
    """
    r = []
    for i in objects:
        if type(i) == str:
            i = named(i)
        if type(i) == list:
            if recurse:
                r.extend(flatten(i,True))
            else:
                r.extend(i)
        else:
            r.append(i)
    return r


def drawable(objects):
    """Filters the drawable objects from a list.

    The input is a list, usually of drawable objects. For each item in the
    list, the following is done:

    - if the item is drawable, it is kept as is,
    - if the item is not drawable but can be converted to a Formex, it is
      converted,
    - if it is neither drawable nor convertible to Formex, it is removed.

    The result is a list of drawable objects (since a Formex is drawable).
    """
    def fltr(i):
        if hasattr(i,'actor'):
            return i
        elif hasattr(i,'toFormex'):
            return i.toFormex()
        else:
            return None
    r = [ fltr(i) for i in objects ]
    return [ i for i in r if i is not None ]
    

def draw(F,
         view=None,bbox=None,
         color='prop',colormap=None,bkcolor=None,bkcolormap=None,alpha=None,
         mode=None,linewidth=None,linestipple=None,shrink=None,marksize=None,
         wait=True,clear=None,allviews=False,
         highlight=False,nolight=False,ontop=False,
         silent=True, **kargs):
    """Draw object(s) with specified settings and direct camera to it.

    The first argument is an object to be drawn. All other arguments are
    settings that influence how  the object is being drawn.

    F is one of:

    - a drawable object (a geometry object like Formex, Mesh or TriSurface),
    - the name of a global pyFormex variable refering to such an object,
    - a list of any of these three items.

    The variables are replaced with their value and the lists are flattened,
    to create a single list of objects. This then filtered for the drawable
    objects, and the resulting list of drawables is drawn using the remaining
    arguments.

    The remaining arguments are drawing options. If None, they are filled in
    from the current viewport drawing options.
    
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
    The special value bbox='auto' will use the bounding box of the objects
    getting drawn (object.bbox()), thus ensuring that the camera will focus
    on these objects.
    The special value bbox=None will use the bounding box of the previous
    drawing operation, thus ensuring that the camera's target volume remains
    unchanged.

    color,colormap,linewidth,alpha,marksize are passed to the
    creation of the 3D actor.

        
    if color is None, it is drawn with the color specified on creation.
    if color == 'prop' and a colormap was installed, props define color.
    else, color should be an array of RGB values, either with shape
    (3,) for a single color, or (nelems,3) for differently colored
    elements 


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
    if 'flat' in kargs:
        utils.warn('warn_flat_removed',DeprecationWarning,stacklevel=2)
        
    # For simplicity of the code, put objects to draw always in a list
    if type(F) is list:
        FL = F
    else:
        FL = [ F ]

    # Flatten the list, replacing named objects with their value
    FL = flatten(FL)
    ntot = len(FL)

    # Transform to list of drawable objects
    FL = drawable(FL)
    nres = len(FL)

    if nres < ntot and not silent:
        raise ValueError,"Data contains undrawable objects (%s/%s)" % (ntot-nres,ntot)

    # Fill in the remaining defaults

    if bbox is None:
        bbox = pf.canvas.options.get('bbox','auto')

    if shrink is None:
        shrink = pf.canvas.options.get('shrink',None)
 
    if marksize is None:
        marksize = pf.canvas.options.get('marksize',pf.cfg.get('marksize',5.0))

    if alpha is None:
        alpha = pf.canvas.options.get('alpha',0.5)

    # Shrink the objects if requested
    if shrink is not None:
        FL = [ _shrink(F,shrink) for F in FL ]

    # Execute the drawlock wait before doing first canvas change
    pf.GUI.drawlock.wait()

    if clear is None:
        clear = pf.canvas.options.get('clear',False)
    if clear:
        clear_canvas()

    if view is not None and view != 'last':
        pf.debug("SETTING VIEW to %s" % view)
        setView(view)

    pf.GUI.setBusy()
    pf.app.processEvents()

    try:
       
        actors = []

        # loop of the objects
        for F in FL:

            # Create the colors
            if color == 'prop':
                try:
                    Fcolor = F.prop
                except:
                    Fcolor = colors.black
            elif color == 'random':
                # create random colors
                Fcolor = numpy.random.rand(F.nelems(),3)
            else:
                Fcolor = color

            # Create the actor
            actor = F.actor(color=Fcolor,colormap=colormap,bkcolor=bkcolor,bkcolormap=bkcolormap,alpha=alpha,mode=mode,linewidth=linewidth,linestipple=linestipple,marksize=marksize,nolight=nolight,ontop=ontop,**kargs)
            
            actors.append(actor)
            
            if actor is not None:
                # Show the actor
                if highlight:
                    pf.canvas.addHighlight(actor)
                else:
                    pf.canvas.addActor(actor)

        # Adjust the camera
        #print "adjusting camera"
        #print "VIEW = %s; BBOX = %s" % (view,bbox)
        if view is not None or bbox not in [None,'last']:
            if view == 'last':
                view = pf.canvas.options['view']
            if bbox == 'auto':
                bbox = coords.bbox(FL)
            #print("SET CAMERA TO: bbox=%s, view=%s" % (bbox,view))
            pf.canvas.setCamera(bbox,view)
            #setView(view)

        pf.canvas.update()
        pf.app.processEvents()
        #pf.debug("AUTOSAVE %s" % image.autoSaveOn())
        if image.autoSaveOn():
            image.saveNext()
        if wait: # make sure next drawing operation is retarded
            pf.GUI.drawlock.lock()
    finally:
        pf.GUI.setBusy(False)
        
    if type(F) is list or len(actors) != 1:
        return actors
    else:
        return actors[0]


def olddraw(F,
         view=None,bbox=None,
         color='prop',colormap=None,bkcolor=None,bkcolormap=None,alpha=None,
         mode=None,linewidth=None,linestipple=None,shrink=None,marksize=None,
         wait=True,clear=None,allviews=False,
         highlight=False,nolight=False,ontop=False,**kargs):
    """_Old recursive draw function"""
    if 'flat' in kargs:
        import warnings
        warnings.warn('warn_flat_removed',DeprecationWarning,stacklevel=2)
        
    # Facility for drawing database objects by name
    if type(F) == str:
        F = named(F)
        if F is None:
            return None

    # We need to get the default for bbox before processing a list,
    # because bbox should be set only once for the whole list of objects
    if bbox is None:
        bbox = pf.canvas.options.get('bbox','auto')
        
    if type(F) == list:
        actor = []
        nowait = False
        save_bbox = bbox
        for Fi in F:
            if Fi is F[-1]:
                nowait = wait
            actor.append(olddraw(Fi,view,bbox,color,colormap,bkcolor,bkcolormap,alpha,mode,linewidth,linestipple,shrink,marksize,wait=nowait,clear=clear,allviews=allviews,highlight=highlight,nolight=nolight,ontop=ontop,**kargs))
            if Fi is F[0]:
                clear = False
                view = None
                bbox = 'last'

        bbox = save_bbox
        if bbox == 'auto':
            bbox = coords.bbox(actor)
            pf.canvas.setCamera(bbox,view)
            pf.canvas.update()
                
        return actor           

    # We now should have a single object to draw
    # Check if it is something we can draw

    
    if not hasattr(F,'actor') and hasattr(F,'toFormex'):
        pf.debug("CONVERTING %s TO FORMEX TO ENABLE DRAWING" %  type(F))
        F = F.toFormex()

    if not hasattr(F,'actor'):
        # Don't know how to draw this object
        raise RuntimeError,"draw() can not draw objects of type %s" % type(F)

    # Fill in the remaining defaults
    if shrink is None:
        shrink = pf.canvas.options.get('shrink',None)
 
    if marksize is None:
        marksize = pf.canvas.options.get('marksize',pf.cfg.get('marksize',5.0))

    if alpha is None:
        alpha = pf.canvas.options.get('alpha',0.5)
       
    # Create the colors
    if color == 'prop':
        if hasattr(F,'p'):
            color = F.prop
        elif hasattr(F,'prop'):
            color = F.prop
        else:
            color = colors.black
    elif color == 'random':
        # create random colors
        color = numpy.random.rand(F.nelems(),3)

    pf.GUI.drawlock.wait()

    if clear is None:
        clear = pf.canvas.options.get('clear',False)
    if clear:
        clear_canvas()

    if view is not None and view != 'last':
        pf.debug("SETTING VIEW to %s" % view)
        setView(view)

    pf.GUI.setBusy()
    pf.app.processEvents()
    if shrink is not None:
        F = _shrink(F,shrink)

    try:
        actor = F.actor(color=color,colormap=colormap,bkcolor=bkcolor,bkcolormap=bkcolormap,alpha=alpha,mode=mode,linewidth=linewidth,linestipple=linestipple,marksize=marksize,nolight=nolight,ontop=ontop,**kargs)
        
        if actor is None:
            return None
        
        if highlight:
            pf.canvas.addHighlight(actor)
        else:
            pf.canvas.addActor(actor)
            if view is not None or bbox not in [None,'last']:
                pf.debug("CHANGING VIEW to %s" % view)
                if view == 'last':
                    view = pf.canvas.options['view']
                if bbox == 'auto':
                    bbox = F.bbox()
                pf.debug("SET CAMERA TO: bbox=%s, view=%s" % (bbox,view))
                pf.canvas.setCamera(bbox,view)
                #setView(view)

        pf.canvas.update()
        pf.app.processEvents()
        #pf.debug("AUTOSAVE %s" % image.autoSaveOn())
        if image.autoSaveOn():
            image.saveNext()
        if wait: # make sure next drawing operation is retarded
            pf.GUI.drawlock.lock()
    finally:
        pf.GUI.setBusy(False)
    return actor

if pf.options.olddraw:
    draw = olddraw


def _setFocus(object,bbox,view):
    """Set focus after a draw operation"""
    if view is not None or bbox not in [None,'last']:
        if view == 'last':
            view = pf.canvas.options['view']
        if bbox == 'auto':
            bbox = coords.bbox(object)
        pf.canvas.setCamera(bbox,view)
    pf.canvas.update()


def focus(object):
    """Move the camera thus that object comes fully into view.

    object can be anything having a bbox() method or a list thereof.
    if no view is given, the default is used.

    The camera is moved with fixed axis directions to a place
    where the whole object can be viewed using a 45. degrees lens opening.
    This technique may change in future!
    """
    pf.canvas.setCamera(bbox=coords.bbox(object))
    pf.canvas.update()

    
def setDrawOptions(kargs0={},**kargs):
    """Set default values for the draw options.

    Draw options are a set of options that hold default values for the
    draw() function arguments and for some canvas settings.
    The draw options can be specified either as a dictionary, or as
    keyword arguments. 
    """
    d = {}
    d.update(kargs0)
    d.update(kargs)
    pf.canvas.setOptions(d)

    
def showDrawOptions():
    pf.message("Current Drawing Options: %s" % pf.canvas.options)
    pf.message("Current Viewport Options: %s" % pf.canvas.settings)


def askDrawOptions(d={}):
    """Interactively ask the Drawing options from the user.
    
    A dictionary may be specified to override the current defaults.
    """
    setDrawOptions(d)
    res = askItems(pf.canvas.options.items())
    setDrawOptions(res)


def reset():
    pf.canvas.resetDefaults()
    pf.canvas.resetOptions()
    pf.GUI.drawwait = pf.cfg['draw/wait']
    clear()
    view('front')


def resetAll():
    wireframe()
    reset()


def shrink(v):
    setDrawOptions({'shrink':v})


def _shrink(F,factor):
    """Return a shrinked object.

    A shrinked object is one where each element is shrinked with a factor
    around its own center.
    """
    if not isinstance(F,Formex):
        F = F.toFormex()
    return F.shrink(factor)


def drawVectors(P,v,size=None,nolight=True,**drawOptions):
    """Draw a set of vectors.

    If size==None, draws the vectors v at the points P.
    If size is specified, draws the vectors size*normalize(v)
    P, v and size are single points
    or sets of points. If sets, they should be of the same size.

    Other drawoptions can be specified and will be passed to the draw function.
    """
    if size is None:
        Q = P + v
    else:
        Q = P + size*normalize(v)
    return draw(connect([P,Q]),nolight=nolight,**drawOptions)


def drawPlane(P,N,size):
    from plugins.tools import Plane
    p = Plane(P,N,size)
    return draw(p,bbox='last')


def drawMarks(X,M,color='black',leader='',ontop=True):
    """Draw a list of marks at points X.

    X is a Coords array.
    M is a list with the same length as X.
    The string representation of the marks are drawn at the corresponding
    3D coordinate.
    """
    _large_ = 20000
    if len(M) > _large_:
        if not ack("You are trying to draw marks at %s points. This may take a long time, and the results will most likely not be readible anyway. If you insist on drawing these marks, anwer YES." % len(M)):
            return None
    M = marks.MarkList(X,M,color=color,leader=leader,ontop=ontop)
    pf.canvas.addAnnotation(M)
    pf.canvas.numbers = M
    pf.canvas.update()
    return M


def drawFreeEdges(M,color='black'):
    """Draw the feature edges of a Mesh"""
    print "DRAW FREE EDGES"
    B = M.getFreeEdgesMesh()
    print B
    draw(B,color=color,nolight=True)
    

def drawNumbers(F,numbers=None,color='black',trl=None,offset=0,leader='',ontop=True):
    """Draw numbers on all elements of F.

    numbers is an array with F.nelems() integer numbers.
    If no numbers are given, the range from 0 to nelems()-1 is used.
    Normally, the numbers are drawn at the centroids of the elements.
    A translation may be given to put the numbers out of the centroids,
    e.g. to put them in front of the objects to make them visible,
    or to allow to view a mark at the centroids.
    If an offset is specified, it is added to the shown numbers.
    """
    try:
        X = F.centroids()
    except:
        return None
    if trl is not None:
        X = X.trl(trl)
    X = X.reshape(-1,3)
    if numbers is None:
        numbers = numpy.arange(X.shape[0])
    return drawMarks(X,numbers+offset,color=color,leader=leader,ontop=ontop)
    

def drawPropNumbers(F,**kargs):
    """Draw property numbers on all elements of F.

    This calls drawNumbers to draw the property numbers on the elements.
    All arguments of drawNumbers except `numbers` may be passed.
    If the object F thus not have property numbers, -1 values are drawn.
    """
    if F.prop is None:
        nrs = [ -1 ] * self.nelems()
    else:
        nrs = F.prop
    drawNumbers(F,nrs,**kargs)
                

def drawVertexNumbers(F,color='black',trl=None,ontop=False):
    """Draw (local) numbers on all vertices of F.

    Normally, the numbers are drawn at the location of the vertices.
    A translation may be given to put the numbers out of the location,
    e.g. to put them in front of the objects to make them visible,
    or to allow to view a mark at the vertices.
    """
    FC = F.coords.reshape((-1,3))
    if trl is not None:
        FC = FC.trl(trl)
    return drawMarks(FC,numpy.resize(numpy.arange(F.coords.shape[-2]),(FC.shape[0])),color=color,ontop=ontop)


def drawText3D(P,text,color=None,font='sans',size=18,ontop=True):
    """Draw a text at a 3D point P."""
    M = marks.TextMark(P,text,color=color,font=font,size=size,ontop=ontop)
    pf.canvas.addAnnotation(M)
    pf.canvas.update()
    return M


def drawAxes(*args,**kargs):
    """Draw the axes of a CoordinateSystem.

    This draws an AxesActor corresponding to the specified Coordinatesystem.
    The arguments are the same as those of the AxesActor constructor.
    """
    A = actors.AxesActor(*args,**kargs)
    drawActor(A)
    return A
        

def drawImage(image,nx=-1,ny=-1,pixel='dot'):
    """Draw an image as a colored Formex

    Draws a raster image as a colored Formex. While there are other and
    better ways to display an image in pyFormex (such as using the imageView
    widget), this function allows for interactive handling the image using
    the OpenGL infrastructure.

    Parameters:

    - `image`: a QImage holding a raster image. An image can be loaded from
      most standard image files using the :func:`loadImage` function
    - `nx`,`ny`: resolution you want to use for the display
    - `pixel`: the Formex representing a single pixel. It should be either
      a single element Formex, or one of the strings 'dot' or 'quad'. If 'dot'
      a single point will be used, if 'quad' a unit square. The difference
      will be important when zooming in. The default is 'dot'. 

    """
    pf.GUI.setBusy()
    from gui.imagecolor import image2glcolor
    w,h = image.width(),image.height()
    if nx <= 0:
        nx = w
    if ny <= 0:
        ny = h
    if nx != w or ny != h:
        image = image.scaled(nx,ny)
    # Create the colors
    color,colortable = image2glcolor(image)

    # Create a 2D grid of nx*ny elements
    # !! THIS CAN PROBABLY BE DONE FASTER
    if isinstance(pixel,Formex) and pixel.nelems()==1:
        F = pixel
    elif pixel == 'quad':
        F = Formex('4:0123')
    else:
        F = Formex(origin())
    F = F.replic2(nx,ny).centered()

    # Draw the grid using the image colors
    FA = draw(F,color=color,colormap=colortable,nolight=True)
    pf.GUI.setBusy(False)
    return FA


def drawViewportAxes3D(pos,color=None):
    """Draw two viewport axes at a 3D position."""
    M = marks.AxesMark(pos,color)
    annotate(M)
    return M


def drawBbox(A):
    """Draw the bbox of the actor A."""
    B = actors.BboxActor(A.bbox())
    annotate(B)
    return B


def drawActor(A):
    """Draw an actor and update the screen."""
    pf.canvas.addActor(A)
    pf.canvas.update()


def undraw(itemlist):
    """Remove an item or a number of items from the canvas.

    Use the return value from one of the draw... functions to remove
    the item that was drawn from the canvas.
    A single item or a list of items may be specified.
    """
    pf.canvas.remove(itemlist)
    pf.canvas.update()
    pf.app.processEvents()
    

def view(v,wait=True):
    """Show a named view, either a builtin or a user defined.

    This shows the current scene from another viewing angle.
    Switching views of a scene is much faster than redrawing a scene.
    Therefore this function is prefered over :func:`draw` when the actors
    in the scene remain unchanged and only the camera viewpoint changes.

    Just like :func:`draw`, this function obeys the drawing lock mechanism,
    and by default it will restart the lock to retard the next draing operation.
    """
    pf.GUI.drawlock.wait()
    if v != 'last':
        angles = pf.canvas.view_angles.get(v)
        if not angles:
            warning("A view named '%s' has not been created yet" % v)
            return
        pf.canvas.setCamera(None,angles)
    setView(v)
    pf.canvas.update()
    if wait:
        pf.GUI.drawlock.lock()


def setTriade(on=None,pos='lb',siz=100):
    """Toggle the display of the global axes on or off.

    If on is True, the axes triade is displayed, if False it is
    removed. The default (None) toggles between on and off.
    """
    pf.canvas.setTriade(on,pos,siz)
    pf.canvas.update()
    pf.app.processEvents()


def drawText(text,x,y,gravity='E',font='helvetica',size=14,color=None,zoom=None):
    """Show a text at position x,y using font."""
    TA = decors.Text(text,x,y,gravity=gravity,font=font,size=size,color=color,zoom=zoom)
    decorate(TA)
    return TA

def annotate(annot):
    """Draw an annotation."""
    pf.canvas.addAnnotation(annot)
    pf.canvas.update()

def unannotate(annot):
    pf.canvas.removeAnnotation(annot)
    pf.canvas.update()

def decorate(decor):
    """Draw a decoration."""
    pf.canvas.addDecoration(decor)
    pf.canvas.update()

def undecorate(decor):
    pf.canvas.removeDecoration(decor)
    pf.canvas.update()


def createView(name,angles,addtogui=False):
    """Create a new named view (or redefine an old).

    The angles are (longitude, latitude, twist).
    By default, the view is local to the script's viewport.
    If gui is True, it is also added to the GUI.
    """
    pf.canvas.view_angles[name] = angles
    if addtogui:
        pf.GUI.createView(name,angles)
    

def setView(name,angles=None):
    """Set the default view for future drawing operations.

    If no angles are specified, the name should be an existing view, or
    the predefined value 'last'.
    If angles are specified, this is equivalent to createView(name,angles)
    followed by setView(name).
    """
    if name != 'last' and angles:
        createView(name,angles)
    setDrawOptions({'view':name})


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


def bgcolor(color,color2=None,mode='h'):
    """Change the background color (and redraw).

    If one color is given, the background is a solid color.
    If two colors are given, the background color will get a vertical
    gradient with color on top and color2 at the bottom.
    """
    pf.canvas.setBgColor(color,color2,mode)
    pf.canvas.display()
    pf.canvas.update()


def fgcolor(color):
    """Set the default foreground color."""
    pf.canvas.setFgColor(color)


def colormap(color=None):
    """Gets/Sets the current canvas color map"""
    return pf.canvas.settings.colormap


def renderMode(mode,avg=False,light=None):
    #print "DRAW.RENDERMODE"
    #print "CANVAS %s" % pf.canvas
    #print "MODE %s" % pf.canvas.rendermode
    # ERROR The following redraws twice !!!
    pf.canvas.setRenderMode(mode,light)
    #print "NEW MODE %s" % pf.canvas.rendermode
    pf.canvas.update()
    toolbar.updateNormalsButton()
    toolbar.updateLightButton()
    pf.GUI.processEvents()
    #print "DONE DRAW>RENDERMODE"
    
    
def wireframe():
    renderMode("wireframe")
    
def smooth():
    renderMode("smooth")

def smoothwire():
    renderMode("smoothwire")
    
def flat():
    renderMode("flat")
    
def flatwire():
    renderMode("flatwire")
    
def smooth_avg():
    renderMode("smooth",True)

## def opacity(alpha):
##     """Set the viewports transparency."""
##     pf.canvas.alpha = float(alpha)

def lights(state=True):
    """Set the lights on or off"""
    pf.canvas.setLighting(state)
    pf.canvas.update()
    toolbar.updateLightButton()
    pf.GUI.processEvents()


def transparent(state=True):
    """Set the transparency mode on or off."""
    pf.canvas.setToggle('alphablend',state)
    pf.canvas.update()
    toolbar.updateTransparencyButton()
    pf.GUI.processEvents()


def perspective(state=True):
    pf.canvas.camera.setPerspective(state)
    pf.canvas.update()
    toolbar.updatePerspectiveButton()
    pf.GUI.processEvents()

    
def timeout(state=None):
    toolbar.timeout(state)


def set_material_value(typ,val):
    """Set the value of one of the material lighting parameters

    typ is one of 'ambient','specular','emission','shininess'
    val is a value between 0.0 and 1.0
    """
    setattr(pf.canvas,typ,val)
    pf.canvas.setLighting(True)
    pf.canvas.update()
    pf.app.processEvents()

def set_light(light,**args):
    light = int(light)
    pf.canvas.lights.set(light,**args)
    pf.canvas.setLighting(True)
    pf.canvas.update()
    pf.app.processEvents()

def set_light_value(light,key,val):
    light = int(light)
    pf.canvas.lights.set_value(light,key,val)
    pf.canvas.setLighting(True)
    pf.canvas.update()
    pf.app.processEvents()


def linewidth(wid):
    """Set the linewidth to be used in line drawings."""
    pf.canvas.setLineWidth(wid)

def linestipple(factor,pattern):
    """Set the linewidth to be used in line drawings."""
    pf.canvas.setLineStipple(factor,pattern)

def pointsize(siz):
    """Set the size to be used in point drawings."""
    pf.canvas.setPointSize(siz)


def canvasSize(width,height):
    """Resize the canvas to (width x height)."""
    pf.canvas.resize(width,height)


def clear_canvas():
    """Clear the canvas.

    This is a low level function not intended for the user.
    """
    pf.canvas.removeAll()
    pf.canvas.clear()


def clear():
    """Clear the canvas.

    Removes everything from the current scene and displays an empty
    background.

    This function waits for the drawing lock to be released, but will
    not reset it.
    """
    pf.GUI.drawlock.wait()
    clear_canvas()
    pf.canvas.update()


def redraw():
    pf.canvas.redrawAll()
    pf.canvas.update()


def delay(s=None):
    """Get/Set the draw delay time.

    Returns the current setting of the draw wait time (in seconds).
    This drawing delay is obeyed by drawing and viewing operations.
    
    A parameter may be given to set the delay time to a new value.
    It should be convertable to a float.
    The function still returns the old setting. This may be practical
    to save that value to restore it later.
    """
    saved = pf.GUI.drawwait
    if s is not None:
        pf.GUI.drawwait = float(s)
    return saved


def wait(relock=True):
    """Wait until the drawing lock is released.

    This uses the drawing lock mechanism to pause. The drawing lock
    ensures that subsequent draws are retarded to give the user the time
    to view. This use of this function is prefered over that of
    :func:`pause` or :func:`sleep`, because it allows your script to
    continue the numerical computations while waiting to draw the next
    screen.

    This function can be used to retard other functions than `
    """
    pf.GUI.drawlock.wait()
    if relock:
        pf.GUI.drawlock.lock()
        
    

def fforward():
    """Releases the drawing lock mechanism indefinely.

    Releasing the drawing lock indefinely means that the lock will not
    be set again and your script will execute till the end.
    """
    pf.GUI.drawlock.free()


def step():
    """Perform one step of a script.

    A step is a set of instructions until the next draw operation.
    If a script is running, this just releases the draw lock.
    Else, it starts the script in step mode.
    """
    if pf.GUI.drawlock.locked:
        pf.GUI.drawlock.release()
    else:
        if ack("""
STEP MODE is currently only possible with specially designed,
very well behaving scripts. If you're not sure what you are
doing, you should cancel the operation now.

Are you REALLY SURE you want to run this script in step mode?
"""):
            play(step=True)
 
    

def pause(msg="Use the Step or Continue button to proceed",timeout=None):
    """Pause the execution until an external event occurs or timeout.

    When the pause statement is executed, execution of the pyformex script
    is suspended until some external event forces it to proceed again.
    Clicking the STEP or CONTINUE button will produce such an event.
    """
    from gui.drawlock import repeat

    def _continue_():
        return pf.GUI.drawlock.locked

    pf.debug("PAUSE ACTIVATED!")
    if msg:
        pf.message(msg)

    pf.GUI.drawlock.release()
    if pf.GUI.drawlock.allowed:
        pf.GUI.drawlock.locked = True
    if timeout is None:
        timeout = widgets.input_timeout
    repeat(_continue_,timeout)
   


################### EXPERIMENTAL STUFF: AVOID! ###############
_wakeup_mode=0
sleeping = False
timer = None
def sleep(timeout=None):
    """Sleep until key/mouse press in the canvas or until timeout"""
    utils.warn('warn_avoid_sleep')
    #
    global sleeping,_wakeup_mode,timer
    if _wakeup_mode > 0 or timeout == 0:  # don't bother
        return
    # prepare for getting wakeup event 
    onSignal(WAKEUP,wakeup)
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
        pf.app.processEvents()
        # And we have to sleep in between, or we would be using to much
        # processor time idling. 0.1 is a good compromise to get some
        # responsitivity while not pegging the cpu
        time.sleep(0.01)
    # ignore further wakeup events
    offSignal(WAKEUP,wakeup)

        
def wakeup(mode=0):
    """Wake up from the sleep function.

    This is the only way to exit the sleep() function.
    Default is to wake up from the current sleep. A mode > 0
    forces wakeup for longer period.
    """
    global timer,sleeping,_wakeup_mode
    if timer:
        timer.cancel()
    sleeping = False
    _wakeup_mode = mode


########################## print information ################################


def printbbox():
    pf.message(pf.canvas.bbox)

def printviewportsettings():
    pf.GUI.viewports.printSettings()
    
def reportCamera():
    print(pf.canvas.camera.report())


#################### camera ##################################

def zoom_factor(factor=None):
    if factor is None:
        factor = pf.cfg['gui/zoomfactor']
    return float(factor)

def pan_factor(factor=None):
    if factor is None:
        factor = pf.cfg['gui/panfactor']
    return float(factor)

def rot_factor(factor=None):
    if factor is None:
        factor = pf.cfg['gui/rotfactor']
    return float(factor)

def zoomIn(factor=None):
    pf.canvas.camera.zoomArea(1./zoom_factor(factor))
    pf.canvas.update()
def zoomOut(factor=None):
    pf.canvas.camera.zoomArea(zoom_factor(factor))
    pf.canvas.update()    
    
def panRight(factor=None):
    pf.canvas.camera.transArea(-pan_factor(factor),0.)
    pf.canvas.update()   
def panLeft(factor=None):
    pf.canvas.camera.transArea(pan_factor(factor),0.)
    pf.canvas.update()   
def panUp(factor=None):
    pf.canvas.camera.transArea(0.,-pan_factor(factor))
    pf.canvas.update()   
def panDown(factor=None):
    pf.canvas.camera.transArea(0.,pan_factor(factor))
    pf.canvas.update()
    
def rotRight(factor=None):
    pf.canvas.camera.rotate(rot_factor(factor),0,1,0)
    pf.canvas.update()   
def rotLeft(factor=None):
    pf.canvas.camera.rotate(-rot_factor(factor),0,1,0)
    pf.canvas.update()   
def rotUp(factor=None):
    pf.canvas.camera.rotate(-rot_factor(factor),1,0,0)
    pf.canvas.update()   
def rotDown(factor=None):
    pf.canvas.camera.rotate(rot_factor(factor),1,0,0)
    pf.canvas.update()   
def twistLeft(factor=None):
    pf.canvas.camera.rotate(rot_factor(factor),0,0,1)
    pf.canvas.update()   
def twistRight(factor=None):
    pf.canvas.camera.rotate(-rot_factor(factor),0,0,1)
    pf.canvas.update()
    
def transLeft(factor=None):
    val = pan_factor(factor) * pf.canvas.camera.getDist()
    pf.canvas.camera.translate(-val,0,0,pf.cfg['draw/localaxes'])
    pf.canvas.update()
def transRight(factor=None):
    val = pan_factor(factor) * pf.canvas.camera.getDist()
    pf.canvas.camera.translate(+val,0,0,pf.cfg['draw/localaxes'])
    pf.canvas.update()
def transDown(factor=None):
    val = pan_factor(factor) * pf.canvas.camera.getDist()
    pf.canvas.camera.translate(0,-val,0,pf.cfg['draw/localaxes'])
    pf.canvas.update()   
def transUp(factor=None):
    val = pan_factor(factor) * pf.canvas.camera.getDist()
    pf.canvas.camera.translate(0,+val,0,pf.cfg['draw/localaxes'])
    pf.canvas.update()
def dollyIn(factor=None):
    pf.canvas.camera.dolly(1./zoom_factor(factor))
    pf.canvas.update()
def dollyOut(factor=None):
    pf.canvas.camera.dolly(zoom_factor(factor))
    pf.canvas.update()

def lockCamera():
    pf.canvas.camera.lock()
def unlockCamera():
    pf.canvas.camera.lock(False)


def zoomRectangle():
    """Zoom a rectangle selected by the user."""
    pf.canvas.start_rectangle_zoom()
    pf.canvas.update()

    
def zoomBbox(bb):
    """Zoom thus that the specified bbox becomes visible."""
    pf.canvas.setBbox(bb)
    pf.canvas.setCamera()
    pf.canvas.update()
    

def zoomAll():
    """Zoom thus that all actors become visible."""
    if pf.canvas.actors:
        zoomBbox(bbox(pf.canvas.actors))

# Can this be replaced with zoomIn/Out?
def zoom(f):
    pf.canvas.zoom(f)
    pf.canvas.update()


def flyAlong(path,upvector=[0.,1.,0.],sleeptime=None):
    """Fly through the current scene along the specified path.

    This function moves the camera through the subsequent points
    of a path, looing at the next point of the path, and keeping
    the upvector of the camera oriented along the specified direction.
    
    - `path`: a PolyLine or plex-2 Formex specifyin the camera path.
    - `upvector`: the direction of the vertical axis of the camera.
    - `sleeptime`: a delay between subsequent images, to slow down
      the camera movement.
    """
    from plugins.curve import PolyLine

    try:
        if not isinstance(path,Formex):
            path = path.toFormex() 
        if not path.nplex() in (2,3):
            raise ValueError
    except:
        raise ValueError,"The camera path should be (convertible to) a plex-2 or plex-3 Formex!"

    if sleeptime is None:
        sleeptime = pf.cfg['draw/flywait']
    saved = delay(sleeptime)
    saved1 = pf.GUI.actions['Continue'].isEnabled()
    pf.GUI.actions['Continue'].setEnabled(True)
    
    for eye,center in path:
        pf.debug("Eye: %s; Center: %s" % (eye,center))
        pf.canvas.camera.lookAt(eye,center,upvector)
        wait()
        pf.canvas.display()
        pf.canvas.update()
        image.saveNext()

    delay(saved)
    pf.GUI.actions['Continue'].setEnabled(saved1)
    pf.canvas.camera.setCenter(*center)
    pf.canvas.camera.setDist(length(center-eye))
    pf.canvas.update()


#################### viewports ##################################

### BEWARE FOR EFFECTS OF SPLITTING pf.canvas and pf.canvas if these
### are called from interactive functions!
   

def viewport(n=None):
    """Select the current viewport.

    n is an integer number in the range of the number of viewports,
    or is one of the viewport objects in pyformex.GUI.viewports

    if n is None, selects the current GUI viewport for drawing
    """
    if n is not None:
        pf.canvas.update()
        pf.GUI.viewports.setCurrent(n)
    pf.canvas = pf.GUI.viewports.current


def layout(nvps=None,ncols=None,nrows=None,pos=None,rstretch=None,cstretch=None):
    """Set the viewports layout."""
    pf.GUI.viewports.changeLayout(nvps,ncols,nrows,pos,rstretch,cstretch)
    viewport()

def addViewport():
    """Add a new viewport."""
    pf.GUI.viewports.addView()
    viewport()

def removeViewport():
    """Remove the last viewport."""
    n = len(pf.GUI.viewports.all)
    if n > 1:
        pf.GUI.viewports.removeView()
    viewport()

def linkViewport(vp,tovp):
    """Link viewport vp to viewport tovp.

    Both vp and tovp should be numbers of viewports. 
    """
    pf.GUI.viewports.link(vp,tovp)
    viewport()

####################

def updateGUI():
    """Update the GUI."""
    pf.GUI.update()
    pf.canvas.update()
    pf.app.processEvents()


######### Highlighting ###############
    
def highlightActors(K):
    """Highlight a selection of actors on the canvas.

    K is Collection of actors as returned by the pick() method.
    colormap is a list of two colors, for the actors not in, resp. in
    the Collection K.
    """
    pf.canvas.removeHighlights()
    for i in K.get(-1,[]):
        pf.message("%s/%s" % (i,len(pf.canvas.actors)))
        actor = pf.canvas.actors[i]
        FA = actors.GeomActor(actor,color=pf.canvas.settings.slcolor)
        pf.canvas.addHighlight(FA)
    pf.canvas.update()


def highlightElements(K):
    """Highlight a selection of actor elements on the canvas.

    K is Collection of actor elements as returned by the pick() method.
    colormap is a list of two colors, for the elements not in, resp. in
    the Collection K.
    """
    pf.canvas.removeHighlights()
    for i in K.keys():
        pf.debug("Actor %s: Selection %s" % (i,K[i]))
        actor = pf.canvas.actors[i]
        FA = actors.GeomActor(actor.select(K[i]),color=pf.canvas.settings.slcolor,linewidth=3)
        pf.canvas.addHighlight(FA)
    pf.canvas.update()


def highlightEdges(K):
    """Highlight a selection of actor edges on the canvas.

    K is Collection of TriSurface actor edges as returned by the pick() method.
    colormap is a list of two colors, for the edges not in, resp. in
    the Collection K.
    """
    pf.canvas.removeHighlights()
    for i in K.keys():
        pf.debug("Actor %s: Selection %s" % (i,K[i]))
        actor = pf.canvas.actors[i]
        FA = actors.GeomActor(Formex(actor.coords[actor.object.getEdges()[K[i]]]),color=pf.canvas.settings.slcolor,linewidth=3)
        pf.canvas.addHighlight(FA)
            
    pf.canvas.update()


def highlightPoints(K):
    """Highlight a selection of actor elements on the canvas.

    K is Collection of actor elements as returned by the pick() method.
    """
    pf.canvas.removeHighlights()
    for i in K.keys():
        pf.debug("Actor %s: Selection %s" % (i,K[i]))
        actor = pf.canvas.actors[i]
        FA = actors.GeomActor(Formex(actor.vertices()[K[i]]),color=pf.canvas.settings.slcolor,marksize=10)
        pf.canvas.addHighlight(FA)
    pf.canvas.update()


def highlightPartitions(K):
    """Highlight a selection of partitions on the canvas.

    K is a Collection of actor elements, where each actor element is
    connected to a collection of property numbers, as returned by the
    partitionCollection() method.
    """
    pf.canvas.removeHighlights()
    for i in K.keys():
        pf.debug("Actor %s: Partitions %s" % (i,K[i][0]))
        actor = pf.canvas.actors[i]
        for j in K[i][0].keys():           
            FA = actors.GeomActor(actor.select(K[i][0][j]),color=j*numpy.ones(len(K[i][0][j]),dtype=int))
            pf.canvas.addHighlight(FA)
    pf.canvas.update()


highlight_funcs = { 'actor': highlightActors,
                    'element': highlightElements,
                    'point': highlightPoints,
                    'edge': highlightEdges,
                    }


def removeHighlights():
    """Remove the highlights from the current viewport"""
    pf.canvas.removeHighlights()
    pf.canvas.update()
    



def pick(mode='actor',filter=None,oneshot=False,func=None):
    """Enter interactive picking mode and return selection.

    See viewport.py for more details.
    This function differs in that it provides default highlighting
    during the picking operation, a button to stop the selection operation

    Parameters:

    - `mode`: one of the pick modes
    - `filter`: one of the `selection_filters`. The default picking filter
      activated on entering the pick mode. All available filters are
      presented in a combobox.
    """
    
    def _set_selection_filter(s):
        """Set the selection filter mode

        This function is used to change the selection filter from the
        selection InputCombo widget.
        s is one of the strings in selection_filters.
        """
        s = str(s)
        if pf.canvas.selection_mode is not None and s in pf.canvas.selection_filters:
            pf.canvas.start_selection(None,s)
        if pf.canvas.selection_mode is not None:
            warning("You need to finish the previous picking operation first!")
            return

    if mode not in pf.canvas.getPickModes():
        warning("Invalid picking mode: %s. Expected one of %s." % (mode,pf.canvas.getPickModes()))
        return

    pick_buttons = widgets.ButtonBox('Selection:',[('Cancel',pf.canvas.cancel_selection),('OK',pf.canvas.accept_selection)])
    
    if mode == 'element':
        filters = pf.canvas.selection_filters
    else:
        filters = pf.canvas.selection_filters[:3]
    filter_combo = widgets.InputCombo('Filter:',None,choices=filters,onselect=_set_selection_filter)
    if filter is not None and filter in pf.canvas.selection_filters:
        filter_combo.setValue(filter)
    
    if func is None:
        func = highlight_funcs.get(mode,None)
    pf.message("Select %s %s" % (filter,mode))

    pf.GUI.statusbar.addWidget(pick_buttons)
    pf.GUI.statusbar.addWidget(filter_combo)
    try:
        sel = pf.canvas.pick(mode,oneshot,func,filter)
    finally:
        # cleanup
        if pf.canvas.selection_mode is not None:
            pf.canvas.finish_selection()
        pf.GUI.statusbar.removeWidget(pick_buttons)
        pf.GUI.statusbar.removeWidget(filter_combo)
    return sel
 
    
def pickActors(filter=None,oneshot=False,func=None):
    return pick('actor',filter,oneshot,func)

def pickElements(filter=None,oneshot=False,func=None):
    return pick('element',filter,oneshot,func)

def pickPoints(filter=None,oneshot=False,func=None):
    return pick('point',filter,oneshot,func)

def pickEdges(filter=None,oneshot=False,func=None):
    return pick('edge',filter,oneshot,func)

def pickNumbers(marks=None):
    if marks:
        pf.canvas.numbers = marks
    return pf.canvas.pickNumbers()


def highlight(K,mode):
    """Highlight a Collection of actor/elements.

    K is usually the return value of a pick operation, but might also
    be set by the user.
    mode is one of the pick modes.
    """
    func = highlight_funcs.get(mode,None)
    if func:
        func(K)


LineDrawing = None
edit_modes = ['undo', 'clear','close']


def set_edit_mode(s):
    """Set the drawing edit mode."""
    s = str(s)
    if s in edit_modes:
        pf.canvas.edit_drawing(s)


def drawLinesInter(mode ='line',single=False,func=None):
    """Enter interactive drawing mode and return the line drawing.

    See viewport.py for more details.
    This function differs in that it provides default displaying
    during the drawing operation and a button to stop the drawing operation.

    The drawing can be edited using the methods 'undo', 'clear' and 'close', which
    are presented in a combobox.
    """
    if pf.canvas.drawing_mode is not None:
        warning("You need to finish the previous drawing operation first!")
        return
    if func == None:
        func = showLineDrawing
    drawing_buttons = widgets.ButtonBox('Drawing:',[('Cancel',pf.canvas.cancel_drawing),('OK',pf.canvas.accept_drawing)])
    pf.GUI.statusbar.addWidget(drawing_buttons)
    edit_combo = widgets.InputCombo('Edit:',None,choices=edit_modes,onselect=set_edit_mode)
    pf.GUI.statusbar.addWidget(edit_combo)
    lines = pf.canvas.drawLinesInter(mode,single,func)
    pf.GUI.statusbar.removeWidget(drawing_buttons)
    pf.GUI.statusbar.removeWidget(edit_combo)
    return lines


def showLineDrawing(L):
    """Show a line drawing.

    L is usually the return value of an interactive draw operation, but
    might also be set by the user.
    """
    global LineDrawing
    if LineDrawing:
        undecorate(LineDrawing)
        LineDrawing = None
    if L.size != 0:
        LineDrawing = decors.LineDrawing(L,color='yellow',linewidth=3)
        decorate(LineDrawing)


################################

def setLocalAxes(mode=True):
    pf.cfg['draw/localaxes'] = mode 
def setGlobalAxes(mode=True):
    setLocalAxes(not mode)
         


#  deprecated alternative spellings
from utils import deprecated,deprecation
@deprecated(zoomAll)
def zoomall(*args,**kargs):
    pass
@deprecated(drawText)
def drawtext(*args,**kargs):
    pass

@deprecation("`drawNormals` is deprecated: use `drawVectors` instead.\nNotice that the argument order is different!")
def drawNormals(v,P,size=None,**drawOptions):
    drawVectors(P,v,size=size,**drawOptions)


#### End
