## $Id$
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
"""Create 3D graphical representations.

The draw module provides the basic user interface to the OpenGL
rendering capabilities of pyFormex.
"""

import pyformex as pf

import threading,os,sys,types,copy,commands,time

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
import coords
from formex import Formex

from script import *
from signals import *

from plugins import trisurface,tools,mesh,fe


        
#################### Interacting with the user ###############################

def closeGui():
    pf.debug("Closing the GUI: currently, this will also terminate pyformex.")
    pf.GUI.close()


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
    """
    w = widgets.TextBox(text,type,actions,modal=modal,mono=mono)
    if modal:
        return w.getResult()
    else:
        w.show()
        return None


def showFile(filename,mono=False):
    """Display a text file.

    This will use the showText() function to display a text read from a
    file. 
    """
    try:
        f = file(filename,'r')
    except IOError:
        return
    showText(f.read(),mono=mono)
    f.close()


# widget and result status of the widget in askItems() function
_dialog_widget = None
_dialog_result = None

def askItems(items,caption=None,timeout=None,legacy=True,**kargs):
    """Ask the value of some items to the user.

    Create an interactive widget to let the user set the value of some items.
    Input is a list of input items (basically [key,value] pairs).
    See the widgets.InputDialog class for complete description of the
    available input items.

    The remaining arguments are keyword arguments that are passed to the
    InputDialog.getResult method.
    A timeout (in seconds) can be specified to have the input dialog
    interrupted automatically.

    Return a dictionary with the results: for each input item there is a
    (key,value) pair. Returns an empty dictionary if the dialog was canceled.
    Sets the dialog timeout and accepted status in global variables.
    """
    global _dialog_widget,_dialog_result
    import warnings
    warnings.warn("""
The default operation of the askItems function will change in version 0.9!
It will then use the new InputDialog, meaning that the input items have
to be dicts.
Though the old InputDialog will still be available for some time when using the
'legacy = True' argument, we advice you to switch to the newer InputItem
format as soon as possible.
""")
    if legacy:
        w = widgets.OldInputDialog(items,caption,**kargs)
    else:
        w = widgets.NewInputDialog(items,caption,**kargs)
        
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
    #pf.debug("cur,fn: %s,%s" % (cur,fn))
    #print("MULTI=%s" % multi)
    w = widgets.FileSelection(cur,filter,exist,multi)
    if fn:
        w.selectFile(fn)
    fn = w.getFilename()
    #print("SELECTED=%s" % fn)
    if fn and change:
        if multi:
            chdir(fn[0])
        else:
            chdir(fn)
    pf.GUI.update()
    pf.canvas.update()
    pf.app.processEvents()
    return fn


def askNewFilename(cur=None,filter="All files (*.*)"):
    """Ask a single new filename.

    This is a convenience function for calling askFilename with the
    arguments exist=False.
    """
    return askFilename(cur=cur,filter=filter,exist=False,multi=False)


def askDirname(cur=None,change=True):
    """Ask for an existing directory name.

    The dialog pops up where the user can interactively select a directory.
    Initially, the dialog will show all the subdirectories in the specified
    path, or by default in the current workdir.
    
    Unless the user cancels the operation, or the change parameter was set to
    False, the selected directory will become the new  working directory.
    """
    if cur is None:
        cur = pf.cfg['workdir']
    cur = os.path.dirname(cur)
    fn = widgets.FileSelection(cur,'*',dir=True).getFilename()
    if fn and change:
        chdir(fn)
    pf.GUI.update()
    pf.canvas.update()
    pf.app.processEvents()
    return fn


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

def draw(F,
         view=None,bbox=None,
         color='prop',colormap=None,bkcolor=None,bkcolormap=None,alpha=None,
         mode=None,linewidth=None,shrink=None,marksize=None,
         wait=True,clear=None,allviews=False,
         highlight=False,flat=False):
    """Draw object(s) with specified settings and direct camera to it.

    The first argument is an object to be drawn. All other arguments are
    settings that influence how  the object is being drawn.

    F is either a Formex or a TriSurface object, or a name of such object
    (global or exported), or a list thereof.
    If F is a list, the draw() function is called repeatedly with each of
    ithe items of the list as first argument and with the remaining arguments
    unchanged.

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
        for Fi in F:
            if Fi is F[-1]:
                nowait = wait
            actor.append(draw(Fi,view,bbox,
                              color,colormap,bkcolor,bkcolormap,alpha,
                              mode,linewidth,shrink,marksize,
                              wait=nowait,clear=clear,allviews=allviews,
                              highlight=highlight,flat=flat))
            if Fi is F[0]:
                clear = False
                view = None
                
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
    if shrink is not None:
        F = _shrink(F,shrink)

    try:
        ## if isinstance(F,tools.Plane):
        ##     return drawPlane(F.point(),F.normal(),F.size())

        actor = F.actor(color=color,colormap=colormap,bkcolor=bkcolor,bkcolormap=bkcolormap,alpha=alpha,mode=mode,linewidth=linewidth,marksize=marksize)

        if actor is None:
            return None
        
        if flat:
            actor.specular = 0.
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
    pf.canvas.setCamera(bbox=bbox(object))
    pf.canvas.update()

    
def setDrawOptions(d):
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


def _shrink(F,factor):
    """Return a shrinked object.

    A shrinked object is one where each element is shrinked with a factor
    around its own center.
    """
    if not isinstance(F,Formex):
        F = F.toFormex()
    return F.shrink(factor)


def drawVectors(P,v,d=1.0,color='red'):
    v = normalize(v)
    Q = P+v
    F = connect([Formex(P),Formex(Q)])
    return draw(F,color=color,linewidth=3)


def drawPlane(P,N,size):
    from plugins.tools import Plane
    p = Plane(P,N,size)
    return draw(p,bbox='last')
    ## actor = actors.PlaneActor(size=size)
    ## actor.create_list(mode=pf.canvas.rendermode)
    ## actor = actors.RotatedActor(actor,N)
    ## actor.create_list(mode=pf.canvas.rendermode)
    ## actor = actors.TranslatedActor(actor,P)
    ## pf.canvas.addActor(actor)
    ## pf.canvas.update()
    ## return actor


def drawMarks(X,M,color='black',leader=''):
    """Draw a list of marks at points X.

    X is a Coords array.
    M is a list with the same length as X.
    The string representation of the marks are drawn at the corresponding
    3D coordinate.
    """
    M = marks.MarkList(X,M,color=color,leader=leader)
    pf.canvas.addAnnotation(M)
    pf.canvas.numbers = M
    pf.canvas.update()
    return M


def drawNumbers(F,color='black',trl=None,offset=0,leader=''):
    """Draw numbers on all elements of F.

    Normally, the numbers are drawn at the centroids of the elements.
    A translation may be given to put the numbers out of the centroids,
    e.g. to put them in front of the objects to make them visible,
    or to allow to view a mark at the centroids.
    If an offset is specified, it is added to the shown numbers.
    """
    if not isinstance(F,coords.Coords):
        F = F.centroids()
    if trl is not None:
        F = F.trl(trl)
    return drawMarks(F,numpy.arange(F.shape[0])+offset,color=color,leader=leader)


def drawVertexNumbers(F,color='black',trl=None):
    """Draw (local) numbers on all vertices of F.

    Normally, the numbers are drawn at the location of the vertices.
    A translation may be given to put the numbers out of the location,
    e.g. to put them in front of the objects to make them visible,
    or to allow to view a mark at the vertices.
    """
    FC = F.coords.reshape((-1,3))
    if trl is not None:
        FC = FC.trl(trl)
    return drawMarks(FC,numpy.resize(numpy.arange(F.coords.shape[1]),(FC.shape[0])),color=color)


def drawNormals(N,P,size=5,**extra):
    """Draw normals N in P. size can be single constant or list."""
    C = numpy.zeros((N.shape[0],2,3))
    C[:,0,:] = P
    C[:,1,:] = P + size*N
    return draw(Formex(C),**extra)


def drawText3D(P,text,color=None,font='sans',size=18):
    """Draw a text at a 3D point P."""
    M = marks.TextMark(P,text,color=color,font=font,size=size)
    pf.canvas.addAnnotation(M)
    pf.canvas.update()
    return M


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
    

def view(v,wait=False):
    """Show a named view, either a builtin or a user defined."""
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
    pf.GUI.setViewAngles(name,angles)   
    

def zoomBbox(bb):
    """Zoom thus that the specified bbox becomes visible."""
    pf.canvas.setBbox(bb)
    pf.canvas.setCamera()
    pf.canvas.update()


def zoomRectangle():
    """Zoom a rectangle selected by the user."""
    pf.canvas.start_rectangle_zoom()
    pf.canvas.update()
    


def zoomAll():
    """Zoom thus that all actors become visible."""
    if pf.canvas.actors:
        zoomBbox(coords.bbox(pf.canvas.actors))

def zoom(f):
    pf.canvas.zoom(f)
    pf.canvas.update()


def bgcolor(color,color2=None):
    """Change the background color (and redraw).

    If one color is given, the background is a solid color.
    If two colors are given, the background color will get a vertical
    gradient with color on top and color2 at the bottom.
    """
    pf.canvas.setBgColor(color,color2)
    pf.canvas.display()
    pf.canvas.update()


def fgcolor(color):
    """Set the default foreground color."""
    pf.canvas.setFgColor(color)


def renderMode(mode,avg=False):
    pf.canvas.setRenderMode(mode)
    toolbar.setNormals(avg)
    toolbar.setLight(pf.canvas.lighting)
    pf.canvas.update()
    pf.app.processEvents()
    
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

def lights(onoff):
    """Set the lights on or off"""
    toolbar.setLight(onoff)


def transparent(state=True):
    toolbar.setTransparency(state)
def perspective(state=True):
    toolbar.setPerspective(state)
def timeout(state=None):
    toolbar.timeout(state)


def set_material_value(typ,val):
    """Set the value of one of the material lighting parameters

    typ is one of 'ambient','specular','emission','shininess'
    val is a value between 0.0 and 1.0
    """
    #print "SETMATVAL %s = %s" % (typ,val)
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
    """Clear the canvas"""
    pf.GUI.drawlock.wait()
    clear_canvas()
    pf.canvas.update()


def redraw():
    pf.canvas.redrawAll()
    pf.canvas.update()



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


def step():
    """Perform one step of a script.

    A step is a set of instructions until the next draw operation.
    If a script is running, this just releases the draw lock.
    Else, it starts the script in step mode.
    """
    import script
    #if script.scriptRunning:
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
        

def fforward():
    pf.GUI.drawlock.free()


def delay(i):
    """Set the draw delay in seconds."""
    i = int(i)
    if i >= 0:
        pf.cfg['draw/wait'] = i
    

        
_wakeup_mode=0
sleeping = False
timer = None
def sleep(timeout=None):
    """Sleep until key/mouse press in the canvas or until timeout"""
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
        #time.sleep(0.1)
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


#################### viewports ##################################

def layout(nvps=None,ncols=None,nrows=None):
    """Set the viewports layout."""
    pf.GUI.viewports.changeLayout(nvps,ncols,nrows)

def addViewport():
    """Add a new viewport."""
    pf.GUI.viewports.addView()

def removeViewport():
    """Remove the last viewport."""
    n = len(pf.GUI.viewports.all)
    if n > 1:
        pf.GUI.viewports.removeView()

def linkViewport(vp,tovp):
    """Link viewport vp to viewport tovp.

    Both vp and tovp should be numbers of viewports. 
    """
    pf.GUI.viewports.link(vp,tovp)

def viewport(n):
    """Select the current viewport"""
    pf.GUI.viewports.setCurrent(n)

####################

def updateGUI():
    """Update the GUI."""
    pf.GUI.update()
    pf.canvas.update()
    pf.app.processEvents()


def flyAlong(path='flypath',upvector=[0.,1.,0.],sleeptime=None):
    """Fly through the current scene along the flypath.

    - `flypath`: a PolyLine or plex-2 Formex.
    """
    from plugins.curve import PolyLine
    
    if type(path) is str:
        path = named(path)
    if not path:
        warning("You have to define a flypath first!")
        return

    if isinstance(path,PolyLine):
        path = path.toFormex() 
    if path.nplex() != 2:
        warning("The flypath should be a plex-2 Formex!")
        
    for eye,center in path:
        pf.debug("Eye: %s; Center: %s" % (eye,center))
        pf.canvas.camera.lookAt(eye,center,upvector)
        pf.canvas.display()
        pf.canvas.update()
        image.saveNext()
        if sleeptime is None:
            sleeptime = pf.cfg['draw/flywait']
        sleeptime = float(sleeptime)
        if sleeptime > 0.0:
            sleep(sleeptime)

    pf.canvas.camera.setCenter(*center)
    pf.canvas.camera.setDist(coords.length(center-eye))
    pf.canvas.update()


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
    


selection_filters = [ 'none', 'single', 'closest', 'connected' ]


def set_selection_filter(i):
    """Set the selection filter mode"""
    if i in range(len(selection_filters)):
        pf.canvas.start_selection(None,selection_filters[i])

    
def pick(mode='actor',filtr=None,oneshot=False,func=None):
    """Enter interactive picking mode and return selection.

    See viewport.py for more details.
    This function differs in that it provides default highlighting
    during the picking operation, a button to stop the selection operation

    If no filter is given, the available filters are presented in a combobox.
    """
    if pf.canvas.selection_mode is not None:
        warning("You need to finish the previous picking operation first!")
        return

    pick_buttons = widgets.ButtonBox('Selection:',[('Cancel',pf.canvas.cancel_selection),('OK',pf.canvas.accept_selection)])
    
    if mode == 'element':
        filters = selection_filters
    else:
        filters = selection_filters[:3]
    filter_combo = widgets.ComboBox('Filter:',filters,set_selection_filter)
    if filtr is not None and filtr in selection_filters:
        i = selection_filters.index(filtr)
        filter_combo.setIndex(i)
    
    if func is None:
        func = highlight_funcs.get(mode,None)
    pf.message("Select %s %s" % (filtr,mode))

    pf.GUI.statusbar.addWidget(pick_buttons)
    pf.GUI.statusbar.addWidget(filter_combo)
    try:
        sel = pf.canvas.pick(mode,oneshot,func,filtr)
    finally:
        # cleanup
        if pf.canvas.selection_mode is not None:
            pf.canvas.finish_selection()
        pf.GUI.statusbar.removeWidget(pick_buttons)
        pf.GUI.statusbar.removeWidget(filter_combo)
    return sel
 
    
def pickActors(filtr=None,oneshot=False,func=None):
    return pick('actor',filtr,oneshot,func)

def pickElements(filtr=None,oneshot=False,func=None):
    return pick('element',filtr,oneshot,func)

def pickPoints(filtr=None,oneshot=False,func=None):
    return pick('point',filtr,oneshot,func)

def pickEdges(filtr=None,oneshot=False,func=None):
    return pick('edge',filtr,oneshot,func)


def highlight(K,mode):
    """Highlight a Collection of actor/elements.

    K is usually the return value of a pick operation, but might also
    be set by the user.
    mode is one of the pick modes.
    """
    func = highlight_funcs.get(mode,None)
    if func:
        func(K)


def pickNumbers(marks=None):
    if marks:
        pf.canvas.numbers = marks
    return pf.canvas.pickNumbers()


LineDrawing = None
edit_modes = ['undo', 'clear','close']


def set_edit_mode(i):
    """Set the drawing edit mode."""
    if i in range(len(edit_modes)):
        pf.canvas.edit_drawing(edit_modes[i])


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
    edit_combo = widgets.ComboBox('Edit:',edit_modes,set_edit_mode)
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
zoomall = zoomAll
drawtext = drawText


#### End
