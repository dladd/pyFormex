## $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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
from __future__ import print_function
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
from plugins import trisurface,tools,fe

from script import *
from colors import *
from signals import *
from formex import *

# these are already imported in script
#from mesh import Mesh
#from plugins.trisurface import TriSurface

#################### Interacting with the user ###############################


def closeGui():
    """Close the GUI.

    Calling this function from a script closes the GUI and terminates
    pyFormex.
    """
    pf.debug("Closing the GUI: currently, this will also terminate pyformex.",pf.DEBUG.GUI)
    pf.GUI.close()


def closeDialog(name):
    """Close the named dialog.

    Closes the InputDialog with the given name. If multiple dialogs are
    open with the same name, all these dialogs are closed.

    This only works for dialogs owned by the pyFormex GUI.
    """
    pf.GUI.closeDialog(name)


def showMessage(text,actions=['OK'],level='info',modal=True,align='00',**kargs):
    """Show a short message widget and wait for user acknowledgement.

    There are three levels of messages: 'info', 'warning' and 'error'.
    They differ only in the icon that is shown next to the test.
    By default, the message widget has a single button with the text 'OK'.
    The dialog is closed if the user clicks a button.
    The return value is the button text.
    """
    w = widgets.MessageBox(text,level=level,actions=actions,**kargs)
    if align == '--':
        w.move(100,100)
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


def showText(text,itemtype='text',actions=[('OK',None)],modal=True,mono=False):
    """Display a text in a dialog window.

    Creates a dialog window displaying some text. The dialog can be modal
    (blocking user input to the main window) or modeless.
    Scrollbars are added if the text is too large to display at once.
    By default, the dialog has a single button to close the dialog.

    Parameters:

    - `text`: a multiline text to be displayed. It can be plain text or html
      or reStructuredText (starts with '..').
    - `itemtype`: an InputItem type that can be used for text display. This
      should be either 'text' of 'info'.
    - `actions`: a list of action button definitions.
    - `modal`: bool: if True, a modal dialog is constructed. Else, the dialog
      is modeless.
    - `mono`: if True, a monospace font will be used. This is only useful for
      plain text, e.g. to show the output of an external command.

    Returns:

    :modal dialog: the result of the dialog after closing.
      The result is a dictionary with a single key: 'text' having the
      displayed text as a value. If an itemtype 'text' was used, this may
      be a changed text.
    :modeless dialog: the open dialog window itself.

    """
    if mono:
        font = "DejaVu Sans Mono"
    else:
        font = None
    w = Dialog(size=(0.75,0.75),
        items=[_I('text',text,itemtype=itemtype,text='',font=font,size=(-1,-1))],
        modal=modal,
        actions=actions,
        caption='pyFormex Text Display',
        )
    if modal:
        return w.getResult()
    else:
        w.show()
        return w


def showFile(filename,mono=True,**kargs):
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


def showDoc(obj=None,rst=True,modal=False):
    """Show the docstring of an object.

    Parameters:

    - `obj`: any object (module, class, method, function) that has a
      __doc__ attribute. If None is specified, the docstring of the current
      application is shown.
    - `rst`: bool. If True (default) the docstring is treated as being
      reStructuredText and will be nicely formatted accordingly.
      If False, the docstring is shown as plain text.
    """
    text = None
    if obj is None:
        if not pf.GUI.canPlay:
            return
        obj = pf.prefcfg['curfile']
        if utils.is_script(obj):
            #print "obj is a script"
            from utils import getDocString
            text = getDocString(obj)
            obj = None
        else:
            import apps
            obj = apps.load(obj)

    if obj:
        text = obj.__doc__

    if text is None:
        raise ValueError,"No documentation found for object %s" % obj

    text = utils.forceReST(text,underline=True)
    if pf.GUI.doc_dialog is None:
        if modal:
            actions=[('OK',None)]
        else:
            actions = [('Close',pf.GUI.close_doc_dialog)]
        pf.GUI.doc_dialog = showText(text,actions=actions,modal=modal)
    else:
        #
        # TODO: check why needed: without sometimes fails
        # RuntimeError: wrapped C/C++ object of %S has been deleted
        # probably when runall?
        #
        try:
            pf.GUI.doc_dialog.updateData({'text':text})
            # pf.GUI.doc_dialog.show()
            pf.GUI.doc_dialog.raise_()
            pf.GUI.doc_dialog.update()
            pf.app.processEvents()
        except:
            pass



def editFile(fn,exist=False):
    """Load a file into the editor.

    Parameters:

    - `fn`: filename. The corresponding file is loaded into the editor.
    - `exist`: bool. If True, only existing filenames will be accepted.

    Loading a file in the editor is done by executing an external command with
    the filename as argument. The command to be used can be set in the
    configuration. If none is set, pyFormex will try to lok at the `EDITOR`
    and `VISUAL` environment settings.

    The main author of pyFormex uses 'emacsclient' as editor command,
    to load the files in a running copy of Emacs.
    """
    print("Edit File: %s" % fn)
    if pf.cfg['editor']:
        if exist and not os.path.exists(fn):
            return
        pid = utils.spawn('%s %s' % (pf.cfg['editor'],fn))
        print(pid)
    else:
        warning('No known editor was found or configured')


# widget and result status of the widget in askItems() function
_dialog_widget = None
_dialog_result = None

def askItems(items,timeout=None,**kargs):
    """Ask the value of some items to the user.

    Create an interactive widget to let the user set the value of some items.
    The items are specified as a list of dictionaries. Each dictionary
    contains the input arguments for a widgets.InputItem. It is often
    convenient to use one of the _I, _G, ot _T functions to create these
    dictionaries. These will respectively create the input for a
    simpleInputItem, a groupInputItem or a tabInputItem.

    For convenience, simple items can also be specified as a tuple.
    A tuple (key,value) will be transformed to a dict
    {'key':key, 'value':value}.

    See the widgets.InputDialog class for complete description of the
    available input items.

    A timeout (in seconds) can be specified to have the input dialog
    interrupted automatically and return the default values.

    The remaining arguments are keyword arguments that are passed to the
    widgets.InputDialog.getResult method.

    Returns a dictionary with the results: for each input item there is a
    (key,value) pair. Returns an empty dictionary if the dialog was canceled.
    Sets the dialog timeout and accepted status in global variables.
    """
    global _dialog_widget,_dialog_result

    w = widgets.InputDialog(items,**kargs)

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


def askFilename(cur=None,filter="All files (*.*)",exist=True,multi=False,change=True,timeout=None):
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
    fn = w.getFilename(timeout)
    if fn and change:
        if multi:
            chdir(fn[0])
        else:
            chdir(fn)
    pf.GUI.update()
    pf.app.processEvents()
    return fn


def askNewFilename(cur=None,filter="All files (*.*)",timeout=None):
    """Ask a single new filename.

    This is a convenience function for calling askFilename with the
    arguments exist=False.
    """
    return askFilename(cur=cur,filter=filter,exist=False,multi=False,timeout=timeout)


def askDirname(path=None,change=True,byfile=False):
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
    if byfile:
        dirmode = 'auto'
    else:
        dirmode = True
    fn = widgets.FileSelection(path,'*',dir=dirmode).getFilename()
    if fn:
        if not os.path.isdir(fn):
            fn = os.path.dirname(fn)
        if change:
            chdir(fn)
    pf.GUI.update()
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
        elif hasattr(i,'toMesh'):
            return i.toMesh()
        else:
            return None
    r = [ fltr(i) for i in objects ]
    return [ i for i in r if i is not None ]


def draw(F,
         color='prop',colormap=None,alpha=None,
         bkcolor=None,bkcolormap=None,bkalpha=None,
         mode=None,linewidth=None,linestipple=None,
         marksize=None,nolight=False,ontop=False,
         view=None,bbox=None,shrink=None,clear=None,
         wait=True,allviews=False,highlight=False,silent=True,
         **kargs):
    """Draw object(s) with specified settings and options.

    This is the main drawing function to get geometry rendered on the OpenGL
    canvas. It has a whole slew of arguments, but in most cases you will only
    need to use a few of them. We divide the arguments in three groups:
    geometry, settings, options.

    Geometry: specifies what objects will be drawn.

    - `F`: all geometry to be drawn is specified in this single argument.
      It can be one of the following:

      - a drawable object (a Geometry object like Formex, Mesh or TriSurface,
        or another object having a proper `actor` method),
      - the name of a global pyFormex variable refering to such an object,
      - a list or nested list of any of the above items.

      The possibility of a nested list means that any complex collections
      of geometry can be drawn in a single operations. The (nested) list
      is recursively flattened, replacing string values by the corresponding
      value from the pyFormex global variables dictionary, until a single list
      of drawable objects results. Next the undrawable items are removed
      from the list. The resulting list of drawable objects will then be
      drawn using the remaining settings and options arguments.

    Settings: specify how the geometry will be drawn. These arguments will
      be passed to the corresponding Actor for the object. The Actor is the
      graphical representation of the geometry. Not all Actors use all of
      the settings that can be specified here. But they all accept specifying
      any setting even if unused. The settings hereafter are thus a
      superset of the settings used by the different Actors.
      Settings have a default value per viewport, and if unspecified, most
      Actors will use the viewport default for that value.

      - `color`, `colormap`: specifies the color of the object (see below)
      - `alpha`: float (0.0..1.0): alpha value to use in transparent mode
      - `bkcolor`, `bkcolormap`: color for the backside of surfaces, if
        different from the front side. Specification as for front color.
      - `bkalpha`: float (0.0..1.0): alpha value for back side.
      - `linewidth`: float, thickness of line drawing
      - `linestipple`: stipple pattern for line drawing
      - `marksize`: float: point size for dot drawing
      - `nolight`: bool: render object as unlighted in modes with lights on
      - `ontop`: bool: render object as if it is on top.
        This will make the object fully visible, even when it is hidden by
        other objects. If more than one objects is drawn with `ontop=True`
        the visibility of the object will depend on the order of drawing.

    Options: these arguments modify the working of the draw functions.
      If None, they are filled in from the current viewport drawing options.
      These can be changed with the :func:`setDrawOptions` function.
      The initial defaults are: view='last', bbox='auto', shrink=False,
      clear=False, shrinkfactor=0.8.

      - `view`: is either the name of a defined view or 'last' or
        None.  Predefined views are 'front', 'back', 'top', 'bottom',
        'left', 'right', 'iso'.  With view=None the camera settings
        remain unchanged (but might be changed interactively through
        the user interface). This may make the drawn object out of
        view!  With view='last', the camera angles will be set to the
        same camera angles as in the last draw operation, undoing any
        interactive changes.  On creation of a viewport, the initial
        default view is 'front' (looking in the -z direction).

      - `bbox`: specifies the 3D volume at which the camera will be
        aimed (using the angles set by `view`). The camera position will
        be set so that the volume comes in view using the current lens
        (default 45 degrees).  bbox is a list of two points or
        compatible (array with shape (2,3)).  Setting the bbox to a
        volume not enclosing the object may make the object invisible
        on the canvas.  The special value bbox='auto' will use the
        bounding box of the objects getting drawn (object.bbox()),
        thus ensuring that the camera will focus on these objects.
        The special value bbox=None will use the bounding box of the
        previous drawing operation, thus ensuring that the camera's
        target volume remains unchanged.

      - `shrink`: bool: if specified, each object will be transformed
        by the :meth:`Coords.shrink` transformation (with the current
        set shrinkfactor as a parameter), thus showing all the elements
        of the object separately.  (Some other softwares call this an
        'exploded' view).

      - `clear`: bool. By default each new draw operation adds the newly
        drawn objects to the shown scene. Using `clear=True` will clear the
        scene before drawing and thus only show the objects of the current
        draw action.

      - `wait`: bool. If True (default) the draw action activates a
        locking mechanism for the next draw action, which will only be
        allowed after `drawdelay` seconds have
        elapsed. This makes it easier to see subsequent renderings and
        is far more efficient than adding an explicit sleep()
        operation, because the script processing can continue up to
        the next drawing instruction. The value of drawdelay can be changed
        in the user settings or using the :func:`delay` function.
        Setting this value to 0 will disable the waiting mechanism for all
        subsequent draw statements (until set > 0 again). But often the user
        wants to specifically disable the waiting lock for some draw
        operation(s). This can be done without changing the `drawdelay`
        setting by specifyin `wait=False`. This means that the *next* draw
        operation does not have to wait.

      - `allviews`: currently not used

      - `highlight`: bool. If True, the object(s) will not be drawn as
        normal geometry, but as highlights (usually on top of other geometry),
        making them removeable by the remove highlight functions

      - `silent`: bool. If True (default), non-drawable objects will be
        silently ignored. If set False, an error is raised if an object
        is not drawable.

      - `**kargs`: any not-recognized keyword parameters are passed to the
        object's Actor constructor. This allows the user to create
        customized Actors with new parameters.

    Specifying color
    ----------------
    Color specification can take many different forms. Some Actors recognize
    up to six different color modes and the draw function adds even another
    mode (property color)

    - no color: `color=None`. The object will be drawn in the current
      viewport foreground color.
    - single color: the whole object is drawn with the specified color.
    - element color: each element of the object has its own color. The
      specified color will normally contain precisely `nelems` colors,
      but will be resized to the required size if not.
    - vertex color: each vertex of each element of the object has its color.
      In smooth shading modes intermediate points will get an interpolated
      color.
    - element index color: like element color, but the color values are not
      specified directly, but as indices in a color table (the `colormap`
      argument).
    - vertex index color: like vertex color, but the colors are indices in a
      color table (the `colormap` argument).
    - property color: as an extra mode in the draw function, if `color='prop'`
      is specified, and the object has an attribute 'prop', that attribute
      will be used as a color index and the object will be drawn in
      element index color mode. If the object has no such attribute, the
      object is drawn in no color mode.

    Element and vertex color modes are usually only used with a single object
    in the `F` parameter, because they require a matching set of colors.
    Though the color set will be automatically resized if not matching, the
    result will seldomly be what the user expects.
    If single colors are specified as a tuple of three float values
    (see below), the correct size of a color array for an object with
    `nelems` elements of plexitude `nplex` would be: (nelems,3) in element
    color mode, and (nelems,nplex,3) in vertex color mode. In the index modes,
    color would then be an integer array with shape respectively (nelems,) and
    (nelems,nplex). Their values are indices in the colormap array, which
    could then have shape (ncolors,3), where ncolors would be larger than the
    highest used value in the index. If the colormap is insufficiently large,
    it will again be wrapped around. If no colormap is specified, the current
    viewport colormap is used. The default contains eight colors: black=0,
    red=1, green=2, blue=3, cyan=4, magenta=5, yellow=6, white=7.

    A color value can be specified in multiple ways, but should be convertible
    to a normalized OpenGL color using the :func:`colors.GLcolor` function.
    The normalized color value is a tuple of three values in the range 0.0..1.0.
    The values are the contributions of the red, green and blue components.
    """

    # For simplicity of the code, put objects to draw always in a list
    if isinstance(F,list):
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

    ## if marksize is None:
    ##     marksize = pf.canvas.options.get('marksize',pf.cfg.get('marksize',5.0))

    # Shrink the objects if requested
    if shrink:
        FL = [ _shrink(F,pf.canvas.options.get('shrink_factor',0.8)) for F in FL ]

    # Execute the drawlock wait before doing first canvas change
    pf.GUI.drawlock.wait()

    if clear is None:
        clear = pf.canvas.options.get('clear',False)
    if clear:
        clear_canvas()

    if view is not None and view != 'last':
        pf.debug("SETTING VIEW to %s" % view,pf.DEBUG.DRAW)
        setView(view)

    pf.GUI.setBusy()
    pf.app.processEvents()

    try:

        actors = []

        # loop over the objects
        for F in FL:

            # Treat special case colors
            #print "COLOR IN",color
            if type(color) is str:
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
            else:
                Fcolor = asarray(color)

            #print "COLOR OUT",Fcolor
            # Create the actor
            actor = F.actor(
                color=Fcolor,colormap=colormap,alpha=alpha,
                bkcolor=bkcolor,bkcolormap=bkcolormap,bkalpha=bkalpha,
                mode=mode,linewidth=linewidth,linestipple=linestipple,
                marksize=marksize,nolight=nolight,ontop=ontop,**kargs)

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
            if bbox == 'last':
                bbox = None
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
    pf.message("Current Viewport Settings: %s" % pf.canvas.settings)


def reset():
    """reset the canvas"""
    pf.canvas.resetDefaults()
    pf.canvas.resetOptions()
    pf.GUI.drawwait = pf.cfg['draw/wait']
    try:
        if len(pf.GUI.viewports) == 1:
            canvasSize(-1,-1)
    except:
        print("Warning: Resetting canvas before initialization?")
    clear()
    view('front')


def resetAll():
    reset()
    wireframe()


def shrink(onoff,factor=None):
    """Set shrinking on or off, and optionally set shrink factor"""
    data = {'shrink':bool(onoff)}
    try:
        data['shrink_factor'] = float(factor)
    except:
        pass
    setDrawOptions(data)


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


def drawPlane(P,N,size,**drawOptions):
    from plugins.tools import Plane
    p = Plane(P,N,size)
    return draw(p,bbox='last',**drawOptions)


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
    #print "DRAW FREE EDGES"
    B = M.getFreeEdgesMesh()
    #print B
    draw(B,color=color,nolight=True)


def drawNumbers(F,numbers=None,color='black',trl=None,offset=0,leader='',ontop=None):
    """Draw numbers on all elements of F.

    numbers is an array with F.nelems() integer numbers.
    If no numbers are given, the range from 0 to nelems()-1 is used.
    Normally, the numbers are drawn at the centroids of the elements.
    A translation may be given to put the numbers out of the centroids,
    e.g. to put them in front of the objects to make them visible,
    or to allow to view a mark at the centroids.
    If an offset is specified, it is added to the shown numbers.
    """
    if ontop is None:
        ontop = getcfg('draw/numbersontop')
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


def drawBbox(F,color=None,linewidth=None):
    """Draw the bounding box of the geometric object F.

    F is any object that has a `bbox` method.
    Returns the drawn Annotation.
    """
    B = actors.BboxActor(F.bbox(),color=color,linewidth=linewidth)
    annotate(B)
    return B


def drawPrincipal(F,weight=None):
    """Draw the principal axes of the geometric object F.

    F is any object that has a `coords` attribute.
    If specified, weight is an array of weights attributed to the points
    of F. It should have the same length as `F.coords`.
    """
    B = actors.PrincipalActor(F,weight)
    annotate(B)
    return B


def drawText3D(P,text,color=None,font='sans',size=18,ontop=True):
    """Draw a text at a 3D point P."""
    M = marks.TextMark(P,text,color=color,font=font,size=size,ontop=ontop)
    pf.canvas.addAnnotation(M)
    pf.canvas.update()
    return M


def drawAxes(CS=None,*args,**kargs):
    """Draw the axes of a CoordinateSystem.

    CS is a CoordinateSystem. If not specified, the global coordinate system
    is used. Other arguments can be added just like in the
    :class:`AxesActor` class.

    While you can draw a CoordinateSystem using the :func:`draw` function,
    this function gives a better result because it has specialized color
    and annotation settings and provides reasonable deafults.
    """
    from coordsys import CoordinateSystem
    if CS is None:
        CS = CoordinateSystem()
    A = actors.AxesActor(CS,*args,**kargs)
    drawActor(A)
    return A


def drawImage3D(image,nx=0,ny=0,pixel='dot'):
    """Draw an image as a colored Formex

    Draws a raster image as a colored Formex. While there are other and
    better ways to display an image in pyFormex (such as using the imageView
    widget), this function allows for interactive handling the image using
    the OpenGL infrastructure.

    Parameters:

    - `image`: a QImage or any data that can be converted to a QImage,
      e.g. the name of a raster image file.
    - `nx`,`ny`: width and height (in cells) of the Formex grid.
      If the supplied image has a different size, it will be rescaled.
      Values <= 0 will be replaced with the corresponding actual size of
      the image.
    - `pixel`: the Formex representing a single pixel. It should be either
      a single element Formex, or one of the strings 'dot' or 'quad'. If 'dot'
      a single point will be used, if 'quad' a unit square. The difference
      will be important when zooming in. The default is 'dot'.

    Returns the drawn Actor.

    See also :func:`drawImage`.
    """
    pf.GUI.setBusy()
    from plugins.imagearray import image2glcolor,resizeImage

    # Create the colors
    image = resizeImage(image,nx,ny)
    nx,ny = image.width(),image.height()
    color,colortable = image2glcolor(image)

    # Create a 2D grid of nx*ny elements
    # !! THIS CAN PROBABLY BE DONE FASTER
    if isinstance(pixel,Formex) and pixel.nelems()==1:
        F = pixel
    elif pixel == 'quad':
        F = Formex('4:0123')
    else:
        F = Formex('1:0')
    F = F.replic2(nx,ny).centered()

    # Draw the grid using the image colors
    FA = draw(F,color=color,colormap=colortable,nolight=True)
    pf.GUI.setBusy(False)
    return FA


def drawImage(image,w=0,h=0,x=-1,y=-1,color=white,ontop=False):
    """Draws an image as a viewport decoration.

    Parameters:

    - `image`: a QImage or any data that can be converted to a QImage,
      e.g. the name of a raster image file. See also the :func:`loadImage`
      function.
    - `w`,`h`: width and height (in pixels) of the displayed image.
      If the supplied image has a different size, it will be rescaled.
      A value <= 0 will be replaced with the corresponding actual size of
      the image.
    - `x`,`y`: position of the lower left corner of the image. If negative,
      the image will be centered on the current viewport.
    - `color`: the color to mix in (AND) with the image. The default (white)
      will make all pixels appear as in the image.
    - `ontop`: determines whether the image will appear as a background
      (default) or at the front of the 3D scene (as on the camera glass).

    Returns the Decoration drawn.

    Note that the Decoration has a fixed size (and position) on the canvas
    and will not scale when the viewport size is changed.
    The :func:`bgcolor` function can be used to draw an image that completely
    fills the background.
    """
    utils.warn("warn_drawImage_changed")
    from plugins.imagearray import image2numpy
    from gui.decors import Rectangle

    image = image2numpy(image,resize=(w,h),indexed=False)
    w,h = image.shape[:2]
    if x < 0:
        x = (pf.canvas.width() - w) // 2
    if y < 0:
        y = (pf.canvas.height() - h) // 2
    R = Rectangle(x,y,x+w,y+h,color=color,texture=image,ontop=ontop)
    decorate(R)
    return R


def drawViewportAxes3D(pos,color=None):
    """Draw two viewport axes at a 3D position."""
    M = marks.AxesMark(pos,color)
    annotate(M)
    return M


def drawActor(A):
    """Draw an actor and update the screen."""
    pf.canvas.addActor(A)
    pf.canvas.update()

def drawAny(A):
    """Draw an Actor/Annotation/Decoration and update the screen."""
    pf.canvas.addAny(A)
    pf.canvas.update()


def undraw(itemlist):
    """Remove an item or a number of items from the canvas.

    Use the return value from one of the draw... functions to remove
    the item that was drawn from the canvas.
    A single item or a list of items may be specified.
    """
    if itemlist:
        pf.canvas.removeAny(itemlist)
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


def saveView(name,addtogui=False):
    pf.GUI.saveView(name)


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


def bgcolor(color=None,image=None):
    """Change the background color and image.

    Parameters:

    - `color`: a single color or a list of 4 colors. A single color sets a
      solid background color. A list of four colors specifies a gradient.
      These 4 colors are those of the Bottom Left, Bottom Right, Top Right
      and Top Left corners respectively.
    - `image`: the name of an image file. If specified, the image will be
      overlayed on the background colors. Specify a solid white background
      color to sea the image unaltered.
    """
    pf.canvas.setBackground(color=color,image=image)
    pf.canvas.display()
    pf.canvas.update()


def fgcolor(color):
    """Set the default foreground color."""
    pf.canvas.setFgColor(color)

def hicolor(color):
    """Set the highlight color."""
    pf.canvas.setSlColor(color)


def colormap(color=None):
    """Gets/Sets the current canvas color map"""
    return pf.canvas.settings.colormap


def colorindex(color):
    """Return the index of a color in the current colormap"""
    cmap = pf.canvas.settings.colormap
    color=array(color)
    i = where((cmap==color).all(axis=1))[0]
    if len(i) > 0:
        return i[0]
    else:
        i = len(cmap)
        pf.message("Add color %s = %s to viewport colormap" % (i,color))
        color = color.reshape(1,3)
        pf.canvas.settings.colormap = concatenate([cmap,color],axis=0)
    return i


def renderModes():
    """Return a list of predefined render profiles."""
    return canvas.CanvasSettings.RenderProfiles.keys()


def renderMode(mode,light=None):
    """Change the rendering profile to a predefined mode.

    Currently the following modes are defined:

    - wireframe
    - smooth
    - smoothwire
    - flat
    - flatwire
    - smooth_avg
    """
    #print "DRAW.RENDERMODE"
    #print "CANVAS %s" % pf.canvas
    #print "MODE %s" % pf.canvas.rendermode
    # ERROR The following redraws twice !!!
    pf.canvas.setRenderMode(mode,light)
    pf.canvas.update()
    toolbar.updateNormalsButton()
    toolbar.updateTransparencyButton()
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
    renderMode("smooth_avg")

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
    """Resize the canvas to (width x height).

    If a negative value is given for either width or height,
    the corresponding size is set equal to the maximum visible size
    (the size of the central widget of the main window).

    Note that changing the canvas size when multiple viewports are
    active is not approved.
    """
    pf.canvas.changeSize(width,height)


def clear_canvas():
    """Clear the canvas.

    This is a low level function not intended for the user.
    """
    pf.canvas.removeAny()
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
    to view. The use of this function is prefered over that of
    :func:`pause` or :func:`sleep`, because it allows your script to
    continue the numerical computations while waiting to draw the next
    screen.

    This function can be used to retard other functions than `draw` and `view`.
    """
    pf.GUI.drawlock.wait()
    if relock:
        pf.GUI.drawlock.lock()


# Functions corresponding with control buttons

def play(refresh=False):
    """Start the current script or if already running, continue it.

    """
    if len(pf.scriptlock) > 0:
        # An application is running
        if pf.GUI.drawlock.locked:
            pf.GUI.drawlock.release()

    else:
        # Start current application
        runAny(refresh=refresh)


def replay():
    """Replay the current app.

    This works pretty much like the play() function, but will
    reload the current application prior to running it.
    This function is especially interesting during development
    of an application.
    If the current application is a script, then it is equivalent with
    play().
    """
    appname = pf.cfg['curfile']
    play(refresh=utils.is_app(appname))



def fforward():
    """Releases the drawing lock mechanism indefinely.

    Releasing the drawing lock indefinely means that the lock will not
    be set again and your script will execute till the end.
    """
    pf.GUI.drawlock.free()


## _repeat_timed_out = False
## _repeat_exit_requested = False

## def repeat(func,duration=-1,maxcount=-1,sleep=0,*args,**kargs):
##     """Repeatedly execute a function.

##     func(*args,**kargs) is repeatedly executed until one of the following
##     conditions is met:
##     - the function returns a value that evaluates to False
##     - duration >= 0 and duration seconds have elapsed
##     - maxcount >=0 and maxcount executions have been reached
##     The default will indefinitely execute the function until it returns False.

##     Between each execution of the function, application events are processed.
##     If sleep > 0, an extra wait time of this length is executed on
##     each step. This avoids high processor load when running idle.
##     """
##     pf.debug("REPEAT: %s, %s" % (duration,maxcount),pf.DEBUG.SCRIPT)
##     global _repeat_timed_out, _repeat_exit_requested
##     _repeat_timed_out = False
##     _repeat_exit_requested = False
##     _repeat_count_reached = False

##     def timeOut():
##         global _repeat_timed_out
##         _repeat_timed_out = True

##     if duration >= 0:
##         timer = threading.Timer(duration,timeOut)
##         timer.start()

##     count = 0
##     while True:
##         pf.app.processEvents()
##         if callable(func):
##             res = func(*args,**kargs)
##             _repeat_exit_requested = not(res)
##         count += 1
##         pf.debug("Count: %s"% count,pf.DEBUG.SCRIPT)
##         if maxcount >= 0:
##              _repeat_count_reached = count >= maxcount
##         if _repeat_exit_requested or _repeat_timed_out or _repeat_count_reached:
##             pf.debug("Count: %s, TimeOut: %s" % (count,_repeat_timed_out),pf.DEBUG.SCRIPT)
##             break
##         if sleep > 0:
##             time.sleep(sleep)

##     pf.debug("EXIT FROM REPEAT",pf.DEBUG.SCRIPT)


## def interrupt():
##     """Interrupt a repeat"""
##     global _repeat_exit_requested
##     _repeat_exit_requested = True


## def wakeup(mode=0):
##     """Wake up from the sleep function.

##     This is the only way to exit the sleep() function.
##     Default is to wake up from the current sleep. A mode > 0
##     forces wakeup for longer period.
##     """
##     global timer,sleeping,_wakeup_mode
##     if timer:
##         timer.cancel()
##     sleeping = False
##     _wakeup_mode = mode

#
# IDEA: The pause() could display a progress bar showing how much time
# is left in the pause,
# maybe also with buttons to repeat, pause indefinitely, ...
#
def pause(timeout=None,msg=None):
    """Pause the execution until an external event occurs or timeout.

    When the pause statement is executed, execution of the pyformex script
    is suspended until some external event forces it to proceed again.
    Clicking the PLAY, STEP or CONTINUE button will produce such an event.
    """
    from drawlock import Repeater
    def _continue_():
        return pf.GUI.drawlock.locked

    if msg is None and timeout is None:
        msg = "Use the Play/Step/Continue button to proceed"

    pf.debug("Pause (%s): %s" % (timeout,msg),pf.DEBUG.SCRIPT)
    if msg:
        print(msg)

    pf.GUI.enableButtons(pf.GUI.actions,['Step','Continue'],True)
    pf.GUI.drawlock.release()
    if pf.GUI.drawlock.allowed:
        pf.GUI.drawlock.locked = True
    if timeout is None:
        timeout = widgets.input_timeout
    R = Repeater(_continue_,timeout)
    #R.start()


################### EXPERIMENTAL STUFF: AVOID! ###############


def sleep(duration,granularity=0.01):
    from drawlock import Repeater
    R = Repeater(None,duration,sleep=granularity)
    #R.start()


## _wakeup_mode=0
## sleeping = False
## timer = None
## def sleep1(timeout=None):
##     """Sleep until key/mouse press in the canvas or until timeout"""
##     utils.warn('warn_avoid_sleep')
##     #
##     global sleeping,_wakeup_mode,timer
##     if _wakeup_mode > 0 or timeout == 0:  # don't bother
##         return
##     # prepare for getting wakeup event
##     onSignal(WAKEUP,wakeup)
##     # create a Timer to wakeup after timeout
##     if timeout and timeout > 0:
##         timer = threading.Timer(timeout,wakeup)
##         timer.start()
##     else:
##         timer = None
##     # go into sleep mode
##     sleeping = True
##     ## while sleeping, we have to process events
##     ## (we could start another thread for this)
##     while sleeping:
##         pf.app.processEvents()
##         # And we have to sleep in between, or we would be using to much
##         # processor time idling. 0.1 is a good compromise to get some
##         # responsitivity while not pegging the cpu
##         time.sleep(0.01)
##     # ignore further wakeup events
##     offSignal(WAKEUP,wakeup)


## def wakeup(mode=0):
##     """Wake up from the sleep function.

##     This is the only way to exit the sleep() function.
##     Default is to wake up from the current sleep. A mode > 0
##     forces wakeup for longer period.
##     """
##     global timer,sleeping,_wakeup_mode
##     if timer:
##         timer.cancel()
##     sleeping = False
##     _wakeup_mode = mode


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
    #pf.canvas.setBbox(bb)
    pf.canvas.setCamera(bbox=bb)
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

    - `path`: a plex-2 or plex-3 Formex (or convertibel to such Formex)
      specifying the paths of camera eye and center (and upvector).
    - `upvector`: the direction of the vertical axis of the camera, in case
      of a 2-plex camera path.
    - `sleeptime`: a delay between subsequent images, to slow down
      the camera movement.

    This function moves the camera through the subsequent elements of the
    Formex. For each element the first point is used as the center of the
    camera and the second point as the eye (the center of the scene looked at).
    For a 3-plex Formex, the third point is used to define the upvector
    (i.e. the vertical axis of the image) of the camera. For a 2-plex
    Formex, the upvector is constant as specified in the arguments.
    """
    try:
        if not isinstance(path,Formex):
            path = path.toFormex()
        if not path.nplex() in (2,3):
            raise ValueError
    except:
        raise ValueError,"The camera path should be (convertible to) a plex-2 or plex-3 Formex!"

    nplex = path.nplex()
    if sleeptime is None:
        sleeptime = pf.cfg['draw/flywait']
    saved = delay(sleeptime)
    saved1 = pf.GUI.actions['Continue'].isEnabled()
    pf.GUI.enableButtons(pf.GUI.actions,['Continue'],True)

    for elem in path:
        eye,center = elem[:2]
        if nplex == 3:
            upv = elem[2] - center
        else:
            upv = upvector
        pf.canvas.camera.lookAt(eye,center,upv)
        wait()
        pf.canvas.display()
        pf.canvas.update()
        image.saveNext()

    delay(saved)
    pf.GUI.enableButtons(pf.GUI.actions,['Continue'],saved1)
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


def nViewports():
    """Return the number of viewports."""
    return len(pf.GUI.viewports.all)

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
    if nViewports() > 1:
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

def highlightActor(i):
    """Highlight the i-th actor in the scene."""
    if pf.options.debuglevel & pf.DEBUG.DRAW:
        print("  Highlighting actor %s/%s" % (i,len(pf.canvas.actors)))
    actor = pf.canvas.actors[i]
    FA = actors.GeomActor(actor,color=pf.canvas.settings.slcolor)
    pf.canvas.addHighlight(FA)


def highlightActors(K):
    """Highlight a selection of actors on the canvas.

    K is Collection of actors as returned by the pick() method.
    colormap is a list of two colors, for the actors not in, resp. in
    the Collection K.
    """
    pf.canvas.removeHighlight()
    [ highlightActor(i) for i in K.get(-1,[]) ]
    pf.canvas.update()


def highlightElements(K):
    """Highlight a selection of actor elements on the canvas.

    K is Collection of actor elements as returned by the pick() method.
    colormap is a list of two colors, for the elements not in, resp. in
    the Collection K.
    """
    pf.canvas.removeHighlight()
    for i in K.keys():
        pf.debug("Actor %s: Selection %s" % (i,K[i]),pf.DEBUG.DRAW)
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
    pf.canvas.removeHighlight()
    for i in K.keys():
        pf.debug("Actor %s: Selection %s" % (i,K[i]),pf.DEBUG.DRAW)
        actor = pf.canvas.actors[i]
        FA = actors.GeomActor(Formex(actor.coords[actor.object.getEdges()[K[i]]]),color=pf.canvas.settings.slcolor,linewidth=3)
        pf.canvas.addHighlight(FA)

    pf.canvas.update()


def highlightPoints(K):
    """Highlight a selection of actor elements on the canvas.

    K is Collection of actor elements as returned by the pick() method.
    """
    pf.canvas.removeHighlight()
    for i in K.keys():
        pf.debug("Actor %s: Selection %s" % (i,K[i]),pf.DEBUG.DRAW)
        actor = pf.canvas.actors[i]
        FA = actors.GeomActor(Formex(actor.points()[K[i]]),color=pf.canvas.settings.slcolor,marksize=10)
        pf.canvas.addHighlight(FA)
    pf.canvas.update()


def highlightPartitions(K):
    """Highlight a selection of partitions on the canvas.

    K is a Collection of actor elements, where each actor element is
    connected to a collection of property numbers, as returned by the
    partitionCollection() method.
    """
    pf.canvas.removeHighlight()
    for i in K.keys():
        pf.debug("Actor %s: Partitions %s" % (i,K[i][0]),pf.DEBUG.DRAW)
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


def removeHighlight():
    """Remove the highlights from the current viewport"""
    pf.canvas.removeHighlight()
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


def resetGUI():
    """Reset the GUI to its default operating mode.

    When an exception is raised during the execution of a script, the GUI
    may be left in a non-consistent state.
    This function may be called to reset most of the GUI components
    to their default operating mode.
    """
    ## resetPick()
    pf.GUI.resetCursor()
    pf.GUI.enableButtons(pf.GUI.actions,['Play','Step'],True)
    pf.GUI.enableButtons(pf.GUI.actions,['Continue','Stop'],False)


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


###########################################################################
# Make _I, _G and _T be included when doing 'from gui.draw import *'
#

__all__ = [ n for n in globals().keys() if not n.startswith('_')] + ['_I','_G','_T']

#### End
