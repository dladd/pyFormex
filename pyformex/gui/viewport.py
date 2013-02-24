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
"""Interactive OpenGL Canvas embedded in a Qt4 widget.

This module implements user interaction with the OpenGL canvas defined in
module :mod:`canvas`.
`QtCanvas` is a single interactive OpenGL canvas, while `MultiCanvas`
implements a dynamic array of multiple canvases.
"""
from __future__ import print_function

import pyformex as pf

from gui import QtCore, QtGui, QtOpenGL
from OpenGL import GL

from collection import Collection
import canvas
import decors
import image
import utils
import toolbar

# TODO
# BV: UGLY! WE SHOULD GET RID OF THIS
if pf.X11:
    from ctypes import cdll
    libGL = cdll.LoadLibrary("libGL.so.1")

import math

from coords import Coords
from numpy import *
from arraytools import isInt


# Some 2D vector operations
# We could do this with the general functions of coords.py,
# but that would be overkill for this simple 2D vectors

def dotpr (v,w):
    """Return the dot product of vectors v and w"""
    return v[0]*w[0] + v[1]*w[1]

def length (v):
    """Return the length of the vector v"""
    return math.sqrt(dotpr(v,v))

def projection(v,w):
    """Return the (signed) length of the projection of vector v on vector w."""
    return dotpr(v,w)/length(w)


# signals
from signals import *

# keys
ESC = QtCore.Qt.Key_Escape
RETURN = QtCore.Qt.Key_Return    # Normal Enter
ENTER = QtCore.Qt.Key_Enter      # Num Keypad Enter

# mouse actions
PRESS = 0
MOVE = 1
RELEASE = 2

# mouse buttons
LEFT = QtCore.Qt.LeftButton
MIDDLE = QtCore.Qt.MidButton
RIGHT = QtCore.Qt.RightButton
_buttons = [LEFT,MIDDLE,RIGHT]

# modifiers
NONE = QtCore.Qt.NoModifier
SHIFT = QtCore.Qt.ShiftModifier
CTRL = QtCore.Qt.ControlModifier
ALT = QtCore.Qt.AltModifier
ALLMODS = SHIFT | CTRL | ALT
_modifiers = [NONE,SHIFT,CTRL,ALT]
_modifiername = ['NONE','SHIFT','CTRL','ALT']

def modifierName(mod):
    try:
        return _modifiername[_modifiers.index(mod)]
    except:
        return 'UNKNOWN'


############### OpenGL Format #################################

opengl_format = None

def setOpenGLFormat():
    """Set the correct OpenGL format.

    On a correctly installed system, the default should do well.
    The default OpenGL format can be changed by command line options::

       --dri   : use the Direct Rendering Infrastructure, if available
       --nodri : do not use the DRI
       --alpha : enable the alpha buffer
    """
    global opengl_format
    pf.debug("Get OpenGL Format",pf.DEBUG.OPENGL)
    fmt = QtOpenGL.QGLFormat.defaultFormat()
    pf.debug("Got OpenGL Format",pf.DEBUG.OPENGL)
    ## print fmt
    ## print "hallo"
    ## print "OpenGL: %s" % fmt.hasOpenGL(),
    ## print "OpenGL Version: %s" % int(fmt.openGLVersionFlags()),
    ## print "OpenGLOverlays: %s" % fmt.hasOpenGLOverlays(),
    ## print "Double Buffer: %s" % fmt.doubleBuffer(),
    ## print "Depth Buffer: %s" % fmt.depth(),
    ## print "RGBA: %s" % fmt.rgba(),
    ## print "Alpha Channel: %s" % fmt.alpha(),
    ## print "Accumulation Buffer: %s" % fmt.accum(),
    ## print "Stencil Buffer: %s" % fmt.stencil(),
    ## print "Stereo: %s" % fmt.stereo(),
    ## print "Direct Rendering: %s" % fmt.directRendering(),
    ## print "Overlay: %s" % fmt.hasOverlay(),
    ## print "Plane: %s" % fmt.plane(),
    ## print "Multisample Buffers: %s" % fmt.sampleBuffers(),
    pf.debug(OpenGLFormat(fmt),pf.DEBUG.OPENGL)
    if pf.options.dri is not None:
        fmt.setDirectRendering(pf.options.dri)
##     if pf.options.alpha:
##         fmt.setAlpha(True)
    pf.debug("Set OpenGL Format",pf.DEBUG.OPENGL)
    pf.debug(OpenGLFormat(fmt),pf.DEBUG.OPENGL)
    QtOpenGL.QGLFormat.setDefaultFormat(fmt)
    #QtOpenGL.QGLFormat.setOverlayFormat(fmt)
    #fmt.setDirectRendering(False)
    opengl_format = fmt
    pf.debug(OpenGLFormat(fmt),pf.DEBUG.OPENGL)
    return fmt

def getOpenGLContext():
    ctxt = QtOpenGL.QGLContext.currentContext()
    if ctxt is not None:
        printOpenGLContext(ctxt)
    return ctxt

def OpenGLFormat(fmt=None):
    """Some information about the OpenGL format."""
    if fmt is None:
        fmt = opengl_format
    s = '\n'.join([
        "OpenGL: %s" % fmt.hasOpenGL(),
        "OpenGL Version: %s" % int(fmt.openGLVersionFlags()),
        "OpenGLOverlays: %s" % fmt.hasOpenGLOverlays(),
        "Double Buffer: %s" % fmt.doubleBuffer(),
        "Depth Buffer: %s" % fmt.depth(),
        "RGBA: %s" % fmt.rgba(),
        "Alpha Channel: %s" % fmt.alpha(),
        "Accumulation Buffer: %s" % fmt.accum(),
        "Stencil Buffer: %s" % fmt.stencil(),
        "Stereo: %s" % fmt.stereo(),
        "Direct Rendering: %s" % fmt.directRendering(),
        "Overlay: %s" % fmt.hasOverlay(),
        "Plane: %s" % fmt.plane(),
        "Multisample Buffers: %s" % fmt.sampleBuffers(),
        ''
        ])
    return s


def printOpenGLContext(ctxt):
    if ctxt:
        print("context is valid: %d" % ctxt.isValid())
        print("context is sharing: %d" % ctxt.isSharing())
    else:
        print("No OpenGL context yet!")


################# Canvas Mouse Event Handler #########################

class CursorShapeHandler(object):
    """A class for handling the mouse cursor shape on the Canvas.

    """
    cursor_shape = { 'default': QtCore.Qt.ArrowCursor,
                     'pick'   : QtCore.Qt.CrossCursor,
                     'busy'   : QtCore.Qt.BusyCursor,
                     }

    def __init__(self,widget):
        """Create a CursorHandler for the specified widget."""
        self.widget = widget

    def setCursorShape(self,shape):
        """Set the cursor shape to shape"""
        if shape not in QtCanvas.cursor_shape.keys():
            shape = 'default'
        self.setCursor(QtCanvas.cursor_shape[shape])


    def setCursorShapeFromFunc(self,func):
        """Set the cursor shape to shape"""
        if func in [ self.mouse_rectangle_zoom,self.mouse_pick ]:
            shape = 'pick'
        else:
            shape = 'default'
        self.setCursorShape(shape)


class CanvasMouseHandler(object):
    """A class for handling the mouse events on the Canvas.

    """
    def setMouse(self,button,func,mod=NONE):
        pf.debug(button,mod,pf.DEBUG.MOUSE)
        self.mousefncsaved[mod][button].append(self.mousefnc[mod][button])
        self.mousefnc[mod][button] = func
        self.setCursorShapeFromFunc(func)
        pf.debug("MOUSE %s" % func,pf.DEBUG.MOUSE)
        pf.debug("MOUSE SAVED %s" % self.mousefncsaved[mod][button],pf.DEBUG.MOUSE)


    def resetMouse(self,button,mod=NONE):
        pf.debug("MOUSE SAVED %s" % self.mousefncsaved[mod][button],pf.DEBUG.MOUSE)
        try:
            func = self.mousefncsaved[mod][button].pop()
        except:
            pf.debug("AAAAAHHH, COULD NOT POP",pf.DEBUG.MOUSE)
            func = None
        self.mousefnc[mod][button] = func
        self.setCursorShapeFromFunc(func)
        pf.debug("RESETMOUSE %s" % func,pf.DEBUG.MOUSE)
        pf.debug("MOUSE SAVED %s" % self.mousefncsaved[mod][button],pf.DEBUG.MOUSE)


    def getMouseFunc(self):
        """Return the mouse function bound to self.button and self.mod"""
        return self.mousefnc.get(int(self.mod),{}).get(self.button,None)




################# Single Interactive OpenGL Canvas ###############


class QtCanvas(QtOpenGL.QGLWidget,canvas.Canvas):
    """A canvas for OpenGL rendering.

    This class provides interactive functionality for the OpenGL canvas
    provided by the :class:`canvas.Canvas` class.

    Interactivity is highly dependent on Qt4. Putting the interactive
    functions in a separate class makes it esier to use the Canvas class
    in non-interactive situations or combining it with other GUI toolsets.

    The QtCanvas constructor may have positional and keyword arguments. The
    positional arguments are passed to the QtOpenGL.QGLWidget constructor,
    while the keyword arguments are passed to the canvas.Canvas constructor.
    """
    cursor_shape = { 'default': QtCore.Qt.ArrowCursor,
                     'pick'   : QtCore.Qt.CrossCursor,
                     'draw'   : QtCore.Qt.CrossCursor,
                     'busy'   : QtCore.Qt.BusyCursor,
                     }

    selection_filters = [ 'none', 'single', 'closest', 'connected', 'closest-connected' ]


    def __init__(self,*args,**kargs):
        """Initialize an empty canvas."""
        QtOpenGL.QGLWidget.__init__(self,*args)
        self.setMinimumSize(32,32)
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        canvas.Canvas.__init__(self,**kargs)
        self.setCursorShape('default')
        self.button = None
        self.mod = NONE
        self.dynamouse = True  # dynamic mouse action works on mouse move
        self.dynamic = None    # what action on mouse move
        self.mousefnc = {}
        self.mousefncsaved = {}
        for mod in _modifiers:
            self.mousefnc[mod] = {}
            self.mousefncsaved[mod] = {}
            for button in _buttons:
                self.mousefnc[mod][button] = None
                self.mousefncsaved[mod][button] = []
        # Initial mouse funcs are dynamic handling
        # ALT is initially same as NONE and should never be changed
        for mod in (NONE,ALT):
            self.setMouse(LEFT,self.dynarot,mod)
            self.setMouse(MIDDLE,self.dynapan,mod)
            self.setMouse(RIGHT,self.dynazoom,mod)
        self.selection_mode = None
        self.selection = Collection()
        self.trackfunc = None
        self.pick_func = {
            'actor'  : self.pick_actors,
            'element': self.pick_elements,
            'face'   : self.pick_faces,
            'edge'   : self.pick_edges,
            'point'  : self.pick_points,
            'number' : self.pick_numbers,
            }
        self.pickable = None
        self.drawmode = None
        self.drawing_mode = None
        self.drawing = None
        # Drawing options
        self.resetOptions()


    def getSize(self):
        """Return the size of this canvas"""
        from guimain import Size
        return Size(self)


    # TODO: negative sizes should probably resize all viewports
    # OR we need to implement frames
    def changeSize(self,width,height):
        """Resize the canvas to (width x height).

        If a negative value is given for either width or height,
        the corresponding size is set equal to the maximum visible size
        (the size of the central widget of the main window).

        Note that this may not have the expected result when multiple
        viewports are used.
        """
        if width < 0 or height < 0:
            w,h = pf.GUI.maxCanvasSize()
            if width < 0:
                width = w
            if height < 0:
                height = h
        self.resize(width,height)


    def getPickModes(self):
        return self.pick_func.keys()


    def resetOptions(self):
        """Reset the Drawing options to some defaults"""
        self.options = dict(
            view = None,       # Keep the current camera angles
            bbox = 'auto',     # Automatically zoom on the drawed object
            clear = False,     # Clear on each drawing action
            shrink = False,
            shrink_factor = 0.8,
            marksize = 5.0,
            )

    def setOptions(self,d):
        """Set the Drawing options to some values"""
        self.options.update(d)


    def setCursorShape(self,shape):
        """Set the cursor shape to shape"""
        if shape not in QtCanvas.cursor_shape.keys():
            shape = 'default'
        #self.setCursor(QtGui.QCursor(QtCanvas.cursor_shape[shape]))
        self.setCursor(QtCanvas.cursor_shape[shape])


    def setCursorShapeFromFunc(self,func):
        """Set the cursor shape to shape"""
        if func in [ self.mouse_rectangle_zoom,self.mouse_pick ]:
            shape = 'pick'
        elif func == self.mouse_draw:
            shape = 'draw'
        else:
            shape = 'default'
        self.setCursorShape(shape)


    def setMouse(self,button,func,mod=NONE):
        pf.debug("setMouse %s %s %s" % (button,mod,func),pf.DEBUG.MOUSE)
        self.mousefncsaved[mod][button].append(self.mousefnc[mod][button])
        self.mousefnc[mod][button] = func
        if button == LEFT and mod == NONE:
            self.setCursorShapeFromFunc(func)
        #print self.mousefnc


    def resetMouse(self,button,mod=NONE):
        pf.debug("resetMouse %s %s" % (button,mod),pf.DEBUG.MOUSE)
        try:
            func = self.mousefncsaved[mod][button].pop()
        except:
            func = None
        self.mousefnc[mod][button] = func
        if button == LEFT and mod == NONE:
            self.setCursorShapeFromFunc(func)
        #print self.mousefnc


    def getMouseFunc(self):
        """Return the mouse function bound to self.button and self.mod"""
        return self.mousefnc.get(int(self.mod),{}).get(self.button,None)


    def start_rectangle_zoom(self):
        self.setMouse(LEFT,self.mouse_rectangle_zoom)


    def finish_rectangle_zoom(self):
        self.update()
        self.resetMouse(LEFT)


    def mouse_rectangle_zoom(self,x,y,action):
        """Process mouse events during interactive rectangle zooming.

        On PRESS, record the mouse position.
        On MOVE, create a rectangular zoom window.
        On RELEASE, zoom to the picked rectangle.
        """
        if action == PRESS:
            self.makeCurrent()
            self.update()
            if self.trackfunc:
                #print "PRESS",self.trackfunc,pf.canvas.camera.ctr
                pf.canvas.camera.setTracking(True)
                x,y,z = pf.canvas.camera.ctr
                self.zplane = pf.canvas.project(x,y,z,True)[2]
                #print 'ZPLANE',self.zplane
                self.trackfunc(x,y,self.zplane)
            self.begin_2D_drawing()
            GL.glEnable(GL.GL_COLOR_LOGIC_OP)
            # An alternative is GL_XOR #
            GL.glLogicOp(GL.GL_INVERT)
            # Draw rectangle
            self.draw_state_rect(x,y)
            self.swapBuffers()

        elif action == MOVE:
            if self.trackfunc:
                #print "MOVE",self.trackfunc
                #print 'ZPLANE',self.zplane
                self.trackfunc(x,y,self.zplane)
            # Remove old rectangle
            self.swapBuffers()
            self.draw_state_rect(*self.state)
            # Draw new rectangle
            self.draw_state_rect(x,y)
            self.swapBuffers()

        elif action == RELEASE:
            GL.glDisable(GL.GL_COLOR_LOGIC_OP)
            self.end_2D_drawing()
            x0 = min(self.statex,x)
            y0 = min(self.statey,y)
            x1 = max(self.statex,x)
            y1 = max(self.statey,y)
            self.zoomRectangle(x0,y0,x1,y1)
            self.finish_rectangle_zoom()

####################### INTERACTIVE PICKING ############################

    def setPickable(self,nrs=None):
        """Set the list of pickable actors"""
        if nrs is None:
            self.pickable = None
        else:
            self.pickable = [ self.actors[i] for i in nrs if i in range(len(self.actors))]


    def start_selection(self,mode,filter):
        """Start an interactive picking mode.

        If selection mode was already started, mode is disregarded and
        this can be used to change the filter method.
        """
        pf.debug("START SELECTION",pf.DEBUG.GUI)
        pf.debug("Mode is %s" % self.selection_mode,pf.DEBUG.GUI)
        if self.selection_mode is None:
            self.setMouse(LEFT,self.mouse_pick)
            self.setMouse(LEFT,self.mouse_pick,SHIFT)
            self.setMouse(LEFT,self.mouse_pick,CTRL)
            self.setMouse(RIGHT,self.emit_done)
            self.setMouse(RIGHT,self.emit_cancel,SHIFT)
            self.connect(self,DONE,self.accept_selection)
            self.connect(self,CANCEL,self.cancel_selection)
            self.selection_mode = mode
            self.selection_front = None

        if filter == 'none':
            filter = None
        self.selection_filter = filter
        if filter is None:
            self.selection_front = None
        self.selection.clear()
        self.selection.setType(self.selection_mode)
        pf.debug("START SELECTION DONE",pf.DEBUG.GUI)


    def wait_selection(self):
        """Wait for the user to interactively make a selection."""
        pf.debug("WAIT SELECTION",pf.DEBUG.GUI)
        self.selection_timer = QtCore.QThread
        self.selection_busy = True
        while self.selection_busy:
            self.selection_timer.msleep(20)
            pf.app.processEvents()
        pf.debug("WAIT SELECTION DONE",pf.DEBUG.GUI)


    def finish_selection(self):
        """End an interactive picking mode."""
        pf.debug("FINISH SELECTION",pf.DEBUG.GUI)
        self.resetMouse(LEFT)
        self.resetMouse(LEFT,SHIFT)
        self.resetMouse(LEFT,CTRL)
        self.resetMouse(RIGHT)
        self.resetMouse(RIGHT,SHIFT)
        self.disconnect(self,DONE,self.accept_selection)
        self.disconnect(self,CANCEL,self.cancel_selection)
        self.selection_mode = None
        pf.debug("FINISH SELECTION DONE",pf.DEBUG.GUI)


    def accept_selection(self,clear=False):
        """Accept or cancel an interactive picking mode.

        If clear == True, the current selection is cleared.
        """
        self.selection_accepted = True
        if clear:
            self.selection.clear()
            self.selection_accepted = False
        self.selection_canceled = True
        self.selection_busy = False

    def cancel_selection(self):
        """Cancel an interactive picking mode and clear the selection."""
        self.accept_selection(clear=True)


    def pick(self,mode='actor',oneshot=False,func=None,filter=None):
        """Interactively pick objects from the viewport.

        - `mode`: defines what to pick : one of
          ``['actor','element','point','number','edge']``
        - `oneshot`: if True, the function returns as soon as the user ends
          a picking operation. The default is to let the user
          modify his selection and only to return after an explicit
          cancel (ESC or right mouse button).
        - `func`: if specified, this function will be called after each
          atomic pick operation. The Collection with the currently selected
          objects is passed as an argument. This can e.g. be used to highlight
          the selected objects during picking.
        - `filter`: defines what elements to retain from the selection: one of
          ``[None,'single','closest,'connected']``.

          - None (default) will return the complete selection.
          - 'closest' will only keep the element closest to the user.
          - 'connected' will only keep elements connected to
            - the closest element (set picked)
            - what is already in the selection (add picked).

            Currently this only works when picking mode is 'element' and
            for Actors having a partitionByConnection method.

        When the picking operation is finished, the selection is returned.
        The return value is always a Collection object.
        """
        self.selection_canceled = False
        self.start_selection(mode,filter)
        while not self.selection_canceled:
            self.wait_selection()
            if not self.selection_canceled:
                # selection by mouse_picking
                self.pick_func[self.selection_mode]()
                if len(self.picked) != 0:
                    if self.selection_filter is None:
                        if self.mod == NONE:
                            self.selection.set(self.picked)
                        elif self.mod == SHIFT:
                            self.selection.add(self.picked)
                        elif self.mod == CTRL:
                            self.selection.remove(self.picked)
                    elif self.selection_filter == 'single':
                        if self.mod == NONE:
                            self.selection.set([self.closest_pick[0]])
                        elif self.mod == SHIFT:
                            self.selection.add([self.closest_pick[0]])
                        elif self.mod == CTRL:
                            self.selection.remove([self.closest_pick[0]])
                    elif self.selection_filter == 'closest':
                        if self.selection_front is None or self.mod == NONE or \
                               (self.mod == SHIFT and self.closest_pick[1] < self.selection_front[1]):
                            self.selection_front = self.closest_pick
                            self.selection.set([self.closest_pick[0]])
                    elif self.selection_filter == 'connected':
                        if self.selection_front is None or self.mod == NONE or len(self.selection.keys()) == 0:
                            self.selection_front = self.closest_pick
                            closest_actor,closest_elem = map(int,self.selection_front[0])
                        elif self.mod == SHIFT:
                            closest_elem = self.selection.get(closest_actor)[0]
                        if self.mod == NONE:
                            self.selection.set(self.picked)
                        elif self.mod == SHIFT:
                            self.selection.add(self.picked)
                        elif self.mod == CTRL:
                            self.selection.remove(self.picked)
                        if self.mod == NONE or self.mod == SHIFT:
                            conn_elems = self.actors[closest_actor].object.connectedElements(closest_elem,self.selection.get(closest_actor))
                            self.selection.set(conn_elems,closest_actor)
                    if func:
                        func(self.selection)
                self.update()
            if oneshot:
                self.accept_selection()
        if func and not self.selection_accepted:
            func(self.selection)
        self.finish_selection()
        return self.selection


    def pickNumbers(self,*args,**kargs):
        """Go into number picking mode and return the selection."""
        return self.pick('numbers',*args,**kargs)

#################### Interactive drawing ####################################

    def idraw(self,mode='point',npoints=-1,zplane=0.,func=None,coords=None,preview=False):
        """Interactively draw on the canvas.

        This function allows the user to interactively create points in 3D
        space and collects the subsequent points in a Coords object. The
        interpretation of these points is left to the caller.

        - `mode`: one of the drawing modes, specifying the kind of objects you
          want to draw. This is passed to the specified `func`.
        - `npoints`: If -1, the user can create any number of points. When >=0,
          the function will return when the total number of points in the
          collection reaches the specified value.
        - `zplane`: the depth of the z-plane on which the 2D drawing is done.
        - `func`: a function that is called after each atomic drawing
          operation. It is typically used to draw a preview using the current
          set of points. The function is passed the current Coords and the
          `mode` as arguments.
        - `coords`: an initial set of coordinates to which the newly created
          points should be added. If specified, `npoints` also counts these
          initial points.
        - `preview`: **Experimental** If True, the preview funcion will also
          be called during mouse movement with a pressed button, allowing to
          preview the result before a point is created.

        The drawing operation is finished when the number of requested points
        has been reached, or when the user clicks the right mouse button or
        hits 'ENTER'.
        The return value is a (n,3) shaped Coords array.
        """
        self.draw_canceled = False
        self.start_draw(mode,zplane,coords)
        try:
            if preview:
                self.previewfunc = func
            else:
                self.previewfunc = None

            while not self.draw_canceled:
                self.wait_selection()
                if not self.draw_canceled:
                    self.drawn = Coords(self.drawn).reshape(-1,3)
                    self.drawing = Coords.concatenate([self.drawing,self.drawn])
                    if func:
                        func(self.drawing,self.drawmode)
                if npoints > 0 and len(self.drawing) >= npoints:
                    self.accept_draw()
            if func and not self.draw_accepted:
                func(self.drawing,self.drawmode)
        finally:
            self.finish_draw()
        return self.drawing


    def start_draw(self,mode,zplane,coords):
        """Start an interactive drawing mode."""
        self.setMouse(LEFT,self.mouse_draw)
        self.setMouse(RIGHT,self.emit_done)
        self.setMouse(RIGHT,self.emit_cancel,SHIFT)
        self.connect(self,DONE,self.accept_draw)
        self.connect(self,CANCEL,self.cancel_draw)
        self.drawmode = mode
        self.zplane = zplane
        self.drawing = Coords(coords)

    def finish_draw(self):
        """End an interactive drawing mode."""
        self.resetMouse(LEFT)
        self.resetMouse(RIGHT)
        self.resetMouse(RIGHT,SHIFT)
        self.disconnect(self,DONE,self.accept_selection)
        self.disconnect(self,CANCEL,self.cancel_selection)
        self.drawmode = None

    def accept_draw(self,clear=False):
        """Cancel an interactive drawing mode.

        If clear == True, the current drawing is cleared.
        """
        self.draw_accepted = True
        if clear:
            self.drawing = Coords()
            self.draw_accepted = False
        self.draw_canceled = True
        self.selection_busy = False

    def cancel_draw(self):
        """Cancel an interactive drawing mode and clear the drawing."""
        self.accept_draw(clear=True)


    def mouse_draw(self,x,y,action):
        """Process mouse events during interactive drawing.

        On PRESS, do nothing.
        On MOVE, do nothing.
        On RELEASE, add the point to the point list.
        """
        if action == PRESS:
            self.makeCurrent()
            self.update()
            if self.trackfunc:
                print("ENABLE TRACKING")
                pf.canvas.camera.setTracking(True)

        elif action == MOVE:
            if pf.app.hasPendingEvents():
                return
            if self.trackfunc:
                self.trackfunc(x,y,self.zplane)
                #pf.app.processEvents()
            if self.previewfunc:
                self.swapBuffers()
                self.drawn = self.unProject(x,y,self.zplane)
                self.drawn = Coords(self.drawn).reshape(-1,3)
                self.previewfunc(Coords.concatenate([self.drawing,self.drawn]),self.drawmode)
                self.swapBuffers()

        elif action == RELEASE:
            self.drawn = self.unProject(x,y,self.zplane)
            self.selection_busy = False

##########################################################################

    def start_drawing(self,mode):
        """Start an interactive line drawing mode."""
        pf.debug("START DRAWING MODE",pf.DEBUG.GUI)
        self.setMouse(LEFT,self.mouse_draw_line)
        self.setMouse(RIGHT,self.emit_done)
        self.setMouse(RIGHT,self.emit_cancel,SHIFT)
        self.connect(self,DONE,self.accept_drawing)
        self.connect(self,CANCEL,self.cancel_drawing)
        #self.setCursorShape('pick')
        self.drawing_mode = mode
        self.edit_mode = None
        self.drawing = empty((0,2,2),dtype=int)

    def wait_drawing(self):
        """Wait for the user to interactively draw a line."""
        self.drawing_timer = QtCore.QThread
        self.drawing_busy = True
        while self.drawing_busy:
            self.drawing_timer.msleep(20)
            pf.app.processEvents()

    def finish_drawing(self):
        """End an interactive drawing mode."""
        pf.debug("END DRAWING MODE",pf.DEBUG.GUI)
        #self.setCursorShape('default')
        self.resetMouse(LEFT)
        self.resetMouse(RIGHT)
        self.resetMouse(RIGHT,SHIFT)
        self.disconnect(self,DONE,self.accept_selection)
        self.disconnect(self,CANCEL,self.cancel_selection)
        self.drawing_mode = None

    def accept_drawing(self,clear=False):
        """Cancel an interactive drawing mode.

        If clear == True, the current drawing is cleared.
        """
        pf.debug("CANCEL DRAWING MODE",pf.DEBUG.GUI)
        self.drawing_accepted = True
        if clear:
            self.drawing = empty((0,2,2),dtype=int)
            self.drawing_accepted = False
        self.drawing_canceled = True
        self.drawing_busy = False

    def cancel_drawing(self):
        """Cancel an interactive drawing mode and clear the drawing."""
        self.accept_drawing(clear=True)

    def edit_drawing(self,mode):
        """Edit an interactive drawing."""
        self.edit_mode = mode
        self.drawing_busy = False

    def drawLinesInter(self,mode='line',oneshot=False,func=None):
        """Interactively draw lines on the canvas.

        - oneshot: if True, the function returns as soon as the user ends
          a drawing operation. The default is to let the user
          draw multiple lines and only to return after an explicit
          cancel (ESC or right mouse button).
        - func: if specified, this function will be called after each
          atomic drawing operation. The current drawing is passed as
          an argument. This can e.g. be used to show the drawing.

        When the drawing operation is finished, the drawing is returned.
        The return value is a (n,2,2) shaped array.
        """
        self.drawing_canceled = False
        self.start_drawing(mode)
        while not self.drawing_canceled:
            self.wait_drawing()
            if not self.drawing_canceled:
                if self.edit_mode: # an edit mode from the edit combo was clicked
                    if self.edit_mode == 'undo' and self.drawing.size != 0:
                        self.drawing = delete(self.drawing,-1,0)
                    elif self.edit_mode == 'clear':
                        self.drawing = empty((0,2,2),dtype=int)
                    elif self.edit_mode == 'close' and self.drawing.size != 0:
                        line = asarray([self.drawing[-1,-1],self.drawing[0,0]])
                        self.drawing = append(self.drawing,line.reshape(-1,2,2),0)
                    self.edit_mode = None
                else: # a line was drawn interactively
                    self.drawing = append(self.drawing,self.drawn.reshape(-1,2,2),0)
                if func:
                    func(self.drawing)
            if oneshot:
                self.accept_drawing()
        if func and not self.drawing_accepted:
            func(self.drawing)
        self.finish_drawing()
        return self.drawing


######## QtOpenGL interface ##############################

    def initializeGL(self):
        if pf.options.debuglevel & pf.DEBUG.GUI:
            p = self.sizePolicy()
            print("Size policy %s,%s,%s,%s" % (p.horizontalPolicy(), p.verticalPolicy(), p.horizontalStretch(), p.verticalStretch()))
        self.initCamera()
        self.glinit()
        self.resizeGL(self.width(),self.height())
        self.setCamera()

    def	resizeGL(self,w,h):
        self.setSize(w,h)

    def	paintGL(self):
        if not self.mode2D:
            self.display()

####### MOUSE EVENT HANDLERS ############################

    # Mouse functions can be bound to any of the mouse buttons
    # LEFT, MIDDLE or RIGHT.
    # Each mouse function should accept three possible actions:
    # PRESS, MOVE, RELEASE.
    # On a mouse button PRESS, the mouse screen position and the pressed
    # button are always saved in self.statex,self.statey,self.button.
    # The mouse function does not need to save these and can directly use
    # their values.
    # On a mouse button RELEASE, self.button is cleared, to avoid further
    # move actions.
    # Functions that change the camera settings should call saveModelView()
    # when they are done.
    # ATTENTION! The y argument is positive upwards, as in normal OpenGL
    # operations!


    def dynarot(self,x,y,action):
        """Perform dynamic rotation operation.

        This function processes mouse button events controlling a dynamic
        rotation operation. The action is one of PRESS, MOVE or RELEASE.
        """
        if action == PRESS:
            w,h = self.getSize()
            self.state = [self.statex-w/2, self.statey-h/2 ]

        elif action == MOVE:
            w,h = self.getSize()
            # set all three rotations from mouse movement
            # tangential movement sets twist,
            # but only if initial vector is big enough
            x0 = self.state        # initial vector
            d = length(x0)
            if d > h/8:
                x1 = [x-w/2, y-h/2]     # new vector
                a0 = math.atan2(x0[0],x0[1])
                a1 = math.atan2(x1[0],x1[1])
                an = (a1-a0) / math.pi * 180
                ds = utils.stuur(d,[-h/4,h/8,h/4],[-1,0,1],2)
                twist = - an*ds
                self.camera.rotate(twist,0.,0.,1.)
                self.state = x1
            # radial movement rotates around vector in lens plane
            x0 = [self.statex-w/2, self.statey-h/2]    # initial vector
            if x0 == [0.,0.]:
                x0 = [1.,0.]
            dx = [x-self.statex, y-self.statey]        # movement
            b = projection(dx,x0)
            if abs(b) > 5:
                val = utils.stuur(b,[-2*h,0,2*h],[-180,0,+180],1)
                rot =  [ abs(val),-dx[1],dx[0],0 ]
                self.camera.rotate(*rot)
                self.statex,self.statey = (x,y)
            self.update()

        elif action == RELEASE:
            self.update()
            self.camera.saveModelView()


    def dynapan(self,x,y,action):
        """Perform dynamic pan operation.

        This function processes mouse button events controlling a dynamic
        pan operation. The action is one of PRESS, MOVE or RELEASE.
        """
        if action == PRESS:
            pass

        elif action == MOVE:
            w,h = self.getSize()
            dx,dy = float(self.statex-x)/w, float(self.statey-y)/h
            self.camera.transArea(dx,dy)
            self.statex,self.statey = (x,y)
            self.update()

        elif action == RELEASE:
            self.update()
            self.camera.saveModelView()


    def dynazoom(self,x,y,action):
        """Perform dynamic zoom operation.

        This function processes mouse button events controlling a dynamic
        zoom operation. The action is one of PRESS, MOVE or RELEASE.
        """
        if action == PRESS:
            self.state = [self.camera.getDist(),self.camera.area.tolist(),pf.cfg['gui/dynazoom']]

        elif action == MOVE:
            w,h = self.getSize()
            dx,dy = float(self.statex-x)/w, float(self.statey-y)/h
            for method,state,value,size in zip(self.state[2],[self.statex,self.statey],[x,y],[w,h]):
                #pf.debug("%s %s %s %s" % (method,state,value,size))
                if method == 'area':
                    d = float(state-value)/size
                    f = exp(4*d)
                    self.camera.zoomArea(f,area=asarray(self.state[1]).reshape(2,2))
                elif method == 'dolly':
                    d = utils.stuur(value,[0,state,size],[5,1,0.2],1.2)
                    #pf.debug(d)
                    self.camera.dist = d*self.state[0]

            self.update()

        elif action == RELEASE:
            self.update()
            self.camera.saveModelView()

    def wheel_zoom(self,delta):
        """Zoom by rotating a wheel over an angle delta"""
        f = 2**(delta/120.*pf.cfg['gui/wheelzoomfactor'])
        if pf.cfg['gui/wheelzoom'] == 'area':
            self.camera.zoomArea(f)
        elif pf.cfg['gui/wheelzoom'] == 'lens':
            self.camera.zoom(f)
        else:
            self.camera.dolly(f)
        self.update()

    def emit_done(self,x,y,action):
        """Emit a DONE event by clicking the mouse.

        This is equivalent to pressing the ENTER button."""
        if action == RELEASE:
            self.emit(DONE,())

    def emit_cancel(self,x,y,action):
        """Emit a CANCEL event by clicking the mouse.

        This is equivalent to pressing the ESC button."""
        if action == RELEASE:
            self.emit(CANCEL,())


    def draw_state_rect(self,x,y):
        """Store the pos and draw a rectangle to it."""
        self.state = x,y
        decors.drawRect(self.statex,self.statey,x,y)


    def mouse_pick(self,x,y,action):
        """Process mouse events during interactive picking.

        On PRESS, record the mouse position.
        On MOVE, create a rectangular picking window.
        On RELEASE, pick the objects inside the rectangle.
        """
        if action == PRESS:
            self.makeCurrent()
            self.update()
            self.begin_2D_drawing()
            #self.swapBuffers()
            GL.glEnable(GL.GL_COLOR_LOGIC_OP)
            # An alternative is GL_XOR #
            GL.glLogicOp(GL.GL_INVERT)
            # Draw rectangle
            self.draw_state_rect(x,y)
            self.swapBuffers()

        elif action == MOVE:
            # Remove old rectangle
            self.swapBuffers()
            self.draw_state_rect(*self.state)
            # Draw new rectangle
            self.draw_state_rect(x,y)
            self.swapBuffers()

        elif action == RELEASE:
            GL.glDisable(GL.GL_COLOR_LOGIC_OP)
            self.swapBuffers()
            self.end_2D_drawing()

            x,y = (x+self.statex)/2., (y+self.statey)/2.
            w,h = abs(x-self.statex)*2., abs(y-self.statey)*2.
            if w <= 0 or h <= 0:
               w,h = pf.cfg.get('draw/picksize',(20,20))
            vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
            self.pick_window = (x,y,w,h,vp)
            self.selection_busy = False


    def pick_actors(self):
        """Set the list of actors inside the pick_window."""
        self.camera.loadProjection(pick=self.pick_window)
        self.camera.loadModelView()
        stackdepth = 1
        npickable = len(self.actors)
        selbuf = GL.glSelectBuffer(npickable*(3+stackdepth))
        GL.glRenderMode(GL.GL_SELECT)
        GL.glInitNames()
        for i,a in enumerate(self.actors):
            GL.glPushName(i)
            GL.glCallList(a.list)
            GL.glPopName()
        libGL.glRenderMode(GL.GL_RENDER)
        # Read the selection buffer
        store_closest = self.selection_filter == 'single' or \
                        self.selection_filter == 'closest'
        self.picked = []
        if selbuf[0] > 0:
            buf = asarray(selbuf).reshape(-1,3+selbuf[0])
            buf = buf[buf[:,0] > 0]
            self.picked = buf[:,3]
            if store_closest:
                w = buf[:,1].argmin()
                self.closest_pick = (self.picked[w], buf[w,1])


    def pick_parts(self,obj_type,max_objects,store_closest=False):
        """Set the list of actor parts inside the pick_window.

        obj_type can be 'element', 'face', 'edge' or 'point'.
        'face' and 'edge' are only available for Mesh type geometry.
        max_objects specifies the maximum number of objects

        The picked object numbers are stored in self.picked.
        If store_closest==True, the closest picked object is stored in as a
        tuple ( [actor,object] ,distance) in self.picked_closest

        A list of actors from which can be picked may be given.
        If so, the resulting keys are indices in this list.
        By default, the full actor list is used.
        """
        self.picked = []
        pf.debug('PICK_PARTS %s %s %s' % (obj_type,max_objects,store_closest),pf.DEBUG.DRAW)
        if max_objects <= 0:
            pf.message("No such objects to be picked!")
            return
        self.camera.loadProjection(pick=self.pick_window)
        self.camera.loadModelView()
        stackdepth = 2
        selbuf = GL.glSelectBuffer(max_objects*(3+stackdepth))
        GL.glRenderMode(GL.GL_SELECT)
        GL.glInitNames()
        if self.pickable is None:
            pickable = self.actors
        else:
            pickable = self.pickable
        for i,a in enumerate(pickable):
            GL.glPushName(i)
            a.pickGL(obj_type)  # this will push the number of the part
            GL.glPopName()
        self.picked = []
        libGL.glRenderMode(GL.GL_RENDER)
        if selbuf[0] > 0:
            buf = asarray(selbuf).reshape(-1,3+selbuf[0])
            buf = buf[buf[:,0] > 0]
            self.picked = buf[:,3:]
            #pf.debug("PICKBUFFER: %s" % self.picked)
            if store_closest and len(buf) > 0:
                w = buf[:,1].argmin()
                self.closest_pick = (self.picked[w], buf[w,1])


    def pick_elements(self):
        """Set the list of actor elements inside the pick_window."""
        npickable = 0
        for a in self.actors:
            npickable += a.nelems()
        self.pick_parts('element',npickable,store_closest=\
                        self.selection_filter == 'single' or\
                        self.selection_filter == 'closest' or\
                        self.selection_filter == 'connected'
                        )


    def pick_points(self):
        """Set the list of actor points inside the pick_window."""
        npickable = 0
        for a in self.actors:
            pf.debug("ADDING %s pickable points"%a.npoints(),pf.DEBUG.DRAW)
            npickable += a.npoints()
        self.pick_parts('point',npickable,store_closest=\
                        self.selection_filter == 'single' or\
                        self.selection_filter == 'closest',
                        )


    def pick_edges(self):
        """Set the list of actor edges inside the pick_window."""
        npickable = 0
        for a in self.actors:
            if hasattr(a,'nedges'):
                npickable += a.nedges()
        self.pick_parts('edge',npickable,store_closest=\
                        self.selection_filter == 'single' or\
                        self.selection_filter == 'closest',
                        )


    def pick_faces(self):
        """Set the list of actor faces inside the pick_window."""
        npickable = 0
        for a in self.actors:
            if hasattr(a,'nfaces'):
                npickable += a.nfaces()
        self.pick_parts('face',npickable,store_closest=\
                        self.selection_filter == 'single' or\
                        self.selection_filter == 'closest',
                        )


    def pick_numbers(self):
        """Return the numbers inside the pick_window."""
        self.camera.loadProjection(pick=self.pick_window)
        self.camera.loadModelView()
        self.picked = [0,1,2,3]
        if self.numbers:
            self.picked = self.numbers.drawpick()


    def draw_state_line(self,x,y):
        """Store the pos and draw a line to it."""
        self.state = x,y
        decors.drawLine(self.statex,self.statey,x,y)


    def mouse_draw_line(self,x,y,action):
        """Process mouse events during interactive drawing.

        On PRESS, record the mouse position.
        On MOVE, draw a line.
        On RELEASE, add the line to the drawing.
        """
        if action == PRESS:
            self.makeCurrent()
            self.update()
            self.begin_2D_drawing()
            self.swapBuffers()
            GL.glEnable(GL.GL_COLOR_LOGIC_OP)
            # An alternative is GL_XOR #
            GL.glLogicOp(GL.GL_INVERT)
            # Draw rectangle
            if self.drawing.size != 0:
                self.statex,self.statey = self.drawing[-1,-1]
            self.draw_state_line(x,y)
            self.swapBuffers()

        elif action == MOVE:
            # Remove old rectangle
            self.swapBuffers()
            self.draw_state_line(*self.state)
            # Draw new rectangle
            self.draw_state_line(x,y)
            self.swapBuffers()

        elif action == RELEASE:
            GL.glDisable(GL.GL_COLOR_LOGIC_OP)
            #self.swapBuffers()
            self.end_2D_drawing()

            self.drawn = asarray([[self.statex,self.statey],[x,y]])
            self.drawing_busy = False


    @classmethod
    def has_modifier(clas,e,mod):
        return ( e.modifiers() & mod ) == mod

    def mousePressEvent(self,e):
        """Process a mouse press event."""
        # Make the clicked viewport the current one
        pf.GUI.viewports.setCurrent(self)
        # on PRESS, always remember mouse position and button
        self.statex,self.statey = e.x(), self.height()-e.y()
        self.button = e.button()
        self.mod = e.modifiers() & ALLMODS
        #pf.debug("PRESS BUTTON %s WITH MODIFIER %s" % (self.button,self.mod))
        func = self.getMouseFunc()
        if func:
            func(self.statex,self.statey,PRESS)
        e.accept()

    def mouseMoveEvent(self,e):
        """Process a mouse move event."""
        # the MOVE event does not identify a button, use the saved one
        func = self.getMouseFunc()
        if func:
            func(e.x(),self.height()-e.y(),MOVE)
        e.accept()

    def mouseReleaseEvent(self,e):
        """Process a mouse release event."""
        func = self.getMouseFunc()
        self.button = None        # clear the stored button
        if func:
            func(e.x(),self.height()-e.y(),RELEASE)
        e.accept()

    def wheelEvent(self,e):
        """Process a wheel event."""
        func = self.wheel_zoom
        if func:
            func(e.delta())
        e.accept()



    # Any keypress with focus in the canvas generates a GUI WAKEUP signal.
    # This is used to break out of a wait status.
    # Events not handled here could also be handled by the toplevel
    # event handler.
    def keyPressEvent (self,e):
        # pf.GUI.signals.WAKEUP.emit()
        # Done byt the top level (GUI) ??
        # Move all of these to GUI ??
        if e.key() == ESC:
            self.emit(CANCEL,())
            e.accept()
        elif e.key() == ENTER or e.key() == RETURN:
            self.emit(DONE,())
            e.accept()
        else:
            e.ignore()

################# Multiple Viewports ###############

class NewiMultiCanvas(QtGui.QGridLayout):
    """An OpenGL canvas with multiple viewports and QT interaction.

    The MultiCanvas implements a central QT widget containing one or more
    QtCanvas widgets.
    """
    def __init__(self,parent=None):
        """Initialize the multicanvas."""
        QtGui.QGridLayout.__init__(self)
        self.all = []
        self.current = None
        self.rowwise = True
        self.parent = parent


    def changeLayout(self,nvps=None,ncols=None,nrows=None,pos=None,rstretch=None,cstretch=None):
        """Change the lay-out of the viewports on the OpenGL widget.

        nvps: number of viewports
        ncols: number of columns
        nrows: number of rows
        pos: list holding the position and span of each viewport
        [[row,col,rowspan,colspan],...]
        rstretch: list holding the stretch factor for each row
        cstretch: list holding the stretch factor for each column
        (rows/columns with a higher stretch factor take more of the
        available space)
        Each of this parameters is optional.

        If pos is given, it specifies all viewports and nvps, nrows and ncols
        are disregarded.

        Else:

        If nvps is given, it specifies the number of viewports in the layout.
        Else, nvps will be set to the current number of viewports.

        If ncols is an int, viewports are laid out rowwise over ncols
        columns and nrows is ignored. If ncols is None and nrows is an int,
        viewports are laid out columnwise over nrows rows.

        If nvps is not equal to the current number of viewports, viewports
        will be added or removed to match the requested number.

        By default they are laid out rowwise over two columns.
        """
        if pos is None:
            # get the new layout definition
            if nvps is None:
                nvps = len(self.all)
            if ncols is None:
                if nrows is None:
                    ncols = self.ncols()
            if isInt(ncols):
                pos = [ divmod(i,ncols) for i in range(nvps) ]
            elif isInt(nrows):
                pos = [ divmod(i,nrows)[::-1] for i in range(nvps) ]
            else:
                return
        else:
            nvps = len(pos)

        while len(self.all) < nvps:
            # create new viewports
            view = self.createView()
            self.all.append(view)

        while len(self.all) > nvps:
            # remove viewports
            self.removeView()

        # remove all views from the canvas
        for w in self.all:
            self.removeWidget(w)
           # w.hide()

        # create the new layout
        for view,args in zip(self.all,pos):
            self.addView(view,*args)
        self.setCurrent(self.all[-1])


    def createView(self,shared=None):
        """Create a new viewport

        If another QtCanvas instance is passed, both will share the same
        display lists and textures.
        """
        if shared is not None:
            pf.debug("SHARING display lists WITH %s" % shared,pf.DEBUG.DRAW)
        view = QtCanvas(self.parent,shared)
        if len(self.all) > 0:
            # copy default settings from previous
            view.resetDefaults(self.all[-1].settings)
        return(view)


    def addView(self,view,row,col,rowspan=1,colspan=1):
        """Add a new viewport and make it visible"""
        self.addWidget(view,row,col,rowspan,colspan)
        view.raise_()
        view.initializeGL()   # Initialize OpenGL context and camera


    def removeView(self,view=None):
        """Remove a view from the canvas

        If view is None, the last one is removed.
        You can not remove a view when there is only one left.
        """
        if len(self.all) > 1:
            if view is None:
                view = self.all.pop()
            else:
                i = self.all.find(view)
                if i < 0:
                    return
                view = self.all.pop(i)
                if self.current == view:
                    self.setCurrent(self.all[i-1])
                self.removeWidget(view)
                view.close()


    def setCurrent(self,view):
        """Make the specified viewport the current one.

        view can be either a viewport or viewport number.
        The current viewport is the one that will be used for drawing
        operations. This may be different from the viewport having GUI
        focus (pf.canvas).
        """
        #print "NEWCANVAS SETTING CURRENT VIEWPORT"
        if isInt(view) and view in range(len(self.all)):
            view = self.all[view]
        if view == self.current:
            return  # already current

        if view in self.all:
            if self.current:
                self.current.focus = False
                self.current.updateGL()
            self.current = view
            self.current.focus = True
            self.current.updateGL()

        pf.canvas = pf.GUI.viewports.current



    def currentView(self):
        return self.all.index(self.current)


    def nrows(self):
        return self.rowCount()


    def ncols(self):
        return self.columnCount()


    def setStretch(self,rowstretch,colstretch):
        """Set the row and column stretch factors.

        rowstretch and colstretch are lists of stretch factors to be applied
        on the subsequent rows/columns. If the lists are shorter than the
        number of rows/columns, the
        """
        if rowstretch:
            for i in range(min(len(rowstretch),self.nrows())):
                self.setRowStretch(i,rowstretch[i])
        if colstretch:
            for i in range(min(len(colstretch),self.ncols())):
                self.setColumnStretch(i,colstretch[i])


    def updateAll(self):
         pf.debug("UPDATING ALL VIEWPORTS",pf.DEBUG.GUI)
         for v in self.all:
             v.update()
         pf.GUI.processEvents()


    def printSettings(self):
        for i,v in enumerate(self.all):
            pf.message("""
## VIEWPORTS ##
Viewport %s;  Current:%s;  Settings:
%s
""" % (i, v == self.current, v.settings))


    def link(self,vp,to):
        """Link viewport vp to to"""
        print("LINK %s to %s" % (vp,to))
        print("LINKING CURRENTLY DISABLED")
        return
        nvps = len(self.all)
        if vp in range(nvps) and to in range(nvps) and vp != to:
            to = self.all[to]
            oldvp = self.all[vp]
            import warnings
            warnings.warn('warn_viewport_linking')
            newvp = self.newView(to)
            self.all[vp] = newvp
            self.removeWidget(oldvp)
            oldvp.close()
            self.showWidget(newvp)
            vp = newvp
            vp.actors = to.actors
            vp.bbox = to.bbox
            vp.show()
            vp.setCamera()
            vp.redrawAll()
            #vp.updateGL()
            pf.GUI.processEvents()



class FramedGridLayout(QtGui.QGridLayout):
    """A QtGui.QGridLayout where each added widget is framed."""

    def __init__(self,parent=None):
        """Initialize the multicanvas."""
        QtGui.QGridLayout.__init__(self)
        self.setContentsMargins(0,0,0,0)
#       self.frames = []


    def addWidget(*args):
#        f = QtGui.QFrame(w)
#        self.frames.append(f)
        QtGui.QGridLayout.addWidget(*args)


    def removeWidget(self,w):
        QtGui.QGridLayout.removeWidget(self,w)


class MultiCanvas(FramedGridLayout):
    """An OpenGL canvas with multiple viewports and QT interaction.

    The MultiCanvas implements a central QT widget containing one or more
    QtCanvas widgets.
    """
    def __init__(self,parent=None):
        """Initialize the multicanvas."""
        FramedGridLayout.__init__(self)
        self.all = []
        self.current = None
        self.ncols = 2
        self.rowwise = True
        self.pos = None
        self.rstretch = None
        self.cstretch = None
        self.parent = parent


    def newView(self,shared=None,settings=None):
        """Create a new viewport

        If another QtCanvas instance is passed, both will share the same
        display lists and textures.
        """
        if shared is not None:
            pf.debug("SHARING display lists WITH %s" % shared,pf.DEBUG.DRAW)
        if settings is None:
            try:
                settings = self.current.settings
            except:
                settings = {}
        pf.debug("Create new viewport with settings:\n%s"%settings,pf.DEBUG.CANVAS)
        ##
        ## BEWARE: shared should be positional, settings should be keyword !
        canv = QtCanvas(self.parent,shared,settings=settings)
        #print(canv.settings)
        return(canv)


    def addView(self):
        """Add a new viewport to the widget"""
        canv = self.newView()
        if len(self.all) > 0:
            # copy default settings from previous
            canv.resetDefaults(self.all[-1].settings)
        self.all.append(canv)
        self.showWidget(canv)
        canv.initializeGL()   # Initialize OpenGL context and camera
        self.setCurrent(canv)


    def setCurrent(self,canv):
        """Make the specified viewport the current one.

        canv can be either a viewport or viewport number.
        """
        #print "SETTING CURRENT VIEWPORT"
        if isInt(canv) and canv in range(len(self.all)):
            canv = self.all[canv]
        if canv == self.current:
            #print "ALREADY CURRENT"
            pass
        elif canv in self.all:
            #print "CHANGING CURRENT"
            if self.current:
                self.current.focus = False
                self.current.updateGL()
            self.current = canv
            self.current.focus = True
            self.current.updateGL()
            toolbar.updateViewportButtons(self.current)

        pf.canvas = self.current


    def viewIndex(self,view):
        """Return the index of the specified view"""
        return self.all.index(view)


    def currentView(self):
        return self.all.index(self.current)


    def showWidget(self,w):
        """Show the view w."""
        ind = self.all.index(w)
        if self.pos is None:
            row,col = divmod(ind,self.ncols)
            if not self.rowwise:
                row,col = col,row
            rspan,cspan = 1,1
        elif ind < len(self.pos):
            row,col,rspan,cspan = self.pos[ind]
        else:
            return
        self.addWidget(w,row,col,rspan,cspan)
        w.raise_()
        # set the stretch factors
        if self.rstretch is not None:
            for i in range(row,row+rspan):
                if i >= len(self.rstretch):
                    self.rstretch.append(1)
                self.setRowStretch(i,self.rstretch[i])
        if self.cstretch is not None:
            for i in range(col,col+cspan):
                if i >= len(self.cstretch):
                    self.cstretch.append(1)
                self.setColumnStretch(i,self.cstretch[i])


    def removeView(self):
        if len(self.all) > 1:
            w = self.all.pop()
            if self.pos is not None:
                self.pos = self.pos[:-1]
            if self.current == w:
                self.setCurrent(self.all[-1])
            self.removeWidget(w)
            w.close()
            # set the stretch factors
            pos = [self.getItemPosition(self.indexOf(w)) for w in self.all]
            if self.rstretch is not None:
                row = max([p[0]+p[2] for p in pos])
                for i in range(row,len(self.rstretch)):
                    self.setRowStretch(i,0)
                self.rstretch = self.rstretch[:row]
            if self.cstretch is not None:
                col = max([p[1]+p[3] for p in pos])
                for i in range(col,len(self.cstretch)):
                    self.setColumnStretch(i,0)
                self.cstretch = self.cstretch[:col]


##     def setCamera(self,bbox,view):
##         self.current.setCamera(bbox,view)

    def updateAll(self):
         pf.debug("UPDATING ALL VIEWPORTS",pf.DEBUG.GUI)
         for v in self.all:
             v.update()
         pf.GUI.processEvents()

    def printSettings(self):
        for i,v in enumerate(self.all):
            pf.message("""
## VIEWPORTS ##
Viewport %s;  Current:%s;  Settings:
%s
""" % (i, v == self.current, v.settings))


    def changeLayout(self,nvps=None,ncols=None,nrows=None,pos=None,rstretch=None,cstretch=None):
        """Change the lay-out of the viewports on the OpenGL widget.

        nvps: number of viewports
        ncols: number of columns
        nrows: number of rows
        pos: list holding the position and span of each viewport
        [[row,col,rowspan,colspan],...]
        rstretch: list holding the stretch factor for each row
        cstretch: list holding the stretch factor for each column
        (rows/columns with a higher stretch factor take more of the
        available space)
        Each of this parameters is optional.

        If a number of viewports is given, viewports will be added
        or removed to match the requested number.
        By default they are laid out rowwise over two columns.

        If ncols is an int, viewports are laid out rowwise over ncols
        columns and nrows is ignored. If ncols is None and nrows is an int,
        viewports are laid out columnwise over nrows rows. Alternatively,
        the pos argument can be used to specify the layout of the viewports.
        """
        # add or remove viewports to match the requested number
        if isInt(nvps):
            while len(self.all) > nvps:
                self.removeView()
            while len(self.all) < nvps:
                self.addView()
        # get the new layout definition
        if isInt(ncols):
            rowwise = True
            pos = None
        elif isInt(nrows):
            ncols = nrows
            rowwise = False
            pos = None
        elif type(pos) == list and len(pos) == len(self.all):
            ncols = None
            rowwise = None
        else:
            return
        # remove the viewport widgets
        for w in self.all:
            self.removeWidget(w)
        # assign the new layout arguments
        self.ncols = ncols
        self.rowwise = rowwise
        self.pos = pos
        self.rstretch = rstretch
        self.cstretch = cstretch
        # add the viewport widgets
        for w in self.all:
            self.showWidget(w)


    def link(self,vp,to):
        """Link viewport vp to to"""
        print("LINK %s to %s" % (vp,to))
        nvps = len(self.all)
        if vp in range(nvps) and to in range(nvps) and vp != to:
            to = self.all[to]
            oldvp = self.all[vp]
            import warnings
            warnings.warn('warn_viewport_linking')
            newvp = self.newView(to)
            self.all[vp] = newvp
            self.removeWidget(oldvp)
            oldvp.close()
            self.showWidget(newvp)
            vp = newvp
            vp.actors = to.actors
            vp.bbox = to.bbox
            vp.show()
            vp.setCamera()
            vp.redrawAll()
            #vp.updateGL()
            pf.GUI.processEvents()


def _auto_initialize():
    global MultiCanvas
    try:
        if pf.options.newviewports:
            MultiCanvas = NewMultiCanvas
    except:
        pass

_auto_initialize()

# End
