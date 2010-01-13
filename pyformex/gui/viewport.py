## $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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
"""Interactive OpenGL Canvas embedded in a Qt4 widget.

This module implements user interaction with the OpenGL canvas defined in
module :mod:`canvas`.
`QtCanvas` is a single interactive OpenGL canvas, while `MultiCanvas`
implements a dynamic array of multiple canvases.
"""

import pyformex as GD

from PyQt4 import QtCore, QtGui, QtOpenGL
from OpenGL import GL

from collection import Collection
import canvas
import decors
import image
import utils
import toolbar

from ctypes import cdll
libGL = cdll.LoadLibrary("libGL.so.1")

import math

from coords import Coords
from numpy import *
#asarray,where,unique,intersect1d,setdiff1d,empty,append,delete


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
    
       --dri   : use the Direct Rendering Infrastructure
       --nodri : do not use the DRI
       --alpha : enable the alpha buffer 
    """
    global opengl_format
    fmt = QtOpenGL.QGLFormat.defaultFormat()
    if GD.options.dri is not None:
        fmt.setDirectRendering(GD.options.dri)
##     if GD.options.alpha:
##         fmt.setAlpha(True)
    QtOpenGL.QGLFormat.setDefaultFormat(fmt)
    opengl_format = fmt
    if GD.options.debug:
        print(OpenGLFormat())
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
    s = """
OpenGL: %s
OpenGL Version: %s
OpenGLOverlays: %s
Double Buffer: %s
Depth Buffer: %s
RGBA: %s
Alpha Channel: %s
Accumulation Buffer: %s
Stencil Buffer: %s
Stereo: %s
Direct Rendering: %s
Overlay: %s
Plane: %s
Multisample Buffers: %s
""" % (fmt.hasOpenGL(),
       int(fmt.openGLVersionFlags()),
       fmt.hasOpenGLOverlays(),
       fmt.doubleBuffer(),fmt.depth(),
       fmt.rgba(),fmt.alpha(),
       fmt.accum(),
       fmt.stencil(),
       fmt.stereo(),
       fmt.directRendering(),
       fmt.hasOverlay(),
       fmt.plane(),
       fmt.sampleBuffers()
       )
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
        self.setCursor(QtGui.QCursor(QtCanvas.cursor_shape[shape]))


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
        #print(button,mod)
        self.mousefncsaved[mod][button].append(self.mousefnc[mod][button])
        self.mousefnc[mod][button] = func
        self.setCursorShapeFromFunc(func)
        #print("MOUSE %s" % func)
        #print("MOUSE SAVED %s" % self.mousefncsaved[mod][button])


    def resetMouse(self,button,mod=NONE):
        #print("MOUSE SAVED %s" % self.mousefncsaved[mod][button])
        try:
            func = self.mousefncsaved[mod][button].pop()
        except:
            #print("AAAAAHHH, COULD NOT POP")
            func = None
        self.mousefnc[mod][button] = func
        self.setCursorShapeFromFunc(func)
        #print("RESETMOUSE %s" % func)
        #print("MOUSE SAVED %s" % self.mousefncsaved[mod][button])
            

    def getMouseFunc(self):
        """Return the mouse function bound to self.button and self.mod"""
        return self.mousefnc.get(int(self.mod),{}).get(self.button,None)




################# Single Interactive OpenGL Canvas ###############


class QtCanvas(QtOpenGL.QGLWidget,canvas.Canvas):
    """A canvas for OpenGL rendering.

    This class provides interactive functionality for the OpenGL canvas
    provided by the canvas.Canvas class.
    
    Interactivity is highly dependent on Qt4. Putting the interactive
    functions in a separate class makes it esier to use the Canvas class
    in non-interactive situations or combining it with other GUI toolsets.
    """
    cursor_shape = { 'default': QtCore.Qt.ArrowCursor,
                     'pick'   : QtCore.Qt.CrossCursor, 
                     'busy'   : QtCore.Qt.BusyCursor,
                     }
    
    def __init__(self,*args):
        """Initialize an empty canvas with default settings."""
        QtOpenGL.QGLWidget.__init__(self,*args)
        self.setMinimumSize(32,32)
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        canvas.Canvas.__init__(self)
        self.setCursorShape('default')
        self.button = None
        self.mod = NONE
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
        self.pick_func = {
            'actor'  : self.pick_actors,
            'element': self.pick_elements,
            'point'  : self.pick_points,
            'edge'   : self.pick_edges,
            'number' : self.pick_numbers,
            }
        self.pickable = None
        self.drawing_mode = None
        self.drawing = None
        # Drawing options
        self.resetOptions()

    def resetOptions(self):
        """Reset the Drawing options to some defaults"""
        self.options = dict(
            view = None,       # Keep the current camera angles
            bbox = 'auto',     # Automatically zoom on the drawed object
            clear = False,      # Clear on each drawing action
            shrink = None,
            )

    def setOptions(self,d):
        """Set the Drawing options to some values"""
        self.options.update(d)
        

    def setCursorShape(self,shape):
        """Set the cursor shape to shape"""
        if shape not in QtCanvas.cursor_shape.keys():
            shape = 'default'
        self.setCursor(QtGui.QCursor(QtCanvas.cursor_shape[shape]))


    def setCursorShapeFromFunc(self,func):
        """Set the cursor shape to shape"""
        if func in [ self.mouse_rectangle_zoom,self.mouse_pick ]:
            shape = 'pick'
        else:
            shape = 'default'
        self.setCursorShape(shape)


    def setMouse(self,button,func,mod=NONE):
        self.mousefncsaved[mod][button].append(self.mousefnc[mod][button])
        self.mousefnc[mod][button] = func
        if button == LEFT and mod == NONE:
            self.setCursorShapeFromFunc(func)


    def resetMouse(self,button,mod=NONE):
        try:
            func = self.mousefncsaved[mod][button].pop()
        except:
            func = None
        self.mousefnc[mod][button] = func
        if button == LEFT and mod == NONE:
            self.setCursorShapeFromFunc(func)
            

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
            self.begin_2D_drawing()
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
        

    def start_selection(self,mode,filtr):
        """Start an interactive picking mode.

        If selection mode was already started, mode is disregarded and
        this can be used to change the filter method.
        """
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

        if filtr == 'none':
            filtr = None
        self.selection_filter = filtr
        if filtr is None:
            self.selection_front = None
        self.selection.clear()
        self.selection.setType(self.selection_mode)


    def wait_selection(self):
        """Wait for the user to interactively make a selection."""
        self.selection_timer = QtCore.QThread
        self.selection_busy = True
        while self.selection_busy:
            self.selection_timer.msleep(20)
            GD.app.processEvents()

    def finish_selection(self):
        """End an interactive picking mode."""
        self.resetMouse(LEFT)
        self.resetMouse(LEFT,SHIFT)
        self.resetMouse(LEFT,CTRL)
        self.resetMouse(RIGHT)
        self.resetMouse(RIGHT,SHIFT)
        self.disconnect(self,DONE,self.accept_selection)
        self.disconnect(self,CANCEL,self.cancel_selection)
        self.selection_mode = None

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

    
    def pick(self,mode='actor',single=False,func=None,filter=None):
        """Interactively pick objects from the viewport.

        - `mode`: defines what to pick : one of
          ``['actor','element','point','number','edge']``
        - `single`: if True, the function returns as soon as the user ends
          a picking operation. The default is to let the user
          modify his selection and only to return after an explicit
          cancel (ESC or right mouse button).
        - `func`: if specified, this function will be called after each
          atomic pick operation. The Collection with the currently selected
          objects is passed as an argument. This can e.g. be used to highlight
          the selected objects during picking.
        - `filter`: defines what elements to retain from the selection: one of
          ``[None,'closest,'connected']``.

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
                            conn_elems = self.actors[closest_actor].connectedElements(closest_elem,self.selection.get(closest_actor))
                            self.selection.set(conn_elems,closest_actor)
                    if func:
                        func(self.selection)
                self.update()
            if single:
                self.accept_selection()
        if func and not self.selection_accepted:
            func(self.selection)
        self.finish_selection()
        return self.selection
    

    def pickNumbers(self,*args,**kargs):
        """Go into number picking mode and return the selection."""
        return self.pick('numbers',*args,**kargs)

#################### Interactive drawing ####################################

    def mouse_draw(self,x,y,action):
        """Process mouse events during interactive drawing.

        On PRESS, do nothing.
        On MOVE, do nothing.
        On RELEASE, add the point to the point list.
        """
        if action == PRESS:
            self.makeCurrent()
            self.update()

        elif action == MOVE:
            self.drawn = self.unProject(x,y,self.zplane)
            if GD.app.hasPendingEvents():
                return
            if self.previewfunc:
                self.swapBuffers()
                self.drawn = Coords(self.drawn).reshape(-1,3)
                self.previewfunc(Coords.concatenate([self.drawing,self.drawn]))
                self.swapBuffers()
            
        elif action == RELEASE:
            self.drawn = self.unProject(x,y,self.zplane)
            self.selection_busy = False


    def start_draw(self,mode,zplane=0.):
        """Start an interactive drawing mode."""
        self.setMouse(LEFT,self.mouse_draw)
        self.setMouse(RIGHT,self.emit_done)
        self.setMouse(RIGHT,self.emit_cancel,SHIFT)
        self.connect(self,DONE,self.accept_draw)
        self.connect(self,CANCEL,self.cancel_draw)
        self.drawmode = mode
        self.zplane = zplane
        self.drawing = Coords()

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

    def draw(self,mode='point',npoints=-1,zplane=0.,func=None, preview=False):
        """Interactively draw on the canvas.

        - single: if True, the function returns as soon as the user ends
        a drawing operation. The default is to let the user
        draw multiple lines and only to return after an explicit
        cancel (ESC or right mouse button).
        - func: if specified, this function will be called after each
        atomic drawing operation. The current drawing is passed as
        an argument. This can e.g. be used to show the drawing.
        When the drawing operation is finished, the drawing is returned.
        The return value is a (n,2,2) shaped array.
        """
        self.draw_canceled = False
        self.start_draw(mode,zplane)
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
                    func(self.drawing)
            if npoints > 0 and len(self.drawing) >= npoints:
                self.accept_draw()                
        if func and not self.draw_accepted:
            func(self.drawing)
        self.finish_draw()
        return self.drawing

##########################################################################

    def start_drawing(self,mode):
        """Start an interactive line drawing mode."""
        GD.debug("START DRAWING MODE")
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
            GD.app.processEvents()

    def finish_drawing(self):
        """End an interactive drawing mode."""
        GD.debug("END DRAWING MODE")
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
        GD.debug("CANCEL DRAWING MODE")
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

    def drawLinesInter(self,mode='line',single=False,func=None):
        """Interactively draw lines on the canvas.

        - single: if True, the function returns as soon as the user ends
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
            if single:
                self.accept_drawing()                
        if func and not self.drawing_accepted:
            func(self.drawing)
        self.finish_drawing()
        return self.drawing


######## QtOpenGL interface ##############################
        
    def initializeGL(self):
        if GD.options.debug:
            p = self.sizePolicy()
            print(p.horizontalPolicy(), p.verticalPolicy(), p.horizontalStretch(), p.verticalStretch())
        self.initCamera()
        self.glinit()
        self.resizeGL(self.width(),self.height())
        self.setCamera()

    def	resizeGL(self,w,h):
        self.setSize(w,h)

    def	paintGL(self):
        if not self.mode2D:
            #GD.debugt("CANVAS DISPLAY")
            self.display()

    def getSize(self):
        return int(self.width()),int(self.height())

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
    # Functions that change the camera settings should call saveMatrix()
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
                # GD.debug(d)
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
            self.camera.saveMatrix()

            
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
            self.camera.saveMatrix()          

            
    def dynazoom(self,x,y,action):
        """Perform dynamic zoom operation.

        This function processes mouse button events controlling a dynamic
        zoom operation. The action is one of PRESS, MOVE or RELEASE.
        """
        if action == PRESS:
            self.state = [self.camera.getDist(),self.camera.area.tolist(),GD.cfg['gui/dynazoom']]

        elif action == MOVE:
            w,h = self.getSize()
            dx,dy = float(self.statex-x)/w, float(self.statey-y)/h
            for method,state,value,size in zip(self.state[2],[self.statex,self.statey],[x,y],[w,h]):
                #GD.debug("%s %s %s %s" % (method,state,value,size))
                if method == 'area':
                    d = float(state-value)/size
                    f = exp(4*d)
                    self.camera.zoomArea(f,area=asarray(self.state[1]).reshape(2,2))
                elif method == 'dolly':
                    d = utils.stuur(value,[0,state,size],[5,1,0.2],1.2)
                    #GD.debug(d)
                    self.camera.setDist(d*self.state[0])
                    
            self.update()

        elif action == RELEASE:
            self.update()
            self.camera.saveMatrix()

    def wheel_zoom(self,delta):
        """Zoom by rotating a wheel over an angle delta"""
        f = 2**(delta/120.*GD.cfg['gui/wheelzoomfactor'])
        if GD.cfg['gui/wheelzoom'] == 'area':
            self.camera.zoomArea(f)
        elif GD.cfg['gui/wheelzoom'] == 'lens':
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
               w,h = GD.cfg.get('pick/size',(20,20))
            vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
            self.pick_window = (x,y,w,h,vp)
            self.selection_busy = False


    def pick_actors(self):
        """Set the list of actors inside the pick_window."""
        self.camera.loadProjection(pick=self.pick_window)
        self.camera.loadMatrix()
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

        obj_type can be 'element', 'edge' or 'point'
        'edge' is only available for mesh type geometry
        max_objects specifies the maximum number of objects

        The picked object numbers are stored in self.picked.
        If store_closest==True, the closest picked object is stored in as a
        tuple ( [actor,object] ,distance) in self.picked_closest

        A list of actors from which can be picked may be given.
        If so, the resulting keys are indices in this list.
        By default, the full actor list is used.
        """
        self.picked = []
        if max_objects <= 0:
            GD.message("No such objects to be picked!")
            return
        self.camera.loadProjection(pick=self.pick_window)
        self.camera.loadMatrix()
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
            #GD.debug("PICKBUFFER: %s" % self.picked)
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
            #GD.debug("ADDING %s pickable points"%a.npoints())
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


    def pick_numbers(self):
        """Return the numbers inside the pick_window."""
        self.camera.loadProjection(pick=self.pick_window)
        self.camera.loadMatrix()
        self.picked = [0,1,2,3]
        if self.numbers:
            self.picked = self.numbers.drawpick()


    def draw_state_line(self,x,y):
        """Store the pos and draw a line to it."""
        self.state = x,y
        #GD.debug("Rect (%s,%s) - (%s,%s)" % (self.statex,self.statey,x,y))
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
        GD.GUI.viewports.setCurrent(self)
        # on PRESS, always remember mouse position and button
        self.statex,self.statey = e.x(), self.height()-e.y()
        self.button = e.button()
        self.mod = e.modifiers() & ALLMODS
        #GD.debug("PRESS BUTTON %s WITH MODIFIER %s" % (self.button,self.mod))
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
        


    # Any keypress with focus in the canvas generates a 'wakeup' signal.
    # This is used to break out of a wait status.
    # Events not handled here could also be handled by the toplevel
    # event handler.
    def keyPressEvent (self,e):
        self.emit(WAKEUP,())
        if e.key() == ESC:
            self.emit(CANCEL,())
            e.accept()
        elif e.key() == ENTER or e.key() == RETURN:
            self.emit(DONE,())
            e.accept()
        else:
            e.ignore()

################# Multiple Viewports ###############


class FramedGridLayout(QtGui.QGridLayout):
    """A QtGui.QGridLayout where each added widget is framed."""

    def __init__(self,parent=None):
        """Initialize the multicanvas."""
        QtGui.QGridLayout.__init__(self)
 #       self.frames = []

        
    def addWidget(*args):
#        f = QtGui.QFrame(w)
#        self.frames.append(f)
        QtGui.QGridLayout.addWidget(*args)

    
    def removeWidget(self,w):
        QtGui.QGridLayout.removeWidget(self,w)



## class FramedGridLayout(QtGui.QSplitter):
##     """A QtGui.QGridLayout where each added widget is framed."""

##     def __init__(self,parent=None):
##         """Initialize the multicanvas."""
##         QtGui.QSplitter.__init__(self)
        
##     def addWidget(self,w,row,col):
##         QtGui.QSplitter.addWidget(self,w,row,col)
    
##     def removeWidget(self,w):
##         QtGui.QSplitter.removeWidget(self,w)

    


#class MultiCanvas(QtGui.QGridLayout):
class MultiCanvas(FramedGridLayout):
    """An OpenGL canvas with multiple viewports and QT interaction.

    The MultiCanvas implements a central QT widget containing one or more
    QtCanvas widgets.
    """
    def __init__(self,parent=None):
        """Initialize the multicanvas."""
        FramedGridLayout.__init__(self)
#        QtGui.QGridLayout.__init__(self)
        self.all = []
        self.active = []
        self.current = None
        self.ncols = 2
        self.rowwise = True
        self.parent = parent


    def setDefaults(self,dict):
        """Update the default settings of the canvas class."""
        GD.debug("Setting canvas defaults:\n%s" % dict)
        canvas.CanvasSettings.default.update(canvas.CanvasSettings.checkDict(dict))

    def newView(self,shared=None):
        "Create a new viewport"
        canv = QtCanvas(self.parent,shared)
        return(canv)
        

    def addView(self):
        """Add a new viewport to the widget"""
        canv = self.newView()
        self.all.append(canv)
        self.active.append(canv)
        self.showWidget(canv)
        canv.initializeGL()   # Initialize OpenGL context and camera
        # DO NOT USE self.setCurrent(canv) HERE, because no camera yet
        #GD.canvas = self.current = canv
        self.setCurrent(canv)
        

    def setCurrent(self,canv):
        """Make the specified viewport the current  one.

        canv can be either a viewport or viewport number.
        """
#        GL.glFlush()
        if type(canv) == int and canv in range(len(self.all)):
            canv = self.all[canv]
        if self.current == canv:
            # alreay current
            return
        if GD.canvas:
            GD.canvas.focus = False
            GD.canvas.updateGL()
        if canv in self.all:
            GD.canvas = self.current = canv
            GD.canvas.focus = True
            toolbar.setTransparency(self.current.alphablend)
            toolbar.setPerspective(self.current.camera.perspective)
            toolbar.setLight(self.current.lighting)
            #toolbar.setNormals(self.current.avgnormals)
#            GL.glFlush()
            GD.canvas.updateGL()


    def currentView(self):
        return self.all.index(GD.canvas)


    def showWidget(self,w):
        """Show the view w"""
        ind = self.all.index(w)
        row,col = divmod(ind,self.ncols)

        ## print("Viewport %s %s,%s,%s" % (self.rowwise,row,col,self.ncols))

        ## if row > 0 and col < self.ncols and ind == len(self.all)-1:
        ##     print("SPANMULTIPLE")
        ##     rspan,cspan = 1,2
        ##     if not self.rowwise:
        ##         row,col = col,row
        ##         rspan,cspan = cspan,rspan
        ##     self.addWidget(w,row,col,rspan,cspan)
        ## else:
        if not self.rowwise:
            row,col = col,row
        self.addWidget(w,row,col)
        w.raise_()


    def removeView(self):
        if len(self.all) > 1:
            w = self.all.pop()
            if self.current == w:
                self.setCurrent(self.all[-1])
            if w in self.active:
                self.active.remove(w)
            self.removeWidget(w)
            w.close()


##     def setCamera(self,bbox,view):
##         self.current.setCamera(bbox,view)
            
    def updateAll(self):
         GD.debug("UPDATING ALL VIEWPORTS")
         for v in self.all:
             v.update()
         GD.app.processEvents()

    def removeAll(self):
        for v in self.active:
            v.removeAll()

##     def clear(self):
##         self.current.clear()  

    def addActor(self,actor):
        for v in self.active:
            v.addActor(actor)


    def printSettings(self):
        for i,v in enumerate(self.all):
            GD.message("""
## VIEWPORTS ##
Viewport %s;  Active:%s;  Current:%s;  Settings:
%s
""" % (i,v in self.active, v == self.current, v.settings))


    def changeLayout(self, nvps=None,ncols=None, nrows=None):
        """Lay out the viewports.

        You can specify the number of viewports and the number of columns or
        rows.

        If a number of viewports is given, viewports will be added
        or removed to match the number requested.
        By default they are layed out rowwise over two columns.

        If ncols is an int, viewports are laid out rowwise over ncols
        columns and nrows is ignored. If ncols is None and nrows is an int,
        viewports are laid out columnwise over nrows rows.
        """
        if type(nvps) == int:
            while len(self.all) > nvps:
                self.removeView()
            while len(self.all) < nvps:
                self.addView()

        if type(ncols) == int:
            rowwise = True
        elif type(nrows) == int:
            ncols = nrows
            rowwise = False
        else:
            return

        for w in self.all:
            self.removeWidget(w)
        self.ncols = ncols
        self.rowwise = rowwise
        for w in self.all:
            self.showWidget(w)


    def link(self,vp,to):
        """Link viewport vp to to"""
        nvps = len(self.all)
        if vp < nvps and to < nvps:
            to = self.all[to]
            oldvp = self.all[vp]
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
            GD.app.processEvents()

                    
# End
