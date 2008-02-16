# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""viewport.py: Interactive OpenGL Canvas.

This module implements user interaction with the OpenGL canvas.
QtCanvas is a single interactive OpenGL canvas, while MultiCanvas
implements a dynamic array of multiple canvases.

The basic OpenGL drawing functionality is implemented in the canvas module.
"""

import globaldata as GD

from PyQt4 import QtCore, QtGui, QtOpenGL
from OpenGL import GL

from numtools import Collection
import canvas
import image
import utils
import toolbar

import math

from coords import Coords

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
CANCEL = QtCore.SIGNAL("Cancel")
DONE   = QtCore.SIGNAL("Done")
WAKEUP = QtCore.SIGNAL("Wakeup")

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

# modifiersdrawSe
NONE = QtCore.Qt.NoModifier
SHIFT = QtCore.Qt.ShiftModifier
CTRL = QtCore.Qt.ControlModifier
ALT = QtCore.Qt.AltModifier
ALLMODS = SHIFT | CTRL | ALT


def modifierName(mod):
    return {NONE:'NONE',SHIFT:'SHIFT',CTRL:'CTRL',ALT:'ALT'}.get(mod,'UNKNOWN')


############### OpenGL Format #################################

def setOpenGLFormat():
    """Set the correct OpenGL format.

    The default OpenGL format can be changed by command line options.
    --dri   : use the DIrect Rendering Infrastructure
    --nodri : do no use the DRI
    --alpha : enable the alpha buffer 
    """
    fmt = QtOpenGL.QGLFormat.defaultFormat()
    fmt.setDirectRendering(GD.options.dri)
    if GD.options.alpha:
        fmt.setAlpha(True)
    if GD.options.debug:
        printOpenGLFormat(fmt)
    QtOpenGL.QGLFormat.setDefaultFormat(fmt)
    return fmt

def getOpenGLContext():
    ctxt = QtOpenGL.QGLContext.currentContext()
    if ctxt is not None:
        printOpenGLContext(ctxt)
    return ctxt

def printOpenGLFormat(fmt):
    """Print some information about the OpenGL format."""
    print "OpenGL: ",fmt.hasOpenGL()
    print "OpenGL Version: %s" % str(fmt.openGLVersionFlags()) 
    print "OpenGLOverlays: ",fmt.hasOpenGLOverlays()
    print "Double Buffer: ",fmt.doubleBuffer()
    print "Depth Buffer: ",fmt.depth()
    print "RGBA: ",fmt.rgba()
    print "Alpha Channel: ",fmt.alpha()
    print "Accumulation Buffer: ",fmt.accum()
    print "Stencil Buffer: ",fmt.stencil()
    print "Stereo: ",fmt.stereo()
    print "Direct Rendering: ",fmt.directRendering()
    print "Overlay: ",fmt.hasOverlay()
    print "Plane: ",fmt.plane()
    print "Multisample Buffers: ",fmt.sampleBuffers()


def printOpenGLContext(ctxt):
    if ctxt:
        print "context is valid: %d" % ctxt.isValid()
        print "context is sharing: %d" % ctxt.isSharing()
    else:
        print "No OpenGL context yet!"


################# Single Interactive OpenGL Canvas ###############

class QtCanvas(QtOpenGL.QGLWidget,canvas.Canvas):
    """A canvas for OpenGL rendering.

    This class provides interactive functionality for the OpenGL canvas
    provided by the canvas.Canvas class.
    
    Interactivity is highly dependent on Qt4. Putting the interactive
    functions in a separate class makes it esier to use the Canvas class
    in non-interactive situations or combining it with other GUI toolsets.
    """

    ##
    ##
    ##  IMPORTANT : The mouse functionality should (and will) be moved
    ##              to the MultiCanvas class!
    ##

    
    def __init__(self,*args):
        """Initialize an empty canvas with default settings."""
        QtOpenGL.QGLWidget.__init__(self,*args)
        self.setMinimumSize(32,32)
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        canvas.Canvas.__init__(self)
        self.button = None
        self.mod = NONE
        self.mousefnc = { NONE:{}, SHIFT:{}, CTRL:{}, ALT:{} }
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
            'number' : self.pick_numbers,
            'edge'   : self.pick_lines,
            }


    def setMouse(self,button,func,mod=NONE):
        self.mousefnc[mod][button] = func


    def getMouseFunc(self):
        """Return the mouse function bound to self.button and self.mod"""
        return self.mousefnc.get(int(self.mod),{}).get(self.button,None)


    def start_selection(self,mode):
        """Start an interactive picking mode."""
        if self.selection_mode is None:
            GD.debug("START SELECTION MODE: %s" % mode)
            self.setMouse(LEFT,self.mouse_pick)
            self.setMouse(LEFT,self.mouse_pick,SHIFT)
            self.setMouse(LEFT,self.mouse_pick,CTRL)
            self.setMouse(RIGHT,self.emit_done)
            self.setMouse(RIGHT,self.emit_cancel,SHIFT)
            self.connect(self,DONE,self.accept_selection)
            self.connect(self,CANCEL,self.cancel_selection)
            self.selection_mode = mode
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

    def wait_selection(self):
        """Wait for the user to interactively make a selection."""
        self.selection_timer = QtCore.QThread
        self.selection_busy = True
        while self.selection_busy:
            self.selection_timer.msleep(20)
            GD.app.processEvents()

    def finish_selection(self):
        """End an interactive picking mode."""
        GD.debug("END SELECTION MODE")
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.setMouse(LEFT,self.dynarot)
        self.setMouse(LEFT,None,SHIFT)
        self.setMouse(LEFT,None,CTRL)
        self.setMouse(RIGHT,self.dynazoom)
        self.setMouse(RIGHT,None,SHIFT)
        self.disconnect(self,DONE,self.accept_selection)
        self.disconnect(self,CANCEL,self.cancel_selection)
        self.selection_mode = None

    def accept_selection(self,clear=False):
        """Cancel an interactive picking mode.

        If clear == True, the current selection is cleared.
        """
        GD.debug("CANCEL SELECTION MODE")
        if clear:
            self.selection.clear()
        self.selection_canceled = True
        self.selection_busy = False

    def cancel_selection(self):
        """Cancel an interactive picking mode and clear the selection."""
        self.accept_selection(clear=True)


    def pick(self,mode='actor',single=False,func=None):
        """Interactively pick objects from the viewport.

        - mode: defines what to pick : one of
        ['actor','element','point','number','edge']
        - single: if True, the function returns as soon as the user ends
        a picking operation. The default is to let the user
        modify his selection and only to return after an explicit
        cancel (ESC or right mouse button).
        - func: if specified, this function will be called after each
        atomic pick operation. The Collection with the currently selected
        objects is passed as an argument.
        This can e.g. be used to highlight the selected objects during picking.

        When the picking operation is finished, the selection is returned.
        The return value is always a Collection object.
        """
        if not single:
            GD.debug('PICK UNTIL ESC/RIGHT MOUSE')
        self.selection.clear()
        self.selection.setType(mode)
        self.selection_canceled = False
        self.start_selection(mode)
        while not self.selection_canceled:
            self.wait_selection()
            if not self.selection_canceled:
                self.pick_func[self.selection_mode]()
                if self.mod == NONE:
                    self.selection.set(self.picked)
                elif self.mod == SHIFT:
                    self.selection.add(self.picked)
                elif self.mod == CTRL:
                    self.selection.remove(self.picked)
            GD.debug("Selection: %s" % self.selection)
            if func:
                func(self.selection)
            if single:
                self.selection_canceled = True
        self.finish_selection()
        return self.selection
    

    def pickNumbers(self,*args,**kargs):
        """Go into number picking mode and return the selection."""
        return self.pick('numbers',*args,**kargs)


#### START OF OBSOLETE FUNCS

    
    def enableSelect(self,shape):
        """Start selection mode."""
        self.selection = []
        if shape == 'lines':
            self.setMouse(LEFT,self.pick_lines)
        self.mousefnc[CTRL][LEFT] = self.mousefnc[NONE][LEFT]
        self.mousefnc[ALT][LEFT] = self.mousefnc[NONE][LEFT]
        self.mousefnc[SHIFT] = {}
        self.mousefnc[SHIFT][LEFT] = self.dynarot


    def makeSelection(self):
        """Wait for the user to make a selection.
        
        When the CTRL button is pressed, add the selected item to selection,
        otherwise, replace selection.
        """
        self.selected = []
        while GD.gui.pushbutton.checkState() == False and len(self.selected) == 0:
            GD.app.processEvents()
        if GD.gui.pushbutton.checkState() == False:
            if int(self.mod) == CTRL:
                if len(self.selection) == 0:
                    self.selection = self.selected
                else:
                    self.selection = Coords.concatenate([self.selection,self.selected])
            else:
                self.selection = self.selected
        return self.selection


    def disableSelect(self):
        """End selection mode."""
        self.setMouse(LEFT,self.dynarot)
        GD.debug("Re-enabling dynarot")
        self.mousefnc[CTRL] = self.mousefnc[ALT] = {}
        self.mousefnc[SHIFT] = self.mousefnc[NONE]


    ##### QtOpenGL interface #####
        
    def initializeGL(self):
        if GD.options.debug:
            p = self.sizePolicy()
            print p.horizontalPolicy(), p.verticalPolicy(), p.horizontalStretch(), p.verticalStretch()
        self.initCamera()
        self.glinit()
        self.resizeGL(self.width(),self.height())
        self.setCamera()

    def	resizeGL(self,w,h):
        self.setSize(w,h)

    def	paintGL(self):
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
            dist = self.camera.getDist() * 0.5
            # get distance from where button was pressed
            dx,dy = (x-self.statex,y-self.statey)
            panx = utils.stuur(dx,[-w,0,w],[-dist,0.,+dist],1.0)
            pany = utils.stuur(dy,[-h,0,h],[-dist,0.,+dist],1.0)
            # print dx,dy,panx,pany
            self.camera.translate(panx,pany,0)
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
            self.state = [self.camera.getDist(),self.camera.fovy]

        elif action == MOVE:
            w,h = self.getSize()
            # hor movement is lens zooming
            f = utils.stuur(x,[0,self.statex,w],[180,self.state[1],0],1.2)
            #print "Lens Zooming: %s" % f
            self.camera.setLens(f)
            # vert movement is dolly zooming
            d = utils.stuur(y,[0,self.statey,h],[5,1,0.2],1.2)
            self.camera.setDist(d*self.state[0])
            self.update()

        elif action == RELEASE:
            self.update()
            self.camera.saveMatrix()


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


##     def pick_finish(self,x,y,action):
##         """Finish the selection mode by clicking a mouse button."""
##         if action == RELEASE:
##             self.finish_selection()


    def mouse_pick(self,x,y,action):
        """Process mouse events during intractive picking.

        On PRESS, record the mouse position.
        On MOVE, create a rectangular picking window.
        On RELEASE, pick the objects inside the rectangle.
        """
        if action == PRESS:
            self.draw_cursor(self.statex,self.statey)
            self.update()
            
        elif action == MOVE:
            self.draw_rectangle(x,y)
            self.update()

        elif action == RELEASE:
            x,y = (x+self.statex)/2., (y+self.statey)/2.
            w,h = abs(x-self.statex)*2., abs(y-self.statey)*2.
            if w <= 0 or h <= 0:
               w,h = GD.cfg.get('pick/size',(20,20))
            vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
            self.pick_window = (x,y,w,h,vp)
            if self.cursor:
                self.removeDecoration(self.cursor)
            self.selection_busy = False
            self.update()

            
    def pick_actors(self):
        """Set the list of actors inside the pick_window."""
        self.camera.loadProjection(pick=self.pick_window)
        self.camera.loadMatrix()
        stackdepth = 1
        npickable = len(self.actors)
        GL.glSelectBuffer(npickable*(3+stackdepth))
        GL.glRenderMode(GL.GL_SELECT)
        GL.glInitNames()
        for i,a in enumerate(self.actors):
            GD.debug("PICK actor %s = %s" % (i,a.list))
            GL.glPushName(i)
            GL.glCallList(a.list)
            GL.glPopName()
        buf = GL.glRenderMode(GL.GL_RENDER)
        self.picked = [ r[2][0] for r in buf]


    def pick_elements(self):
        """Set the list of actor elements inside the pick_window."""
        self.camera.loadProjection(pick=self.pick_window)
        self.camera.loadMatrix()
        stackdepth = 2
        npickable = 0
        for a in self.actors:
             npickable += a.nelems()
        GL.glSelectBuffer(npickable*(3+stackdepth))
        GL.glRenderMode(GL.GL_SELECT)
        GL.glInitNames()
        for i,a in enumerate(self.actors):
            GL.glPushName(i)
            a.pickGL('element')
            GL.glPopName()
        buf = GL.glRenderMode(GL.GL_RENDER)
        self.picked = [ r[2] for r in buf ]


    def pick_points(self):
        """Set the list of actor points inside the pick_window."""
        self.camera.loadProjection(pick=self.pick_window)
        self.camera.loadMatrix()
        stackdepth = 2
        npickable = 0
        for a in self.actors:
             npickable += a.npoints()
        GL.glSelectBuffer(npickable*(3+stackdepth))
        GL.glRenderMode(GL.GL_SELECT)
        GL.glInitNames()
        for i,a in enumerate(self.actors):
            GL.glPushName(i)
            a.pickGL('point')
            GL.glPopName()
        buf = GL.glRenderMode(GL.GL_RENDER)
        self.picked = [ r[2] for r in buf ]


    def pick_numbers(self):
        """Return the numbers inside the pick_window."""
        self.camera.loadProjection(pick=self.pick_window)
        self.camera.loadMatrix()
        self.picked = [0,1,2,3]
        if self.numbers:
            self.picked = self.numbers.drawpick()

    
    def pick_lines(self,x,y,action):
        """Return the lines close to the mouse pointer."""
        if action == PRESS:
            GD.debug("Start picking mode")
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
            self.draw_cursor(self.statex,self.statey)
            self.draw_square(self.statex-5,self.statey-5,self.statex+5,self.statey+5)
            self.update()
            
        elif action == RELEASE:
            GD.debug("End picking mode")
            if self.cursor:
                self.removeDecoration(self.cursor)
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self.update()
            
            GD.debug((x,y))
            GD.debug((self.statex,self.statey))
            x,y = self.statex, self.statey
            w,h = 10., 10.
            
            vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
            GD.debug("PICK: cursor %s, viewport %s" % ((x,y,w,h),vp))
            self.camera.loadProjection(pick=(x,y,w,h,vp))
            self.camera.loadMatrix()
            self.selected = self.actors[-1].drawpick('lines')
    
    
    def pick_elements_3d(self,x,y,action):
        """Return the elements close to the mouse pointer."""
        if action == PRESS:
            GD.debug("Start picking mode")
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
            self.draw_cursor(self.statex,self.statey)
            self.update()
            
        elif action == MOVE:
            GD.debug("Move picking window")
            self.draw_rectangle(x,y)
            self.update()
            
        elif action == RELEASE:
            GD.debug("End picking mode")
            if self.cursor:
                self.removeDecoration(self.cursor)
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self.update()
            
            GD.debug((x,y))
            GD.debug((self.statex,self.statey))
            x,y = (x+self.statex)/2., (y+self.statey)/2.
            w,h = abs(x-self.statex)*2., abs(y-self.statey)*2.
            if w <= 0 or h <= 0:
               w,h = GD.cfg.get('pick/size',(20,20))
            
            vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
            GD.debug("PICK: cursor %s, viewport %s" % ((x,y,w,h),vp))
            self.camera.loadProjection(pick=(x,y,w,h,vp))
            self.camera.loadMatrix()
            if int(self.mod) == ALT:
                self.selected = self.actors[0].drawpick('elements_3d')
            else:
                self.selected = self.actors[-1].drawpick('elements_3d')




    @classmethod
    def has_modifier(clas,e,mod):
        return ( e.modifiers() & mod ) == mod
        
    def mousePressEvent(self,e):
        """Process a mouse press event."""
        GD.gui.viewports.setCurrent(self)
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


class MultiCanvas(QtGui.QGridLayout):
    """A viewport that can be splitted."""

    def __init__(self,parent=None):
        """Initialize the multicanvas."""
        QtGui.QGridLayout.__init__(self)
        self.all = []
        self.active = []
        self.current = None
        self.ncols = 2
        self.rowwise = True
        self.parent = parent
        # With QtOPenGL, the context should refer to the QGLWidget,
        # therefore we can not use one single context
        #self.context = context
	OGLfmt = setOpenGLFormat()
	#OGLctxt = getOpenGLContext()
	#print OGLctxt

        
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
        GD.canvas = self.current = canv


    def setCurrent(self,canv):
        if type(canv) == int and canv in range(len(self.all)):
            canv = self.all[canv]
        if canv in self.all:
            GD.canvas = self.current = canv
            toolbar.setTransparency(self.current.alphablend)
            toolbar.setPerspective(self.current.camera.perspective)
            self.current.display()
            

    def currentView(self):
        return self.all.index(GD.canvas)


    def showWidget(self,w):
        """Show the view w"""
        row,col = divmod(self.all.index(w),self.ncols)
        if self.rowwise:
            self.addWidget(w,row,col)
        else:
            self.addWidget(w,col,row)
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


if __name__ == "__main__":
    print "Testing the Collection object"
    a = Collection()
    a.add(range(7),3)
    a.add(range(4))
    a.remove([2,4],3)
    print a
    a.add([[2,0],[2,3],[-1,7],[3,88]])
    print a
    a[2] = [1,2,3]
    print a
    a[2] = []
    print a
    a.set([[2,0],[2,3],[-1,7],[3,88]])
    print a

                    
# End
