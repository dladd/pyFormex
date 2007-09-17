#!/usr/bin/env python
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
"""viewport.py: Interactive OpenGL Canvas.

This module implements user interaction with the OpenGL canvas.
QtCanvas is a single interactive OpenGL canvas, while MultiCanvas
implements a dynamic array of multiple canvases.

The basic OpenGL drawing functionality is implemented in the canvas module.
"""

import globaldata as GD

from PyQt4 import QtCore, QtGui, QtOpenGL
from OpenGL import GL

import canvas
import image
import utils
import toolbar

import math

# Some 2D vector operations

def dotpr (v,w):
    """Return the dot product of vectors v and w"""
    return v[0]*w[0] + v[1]*w[1]

def length (v):
    """Return the length of the vector v"""
    return math.sqrt(dotpr(v,v))

def projection(v,w):
    """Return the (signed) length of the projection of vector v on vector w."""
    return dotpr(v,w)/length(w)

# mouse actions
PRESS = 0
MOVE = 1
RELEASE = 2

# mouse buttons
LEFT = QtCore.Qt.LeftButton
MIDDLE = QtCore.Qt.MidButton
RIGHT = QtCore.Qt.RightButton

# modifiers
NONE = QtCore.Qt.NoModifier
SHIFT = QtCore.Qt.ShiftModifier
CONTROL = QtCore.Qt.ControlModifier
ALT = QtCore.Qt.AltModifier
ALLMODS = SHIFT | CONTROL | ALT


############### OpenGL Format #################################

def setOpenglFormat():
    """Set the correct OpenGL format.

    The default OpenGL format can be changed by command line options.
    --dri   : use the DIrect Rendering Infrastructure
    --nodri : do no use the DRI
    --alpha : enable the alpha buffer 
    """
    fmt = QtOpenGL.QGLFormat.defaultFormat()
    if GD.options.dri:
        fmt.setDirectRendering(True)
    if GD.options.nodri:
        fmt.setDirectRendering(False)
    if GD.options.alpha:
        fmt.setAlpha(True)
    #fmt.setRgba(False)
    if GD.options.debug:
        printOpenglFormat(fmt)
    QtOpenGL.QGLFormat.setDefaultFormat(fmt)

def printOpenglFormat(fmt):
    """Print some information about the OpenGL format."""
    print "OpenGL: ",fmt.hasOpenGL()
    print "OpenGLOverlays: ",fmt.hasOpenGLOverlays()
    print "Double Buffer: ",fmt.doubleBuffer()
    print "Depth Buffer: ",fmt.depth()
    print "RGBA: ",fmt.rgba()
    print "Alpha Channel: ",fmt.alpha()
    print "Accumulation Buffer: ",fmt.accum()
    print "Stencil Buffer: ",fmt.stencil()
    print "Stereao: ",fmt.stereo()
    print "Direct Rendering: ",fmt.directRendering()
    print "Overlay: ",fmt.hasOverlay()
    print "Plane: ",fmt.plane()
    print "Multisample Buffers: ",fmt.sampleBuffers()


################# Single Interactive OpenGL Canvas ###############

class QtCanvas(QtOpenGL.QGLWidget,canvas.Canvas):
    """A canvas for OpenGL rendering.

    This class provides interactive functionality for the OpenGL canvas
    provided by the canvas.Canvas class.
    
    Interactivity is highly dependent on Qt4. Putting the interactive
    functions in a separate class makes it esier to use the Canvas class
    in non-interactive situations or combining it with other GUI toolsets.
    """
    
    def __init__(self,*args):
        """Initialize an empty canvas with default settings."""
        QtOpenGL.QGLWidget.__init__(self,*args)
        if not self.isValid():
            raise RuntimeError,"Could not create a valid OpenGL widget"
        self.setMinimumSize(32,32)
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        canvas.Canvas.__init__(self)
        self.button = None
        self.mousefunc = {}
        self.mouseshiftfunc = {}
        self.setMouse(LEFT,self.dynarot) 
        self.setMouse(MIDDLE,self.dynapan) 
        self.setMouse(RIGHT,self.dynazoom)
        self.mouseshiftfunc.update(self.mousefunc) # initially the same
        self.mod = NONE
        

    def setMouse(self,button,func,mod=None):
        if mod == SHIFT:
            self.mouseshiftfunc[button] = func
        else:
            self.mousefunc[button] = func


    def pick(self):
        """Go into picking mode and return the selection."""
        self.setMouse(LEFT,self.pick_actors)  
        self.selection =[]
        timer = QtCore.QThread
        while len(self.selection) == 0:
            timer.usleep(200)
            GD.app.processEvents()
        return GD.canvas.selection

    def pickNumbers(self):
        """Go into number picking mode and return the selection."""
        self.setMouse(LEFT,self.pick_numbers)  
        self.selection =[]
        timer = QtCore.QThread
        while len(self.selection) == 0:
            timer.usleep(200)
            GD.app.processEvents()
        return GD.canvas.selection


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
        GD.debug("resizeGL: %s x %s" % (w,h))
        self.setSize(w,h)

    def	paintGL(self):
        self.display()


####### MOUSE EVENT HANDLERS ############################

    # Mouse functions can be bound to any of the mousse buttons
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
            w,h = self.width(),self.height()
            self.state = [self.statex-w/2, self.statey-h/2 ]

        elif action == MOVE:
            w,h = self.width(),self.height()
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
            dx = [x-self.statex, y-self.statey]         # movement
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
            w,h = self.width(),self.height()
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
            w,h = self.width(),self.height()
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

       

    def pick_actors(self,x,y,action):
        """Return the actors close to the mouse pointer."""
        if action == PRESS:
            GD.debug("Start picking mode")
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
            self.draw_cursor(self.statex,self.statey)
            self.selection = []
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
            GL.glSelectBuffer(16+3*len(self.actors))
            GL.glRenderMode(GL.GL_SELECT)
            GL.glInitNames() # init the name stack
            GD.debug((x,y))
            GD.debug((self.statex,self.statey))
            x,y = (x+self.statex)/2., (y+self.statey)/2.
            w,h = abs(x-self.statex)*2., abs(y-self.statey)*2.
            if w <= 0 or h <= 0:
               w,h = GD.cfg.get('pick/size',(20,20))
            GD.debug((x,y,w,h))
            self.camera.loadProjection(pick=[x,y,w,h])
            self.camera.loadMatrix()
            for i,actor in enumerate(self.actors):
                #print "Adding name %s" % i
                GL.glPushName(i)
                GL.glCallList(actor.list)
                GL.glPopName()
            buf = GL.glRenderMode(GL.GL_RENDER)
            self.selection = []
            for r in buf:
                GD.debug(r)
                for i in r[2]:
                    self.selection.append(self.actors[i])
            self.setMouse(LEFT,self.dynarot)
            GD.debug("Re-enabling dynarot")
            self.update()

            
    def pick_numbers(self,x,y,action):
        """Return the numbers close to the mouse pointer."""
        if action == PRESS:
            GD.debug("Start picking mode")
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
            self.draw_cursor(self.statex,self.statey)
            self.selection = []
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
            GD.debug((x,y,w,h))
            self.camera.loadProjection(pick=[x,y,w,h])
            self.camera.loadMatrix()
            if self.numbers:
                self.selection = self.numbers.drawpick()
            self.setMouse(LEFT,self.dynarot)
            GD.debug("Re-enabling dynarot")
            self.update()

    @classmethod
    def has_modifier(clas,e,mod):
        return ( e.modifiers() & mod ) == mod

    def getMouseFunc(self):
        if self.mod == SHIFT:
            return self.mouseshiftfunc.get(self.button,None)
        else:
            return self.mousefunc.get(self.button,None)
        
    def mousePressEvent(self,e):
        """Process a mouse press event."""
        GD.gui.viewports.set_current(self)
        # on PRESS, always remember mouse position and button
        self.statex,self.statey = e.x(), self.height()-e.y()
        self.button = e.button()
        self.mod = e.modifiers() & ALLMODS
        func = self.getMouseFunc()
        if func:
            func(self.statex,self.y,PRESS)
        
    def mouseMoveEvent(self,e):
        """Process a mouse move event."""
        # the MOVE event does not identify a button, use the saved one
        func = self.getMouseFunc()
        if func:
            func(e.x(),self.height()-e.y(),MOVE)

    def mouseReleaseEvent(self,e):
        """Process a mouse release event."""
        func = self.getMouseFunc()
        self.button = None        # clear the stored button
        if func:
            func(e.x(),self.height()-e.y(),RELEASE)


    # Any keypress with focus in the canvas generates a 'wakeup' signal.
    # This is used to break out of a wait status.
    # Events not handled here could also be handled by the toplevel
    # event handler.
    def keyPressEvent (self,e):
        self.emit(QtCore.SIGNAL("Wakeup"),())
        e.ignore()


################# Multiple Viewports ###############

def vpfocus(canv):
    print "vpfocus %s" % canv
    GD.gui.viewports.set_current(canv)

class MultiCanvas(QtGui.QGridLayout):
    """A viewport that can be splitted."""

    def __init__(self):
        QtGui.QGridLayout.__init__(self)
        self.all = []
        self.active = []
        self.current = None
        #self.addView(0,0)


    def setDefaults(self,dict):
        """Update the default settings of the canvas class."""
        GD.debug("Setting canvas defaults:\n%s" % dict)
        canvas.CanvasSettings.default.update(canvas.CanvasSettings.checkDict(dict))

    def newView(self):
        "Adding a View"
        canv = QtCanvas()
        #QtCore.QObject.connect(canv,QtCore.SIGNAL("VPFocus"),vpfocus)
        #canv.initCamera() # THis is already done in the __init__()
        self.all.append(canv)
        self.active.append(canv)
        # DO NOT USE self.set_current(canv) HERE, because no camera yet
        GD.canvas = self.current = canv
        return(canv)

    def set_current(self,canv):
        #print self.all
        #print self.current
        if canv in self.all:
            GD.canvas = self.current = canv
            toolbar.setTransparency(self.current.alphablend)
            toolbar.setPerspective(self.current.camera.perspective)
            

    def currentView(self):
        return self.all.index(GD.canvas)
 
    def addView(self,row,col):
        w = self.newView()
        self.addWidget(w,row,col)
        w.raise_()

    def removeView(self):
        if len(self.all) > 1:
            w = self.all.pop()
            if self.current == w:
                self.set_current(self.all[-1])
            if w in self.active:
                self.active.remove(w)
            self.removeWidget(w)
            w.close()

##     def setCamera(self,bbox,view):
##         self.current.setCamera(bbox,view)
            
    def update(self):
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
                       
#### End
