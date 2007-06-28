#!/usr/bin/env python
# $Id$
"""viewport.py: Interactive OpenGL Canvas.

This module implements user interaction with the OpenGL canvas.
QtCanvas is a single interactive OpenGL canvas, while MultiCanvas
implements a dynamic array of multiple canvases.

The basic OpenGL drawing functionality is implemented in the canvas module.
"""

import globaldata as GD

from PyQt4 import QtCore, QtGui, QtOpenGL

import canvas
import image


############### OpenGL Format #################################

def setOpenglFormat():
    """Set the correct OpenGL format.

    The default OpenGL format can be changed by command line options.
    --dri   : use the DIrect Rendering Infrastructure
    --nodri : do no use the DRI
    """
    fmt = QtOpenGL.QGLFormat.defaultFormat()
    if GD.options.dri:
        fmt.setDirectRendering(True)
    if GD.options.nodri:
        fmt.setDirectRendering(False)
    #fmt.setRgba(False)
    if GD.options.debug:
        printOpenglFormat(fmt)
    QtOpenGL.QGLFormat.setDefaultFormat(fmt)

def printOpenglFormat(fmt):
    """Print some information about the OpenGL format."""
    print "OpenGL: ",fmt.hasOpenGL()
    print "OpenGLOverlays: ",fmt.hasOpenGLOverlays()
    print "Overlay: ",fmt.hasOverlay()
    print "Plane: ",fmt.plane()
    print "Direct Rendering: ",fmt.directRendering()
    print "Double Buffer: ",fmt.doubleBuffer()
    print "Depth Buffer: ",fmt.depth()
    print "RGBA: ",fmt.rgba()
    print "Alpha: ",fmt.alpha()


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
        """Initialize an empty canvas with default settings.
        """
        QtOpenGL.QGLWidget.__init__(self,*args)
        if not self.isValid():
            raise RuntimeError,"Could not create a valid OpenGL widget"
        self.setMinimumSize(32,32)
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        canvas.Canvas.__init__(self)
        
    def initializeGL(self):
        if GD.options.debug:
            p = self.sizePolicy()
            print p.horizontalPolicy(), p.verticalPolicy(), p.horizontalStretch(), p.verticalStretch()
        self.initCamera()
        #print self.view_angles
        self.glinit()

    def	resizeGL(self,w,h):
        GD.debug("resizeGL: %s x %s" % (w,h))
        self.setSize(w,h)

    def	paintGL(self):
        self.display()

    def save(self,*args):
        return image.save(self,*args)

##    def update(self):
##        print "Updating QtCanvas"
##        QtOpenGL.QGLWidget.update(self)
##        #canvas.Canvas.update(self)


################# Multiiple Viewports ###############

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

    def newView(self):
        "Adding a View"
        canv = QtCanvas()
        #QtCore.QObject.connect(canv,QtCore.SIGNAL("VPFocus"),vpfocus)
        canv.initCamera()
        self.all.append(canv)
        self.active.append(canv)
        self.set_current(canv)
        return(canv)

    def set_current(self,canv):
        #print self.all
        #print self.current
        if canv in self.all:
            GD.canvas = self.current = canv
 
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

        
##     def setCamera(self,bbox,view):
##         self.current.setCamera(bbox,view)
            
    def update(self):
        for v in self.all:
            v.update()
        print "Processing events"
        GD.app.processEvents()

    def removeAll(self):
        for v in self.active:
            v.removeAll()

##     def clear(self):
##         self.current.clear()  

    def addActor(self,actor):
        for v in self.active:
            v.addActor(actor)



#### End
