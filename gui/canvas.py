# canvas.py
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""This implements an OpenGL drawing widget for painting 3D scenes."""

import globaldata as GD

from OpenGL import GL,GLU
from PyQt4 import QtCore, QtGui # needed for signals, threads, cursors

# mouse buttons
LEFT = QtCore.Qt.LeftButton
MIDDLE = QtCore.Qt.MidButton
RIGHT = QtCore.Qt.RightButton
# mouse actions
PRESS = 0
MOVE = 1
RELEASE = 2



import colors
import camera
import actors
import decors
import marks
import utils

import numpy

import math
import vector


class ActorList(list):

    def __init__(self,canvas,useDisplayLists=True):
        self.canvas = canvas
        self.uselists = useDisplayLists
        list.__init__(self)
        
    def add(self,actor):
        """Add an actor to an actorlist."""
        if self.uselists:
            self.canvas.makeCurrent()
            actor.list = GL.glGenLists(1)
            GL.glNewList(actor.list,GL.GL_COMPILE)
            actor.draw(self.canvas.rendermode)
            GL.glEndList()
        self.append(actor)

    def delete(self,actor):
        """Remove an actor from an actorlist."""
        if actor in self:
            self.remove(actor)
            if self.uselists and actor.list:
                self.canvas.makeCurrent()
                GL.glDeleteLists(actor.list,1)


    def redraw(self,actorlist=None):
        """Redraw (some) actors in the scene.

        This redraws the specified actors (recreating their display list).
        This should e.g. be used after changing an actor's properties.
        Only actors that are in the current actor list will be redrawn.
        If no actor list is specified, the whole current actorlist is redrawn.
        """
        if actorlist is None:
            actorlist = self
        if self.uselists:
            for actor in actorlist:
                if actor.list:
                    GL.glDeleteLists(actor.list,1)
                actor.list = GL.glGenLists(1)
                GL.glNewList(actor.list,GL.GL_COMPILE)
                actor.draw(self.canvas.rendermode)
                GL.glEndList()

                
##################################################################
#
#  The Canvas
#
class Canvas(object):
    """A canvas for OpenGL rendering."""
    
    # default light
    default_light = { 'ambient':0.5, 'diffuse': 1.0, 'specular':0.5, 'position':(0.,0.,1.,0.)}
    

    def __init__(self):
        """Initialize an empty canvas with default settings."""
        self.actors = ActorList(self)       # start with an empty scene
        self.annotations = ActorList(self)  # without annotations
        self.decorations = ActorList(self)  # and no decorations either
        self.lights = []
        self.setBbox()
        self.bgcolor = colors.mediumgrey
        self.fgcolor = colors.black
        self.slcolor = colors.red
        self.rendermode = 'wireframe'
        self.dynamouse = True  # dynamic mouse action works on mouse move
        self.dynamic = None    # what action on mouse move
        self.mousefunc = {}
        self.setMouse(LEFT,self.dynarot) 
        self.setMouse(MIDDLE,self.dynapan) 
        self.setMouse(RIGHT,self.dynazoom) 
        self.camera = None
        self.view_angles = camera.view_angles


    def setMouse(self,button,func):
        self.mousefunc[button] = func
    
    def addLight(self,position,ambient,diffuse,specular):
        """Adds a new light to the scene."""
        pass
    

    def initCamera(self):
        if GD.options.makecurrent:
            self.makeCurrent()  # we need correct OpenGL context for camera
        self.camera = camera.Camera()
        GD.debug("camera.rot = %s" % self.camera.rot)
        GD.debug("view angles: %s" % self.view_angles)


##    def update(self):
##        GD.app.processEvents()


    def glinit(self,mode=None):
        GD.debug("canvas GLINIT")
        if mode:
            self.rendermode = mode
            
        GL.glClearColor(*colors.RGBA(self.bgcolor))# Clear The Background Color
        GL.glClearDepth(1.0)	       # Enables Clearing Of The Depth Buffer
        GL.glDepthFunc(GL.GL_LESS)	       # The Type Of Depth Test To Do
        GL.glEnable(GL.GL_DEPTH_TEST)	       # Enables Depth Testing
        #GL.glEnable(GL.GL_CULL_FACE)
        #GL.glPolygonMode(GL.GL_FRONT_AND_BACK,GL.GL_LINE) # WIREFRAME!
        

        if self.rendermode == 'wireframe':
            GL.glShadeModel(GL.GL_FLAT)      # Enables Flat Color Shading
            GL.glDisable(GL.GL_LIGHTING)
        elif self.rendermode.startswith('flat'):
            GL.glShadeModel(GL.GL_FLAT)      # Enables Flat Color Shading
            GL.glDisable(GL.GL_LIGHTING)
        elif self.rendermode.startswith('smooth'):
            GL.glShadeModel(GL.GL_SMOOTH)    # Enables Smooth Color Shading
            GL.glEnable(GL.GL_LIGHTING)
            for l,i in zip(['light0','light1'],[GL.GL_LIGHT0,GL.GL_LIGHT1]):
                key = 'render/%s' % l
                light = GD.cfg.get(key,self.default_light)
                GD.debug("  set up %s %s" % (l,light))
                GL.glLightModel(GL.GL_LIGHT_MODEL_AMBIENT,colors.GREY(GD.cfg['render/ambient']))
                GL.glLightModel(GL.GL_LIGHT_MODEL_TWO_SIDE, GL.GL_TRUE)
                GL.glLightModel(GL.GL_LIGHT_MODEL_LOCAL_VIEWER, 0)
                GL.glLightfv(i,GL.GL_AMBIENT,colors.GREY(light['ambient']))
                GL.glLightfv(i,GL.GL_DIFFUSE,colors.GREY(light['diffuse']))
                GL.glLightfv(i,GL.GL_SPECULAR,colors.GREY(light['specular']))
                GL.glLightfv(i,GL.GL_POSITION,colors.GREY(light['position']))
                GL.glEnable(i)
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK,GL.GL_SPECULAR,colors.GREY(GD.cfg['render/specular']))
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK,GL.GL_EMISSION,colors.GREY(GD.cfg['render/emission']))
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK,GL.GL_SHININESS,GD.cfg['render/shininess'])
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK,GL.GL_AMBIENT_AND_DIFFUSE)
            GL.glEnable(GL.GL_COLOR_MATERIAL)
        else:
            raise RuntimeError,"Unknown rendering mode"

    
    def setSize (self,w,h):
        if h == 0:	# Prevent A Divide By Zero 
            h = 1
        GL.glViewport(0, 0, w, h)
        self.aspect = float(w)/h
        self.camera.setLens(aspect=self.aspect)
        self.display()


    def clear(self):
        """Clear the canvas to the background color."""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glClearColor(*colors.RGBA(self.bgcolor))


    def display(self):
        """(Re)display all the actors in the scene.

        This should e.g. be used when actors are added to the scene,
        or after changing  camera position/orientation or lens.
        """
        self.clear()
        # Draw Scene Actors
        self.camera.loadProjection()
        self.camera.loadMatrix()
        for actor in self.actors:
            GL.glCallList(actor.list)
        for actor in self.annotations:
            GL.glCallList(actor.list)
            #actor.draw()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        # Plot viewport decorations
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluOrtho2D(0, self.width(), 0, self.height())
        for actor in self.decorations:
            GL.glCallList(actor.list)
        # end plot viewport decorations
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()
##         # Display angles
##         self.camera.getCurrentAngles()
        

    def setLinewidth(self,lw):
        """Set the linewidth for line rendering."""
        GL.glLineWidth(lw)


    def setBgColor(self,bg):
        """Set the background color."""
        self.bgcolor = bg


    def setFgColor(self,fg):
        """Set the default foreground color."""
        self.fgcolor = fg
        GL.glColor3fv(self.fgcolor)

        
    def setBbox(self,bbox=None):
        """Set the bounding box of the scene you want to be visible."""
        # TEST: use last actor
        if bbox is None:
            if len(self.actors) > 0:
                self.bbox = self.actors[-1].bbox()
            else:
                self.bbox = [[-1.,-1.,-1.],[1.,1.,1.]]
        else:
            self.bbox = bbox

         
    def addActor(self,actor):
        """Add a 3D actor to the 3D scene."""
        self.actors.add(actor)

    def removeActor(self,actor):
        """Remove a 3D actor from the 3D scene."""
        self.actors.delete(actor)

         
    def addMark(self,actor):
        """Add an annotation to the 3D scene."""
        self.annotations.add(actor)

    def removeMark(self,actor):
        """Remove an annotation from the 3D scene."""
        self.annotations.delete(actor)

         
    def addDecoration(self,actor):
        """Add a 2D decoration to the canvas."""
        self.decorations.add(actor)

    def removeDecoration(self,actor):
        """Remove a 2D decoration from the canvas."""
        self.decorations.delete(actor)

    def remove(self,itemlist):
        """Remove a list of any actor/annotation/decoration items.

        itemlist can also be a single item instead of a list.
        """
        if not type(itemlist) == list:
            itemlist = [ itemlist ]
        for item in itemlist:
            if isinstance(item,actors.Actor):
                self.actors.delete(item)
            elif isinstance(item,marks.Mark):
                self.annotations.delete(item)
            elif isinstance(item,decors.Decoration):
                self.decorations.delete(item)
        

    def removeActors(self,actorlist=None):
        """Remove all actors in actorlist (default = all) from the scene."""
        if actorlist == None:
            actorlist = self.actors[:]
        for actor in actorlist:
            self.removeActor(actor)
        self.setBbox()


    def removeMarks(self,actorlist=None):
        """Remove all actors in actorlist (default = all) from the scene."""
        if actorlist == None:
            actorlist = self.annotations[:]
        for actor in actorlist:
            self.removeMark(actor)


    def removeDecorations(self,actorlist=None):
        """Remove all decorations in actorlist (default = all) from the scene."""
        if actorlist == None:
            actorlist = self.decorations[:]
        for actor in actorlist:
            self.removeDecoration(actor)


    def removeAll(self):
        """Remove all actors and decorations"""
        self.removeActors()
        self.removeMarks()
        self.removeDecorations()
        self.display()


    def redrawAll(self):
        """Redraw all actors in the scene."""
        self.actors.redraw()
        self.annotations.redraw()
        self.decorations.redraw()
        self.display()

        
##     def setView(self,bbox=None,side=None):
##         """Sets the camera looking from one of the named views."""
## ##         # select view angles: if undefined use (0,0,0)
## ##         if side:
## ##             angles = self.camera.getAngles(side)
## ##         else:
## ##             angles = None
##         self.setCamera(bbox,angles)

        
    def setCamera(self,bbox=None,angles=None):
        """Sets the camera looking under angles at bbox.

        This function sets the camera angles and adjusts the zooming.
        The camera distance remains unchanged.
        If a bbox is specified, the camera will be zoomed to make the whole
        bbox visible.
        If no bbox is specified, the current scene bbox will be used.
        If no current bbox has been set, it will be calculated as the
        bbox of the whole scene.
        """
        self.makeCurrent()
        # go to a distance to have a good view with a 45 degree angle lens
        if bbox is None:
            bbox = self.bbox
        else:
            self.bbox = bbox
        center,size = vector.centerDiff(bbox[0],bbox[1])
        # calculating the bounding circle: this is rather conservative
        dist = vector.length(size)
        if dist <= 0.0:
            dist = 1.0
        self.camera.setCenter(*center)
        if angles:
            self.camera.setAngles(angles)
#            self.camera.setRotation(*angles)
        self.camera.setDist(dist)
        self.camera.setLens(45.,self.aspect)
        self.camera.setClip(0.01*dist,100.*dist)


    def zoom(self,f):
        """Dolly zooming."""
        self.camera.setDist(f*self.camera.getDist())


    def pick(self):
        """Go into picking mode and return the selection."""
        self.setMouse(LEFT,self.pick_actors)  
        self.selection =[]
        timer = QtCore.QThread
        while len(self.selection) == 0:
            timer.usleep(200)
            GD.app.processEvents()
        return GD.canvas.selection



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
            self.state = [self.statex-self.width()/2, -(self.statey-self.height()/2), 0.]

        elif action == MOVE:
            w,h = self.width(),self.height()
            # set all three rotations from mouse movement
            # tangential movement sets twist,
            # but only if initial vector is big enough
            x0 = self.state        # initial vector
            d = vector.length(x0)
            if d > h/8:
                x1 = [x-w/2, y-h/2, 0]     # new vector
                a0 = math.atan2(x0[0],x0[1])
                a1 = math.atan2(x1[0],x1[1])
                an = (a1-a0) / math.pi * 180
                ds = utils.stuur(d,[-h/4,h/8,h/4],[-1,0,1],2)
                twist = - an*ds
                self.camera.rotate(twist,0.,0.,1.)
                self.state = x1
            # radial movement rotates around vector in lens plane
            x0 = [self.statex-w/2, self.statey-h/2, 0]    # initial vector
            dx = [x-self.statex, y-self.statey,0]         # movement
            b = vector.projection(dx,x0)
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


    def draw_cursor(self,x,y):
        if self.cursor:
            self.removeDecoration(self.cursor)
        w,h = GD.cfg.get('pick/size',(20,20))
        self.cursor = decors.Grid(x-w/2,y-h/2,x+w/2,y+h/2,color='cyan',linewidth=1)
        self.addDecoration(self.cursor)

    def draw_rectangle(self,x,y):
        if self.cursor:
            self.removeDecoration(self.cursor)
        self.cursor = decors.Grid(self.statex,self.statey,x,y,color='cyan',linewidth=1)
        self.addDecoration(self.cursor)
       

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

        
    def mousePressEvent(self,e):
        """Process a mouse press event."""
        GD.gui.viewports.set_current(self)
        # on PRESS, always remember mouse position and button
        self.statex,self.statey = e.x(), self.height()-e.y()
        self.button = e.button()
        func = self.mousefunc.get(self.button,None)
        if func:
            func(self.statex,self.y,PRESS)
        
    def mouseMoveEvent(self,e):
        """Process a mouse move event."""
        # the MOVE event does not identify a button, use the saved one
        func = self.mousefunc.get(self.button,None)
        if func:
            func(e.x(),self.height()-e.y(),MOVE)

    def mouseReleaseEvent(self,e):
        """Process a mouse release event."""
        # clear the stored button
        self.button = None
        func = self.mousefunc.get(e.button(),None)
        if func:
            func(e.x(),self.height()-e.y(),RELEASE)



    # Any keypress with focus in the canvas generates a 'wakeup' signal.
    # This is used to break out of a wait status.
    # Events not handled here could also be handled by the toplevel
    # event handler.
    def keyPressEvent (self,e):
        self.emit(QtCore.SIGNAL("Wakeup"),())
        e.ignore()


### End
