# $Id$
##
## This file is part of pyFormex 0.5 Release Mon Jul 30 13:38:48 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""This implements an OpenGL drawing widget for painting 3D scenes."""

import globaldata as GD

from numpy import *
from OpenGL import GL,GLU

import string
keys = [ k for k in GL.__dict__.keys() if k.startswith('gl') ]
keys = sort(keys)
print keys

from formex import length
import colors
import camera
import actors
import decors
import marks
import utils

class ActorList(list):

    def __init__(self,canvas,useDisplayLists=True):
        self.canvas = canvas
        self.uselists = useDisplayLists
        list.__init__(self)
        
    def add(self,actor):
        """Add an actor to an actorlist."""
        if self.uselists:
            self.canvas.makeCurrent()
            self.canvas.setDefaults()
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
            self.canvas.makeCurrent()
            for actor in actorlist:
                self.canvas.setDefaults()
                if actor.list:
                    GL.glDeleteLists(actor.list,1)
                actor.list = GL.glGenLists(1)
                GL.glNewList(actor.list,GL.GL_COMPILE)
                actor.draw(self.canvas.rendermode)
                GL.glEndList()

                
##################################################################
#
#  The Canvas Settings
#

class CanvasSettings(object):
    """A collection of settings for an OpenGL Canvas."""

    default = dict(
        linewidth = 1.0,
        bgcolor = colors.mediumgrey,
        fgcolor = colors.black,
        slcolor = colors.red,     # color for selected items
        transparency = 1.0,       # opaque
        # colormap for mapping property values
        propcolors = [ colors.black, colors.red, colors.green, colors.blue,
                       colors.cyan, colors.magenta, colors.yellow, colors.white ],
        )

    @classmethod
    def checkDict(clas,dict):
        """Transform a dict to acceptable settings."""
        ok = {}
        keys = dict.keys()
##        if 'rendermode' in keys:
##            ok['rendermode'] = dict['rendermode']
        if 'linewidth' in keys:
            ok['linewidth'] =  float(dict['linewidth'])
        for c in [ 'bgcolor', 'fgcolor', 'slcolor' ]:
            if c in keys:
                ok[c] = colors.GLColor(dict[c])
        if 'propcolors' in keys:
            ok['propcolors'] = map(colors.GLColor,dict['propcolors'])
        return ok
        
    def __init__(self,dict={}):
        """Create a new set of CanvasSettings, possibly changing defaults."""
        self.reset(dict)

    def reset(self,dict={}):
        """Reset to default settings

        If a dict is specified, these settings will override defaults.
        """
        self.__dict__.update(CanvasSettings.checkDict(CanvasSettings.default))
        if dict:
            self.__dict__.update(CanvasSettings.checkDict(dict))
    
    def __str__(self):
        return utils.formatDict(self.__dict__)
                
##################################################################
#
#  The Canvas
#
class Canvas(object):
    """A canvas for OpenGL rendering."""

    rendermodes = ['wireframe','flat','flatwire','smooth','smoothwire']
  
    # default light
    default_light = { 'ambient':0.5, 'diffuse': 1.0, 'specular':0.5, 'position':(0.,0.,1.,0.)}
    

    def __init__(self):
        """Initialize an empty canvas with default settings."""
        self.actors = ActorList(self)       # start with an empty scene
        self.annotations = ActorList(self)  # without annotations
        self.decorations = ActorList(self)  # and no decorations either
        self.triade = None
        self.lights = []
        self.setBbox()
        self.settings = CanvasSettings()
        self.rendermode = 'wireframe'
        self.alphablend = False
        self.dynamouse = True  # dynamic mouse action works on mouse move
        self.dynamic = None    # what action on mouse move
        self.camera = None
        self.view_angles = camera.view_angles
        GD.debug("Canvas Setting:\n%s"% self.settings)


    def resetDefaults(self,dict={}):
        """Return all the settings to their default values."""
        self.settings.reset(dict)


    def setRenderMode(self,rm):
        """Set the rendermode.

        This changes the rendermode and redraws everything with the new mode.
        """
        GD.debug("Changing Render Mode to %s" % rm)
        if rm != self.rendermode:
            if rm not in Canvas.rendermodes:
                rm = Canvas.rendermodes[0]
            self.rendermode = rm
            GD.debug("Redrawing with mode %s" % self.rendermode)
            self.glinit(self.rendermode)
            self.redrawAll()


    def setTransparency(self,mode):
        self.alphablend = mode

    def setLineWidth(self,lw):
        """Set the linewidth for line rendering."""
        self.settings.linewidth = float(lw)


    def setBgColor(self,bg):
        """Set the background color."""
        self.settings.bgcolor = colors.GLColor(bg)
        self.clear()
        self.redrawAll()
        

    def setFgColor(self,fg):
        """Set the default foreground color."""
        self.settings.fgcolor = colors.GLColor(fg)

    
    def addLight(self,position,ambient,diffuse,specular):
        """Adds a new light to the scene."""
        pass
    

    def initCamera(self):
##         if GD.options.makecurrent:
##             self.makeCurrent()  # we need correct OpenGL context for camera
        self.camera = camera.Camera()
        GD.debug("camera.rot = %s" % self.camera.rot)
        GD.debug("view angles: %s" % self.view_angles)


    def glinit(self,mode=None):
        if mode:
            self.rendermode = mode

        self.clear()
        #GL.glClearColor(*colors.RGBA(self.default.bgcolor))# Clear The Background Color
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
            if self.rendermode.find('trans') >= 0:
                GL.glEnable (GL.GL_BLEND)              # Enables Alpha Transparency
                GL.glBlendFunc (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glDisable(GL.GL_DEPTH_TEST)	 # Disable Depth Testing
               
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


    def glupdate(self):
        """Flush all OpenGL commands, making sure the display is updated."""
        GL.glFlush()
        

    def clear(self):
        """Clear the canvas to the background color."""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glClearColor(*colors.RGBA(self.settings.bgcolor))
        self.setDefaults()


    def setDefaults(self):
        """Activate the canvas settings in the GL machine."""
        GL.glColor3fv(self.settings.fgcolor)
        GL.glLineWidth(self.settings.linewidth)

    
    def setSize (self,w,h):
        if h == 0:	# Prevent A Divide By Zero 
            h = 1
        GL.glViewport(0, 0, w, h)
        self.aspect = float(w)/h
        self.camera.setLens(aspect=self.aspect)
        self.display()


    def display(self):
        """(Re)display all the actors in the scene.

        This should e.g. be used when actors are added to the scene,
        or after changing  camera position/orientation or lens.
        """
        self.clear()
        # Draw Scene Actors
        # GD.debug("%s / %s" % (len(self.actors),len(self.annotations)))
        self.camera.loadProjection()
        self.camera.loadMatrix()
        if self.alphablend:
            print "ENABLE TRANS"
            opaque = [ a for a in self.actors if not a.trans ]
            transp = [ a for a in self.actors if a.trans ]
            for actor in opaque:
               GL.glCallList(actor.list)
            GL.glEnable (GL.GL_BLEND)
            GL.glDepthMask (GL.GL_FALSE)
            GL.glBlendFunc (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            for actor in transp:
                GL.glCallList(actor.list)
            GL.glDepthMask (GL.GL_TRUE)
            GL.glDisable (GL.GL_BLEND)
        else:
            for actor in self.actors:
                GL.glCallList(actor.list)

        for actor in self.annotations:
            GL.glCallList(actor.list)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        # Plot viewport decorations
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluOrtho2D(0,self.width(),0,self.height())
        for actor in self.decorations:
            GL.glCallList(actor.list)
        # end plot viewport decorations
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()
##         # Display angles
##         self.camera.getCurrentAngles()
        # ADDED TO MAKE SURE SCENE IS UPTODATE
        GL.glFlush()

        
    def setBbox(self,bbox=None):
        """Set the bounding box of the scene you want to be visible."""
        # TEST: use last actor
        if bbox is None:
            if len(self.actors) > 0:
                bbox = self.actors[-1].bbox()
            else:
                bbox = [[-1.,-1.,-1.],[1.,1.,1.]]
        self.bbox = asarray(bbox)

         
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
        if actor == self.triade:
            self.triade = None
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

        If no camera angles are given, the camera orientation is kept.
        """
        self.makeCurrent()
        # go to a distance to have a good view with a 45 degree angle lens
        if not bbox is None:
            self.setBbox(bbox)
        bbox = self.bbox
        center = (bbox[0]+bbox[1]) / 2
        size = bbox[1] - bbox[0]
        # calculating the bounding circle: this is rather conservative
        dist = length(size)
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


    def draw_cursor(self,x,y):
        if self.cursor:
            self.removeDecoration(self.cursor)
        w,h = GD.cfg.get('pick/size',(20,20))
        col = GD.cfg.get('pick/color','yellow')
        self.cursor = decors.Grid(x-w/2,y-h/2,x+w/2,y+h/2,color=col,linewidth=1)
        self.addDecoration(self.cursor)

    def draw_rectangle(self,x,y):
        if self.cursor:
            self.removeDecoration(self.cursor)
        col = GD.cfg.get('pick/color','yellow')
        self.cursor = decors.Grid(self.statex,self.statey,x,y,color=col,linewidth=1)
        self.addDecoration(self.cursor)


### End
