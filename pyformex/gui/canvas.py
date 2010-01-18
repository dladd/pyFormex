# $Id$
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
"""This implements an OpenGL drawing widget for painting 3D scenes."""

import pyformex as GD
from coords import tand

from numpy import *
from OpenGL import GL,GLU

from formex import length
import colors
import camera
import actors
import decors
import marks
import utils


def gl_pickbuffer():
    "Return a list of the 2nd numbers in the openGL pick buffer."
    buf = GL.glRenderMode(GL.GL_RENDER)
    #GD.debugt("translate getpickbuf")
    return asarray([ r[2] for r in buf ])


fill_modes = [ GL.GL_FRONT_AND_BACK, GL.GL_FRONT, GL.GL_BACK ]
fill_mode = GL.GL_FRONT_AND_BACK

def glFillMode(mode):
    global fill_mode
    if mode in fill_modes:
        fill_mode = mode
def glFrontFill():
    glFillMode(GL.GL_FRONT)
def glBackFill():
    glFillMode(GL.GL_BACK)
def glBothFill():
    glFillMode(GL.GL_FRONT_AND_BACK)
def glFill():
    GL.glPolygonMode(fill_mode,GL.GL_FILL)
def glLine():
    GL.glPolygonMode(GL.GL_FRONT_AND_BACK,GL.GL_LINE)

## def glLight(onoff):
##     """Toggle lights on/off."""
##     if onoff:
##         GL.glEnable(GL.GL_LIGHTING)
##     else:
##         GL.glDisable(GL.GL_LIGHTING)

def glSmooth():
    """Enable smooth shading"""
    GL.glShadeModel(GL.GL_SMOOTH)
def glFlat():
    """Disable smooth shading"""
    GL.glShadeModel(GL.GL_FLAT)

def glCulling():
    """Enable culling"""
    GL.glEnable(GL.GL_CULL_FACE)
def glNoCulling():
    """Disable culling"""
    GL.glDisable(GL.GL_CULL_FACE)


    
class ActorList(list):

    def __init__(self,canvas,atype):
        self.canvas = canvas
        self.atype = atype
        list.__init__(self)
        
    def add(self,actor):
        """Add an actor to an actorlist."""
        self.append(actor)

    def delete(self,actor):
        """Remove an actor from an actorlist."""
##         if self.atype == 'annotation':
##             print("DELETE")
##             print(self)
##             print(actor)
        if actor in self:
##             if self.atype == 'annotation':
##                 print("REMOVE %s" % actor)
            self.remove(actor)
##             if self.atype == 'annotation':
##                 print(self)

    def redraw(self):
        """Redraw (some) actors in the scene.

        This redraws the specified actors (recreating their display list).
        This should e.g. be used after changing an actor's properties.
        """
        for actor in self:
            actor.redraw(mode=self.canvas.rendermode)



##################################################################
#
#  The Canvas Settings
#

class CanvasSettings(object):
    """A collection of settings for an OpenGL Canvas."""

    default = dict(
        linewidth = 1.0,
        bgcolor = colors.mediumgrey,
        bgcolor2 = None,
        fgcolor = colors.black,
        slcolor = colors.red,     # color for selected items
        transparency = 1.0,       # opaque
        # colormap for mapping property values
        propcolors = [
            colors.black, colors.red, colors.green, colors.blue,
            colors.cyan, colors.magenta, colors.yellow, colors.white
            ],
        )

    @classmethod
    def checkDict(clas,dict):
        """Transform a dict to acceptable settings."""
        ok = {}
        for k,v in dict.items():
            if k in [ 'bgcolor', 'fgcolor', 'slcolor' ]:
                ok[k] = colors.GLColor(v)
            elif k == 'bgcolor2':
                if v is None or v == ok['bgcolor']:
                    ok[k] = None
                else:
                    ok[k] = colors.GLColor(v)
            elif k == 'propcolors':
                ok[k] = map(colors.GLColor,v)
            elif k == 'linewidth':
                ok[k] = float(v)
        return ok
        
    def __init__(self,dict={}):
        """Create a new set of CanvasSettings, possibly changing defaults."""
        self.reset(dict)

    def reset(self,dict={}):
        """Reset to defaults

        If a dict is specified, these settings will override defaults.
        """
        self.update(self.default)
        if dict:
            self.update(dict)

    def update(self,dict):
        """Update current values with the specified settings

        Returns the sanitized update values.
        """
        ok = self.checkDict(dict)
        self.__dict__.update(ok)
        return ok
    
    def __str__(self):
        return utils.formatDict(self.__dict__)


############### OpenGL Lighting #################################
    

class Light(object):

    def __init__(self,nr,*args,**kargs):
        self.light = GL.GL_LIGHT0 + (nr % GL.GL_MAX_LIGHTS)
        self.set(**kargs)

    def set(self,ambient=0.5,diffuse=0.5,specular=0.5,position=(0.,0.,1.,0.)):
        self.ambient = colors.GREY(ambient)
        self.diffuse = colors.GREY(diffuse)
        self.specular = colors.GREY(specular)
        self.position = position

    def enable(self):
        GL.glLightfv(self.light,GL.GL_POSITION,self.position)
        GL.glLightfv(self.light,GL.GL_AMBIENT,self.ambient)
        GL.glLightfv(self.light,GL.GL_DIFFUSE,self.diffuse)
        GL.glLightfv(self.light,GL.GL_SPECULAR,self.specular)
        GL.glEnable(self.light)

    def disable(self):
        GL.glDisable(self.light)


##################################################################
#
#  The Canvas
#
class Canvas(object):
    """A canvas for OpenGL rendering."""

    rendermodes = ['wireframe','flat','flatwire','smooth','smoothwire',
                   'smooth_avg']

    def __init__(self):
        """Initialize an empty canvas with default settings."""
        self.actors = ActorList(self,'actor')
        self.highlights = ActorList(self,'highlight')
        self.annotations = ActorList(self,'annotation')
        self.decorations = ActorList(self,'decoration')
        self.triade = None
        self.background = None
        self.bbox = None
        self.resetLighting()
        self.resetLights()
        self.setBbox()
        self.settings = CanvasSettings(GD.cfg['canvas'])
        self.mode2D = False
        self.rendermode = GD.cfg['render/mode']
        self.polygonfill = False
        self.lighting = True
        self.avgnormals = False
        self.alphablend = False
        self.dynamouse = True  # dynamic mouse action works on mouse move
        self.dynamic = None    # what action on mouse move
        self.camera = None
        self.view_angles = camera.view_angles
        self.cursor = None
        self.focus = False
        GD.debug("Canvas Setting:\n%s"% self.settings)


    def Size(self):
        return self.width(),self.height()
    

    def glMatSpec(self):
        GL.glMaterialfv(fill_mode,GL.GL_SPECULAR,colors.GREY(self.specular))
        GL.glMaterialfv(fill_mode,GL.GL_EMISSION,colors.GREY(self.emission))
        GL.glMaterialfv(fill_mode,GL.GL_SHININESS,self.shininess)


    def glLightSpec(self):
        GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT,colors.GREY(self.ambient))
        GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, 1)
        GL.glLightModeli(GL.GL_LIGHT_MODEL_LOCAL_VIEWER, 0)
        if GD.canvas:
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glPushMatrix()
            GL.glLoadIdentity()
            for l in GD.canvas.lights:
                l.enable()
            GL.glPopMatrix()
        self.glMatSpec()
        GL.glColorMaterial(fill_mode,GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glEnable(GL.GL_COLOR_MATERIAL)


    def glLight(self,onoff):
        """Toggle lights on/off."""
        if onoff:
            self.glLightSpec()
            GL.glEnable(GL.GL_LIGHTING)
        else:
            GL.glDisable(GL.GL_LIGHTING)


    def hasLight(self):
        """Return the status of the lighting."""
        return GL.glIsEnabled(GL.GL_LIGHTING)
        

    def resetDefaults(self,dict={}):
        """Return all the settings to their default values."""
        self.settings.reset(dict)
        self.resetLights()


    def resetLighting(self):
        self.ambient = GD.cfg['render/ambient']
        self.specular = GD.cfg['render/specular']
        self.emission = GD.cfg['render/emission']
        self.shininess = GD.cfg['render/shininess']

        
    def resetLights(self):
        self.lights = []
        for i in range(8):
            light = GD.cfg.get('render/light%d' % i, None)
            if light is not None:
                self.lights.append(Light(i,light))


    def setRenderMode(self,rm):
        """Set the rendermode.

        This changes the rendermode and redraws everything with the new mode.
        """
        #GD.debug("Changing Render Mode to %s" % rm)
        if rm != self.rendermode:
            if rm not in Canvas.rendermodes:
                rm = Canvas.rendermodes[0]
            self.rendermode = rm
            #GD.debug("Redrawing with mode %s" % self.rendermode)
            self.glinit(self.rendermode)
            self.redrawAll()


    def setTransparency(self,mode):
        self.alphablend = mode

    def setLighting(self,mode):
        self.lighting = mode
        self.glLight(self.lighting)

    def setAveragedNormals(self,mode):
        self.avgnormals = mode
        change = (self.rendermode == 'smooth' and self.avgnormals) or \
                 (self.rendermode == 'smooth_avg' and not self.avgnormals)
        if change:
            if self.avgnormals:
                self.rendermode = 'smooth_avg'
            else:
                self.rendermode = 'smooth'
            self.actors.redraw()
            self.display()
       

    def setLineWidth(self,lw):
        """Set the linewidth for line rendering."""
        self.settings.linewidth = float(lw)


    def setBgColor(self,color1,color2=None):
        """Set the background color.

        If one color is specified, a solid background is set.
        If two colors are specified, a graded background is set
        and an object is created to display the background.
        """
        self.settings.bgcolor = colors.GLColor(color1)
        if color2 is None:
            GD.debug("Clearing twocolor background")
            self.settings.bgcolor2 = None
            self.background = None
        else:
            self.settings.bgcolor2 = colors.GLColor(color2)
            self.createBackground()
            glSmooth()
            glFill()
        self.clear()
        self.redrawAll()


    def createBackground(self):
        """Create the background object."""
        x1,y1 = 0,0
        x2,y2 = self.Size()
        color4 = [self.settings.bgcolor2,self.settings.bgcolor2,self.settings.bgcolor,self.settings.bgcolor]
        self.background = decors.Rectangle(x1,y1,x2,y2,color=color4)
        

    def setFgColor(self,color):
        """Set the default foreground color."""
        self.settings.fgcolor = colors.GLColor(color)
        

    def setSlColor(self,color):
        """Set the highlight color."""
        self.settings.slcolor = colors.GLColor(color)


    def updateSettings(self,settings):
        """Update the viewport settings"""
        for k,v in settings.items():
            if k == 'linewidth':
                self.setLineWidth(v)
            elif k == 'bgcolor':
                if 'bgcolor2' in settings:
                    self.setBgColor(v,settings.get('bgcolor2',None))
            elif k == 'bgcolor2':
                if 'bgcolor' in settings:
                    pass
                else:
                    self.setBgColor(self.settings.bgcolor,v)
            elif k == 'fgcolor':
                self.setFgColor(v)
            elif k == 'slcolor':
                self.setSlColor(v)
        
        

    
    def setLight(self,nr,ambient,diffuse,specular,position):
        """(Re)Define a light on the scene."""
        self.lights[nr].set(ambient,diffuse,specular,position)

    def enableLight(self,nr):
        """Enable an existing light."""
        self.lights[nr].enable()

    def disableLight(self,nr):
        """Disable an existing light."""
        self.lights[nr].disable()


    def setTriade(self,on=None,size=1.0,pos=[0.0,0.0,0.0]):
        """Toggle the display of the global axes on or off.

        If on is True, a triade of global axes is displayed, if False it is
        removed. The default (None) toggles between on and off.
        """
        if on is None:
            on = self.triade is None
        GD.debug("SETTING TRIADE %s" % on)
        if self.triade:
            self.removeAnnotation(self.triade)
            self.triade = None
        if on:
            self.triade = actors.TriadeActor(size,pos)
            self.addAnnotation(self.triade)
    

    def initCamera(self):
        #if GD.options.makecurrent:
        self.makeCurrent()  # we need correct OpenGL context for camera
        self.camera = camera.Camera()
        #GD.debug("camera.rot = %s" % self.camera.rot)
        #GD.debug("view angles: %s" % self.view_angles)


    def glinit(self,mode=None):
        if mode:
            self.rendermode = mode


        if self.settings.bgcolor != self.settings.bgcolor2:
            self.setBgColor(self.settings.bgcolor,self.settings.bgcolor2)

        self.clear()
        #GL.glClearColor(*colors.RGBA(self.default.bgcolor))# Clear The Background Color
        GL.glClearDepth(1.0)	       # Enables Clearing Of The Depth Buffer
        GL.glDepthFunc(GL.GL_LESS)	       # The Type Of Depth Test To Do
        GL.glEnable(GL.GL_DEPTH_TEST)	       # Enables Depth Testing
        #GL.glEnable(GL.GL_CULL_FACE)
        

        # On initializing a rendering mode, we also set default lighting
        if self.rendermode == 'wireframe':
            if self.background:
                glSmooth()
                glFill()
            else:
                #glFlat()
                glLine()
            self.lighting = False
            self.glLight(False)

                
        elif self.rendermode.startswith('flat'):
            if self.background:
                glSmooth()
            else:
                glFlat()
            glFill()
            self.lighting = False
            self.glLight(False)
               
        elif self.rendermode.startswith('smooth'):
            glSmooth()
            glFill()
            self.lighting = True
            self.glLight(True)
            
        else:
            raise RuntimeError,"Unknown rendering mode"

        if self.rendermode.endswith('wire'):
            GL.glPolygonOffset(1.0,1.0) 
        

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
        if self.rendermode.startswith('smooth'):
            self.glMatSpec()

    
    def setSize (self,w,h):
        if h == 0:	# prevent divide by zero 
            h = 1
        GL.glViewport(0, 0, w, h)
        self.aspect = float(w)/h
        self.camera.setLens(aspect=self.aspect)
        if self.background:
            # recreate the background to match the current size
            self.createBackground()
        self.display()


    def display(self):
        """(Re)display all the actors in the scene.

        This should e.g. be used when actors are added to the scene,
        or after changing  camera position/orientation or lens.
        """
        #GD.debugt("UPDATING CURRENT OPENGL CANVAS")
        self.makeCurrent()
        self.clear()
        self.glLight(self.lighting)
        
        # decorations are drawn in 2D mode
        self.begin_2D_drawing()
        if self.background:
            GD.debug("Displaying background")
            self.background.draw(mode='smooth')

        if len(self.decorations) > 0:
            for actor in self.decorations:
                self.setDefaults()
                ## if hasattr(actor,'zoom'):
                ##     self.zoom_2D(actor.zoom)
                actor.draw(mode=self.rendermode)
                ## if hasattr(actor,'zoom'):
                ##     self.zoom_2D()

        # draw the focus rectangle if more than one viewport
        if len(GD.GUI.viewports.all) > 1:
            if self.hasFocus():
                self.draw_focus_rectangle(2)
            elif self.focus:
                self.draw_focus_rectangle(1)
        self.end_2D_drawing()
        
        # draw the highlighted actors
        self.camera.loadProjection()
        self.camera.loadMatrix()
        if self.highlights:
            for actor in self.highlights:
                self.setDefaults()
                actor.draw(mode=self.rendermode)

        # draw the scene actors
        if self.alphablend:
            opaque = [ a for a in self.actors if not a.trans ]
            transp = [ a for a in self.actors if a.trans ]
            for actor in opaque:
                self.setDefaults()
                actor.draw(mode=self.rendermode)
            GL.glEnable (GL.GL_BLEND)
            GL.glDepthMask (GL.GL_FALSE)
            GL.glBlendFunc (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            for actor in transp:
                self.setDefaults()
                actor.draw(mode=self.rendermode)
            GL.glDepthMask (GL.GL_TRUE)
            GL.glDisable (GL.GL_BLEND)
        else:
            for actor in self.actors:
                self.setDefaults()
                actor.draw(mode=self.rendermode)

        # annotations are drawn in 3D space
        for actor in self.annotations:
            self.setDefaults()
            actor.draw(mode=self.rendermode)

        # decorations are drawn in 2D mode
        # decoration drawing was moved to the start of the drawing cycle

        # make sure canvas is updated
        GL.glFlush()


    def zoom_2D(self,zoom=None):
        if zoom is None:
            zoom = (0,self.width(),0,self.height())
        GLU.gluOrtho2D(*zoom)
            

    def begin_2D_drawing(self):
        """Set up the canvas for 2D drawing on top of 3D canvas.

        The 2D drawing operation should be ended by calling end_2D_drawing. 
        It is assumed that you will not try to change/refresh the normal
        3D drawing cycle during this operation.
        """
        #GD.debug("Start 2D drawing")
        if self.mode2D:
            #GD.debug("WARNING: ALREADY IN 2D MODE")
            return
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        self.zoom_2D()
        GL.glDisable(GL.GL_DEPTH_TEST)
        self.glLight(False)
        self.mode2D = True

 
    def end_2D_drawing(self):
        """Cancel the 2D drawing mode initiated by begin_2D_drawing."""
        #GD.debug("End 2D drawing")
        if self.mode2D:
            GL.glEnable(GL.GL_DEPTH_TEST)    
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glPopMatrix()
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glPopMatrix()
            self.glLight(self.lighting)
            self.mode2D = False
       
        
    def setBbox(self,bbox=None):
        """Set the bounding box of the scene you want to be visible."""
        # TEST: use last actor
        #GD.debug("BBOX WAS: %s" % self.bbox)
        if bbox is None:
            if len(self.actors) > 0:
                bbox = self.actors[-1].bbox()
            else:
                bbox = [[-1.,-1.,-1.],[1.,1.,1.]]
        bbox = asarray(bbox)
        if bbox.any() == nan:
            GD.message("Invalid Bbox: %s" % bbox)
            return
        self.bbox = nan_to_num(bbox)
        #GD.debug("BBOX BECOMES: %s" % self.bbox)

         
    def addActor(self,actor):
        """Add a 3D actor to the 3D scene."""
        self.actors.add(actor)

    def removeActor(self,actor):
        """Remove a 3D actor from the 3D scene."""
        self.actors.delete(actor)
        #self.highlights.delete(actor)

    def addHighlight(self,actor):
        """Add a 3D actor highlight to the 3D scene."""
        self.highlights.add(actor)

    def removeHighlight(self,actor):
        """Remove a 3D actor highlight from the 3D scene."""
        self.highlights.delete(actor)
         
    def addAnnotation(self,actor):
        """Add an annotation to the 3D scene."""
        self.annotations.add(actor)

    def removeAnnotation(self,actor):
        """Remove an annotation from the 3D scene."""
##         if actor == self.triade:
##             self.triade = None
        self.annotations.delete(actor)
         
    def addDecoration(self,actor):
        """Add a 2D decoration to the canvas."""
        self.decorations.add(actor)

    def removeDecoration(self,actor):
        """Remove a 2D decoration from the canvas."""
        self.decorations.delete(actor)

    def remove(self,itemlist):
        """Remove a list of any actor/highlights/annotation/decoration items.

        This will remove the items from any of the canvas lists in which the
        item appears.
        itemlist can also be a single item instead of a list.
        """
        if not type(itemlist) == list:
            itemlist = [ itemlist ]
        for item in itemlist:
            self.actors.delete(item)
            self.highlights.delete(item)
            self.annotations.delete(item)
            self.decorations.delete(item)
        

    def removeActors(self,actorlist=None):
        """Remove all actors in actorlist (default = all) from the scene."""
        if actorlist == None:
            actorlist = self.actors[:]
        for actor in actorlist:
            if actor is self.triade:
                self.triade == None
            self.removeActor(actor)
        self.setBbox()
        

    def removeHighlights(self,actorlist=None):
        """Remove all highlights in actorlist (default = all) from the scene."""
        if actorlist == None:
            actorlist = self.highlights[:]
        for actor in actorlist:
            self.removeHighlight(actor)


    def removeAnnotations(self,actorlist=None):
        """Remove all actors in actorlist (default = all) from the scene."""
        GD.debug("REMOVING ALL ACTORS")
        if actorlist == None:
            actorlist = self.annotations[:]
        for actor in actorlist:
            if actor == self.triade:
                GD.debug("REMOVING TRIADE")
                self.triade = None
            self.removeAnnotation(actor)


    def removeDecorations(self,actorlist=None):
        """Remove all decorations in actorlist (default = all) from the scene."""
        if actorlist == None:
            actorlist = self.decorations[:]
        for actor in actorlist:
            self.removeDecoration(actor)


    def removeAll(self):
        """Remove all actors and decorations"""
        self.removeActors()
        self.removeHighlights()
        self.removeAnnotations()
        self.removeDecorations()
        self.triade = None


    def redrawAll(self):
        """Redraw all actors in the scene."""
        self.actors.redraw()
        self.highlights.redraw()
        self.annotations.redraw()
        self.decorations.redraw()
        self.display()

        
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
        angles can be a set of 3 angles, or a string
        """
        self.makeCurrent()
        # go to a distance to have a good view with a 45 degree angle lens
        if bbox is not None:
            GD.debug("SETTING BBOX: %s" % self.bbox)
            self.setBbox(bbox)
        GD.debug("USING BBOX: %s" % self.bbox)
        X0,X1 = self.bbox
        center = 0.5*(X0+X1)
        # calculating the bounding circle: this is rather conservative
        self.camera.setCenter(*center)
        if type(angles) is str:
            angles = self.view_angles.get(angles)
        if angles is not None:
            try:
                self.camera.setAngles(angles)
            except:
                raise ValueError,'Invalid view angles specified'
        # Currently, we keep the default fovy/aspect
        # and change the camera distance to focus
        fovy = self.camera.fovy
        #GD.debug("FOVY: %s" % fovy)
        self.camera.setLens(fovy,self.aspect)
        # Default correction is sqrt(3)
        correction = float(GD.cfg.get('gui/autozoomfactor',1.732))
        tf = tand(fovy/2.)

        import simple,coords
        bbix = simple.regularGrid(X0,X1,[1,1,1])
        bbix = dot(bbix,self.camera.rot[:3,:3])
        bbox = coords.Coords(bbix).bbox()
        dx,dy,dz = bbox[1] - bbox[0]
        vsize = max(dx/self.aspect,dy)
        offset = dz
        dist = (vsize/tf + offset) / correction
        
        if dist == nan or dist == inf:
            GD.debug("DIST: %s" % dist)
            return
        if dist <= 0.0:
            dist = 1.0
        self.camera.setDist(dist)
        self.camera.setClip(0.01*dist,100.*dist)
        self.camera.resetArea()


    def zoom(self,f,dolly=True):
        """Dolly zooming."""
        if dolly:
            self.camera.dolly(f)


    def project(self,x,y,z):
        "Map the object coordinates (x,y,z) to window coordinates."""
        self.makeCurrent()
        self.camera.loadProjection()
        model = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        proj = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        view = GL.glGetIntegerv(GL.GL_VIEWPORT)
        winx,winy,winz = GLU.gluProject(x,y,z,model,proj,view)
        #GD.debug("Object coordinates: ",x,y,z," map to ",winx,winy,winz)
        return winx,winy,winz


    def unProject(self,x,y,z):
        "Map the window coordinates (x,y,z) to object coordinates."""
        self.makeCurrent()
        self.camera.loadProjection()
        model = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        proj = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        view = GL.glGetIntegerv(GL.GL_VIEWPORT)
        objx, objy, objz = GLU.gluUnProject(x,y,z,model,proj,view)
        #GD.debug("Window coordinates: ",x,y,z," map to ",objx,objy,objz)
        return (objx,objy,objz)


    def zoomRectangle(self,x0,y0,x1,y1):
        """Rectangle zooming

        x0,y0,x1,y1 are pixel coordinates of the lower left and upper right
        corners of the area to zoom to the full window
        """
        ## WE SHOULD ADD FACILITIES TO KEEP THE ASPECT RATIO
        w,h = float(self.width()),float(self.height())
        self.camera.setArea(x0/w,y0/h,x1/w,y1/h)


    def zoomAll(self):
        """Rectangle zooming

        x0,y0,x1,y1 are relative corners in (0,0)..(1,1) space
        """
        self.camera.resetArea()


    def saveBuffer(self):
        """Save the current OpenGL buffer"""
        self.save_buffer = GL.glGetIntegerv(GL.GL_DRAW_BUFFER)

    def showBuffer(self):
        """Show the saved buffer"""
        pass

    def draw_focus_rectangle(self,width=2):
        """Draw the focus rectangle.

        The specified width is HALF of the line width"""
        lw = width
        w,h = self.width(),self.height()
        self._focus = decors.Grid(lw,lw,w-lw,h-lw,color=colors.pyformex_pink,linewidth=2*lw)
        self._focus.draw()

    def draw_cursor(self,x,y):
        """draw the cursor"""
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
