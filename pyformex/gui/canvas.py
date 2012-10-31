# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""This implements an OpenGL drawing widget for painting 3D scenes.

"""
from __future__ import print_function

import pyformex as pf
import coords

from numpy import *
from OpenGL import GL,GLU

from formex import length
from drawable import saneColor,glColor
import colors
import camera
import actors
import decors
import marks
import utils
from mydict import Dict



def gl_pickbuffer():
    "Return a list of the 2nd numbers in the openGL pick buffer."
    buf = GL.glRenderMode(GL.GL_RENDER)
    return asarray([ r[2] for r in buf ])


from OpenGL.GL import glLineWidth as glLinewidth, glPointSize as glPointsize

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

def glFill(fill=True):
    if fill:
        GL.glPolygonMode(fill_mode,GL.GL_FILL)
    else:
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK,GL.GL_LINE)

def glLineSmooth(onoff):
    if onoff is True:
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
    elif onoff is False:
        GL.glDisable(GL.GL_LINE_SMOOTH)

    
def glLineStipple(factor,pattern):
    """Set the line stipple pattern.

    When drawing lines, OpenGl can use a stipple pattern. The stipple
    is defined by two values: a pattern (on/off) of maximum 16 bits,
    used on the pixel level, and a multiplier factor for each bit.

    If factor <= 0, the stippling is disabled.
    """
    if factor > 0:
        GL.glLineStipple(factor,pattern)
        GL.glEnable(GL.GL_LINE_STIPPLE)    
    else:
        GL.glDisable(GL.GL_LINE_STIPPLE)

def glSmooth(smooth=True):
    """Enable smooth shading"""
    if smooth:
        GL.glShadeModel(GL.GL_SMOOTH)
    else:
        GL.glShadeModel(GL.GL_FLAT)
        
def glFlat():
    """Disable smooth shading"""
    glSmooth(False)
    

def onOff(onoff):
    """Convert On/Off strings to a boolean"""
    if type(onoff) is str:
        return (onoff.lower() == 'on')
    else:
        if onoff:
            return True
        else:
            return False


def glEnable(facility,onoff):
    """Enable/Disable an OpenGL facility, depending on onoff value

    facility is an OpenGL facility.
    onoff can be True or False to enable, resp. disable the facility, or
    None to leave it unchanged.
    """
    pf.debug("%s: %s" % (facility,onoff),pf.DEBUG.DRAW)
    if onOff(onoff):
        pf.debug("ENABLE",pf.DEBUG.DRAW)
        GL.glEnable(facility)
    else:
        pf.debug("DISABLE",pf.DEBUG.DRAW)
        GL.glDisable(facility)
        

def glCulling(onoff=True):
    glEnable(GL.GL_CULL_FACE,onoff)
def glNoCulling():
    glCulling(False)

def glLighting(onoff):
    glEnable(GL.GL_LIGHTING,onoff)


def glPolygonFillMode(mode):
    if type(mode) is str:
        mode = mode.lower()
        if mode == 'Front and Back':
            glBothFill()
        elif mode == 'Front':
            glFrontFill()
        elif mode == 'Back':
            glBackFill()

            
def glPolygonMode(mode):
    if type(mode) is str:
        mode = mode.lower()
        glFill(mode == 'fill')
    

def glShadeModel(model):
    if type(model) is str:
        model = model.lower()
        if model == 'smooth':
            glSmooth()
        elif model == 'flat':
            glFlat()

    
class ActorList(list):
    """A list of drawn objects of the same kind.

    This is used to collect the Actors, Decorations and Annotations
    in a scene.
    Currently the implementation does not check that the objects are of
    the proper type.
    """
    
    def __init__(self,canvas):
        self.canvas = canvas
        list.__init__(self)
        
    def add(self,actor):
        """Add an actor or a list thereof to a ActorList."""
        if type(actor) is list:
            self.extend(actor)
        else:
            self.append(actor)

    def delete(self,actor):
        """Remove an actor or a list thereof from an ActorList."""
        if not type(actor) in (list,tuple):
            actor = [ actor ]
        for a in actor:
            if a in self:
                self.remove(a)

    def redraw(self):
        """Redraw all actors in the list.

        This redraws the specified actors (recreating their display list).
        This could e.g. be used after changing an actor's properties.
        """
        for actor in self:
            actor.redraw()



############### OpenGL Lighting #################################

class Material(object):
    def __init__(self,name,ambient=0.2,diffuse=0.2,specular=0.9,emission=0.1,shininess=2.0):
        self.name = str(name)
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)
        self.specular = float(specular)
        self.emission = float(emission)
        self.shininess = float(shininess)


    def setValues(self,**kargs):
        #print "setValues",kargs
        for k in kargs:
            #print k,kargs[k]
            if hasattr(self,k):
                #print getattr(self,k)
                setattr(self,k,float(kargs[k]))
                #print getattr(self,k)


    def activate(self):   
        GL.glMaterialfv(fill_mode,GL.GL_AMBIENT,colors.GREY(self.ambient))
        GL.glMaterialfv(fill_mode,GL.GL_DIFFUSE,colors.GREY(self.diffuse))
        GL.glMaterialfv(fill_mode,GL.GL_SPECULAR,colors.GREY(self.specular))
        GL.glMaterialfv(fill_mode,GL.GL_EMISSION,colors.GREY(self.emission))
        GL.glMaterialfv(fill_mode,GL.GL_SHININESS,self.shininess)


    def dict(self):
        """Return the material light parameters as a dict"""
        return dict([(k,getattr(self,k)) for k in ['ambient','diffuse','specular','emission','shininess']])
    

    def __str__(self):
        return """MATERIAL: %s
    ambient:  %s
    diffuse:  %s
    specular: %s
    emission: %s
    shininess: %s
""" % (self.name,self.ambient,self.diffuse,self.specular,self.emission,self.shininess)
        


def getMaterials():
    mats = pf.refcfg['material']
    mats.update(pf.prefcfg['material'])
    mats.update(pf.cfg['material'])
    return mats


def createMaterials():
    mats = getMaterials()
    matdb = {}
    for m in mats:
        matdb[m] = Material(m,**mats[m])
    return matdb


class Light(object):

    def __init__(self,nr,**kargs):
        self.light = GL.GL_LIGHT0 + (nr % GL.GL_MAX_LIGHTS)
        self.set(ambient=0.5,diffuse=0.5,specular=0.5,position=[0.,0.,1.,0.],enabled=False)

    def set(self,**kargs):
        for k in kargs:
            self.set_value(k,kargs[k])

    def set_value(self,key,value):
        if key in [ 'ambient','diffuse','specular' ]:
            value = colors.GREY(value)
        setattr(self,key,value)

    def enable(self):
        GL.glEnable(self.light)
        GL.glLightfv(self.light,GL.GL_AMBIENT,self.ambient)
        GL.glLightfv(self.light,GL.GL_DIFFUSE,self.diffuse)
        GL.glLightfv(self.light,GL.GL_SPECULAR,self.specular)
        GL.glLightfv(self.light,GL.GL_POSITION,self.position)

    def disable(self):
        GL.glDisable(self.light)

    def __str__(self):
        return """LIGHT %s:
    ambient color:  %s
    diffuse color:  %s
    specular color: %s
    position: %s
""" % (self.light-GL.GL_LIGHT0,self.ambient,self.diffuse,self.specular,self.position)


class LightProfile(object):

    light_model = {
        'ambient': GL.GL_AMBIENT,
        'diffuse': GL.GL_DIFFUSE,
        'ambient and diffuse': GL.GL_AMBIENT_AND_DIFFUSE,
        'emission': GL.GL_EMISSION,
        'specular': GL.GL_SPECULAR,
        }
    
    def __init__(self,model,ambient,lights):
        self.model = self.light_model[model]
        self.ambient = ambient
        self.lights = lights

    def activate(self):
        #GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(fill_mode,self.model)
        GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT,colors.GREY(self.ambient))
        GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, 1)
        GL.glLightModeli(GL.GL_LIGHT_MODEL_LOCAL_VIEWER, 0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        for light in self.lights:
            light.enable()
        GL.glPopMatrix()
        


##################################################################
#
#  The Canvas Settings
#

class CanvasSettings(Dict):
    """A collection of settings for an OpenGL Canvas.

    The canvas settings are a collection of settings and default values
    affecting the rendering in an individual viewport. There are two type of
    settings:
    
    - mode settings are set during the initialization of the canvas and
      can/should not be changed during the drawing of actors and decorations;
    - default settings can be used as default values but may be changed during
      the drawing of actors/decorations: they are reset before each individual
      draw instruction.
      
    Currently the following mode settings are defined:
    
    - bgcolor: the viewport background color: a single color or a list of
      colors (max. 4 are used).
    - bgimage: background image filename
    - slcolor: the highlight color
    - alphablend: boolean (transparency on/off)

    The list of default settings includes:

    - fgcolor: the default drawing color
    - bkcolor: the default backface color
    - colormap: the default color map to be used if color is an index
    - bklormap: the default color map to be used if bkcolor is an index
    - smooth: boolean (smooth/flat shading)
    - lighting: boolean (lights on/off)
    - culling: boolean
    - transparency: float (0.0..1.0)
    - avgnormals: boolean
    - edges: 'none', 'feature' or 'all' 
    - pointsize: the default size for drawing points
    - marksize: the default size for drawing markers
    - linewidth: the default width for drawing lines

    Any of these values can be set in the constructor using a keyword argument.
    All items that are not set, will get their value from the configuration
    file(s).
    """

    # A collection of default rendering profiles.
    # These contain the values different from the overall defaults
    RenderProfiles = {
        'wireframe': Dict({
            'smooth': False,
            'fill': False,
            'lighting': False,
            'alphablend': False,
            'transparency': 1.0,
            'edges': 'none',
            'avgnormals': False,
            }),
        'smooth': Dict({
            'smooth': True,
            'fill': True,
            'lighting': True,
            'alphablend': False,
            'transparency': 0.5,
            'edges': 'none',
            'avgnormals': False,
            }),
        'smooth_avg': Dict({
            'smooth': True,
            'fill': True,
            'lighting': True,
            'alphablend': False,
            'transparency': 0.5,
            'edges': 'none',
            'avgnormals': True,
            }),
        'smoothwire': Dict({
            'smooth': True,
            'fill': True,
            'lighting': True,
            'alphablend': False,
            'transparency': 0.5,
            'edges': 'all',
            'avgnormals': False,
            }),
        'flat': Dict({
            'smooth': False,
            'fill': True,
            'lighting': False,
            'alphablend': False,
            'transparency': 0.5,
            'edges': 'none',
            'avgnormals': False,
            }),
        'flatwire': Dict({
            'smooth': False,
            'fill': True,
            'lighting': False,
            'alphablend': False,
            'transparency': 0.5,
            'edges': 'all',
            'avgnormals': False,
            }),
        }
    edge_options = [ 'none','feature','all' ]
    
    def __init__(self,**kargs):
        """Create a new set of CanvasSettings."""
        Dict.__init__(self)
        self.reset(kargs)

    def reset(self,d={}):
        """Reset the CanvasSettings to its defaults.

        The default values are taken from the configuration files.
        An optional dictionary may be specified to override (some of) these defaults.
        """
        self.update(pf.refcfg['canvas'])
        self.update(self.RenderProfiles[pf.prefcfg['draw/rendermode']])
        self.update(pf.prefcfg['canvas'])
        self.update(pf.cfg['canvas'])
        if d:
            self.update(d)

    def update(self,d,strict=True):
        """Update current values with the specified settings

        Returns the sanitized update values.
        """
        ok = self.checkDict(d,strict)
        Dict.update(self,ok)

    @classmethod
    def checkDict(clas,dict,strict=True):
        """Transform a dict to acceptable settings."""
        ok = {}
        for k,v in dict.items():
            try:
                if k in [ 'bgcolor', 'fgcolor', 'bkcolor', 'slcolor',
                          'colormap','bkcolormap' ]:
                    if v is not None:
                        v = saneColor(v)
                elif k in ['bgimage']:
                    v = str(v)
                elif k in ['smooth', 'fill', 'lighting', 'culling',
                           'alphablend', 'avgnormals',]:
                    v = bool(v)
                elif k in ['linewidth', 'pointsize', 'marksize']:
                    v = float(v)
                elif k == 'linestipple':
                    v = map(int,v)
                elif k == 'transparency':
                    v = max(min(float(v),1.0),0.0)
                elif k == 'edges':
                    v = str(v).lower()
                    if not v in clas.edge_options:
                        raise
                elif k == 'marktype':
                    pass
                else:
                    raise
                ok[k] = v
            except:
                if strict:
                    raise ValueError,"Invalid key/value for CanvasSettings: %s = %s" % (k,v)
        return ok
    
    def __str__(self):
        return utils.formatDict(self)


    def setMode(self):
        """Activate the mode canvas settings in the GL machine."""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.bgcolor.ndim > 1:
            color = self.bgcolor[0]
        else:
            color = self.bgcolor
        GL.glClearColor(*colors.RGBA(color))
 

    def activate(self):
        """Activate the default canvas settings in the GL machine."""
        self.glOverride(self,self)


    @staticmethod
    def glOverride(settings,default):
        #if settings != default:
        #print("OVERRIDE CANVAS SETINGS %s" % settings['fill'])
        for k in settings:
            if k in ['fgcolor','transparency']:
                c = settings.get('fgcolor',default.fgcolor)
                t = settings.get('transparency',default.transparency)
                glColor(c,t)
            elif k == 'linestipple':
                glLineStipple(*settings[k])
            elif k in ['smooth','fill','lighting','linewidth','pointsize']:
                func = globals()['gl'+k.capitalize()]
                func(settings[k])
            ## else:
            ##     print("CAN NOT SET %s" % k)


### OLD: to be rmoved
def glSettings(settings):
    pf.debug("GL SETTINGS: %s" % settings,pf.DEBUG.DRAW)
    glShadeModel(settings.get('Shading',None))
    glCulling(settings.get('Culling',None))
    glLighting(settings.get('Lighting',None))
    glLineSmooth(onOff(settings.get('Line Smoothing',None)))
    glPolygonFillMode(settings.get('Polygon Fill',None))
    glPolygonMode(settings.get('Polygon Mode',None))
    pf.canvas.update()
    
            

def extractCanvasSettings(d):
    """Split a dict in canvas settings and other items.

    Returns a tuple of two dicts: the first one contains the items
    that are canvas settings, the second one the rest.
    """
    return utils.select(d,pf.refcfg['canvas']),utils.remove(d,pf.refcfg['canvas'])


##################################################################
#
#  The Canvas
#

def print_camera(self):
    print(self.report())

    
class Canvas(object):
    """A canvas for OpenGL rendering.

    The Canvas is a class holding all global data of an OpenGL scene rendering.
    This includes colors, line types, rendering mode.
    It also keeps lists of the actors and decorations in the scene.
    The canvas has a Camera object holding important viewing parameters.
    Finally, it stores the lighting information.
    
    It does not however contain the viewport size and position.
    """

    def __init__(self,settings={}):
        """Initialize an empty canvas with default settings."""
        self.actors = ActorList(self)
        self.highlights = ActorList(self)
        self.annotations = ActorList(self)
        self.decorations = ActorList(self)
        self.triade = None
        self.background = None
        self.bbox = None
        self.setBbox()
        self.settings = CanvasSettings(**settings)
        self.resetLighting()
        self.mode2D = False
        self.rendermode = pf.cfg['draw/rendermode']
        self.setRenderMode(pf.cfg['draw/rendermode'])
        #print("INIT: %s, %s" %(self.rendermode,self.settings.fill))
        self.camera = None
        self.view_angles = camera.view_angles
        self.cursor = None
        self.focus = False
        pf.debug("Canvas Setting:\n%s"% self.settings,pf.DEBUG.DRAW)


    def enable_lighting(self,state):
        """Toggle lights on/off."""
        if state:
            self.lightprof.activate()
            self.material.activate()
            #print("ENABLE LIGHT")
            GL.glEnable(GL.GL_LIGHTING)
        else:
            #print("DISABLE LIGHT")
            GL.glDisable(GL.GL_LIGHTING)


    def has_lighting(self):
        """Return the status of the lighting."""
        return GL.glIsEnabled(GL.GL_LIGHTING)
        

    def resetDefaults(self,dict={}):
        """Return all the settings to their default values."""
        self.settings.reset(dict)
        self.resetLighting()
        ## self.resetLights()

    def setAmbient(self,ambient):
        """Set the global ambient lighting for the canvas"""
        self.lightprof.ambient = float(ambient)
        
    def setMaterial(self,matname):
        """Set the default material light properties for the canvas"""
        self.material = pf.GUI.materials[matname]
        

    def resetLighting(self):
        """Change the light parameters"""
        self.lightmodel = pf.cfg['render/lightmodel']
        self.setMaterial(pf.cfg['render/material'])
        self.lightset = pf.cfg['render/lights']
        lights = [ Light(int(light[-1:]),**pf.cfg['light/%s' % light]) for light in self.lightset ]
        self.lightprof = LightProfile(self.lightmodel,pf.cfg['render/ambient'],lights)


    def setRenderMode(self,mode,lighting=None):
        """Set the rendering mode.

        This sets or changes the rendermode and lighting attributes.
        If lighting is not specified, it is set depending on the rendermode.
        
        If the canvas has not been initialized, this merely sets the
        attributes self.rendermode and self.settings.lighting.
        If the canvas was already initialized (it has a camera), and one of
        the specified settings is fdifferent from the existing, the new mode
        is set, the canvas is re-initialized according to the newly set mode,
        and everything is redrawn with the new mode.
        """
        #print("Setting rendermode to %s" % mode)
        if mode not in CanvasSettings.RenderProfiles:
            raise ValueError,"Invalid render mode %s" % mode

        self.settings.update(CanvasSettings.RenderProfiles[mode])
        #print(self.settings)
        if lighting is None:
            lighting = self.settings.lighting
            
        if mode != self.rendermode or lighting != self.settings.lighting:
            #print("SWITCHING MODE")
            self.rendermode = mode
            self.settings.lighting = lighting
            self.glinit()


    def setToggle(self,attr,state):
        """Set or toggle a boolean settings attribute

        Furthermore, if a Canvas method do_ATTR is defined, it will be called
        with the old and new toggle state as a parameter.
        """
        #print("Toggling %s = %s"%(attr,state),pf.DEBUG.CANVAS)
        oldstate = self.settings[attr]
        if state not in [True,False]:
            state = not oldstate
        self.settings[attr] = state
        try:
            func = getattr(self,'do_'+attr)
            func(state,oldstate)
        except:
            pass


    def setLighting(self,onoff):
        self.setToggle('lighting',onoff)


    def do_lighting(self,state,oldstate=None):
        """Toggle lights on/off."""
        #print("TOGGLING LIGHTING %s %s"%(state,oldstate))
        if state != oldstate:
            self.enable_lighting(state)


    def do_avgnormals(self,state,oldstate):
        #print("Toggling avgnormals",self.rendermode,state,oldstate)
        if state!=oldstate and self.rendermode.startswith('smooth'):
            if self.settings.avgnormals:
                self.rendermode = 'smooth_avg'
            else:
                self.rendermode = 'smooth'
            #print("REDRAW")
            self.actors.redraw()
            self.display()
       

    def setLineWidth(self,lw):
        """Set the linewidth for line rendering."""
        self.settings.linewidth = float(lw)
       

    def setLineStipple(self,repeat,pattern):
        """Set the linestipple for line rendering."""
        self.settings.update({'linestipple':(repeat,pattern)})
        

    def setPointSize(self,sz):
        """Set the size for point drawing."""
        self.settings.pointsize = float(sz)


    def setBackground(self,color=None,image=None):
        """Set the color(s) and image.

        Change the background settings according to the specified parameters
        and set the canvas background accordingly. Only (and all) the specified
        parameters get a new value.
        
        Parameters:

        - `color`: either a single color, a list of two colors or a list of
          four colors.
        - `image`: an image to be set. 
        """
        self.settings.update(dict(bgcolor=color,bgimage=image))
        color = self.settings.bgcolor
        if color.ndim == 1 and not self.settings.bgimage:
            pf.debug("Clearing fancy background",pf.DEBUG.DRAW)
            self.background = None
        else:
            self.createBackground()
            #glSmooth()
            #glFill()
        self.clear()
        self.redrawAll()
        #self.update()


    def createBackground(self):
        """Create the background object."""
        x1,y1 = 0,0
        x2,y2 = self.getSize()
        from gui.drawable import saneColorArray
        color = saneColorArray(self.settings.bgcolor,(4,))
        #print color.shape,color
        image = None
        if self.settings.bgimage:
            from gui.imagearray import image2numpy
            try:
                image = image2numpy(self.settings.bgimage,indexed=False)
            except:
                pass
        #print("BACKGROUN %s,%s"%(x2,y2))
        self.background = decors.Rectangle(x1,y1,x2,y2,color=color,texture=image)
        

    def setFgColor(self,color):
        """Set the default foreground color."""
        self.settings.fgcolor = colors.GLColor(color)
        

    def setSlColor(self,color):
        """Set the highlight color."""
        self.settings.slcolor = colors.GLColor(color)



    def setTriade(self,on=None,pos='lb',siz=100):
        """Toggle the display of the global axes on or off.

        If on is True, a triade of global axes is displayed, if False it is
        removed. The default (None) toggles between on and off.
        """
        if on is None:
            on = self.triade is None
        pf.debug("SETTING TRIADE %s" % on,pf.DEBUG.DRAW)
        if self.triade:
            self.removeAnnotation(self.triade)
            self.triade = None
        if on:
            self.triade = decors.Triade(pos,siz)
            self.addAnnotation(self.triade)
    

    def initCamera(self):
        self.makeCurrent()  # we need correct OpenGL context for camera
        self.camera = camera.Camera()
        if pf.options.testcamera:
            self.camera.modelview_callback = print_camera
            self.camera.projection_callback = print_camera

            
    def clear(self):
        """Clear the canvas to the background color."""
        self.settings.setMode()
        self.setDefaults()

    
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


    def drawit(self,a):
        """_Perform the drawing of a single item"""
        self.setDefaults()
        a.draw(self)


    def setDefaults(self):
        """Activate the canvas settings in the GL machine."""
        self.settings.activate()
        self.enable_lighting(self.settings.lighting)
        GL.glDepthFunc(GL.GL_LESS)


    def overrideMode(self,mode):
        """Override some settings"""
        settings = CanvasSettings.RenderProfiles[mode]
        CanvasSettings.glOverride(settings,self.settings)
        

    def glinit(self):
        """Initialize the rendering machine.

        The rendering machine is initialized according to self.settings:
        - self.rendermode: one of
        - self.lighting
        """
        self.setDefaults()
        self.setBackground(self.settings.bgcolor,self.settings.bgimage)
        self.clear()
        GL.glClearDepth(1.0)	       # Enables Clearing Of The Depth Buffer
        GL.glEnable(GL.GL_DEPTH_TEST)	       # Enables Depth Testing
        #GL.glEnable(GL.GL_CULL_FACE)

        if self.rendermode.endswith('wire'):
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(1.0,1.0) 
        else:
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            

    def glupdate(self):
        """Flush all OpenGL commands, making sure the display is updated."""
        GL.glFlush()
        

    def display(self):
        """(Re)display all the actors in the scene.

        This should e.g. be used when actors are added to the scene,
        or after changing  camera position/orientation or lens.
        """
        #pf.debugt("UPDATING CURRENT OPENGL CANVAS",pf.DEBUG.DRAW)
        self.makeCurrent()
        self.clear()
        
        # draw background decorations in 2D mode
        self.begin_2D_drawing()
        
        if self.background:
            #pf.debug("Displaying background",pf.DEBUG.DRAW)
            # If we have a shaded background, we need smooth/fill anyhow
            glSmooth()
            glFill()
            self.background.draw(mode='smooth')

        # background decorations
        back_decors = [ d for d in self.decorations if not d.ontop ]
        for actor in back_decors:
            self.setDefaults()
            ## if hasattr(actor,'zoom'):
            ##     self.zoom_2D(actor.zoom)
            actor.draw(canvas=self)
            ## if hasattr(actor,'zoom'):
            ##     self.zoom_2D()

        # draw the focus rectangle if more than one viewport
        if len(pf.GUI.viewports.all) > 1 and pf.cfg['gui/showfocus']:
            if self.hasFocus(): # QT focus
                self.draw_focus_rectangle(0,color=colors.blue)
            if self.focus:      # pyFormex DRAW focus
                self.draw_focus_rectangle(2,color=colors.red)
                
        self.end_2D_drawing()

        # start 3D drawing
        self.camera.set3DMatrices()
        
        # draw the highlighted actors
        if self.highlights:
            for actor in self.highlights:
                self.setDefaults()
                actor.draw(canvas=self)

        # draw the scene actors and annotations
        sorted_actors =  [ a for a in self.actors if not a.ontop ] + [ a for a in self.actors if a.ontop ] + self.annotations
        if self.settings.alphablend:
            opaque = [ a for a in sorted_actors if a.opak ]
            transp = [ a for a in sorted_actors if not a.opak ]
            
            for actor in opaque:
                self.setDefaults()
                actor.draw(canvas=self)
            GL.glEnable (GL.GL_BLEND)
            GL.glDepthMask (GL.GL_FALSE)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glBlendFunc (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            for actor in transp:
                self.setDefaults()
                actor.draw(canvas=self)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthMask (GL.GL_TRUE)
            GL.glDisable (GL.GL_BLEND)
        else:
            for actor in sorted_actors:
                self.setDefaults()
                actor.draw(canvas=self)

        ## # annotations are decorations drawn in 3D space
        ## for actor in self.annotations:
        ##     self.setDefaults()
        ##     actor.draw(canvas=self)


        # draw foreground decorations in 2D mode
        self.begin_2D_drawing()
        decors = [ d for d in self.decorations if d.ontop ]
        for actor in decors:
            self.setDefaults()
            ## if hasattr(actor,'zoom'):
            ##     self.zoom_2D(actor.zoom)
            actor.draw(canvas=self)
            ## if hasattr(actor,'zoom'):
            ##     self.zoom_2D()
        self.end_2D_drawing()

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
        #pf.debug("Start 2D drawing",pf.DEBUG.DRAW)
        if self.mode2D:
            #pf.debug("WARNING: ALREADY IN 2D MODE",pf.DEBUG.DRAW)
            return
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        self.zoom_2D()
        GL.glDisable(GL.GL_DEPTH_TEST)
        self.enable_lighting(False)
        self.mode2D = True

 
    def end_2D_drawing(self):
        """Cancel the 2D drawing mode initiated by begin_2D_drawing."""
        #pf.debug("End 2D drawing",pf.DEBUG.DRAW)
        if self.mode2D:
            GL.glEnable(GL.GL_DEPTH_TEST)    
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glPopMatrix()
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glPopMatrix()
            self.enable_lighting(self.settings.lighting)
            self.mode2D = False
       
        
    def setBbox(self,bbox=None):
        """Set the bounding box of the scene you want to be visible.

        bbox is a (2,3) shaped array specifying a bounding box.
        If no bbox is given, the bounding box of all the actors in the
        scene is used, or if the scene is empty, a default unit bounding box.
        """
        if bbox is None:
            if len(self.actors) > 0:
                bbox = self.actors[-1].bbox()
            else:
                bbox = [[-1.,-1.,-1.],[1.,1.,1.]]
        bbox = asarray(bbox)
        try:
            self.bbox = nan_to_num(bbox)
        except:
            pf.message("Invalid Bbox: %s" % bbox)


    def addActor(self,itemlist):
        """Add a 3D actor or a list thereof to the 3D scene."""
        self.actors.add(itemlist)

    def addHighlight(self,itemlist):
        """Add a highlight or a list thereof to the 3D scene."""
        self.highlights.add(itemlist)
         
    def addAnnotation(self,itemlist):
        """Add an annotation or a list thereof to the 3D scene."""
        self.annotations.add(itemlist)
         
    def addDecoration(self,itemlist):
        """Add a 2D decoration or a list thereof to the canvas."""
        self.decorations.add(itemlist)

    def addAny(self,itemlist=None):
        """Add any  item or list.

        This will add any actor/annotation/decoration item or a list
        of any such items  to the canvas. This is the prefered method to add
        an item to the canvas, because it makes sure that each item is added
        to the proper list. It can however not be used to add highlights.

        If you have a long list of a single type, it is more efficient to
        use one of the type specific add methods.
        """
        if type(itemlist) not in (tuple,list):
            itemlist = [ itemlist ]
        self.addActor([ i for i in itemlist if isinstance(i,actors.Actor)])
        self.addAnnotation([ i for i in itemlist if isinstance(i,marks.Mark)])
        self.addDecoration([ i for i in itemlist if isinstance(i,decors.Decoration)])


    def removeActor(self,itemlist=None):
        """Remove a 3D actor or a list thereof from the 3D scene.

        Without argument, removes all actors from the scene.
        This also resets the bounding box for the canvas autozoom.
        """
        if itemlist == None:
            itemlist = self.actors[:]
        self.actors.delete(itemlist)
        self.setBbox()
        
    def removeHighlight(self,itemlist=None):
        """Remove a highlight or a list thereof from the 3D scene.
        
        Without argument, removes all highlights from the scene.
        """
        if itemlist == None:
            itemlist = self.highlights[:]
        self.highlights.delete(itemlist)

    def removeAnnotation(self,itemlist=None):
        """Remove an annotation or a list thereof from the 3D scene.

        Without argument, removes all annotations from the scene.
        """
        if itemlist == None:
            itemlist = self.annotations[:]
        #
        # TODO: check whether the removal of the following code
        # does not have implications
        #
        ## if self.triade in itemlist:
        ##     pf.debug("REMOVING TRIADE",pf.DEBUG.DRAW)
        ##     self.triade = None
        self.annotations.delete(itemlist)

    def removeDecoration(self,itemlist=None):
        """Remove a 2D decoration or a list thereof from the canvas.

        Without argument, removes all decorations from the scene.
        """
        if itemlist == None:
            itemlist = self.decorations[:]
        self.decorations.delete(itemlist)


    def removeAny(self,itemlist=None):
        """Remove a list of any actor/highlights/annotation/decoration items.

        This will remove the items from any of the canvas lists in which the
        item appears.
        itemlist can also be a single item instead of a list.
        If None is specified, all items from all lists will be removed.
        """
        self.removeActor(itemlist)
        self.removeHighlight(itemlist)
        self.removeAnnotation(itemlist)
        self.removeDecoration(itemlist)
    

    def redrawAll(self):
        """Redraw all actors in the scene."""
        self.actors.redraw()
        self.highlights.redraw()
        self.annotations.redraw()
        self.decorations.redraw()
        self.display()

        
    def setCamera(self,bbox=None,angles=None):
        """Sets the camera looking under angles at bbox.

        This function sets the camera parameters to view the specified
        bbox volume from the specified viewing direction.

        Parameters:

        - `bbox`: the bbox of the volume looked at
        - `angles`: the camera angles specifying the viewing direction.
          It can also be a string, the key of one of the predefined
          camera directions

        If no angles are specified, the viewing direction remains constant.
        The scene center (camera focus point), camera distance, fovy and
        clipping planes are adjusted to make the whole bbox viewed from the
        specified direction fit into the screen.

        If no bbox is specified, the following remain constant:
        the center of the scene, the camera distance, the lens opening
        and aspect ratio, the clipping planes. In other words the camera
        is moving on a spherical surface and keeps focusing on the same
        point.

        If both are specified, then first the scene center is set,
        then the camera angles, and finally the camera distance.

        In the current implementation, the lens fovy and aspect are not
        changed by this function. Zoom adjusting is performed solely by
        changing the camera distance.
        """
        #
        # TODO: we should add the rectangle (digital) zooming to
        #       the docstring
        
        self.makeCurrent()
        
        # set scene center
        if bbox is not None:
            pf.debug("SETTING BBOX: %s" % self.bbox,pf.DEBUG.DRAW)
            self.setBbox(bbox)
            
            X0,X1 = self.bbox
            center = 0.5*(X0+X1)
            self.camera.setCenter(*center)

        # set camera angles
        if type(angles) is str:
            angles = self.view_angles.get(angles)
        if angles is not None:
            try:
                self.camera.setAngles(angles)
            except:
                raise ValueError,'Invalid view angles specified'

        # set camera distance and clipping planes
        if bbox is not None:
            # Currently, we keep the default fovy/aspect
            # and change the camera distance to focus
            fovy = self.camera.fovy
            #pf.debug("FOVY: %s" % fovy,pf.DEBUG.DRAW)
            self.camera.setLens(fovy,self.aspect)
            # Default correction is sqrt(3)
            correction = float(pf.cfg.get('gui/autozoomfactor',1.732))
            tf = coords.tand(fovy/2.)

            import simple
            bbix = simple.regularGrid(X0,X1,[1,1,1])
            bbix = dot(bbix,self.camera.rot[:3,:3])
            bbox = coords.Coords(bbix).bbox()
            dx,dy,dz = bbox[1] - bbox[0]
            vsize = max(dx/self.aspect,dy)
            dsize = bbox.dsize()
            offset = dz
            dist = (vsize/tf + offset) / correction

            if dist == nan or dist == inf:
                pf.debug("DIST: %s" % dist,pf.DEBUG.DRAW)
                return
            if dist <= 0.0:
                dist = 1.0
            self.camera.setDist(dist)

            ## print "vsize,dist = %s, %s" % (vsize,dist)
            ## near,far = 0.01*dist,100.*dist
            ## print "near,far = %s, %s" % (near,far)
            #near,far = dist-1.2*offset/correction,dist+1.2*offset/correction
            near,far = dist-1.0*dsize,dist+1.0*dsize
            # print "near,far = %s, %s" % (near,far)
            #print (0.0001*vsize,0.01*dist,near)
            # make sure near is positive
            near = max(near,0.0001*vsize,0.01*dist,finfo(coords.Float).tiny)
            # make sure far > near
            if far <= near:
                far += finfo(coords.Float).eps
            #print "near,far = %s, %s" % (near,far)
            self.camera.setClip(near,far)
            self.camera.resetArea()


    def project(self,x,y,z,locked=False):
        "Map the object coordinates (x,y,z) to window coordinates."""
        locked=False
        if locked:
            model,proj,view = self.projection_matrices
        else:
            self.makeCurrent()
            self.camera.loadProjection()
            model = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
            proj = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
            view = GL.glGetIntegerv(GL.GL_VIEWPORT)
        winx,winy,winz = GLU.gluProject(x,y,z,model,proj,view)
        return winx,winy,winz
        return self.camera.project(x,y,z)


    def unProject(self,x,y,z,locked=False):
        "Map the window coordinates (x,y,z) to object coordinates."""
        locked=False
        if locked:
            model,proj,view = self.projection_matrices
        else:
            self.makeCurrent()
            self.camera.loadProjection()
            model = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
            proj = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
            view = GL.glGetIntegerv(GL.GL_VIEWPORT)
        objx, objy, objz = GLU.gluUnProject(x,y,z,model,proj,view)
        return (objx,objy,objz)
        return self.camera.unProject(x,y,z)


    def zoom(self,f,dolly=True):
        """Dolly zooming.

        Zooms in with a factor `f` by moving the camera closer
        to the scene. This does noet change the camera's FOV setting.
        It will change the perspective view though.
        """
        if dolly:
            self.camera.dolly(f)


    def zoomRectangle(self,x0,y0,x1,y1):
        """Rectangle zooming.

        Zooms in/out by changing the area and position of the visible
        part of the lens.
        Unlike zoom(), this does not change the perspective view.

        `x0,y0,x1,y1` are pixel coordinates of the lower left and upper right
        corners of the area of the lens that will be mapped on the
        canvas viewport.
        Specifying values that lead to smaller width/height will zoom in.
        """
        w,h = float(self.width()),float(self.height())
        self.camera.setArea(x0/w,y0/h,x1/w,y1/h)


    def zoomCentered(self,w,h,x=None,y=None):
        """Rectangle zooming with specified center.

        This is like zoomRectangle, but the zoom rectangle is specified
        by its center and size, which may be more appropriate when using
        off-center zooming.
        """
        self.zoomRectangle(x-w/2,y-h/2,x+w/2,y+w/2)


    def zoomAll(self):
        """Rectangle zoom to make full scene visible.

        """
        self.camera.resetArea()


    def saveBuffer(self):
        """Save the current OpenGL buffer"""
        self.save_buffer = GL.glGetIntegerv(GL.GL_DRAW_BUFFER)

    def showBuffer(self):
        """Show the saved buffer"""
        pass

    def draw_focus_rectangle(self,ofs=0,color=colors.pyformex_pink):
        """Draw the focus rectangle.

        The specified width is HALF of the line width
        """
        w,h = self.width(),self.height()
        self._focus = decors.Grid(1+ofs,ofs,w-ofs,h-1-ofs,color=color,linewidth=1)
        self._focus.draw()

    def draw_cursor(self,x,y):
        """draw the cursor"""
        if self.cursor:
            self.removeDecoration(self.cursor)
        w,h = pf.cfg.get('draw/picksize',(20,20))
        col = pf.cfg.get('pick/color','yellow')
        self.cursor = decors.Grid(x-w/2,y-h/2,x+w/2,y+h/2,color=col,linewidth=1)
        self.addDecoration(self.cursor)

    def draw_rectangle(self,x,y):
        if self.cursor:
            self.removeDecoration(self.cursor)
        col = pf.cfg.get('pick/color','yellow')
        self.cursor = decors.Grid(self.statex,self.statey,x,y,color=col,linewidth=1)
        self.addDecoration(self.cursor)


### End
