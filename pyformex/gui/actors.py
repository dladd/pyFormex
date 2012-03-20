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
"""OpenGL actors for populating the 3D scene."""

import pyformex as pf
import sys
from OpenGL import GL,GLU

from drawable import *
from formex import *
from elements import elementType,elementName
from mesh import Mesh

from plugins.trisurface import TriSurface
from plugins.nurbs import NurbsCurve,NurbsSurface
from marks import TextMark

import timer

### Actors ###############################################

class Actor(Drawable):
    """An Actor is anything that can be drawn in an OpenGL 3D Scene.

    The visualisation of the Scene Actors is dependent on camera position and
    angles, clipping planes, rendering mode and lighting.
    
    An Actor subclass should minimally reimplement the following methods:
    
    - `bbox()`: return the actors bounding box.
    - `drawGL(mode)`: to draw the actor. Takes a mode argument so the
      drawing function can act differently depending on the mode. There are
      currently 5 modes: wireframe, flat, smooth, flatwire, smoothwire.
      drawGL should only contain OpenGL calls that are allowed inside a
      display list. This may include calling the display list of another
      actor but *not* creating a new display list.

    The interactive picking functionality requires the following methods,
    for which we porvide do-nothing defaults here:
    
    - `npoints()`:
    - `nelems()`:
    - `pickGL()`:
    """
    
    def __init__(self,**kargs):
        Drawable.__init__(self,**kargs)

    def bbox(self):
        """Default implementation for bbox()."""
        try:
            return self.coords.bbox()
        except:
            raise ValueError,"No bbox() defined and no coords attribute"
            
    def npoints(self):
        return 0
    def nelems(self):
        return 0
    def pickGL(self,mode):
        pass


class TranslatedActor(Actor):
    """An Actor translated to another position."""

    def __init__(self,A,trl=(0.,0.,0.),**kargs):
        Actor.__init__(self,**kargs)
        self.actor = A
        self.trans = A.trans
        self.trl = asarray(trl)

    def bbox(self):
        return self.actor.bbox() + self.trl

    def redraw(self,**kargs):
        self.actor.redraw(**kargs)
        Drawable.redraw(self,**kargs)

    def drawGL(self,**kargs):
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glTranslate(*self.trl)
        self.actor.use_list()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()


class RotatedActor(Actor):
    """An Actor rotated to another position."""

    def __init__(self,A,rot=(1.,0.,0.),twist=0.0,**kargs):
        """Created a new rotated actor.

        If rot is an array with shape (3,), the rotation is specified
        by the direction of the local 0 axis of the actor.
        If rot is an array with shape (4,4), the rotation is specified
        by the direction of the local 0, 1 and 2 axes of the actor.
        """
        Actor.__init__(self,**kargs)
        self.actor = A
        self.trans = A.trans
        if shape(rot) == (3,):
            self.rot = rotMatrix(rot,n=4)
        else:
            self.rot = rot

    def bbox(self):
        return self.actor.bbox() # TODO : rotate the bbox !

    def redraw(self,**kargs):
        self.actor.redraw(**kargs)
        Drawable.redraw(self,**kargs)

    def drawGL(self,**kargs):
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glMultMatrixf(self.rot)
        self.actor.use_list()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()


class CubeActor(Actor):
    """An OpenGL actor with cubic shape and 6 colored sides."""

    def __init__(self,size=1.0,color=[red,cyan,green,magenta,blue,yellow],**kargs):
        Actor.__init__(self,**kargs)
        self.size = size
        self.color = color

    def bbox(self):
        return (0.5 * self.size) * array([[-1.,-1.,-1.],[1.,1.,1.]])

    def drawGL(self,**kargs):
        """Draw the cube."""
        drawCube(self.size,self.color)


class SphereActor(Actor):
    """An OpenGL actor representing a sphere."""

    def __init__(self,size=1.0,color=None,**kargs):
        Actor.__init__(self)
        self.size = size
        self.color = color

    def bbox(self):
        return (0.5 * self.size) * array([[-1.,-1.,-1.],[1.,1.,1.]])

    def drawGL(self,**kargs):
        """Draw the cube."""
        drawSphere(self.size,self.color)


# This could be subclassed from GridActor
class BboxActor(Actor):
    """Draws a bbox."""

    def __init__(self,bbox,color=None,linewidth=None,**kargs):
        Actor.__init__(self,**kargs)
        self.color = color
        self.linewidth = linewidth
        self.bb = bbox
        Hex8 = elementType('hex8')
        self.vertices = Hex8.vertices * (bbox[1]-bbox[0]) + bbox[0]
        self.edges = Hex8.edges
        self.facets = Hex8.faces

    def bbox(self):
        return self.bb

    def drawGL(self,**kargs):
        """Always draws a wireframe model of the bbox."""
        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)
        drawLines(self.vertices,self.edges,self.color)

 
class AxesActor(Actor):
    """An actor showing the three axes of a CoordinateSystem.

    If no CoordinateSystem is specified, the global coordinate system is drawn.
    
    The default actor consists of three colored lines of unit length along
    the unit vectors of the axes and three colored triangles representing the
    coordinate planes. This can be modified by the following parameters:

    size: scale factor for the unit vectors.
    color: a set of three colors to use for x,y,z axes.
    colored_axes = False: draw black axes.
    draw_planes = False: do not draw the coordinate planes.
    """

    def __init__(self,cs=None,size=1.0,color=[red,green,blue],colored_axes=True,draw_planes=False,linewidth=None,**kargs):
        Actor.__init__(self,**kargs)
        if cs is None:
            cs = CoordinateSystem()
        self.cs = cs
        self.color = saneColorArray(saneColor(color),(3,1))
        self.colored_axes = colored_axes
        self.draw_planes = draw_planes
        self.linewidth = linewidth
        self.setSize(size)

    def bbox(self):
        origin = self.cs[3]
        return array([origin-self.size,origin+self.size])

    def setSize(self,size):
        size = float(size)
        if size > 0.0:
            self.size = size
        self.delete_list()

    def drawGL(self,**kargs):
        """Draw the axes."""
        x = self.cs.trl(-self.cs[3]).scale(self.size).trl(self.cs[3])
        if self.draw_planes:
            e = array([[3,1,2],[3,2,0],[3,0,1]])
            drawPolygons(x,e,'flat')
        
        e = array([[3,0],[3,1],[3,2]])
        if self.colored_axes:
            c = self.color
        else:
            c = None
        if self.linewidth:
            GL.glLineWidth(self.linewidth)
        drawLines(x,e,c)

  
class GridActor(Actor):
    """Draws a (set of) grid(s) in one of the coordinate planes."""

    def __init__(self,nx=(1,1,1),ox=(0.0,0.0,0.0),dx=(1.0,1.0,1.0),linecolor=black,linewidth=None,planecolor=white,alpha=0.2,lines=True,planes=True,**kargs):
        Actor.__init__(self,**kargs)
        self.linecolor = saneColor(linecolor)
        self.planecolor = saneColor(planecolor)
        self.linewidth = linewidth
        self.alpha = alpha
        self.trans = True
        self.lines = lines
        self.planes = planes
        self.nx = asarray(nx)
        self.x0 = asarray(ox)
        self.x1 = self.x0 + self.nx * asarray(dx)

    def bbox(self):
        return array([self.x0,self.x1])

    def drawGL(self,**kargs):
        """Draw the grid."""

        if self.lines:
            if self.linewidth:
                GL.glLineWidth(self.linewidth)
            glColor(self.linecolor)
            drawGridLines(self.x0,self.x1,self.nx)
        
        if self.planes:
            glColor(self.planecolor,self.alpha)
            drawGridPlanes(self.x0,self.x1,self.nx)


class CoordPlaneActor(Actor):
    """Draws a set of 3 coordinate planes."""

    def __init__(self,nx=(1,1,1),ox=(0.0,0.0,0.0),dx=(1.0,1.0,1.0),linecolor=black,linewidth=None,planecolor=white,alpha=0.5,lines=True,planes=True,**kargs):
        Actor.__init__(self,**kargs)
        self.linecolor = saneColor(linecolor)
        self.planecolor = saneColor(planecolor)
        self.linewidth = linewidth
        self.alpha = alpha
        self.trans = True
        self.lines = lines
        self.planes = planes
        self.nx = asarray(nx)
        self.x0 = asarray(ox)
        self.x1 = self.x0 + self.nx * asarray(dx)

    def bbox(self):
        return array([self.x0,self.x1])

    def drawGL(self,**kargs):
        """Draw the grid."""

        for i in range(3):
            nx = self.nx.copy()
            nx[i] = 0
            
            if self.lines:
                if self.linewidth:
                    GL.glLineWidth(self.linewidth)
                glColor(self.linecolor)
                drawGridLines(self.x0,self.x1,nx)

            if self.planes:
                glColor(self.planecolor,self.alpha)
                drawGridPlanes(self.x0,self.x1,nx)


class PlaneActor(Actor):
    """A plane in a 3D scene."""

    def __init__(self,nx=(2,2,2),ox=(0.,0.,0.),size=((0.0,1.0,1.0),(0.0,1.0,1.0)),linecolor=black,linewidth=None,planecolor=white,alpha=0.5,lines=True,planes=True,**kargs):
        """A plane perpendicular to the x-axis at the origin."""
        Actor.__init__(self,**kargs)
        self.linecolor = saneColor(linecolor)
        self.planecolor = saneColor(planecolor)
        self.linewidth = linewidth
        self.alpha = alpha
        self.trans = True
        self.lines = lines
        self.planes = planes
        self.nx = asarray(nx)
        ox = asarray(ox)
        sz = asarray(size)
        self.x0,self.x1 = ox-sz[0], ox+sz[1]
        

    def bbox(self):
        return array([self.x0,self.x1])

    def drawGL(self,**kargs):
        """Draw the grid."""

        for i in range(3):
            nx = self.nx.copy()
            nx[i] = 0
            
            if self.lines:
                if self.linewidth is not None:
                    GL.glLineWidth(self.linewidth)
                color = self.linecolor
                if color is None:
                    color = canvas.settings.fgcolor
                glColor(color)
                drawGridLines(self.x0,self.x1,nx)
            
            if self.planes:
                glColor(self.planecolor,self.alpha)
                drawGridPlanes(self.x0,self.x1,nx)


###########################################################################


class GeomActor(Actor):
    """An OpenGL actor representing a geometrical model.

    The model can either be in Formex or Mesh format.
    """
    mark = False

    def __init__(self,data,elems=None,eltype=None,mode=None,color=None,colormap=None,bkcolor=None,bkcolormap=None,alpha=1.0,bkalpha=1.0,linewidth=None,linestipple=None,marksize=None,texture=None,**kargs):
        """Create a geometry actor.

        The geometry is either in Formex model: a coordinate block with
        shape (nelems,nplex,3), or in Mesh format: a coordinate block
        with shape (npoints,3) and an elems block with shape (nelems,nplex).

        In both cases, an eltype may be specified if the default is not
        suitable. Default eltypes are Point for plexitude 1, Line for
        plexitude 2 and Triangle for plexitude 3 and Polygon for all higher
        plexitudes. Actually, Triangle is just a special case of Polygon.

        Here is a list of possible eltype values (which should match the
        corresponding plexitude):

        =========   ===========   ============================================
        plexitude   `eltype`      element type
        =========   ===========   ============================================
        4           ``tet4``      a tetrahedron
        6           ``wedge6``    a wedge (triangular prism)
        8           ``hex8``      a hexahedron
        =========   ===========   ============================================
        
        The colors argument specifies a list of OpenGL colors for each
        of the property values in the Formex. If the list has less
        values than the PropSet, it is wrapped around. It can also be
        a single OpenGL color, which will be used for all elements.
        For surface type elements, a bkcolor color can be given for
        the backside of the surface. Default will be the same
        as the front color.
        The user can specify a linewidth to be used when drawing
        in wireframe mode.
        """
        Actor.__init__(self,**kargs)

        # Store a reference to the drawn object
        self.object = data
        
        if isinstance(data,GeomActor) or isinstance(data,Mesh):
            self.coords = data.coords
            self.elems = data.elems
            self.eltype = data.eltype
            
        elif isinstance(data,Formex):
            self.coords = data.coords
            self.elems = None
            self.eltype = data.eltype

        else:
            self.coords = data
            self.elems = elems
            self.eltype = eltype

        self.mode = mode
        self.setColor(color,colormap)
        self.setBkColor(bkcolor,bkcolormap)
        self.setAlpha(alpha,bkalpha)
        self.setLineWidth(linewidth)
        self.setLineStipple(linestipple)
        self.marksize = marksize
        self.setTexture(texture)


    def getType(self):
        return self.object.__class__

    def nplex(self):
        return self.shape()[1]

    def nelems(self):
        return self.shape()[0]

    def shape(self):
        if self.elems is None:
            return self.coords.shape[:-1]
        else:
            return self.elems.shape

    def npoints(self):
        return self.vertices().shape[0]

    def nedges(self):
        # This is needed to be able to pick edges!!
        try:
            return self.object.nedges()
        except:
            try:
                return self.object.getEdges().shape[0]
            except:
                return 0

    def vertices(self):
        """Return the vertives as a 2-dim array."""
        return self.coords.reshape(-1,3)

 
    def setColor(self,color,colormap=None):
        """Set the color of the Actor."""
        self.color,self.colormap = saneColorSet(color,colormap,self.shape())


    def setBkColor(self,color,colormap=None):
        """Set the backside color of the Actor."""
        self.bkcolor,self.bkcolormap = saneColorSet(color,colormap,self.shape())


    def setAlpha(self,alpha,bkalpha):
        """Set the Actors alpha value."""
        self.alpha = float(alpha)
        self.bkalpha = float(bkalpha)
        # TODO: not sure about the following
        self.trans = self.alpha < 1.0 # or self.bkalpha < 1.0 
            

    def bbox(self):
        return self.coords.bbox()


    def draw(self,**kargs):
        
        mode = self.mode
        if mode is None:
            if 'mode' in kargs:
                mode = kargs['mode']
            else:
                canvas = kargs.get('canvas',pf.canvas)
                mode = canvas.rendermode

        if mode.endswith('wire'):
            if not hasattr(self,'wire'):
                import copy
                wire = copy.copy(self)
                wire.mode = 'wireframe' # Lock the mode
                wire.nolight = True
                wire.ontop = False # True will make objects transparent for edges
                wire.list = None
                Drawable.prepare_list(wire,color=asarray(black))
                self.wire = wire

            # Add the existing wire to the extra list, and then draw w/o wire
            if self.wire not in self.extra:
                self.extra.append(self.wire)
                # AVOID RECURSION
                self.wire.extra = []
                
            mode = mode[:-4]

        else:
            if hasattr(self,'wire') and self.wire in self.extra:
                self.extra.remove(self.wire)

        if self.list is None or mode != self.listmode:
            kargs['mode'] = mode
            self.delete_list()
            self.list = self.create_list(**kargs)
            self.listmode = mode

        self.use_list()


    def drawGL(self,canvas=None,mode=None,color=None,**kargs):
        """Draw the geometry on the specified canvas.

        The drawing parameters not provided by the Actor itself, are
        derived from the canvas defaults.

        mode and color can be overridden for the sole purpose of allowing
        the recursive use for modes ending on 'wire' ('smoothwire' or
        'flatwire'). In these cases, two drawing operations are done:
        one with mode='wireframe' and color=black, and one with mode=mode[:-4].
        """
        from canvas import glLineStipple
        if canvas is None:
            canvas = pf.canvas
        if mode is None:
           mode = self.mode
        if mode is None:
            mode = canvas.rendermode

        if mode.endswith('wire'):
            mode = mode[:-4]
                            
        ############# set drawing attributes #########
        alpha = self.alpha
        if alpha is None:
            alpha = canvas.settings.alpha
        bkalpha = self.bkalpha
        if bkalpha is None:
            bkalpha = canvas.settings.alpha
        
        if color is None:
            color,colormap = self.color,self.colormap
            bkcolor, bkcolormap = self.bkcolor,self.bkcolormap
        else:
            # THIS OPTION IS ONLY MEANT FOR OVERRIDING THE COLOR
            # WITH THE EDGECOLOR IN ..wire DRAWING MODES
            # SO NO NEED TO SET bkcolor
            color,colormap = saneColor(color),None
            bkcolor, bkcolormap = None,None

        # convert color index to full colors
        if color is not None and color.dtype.kind == 'i':
            color = colormap[color]

        if bkcolor is not None and bkcolor.dtype.kind == 'i':
            bkcolor = bkcolormap[bkcolor]
        
        linewidth = self.linewidth
        if linewidth is None:
            linewidth = canvas.settings.linewidth

        if self.linewidth is not None:
            GL.glLineWidth(self.linewidth)

        if self.linestipple is not None:
            glLineStipple(*self.linestipple)

        if mode.startswith('smooth'):
            if hasattr(self,'specular'):
                fill_mode = GL.GL_FRONT
                import colors
                if color is not None:
                    spec = color * self.specular# *  pf.canvas.specular
                    spec = append(spec,1.)
                else:
                    spec = colors.GREY(self.specular)# *  pf.canvas.specular
                GL.glMaterialfv(fill_mode,GL.GL_SPECULAR,spec)
                GL.glMaterialfv(fill_mode,GL.GL_EMISSION,spec)
                GL.glMaterialfv(fill_mode,GL.GL_SHININESS,self.specular)

        ################## draw the geometry #################
        nplex = self.nplex()
        
        if nplex == 1:
            marksize = self.marksize
            if marksize is None:
                marksize = canvas.settings.pointsize
            # THIS SHOULD GO INTO drawPoints
            if self.elems is None:
                coords = self.coords
            else:
                coords = self.coords[self.elems]
            drawPoints(coords,color,alpha,marksize)

        elif nplex == 2:
            drawLines(self.coords,self.elems,color)
            
        # beware: some Formex eltypes are strings and may not
        # represent a valid Mesh elementType
        # THis is only here for Formex type.
        # We can probably remove it if we avoid eltype 'curve'
        elif nplex == 3 and self.eltype in ['curve','line3']:
            drawQuadraticCurves(self.coords,self.elems,color)

        elif self.eltype is None:
            # polygons
            if mode=='wireframe' :
                drawPolyLines(self.coords,self.elems,color)
            else:
                if bkcolor is not None:
                    GL.glEnable(GL.GL_CULL_FACE)
                    GL.glCullFace(GL.GL_BACK)
                    
                drawPolygons(self.coords,self.elems,mode,color,alpha,self.texture,None)
                if bkcolor is not None:
                    GL.glCullFace(GL.GL_FRONT)
                    drawPolygons(self.coords,self.elems,mode,bkcolor,alpha)
                    GL.glDisable(GL.GL_CULL_FACE)
                    
        else:
            el = elementType(self.eltype)
            
            if mode=='wireframe' or el.ndim < 2:
                for edges in el.getDrawEdges(el.name() in pf.cfg['draw/quadline']):
                    drawEdges(self.coords,self.elems,edges,edges.eltype,color)    
            else:
                for faces in el.getDrawFaces(el.name() in pf.cfg['draw/quadsurf']):
                    if bkcolor is not None:
                        # Enable drawing front and back with different colors
                        GL.glEnable(GL.GL_CULL_FACE)
                        GL.glCullFace(GL.GL_BACK)

                    drawFaces(self.coords,self.elems,faces,faces.eltype,mode,color,alpha,self.texture)

                    if bkcolor is not None:
                        # Draw the back sides
                        GL.glCullFace(GL.GL_FRONT)
                        drawFaces(self.coords,self.elems,faces,faces.eltype,mode,bkcolor,alpha)
                        GL.glDisable(GL.GL_CULL_FACE)

   

    def pickGL(self,mode):
        """ Allow picking of parts of the actor.

        mode can be 'element', 'edge' or 'point'
        """
        if mode == 'element':
            pickPolygons(self.coords,self.elems)

        elif mode == 'edge':
            edges = self.object.getEdges()
            if edges is not None:
                pickPolygons(self.coords,edges)
                
        elif mode == 'point':
            pickPoints(self.coords)


    def select(self,sel):
        """Return a GeomActor with a selection of this actor's elements

        Currently, the resulting Actor will not inherit the properties
        of its parent, but the eltype will be retained.
        """
        # This selection should be reworked to allow edge and point selections
        if self.elems is None:
            x = self.coords[sel]
            e = self.elems
        else:
            x = self.coords
            e = self.elems[sel]
        return GeomActor(x,e,eltype=self.eltype)


class NurbsActor(Actor):

    def __init__(self,data,color=None,colormap=None,bkcolor=None,bkcolormap=None,**kargs):
        from gui.drawable import saneColor
        Actor.__init__(self,**kargs)
        self.object = data
        self.setColor(color,colormap)
        self.setBkColor(bkcolor,bkcolormap)
        if isinstance(self.object,NurbsCurve):
            self.samplingTolerance = 5.0
        elif isinstance(self.object,NurbsSurface):
            self.samplingTolerance = 10.0
        self.list = None


    def shape(self):
        return self.object.coords.shape[:-1]

 
    def setColor(self,color,colormap=None):
        """Set the color of the Actor."""
        self.color,self.colormap = saneColorSet(color,colormap,self.shape())


    def setBkColor(self,color,colormap=None):
        """Set the backside color of the Actor."""
        self.bkcolor,self.bkcolormap = saneColorSet(color,colormap,self.shape())

        
    def bbox(self):
        return self.object.bbox()

        
    def drawGL(self,canvas=None,**kargs):
        if canvas is None:
            canvas = pf.canvas

        mode = canvas.rendermode

        if mode.endswith('wire'):
            mode = mode[:-4]
            
        if isinstance(self.object,NurbsCurve):
            drawNurbsCurves(self.object.coords,self.object.knots,color=self.color,samplingTolerance=self.samplingTolerance)
        elif isinstance(self.object,NurbsSurface):
            if mode == 'wireframe':
                pass
            else:
                drawNurbsSurfaces(self.object.coords,self.object.vknots,self.object.uknots,color=self.color,normals='auto',samplingTolerance=self.samplingTolerance)


# End
