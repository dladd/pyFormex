# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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

"""Interactive 2D drawing in a 3D space

This pyFormex plugin provides some interactive 2D drawing functions.
While the drawing operations themselves are in 2D, they can be performed
on a plane with any orientation in space. The constructed geometry always
has 3D coordinates in the global cartesian coordinate system.
"""

from simple import circle
from odict import ODict
from plugins.geomtools import triangleCircumCircle
from plugins.curve import *
from plugins.nurbs import *
from plugins.tools_menu import *
from plugins import objects

draw_mode_2d = ['point','polyline','curve','nurbs','circle']
autoname = ODict([ (obj,utils.NameSequence(obj)) for obj in draw_mode_2d ])

_preview = False


def set_preview(onoff=True):
    global _preview
    _preview = onoff
    

def toggle_preview(onoff=None):
    global _preview
    if onoff is None:
        try:
            onoff = pf.GUI.menu.item(_menu).item('preview').isChecked()
        except:
            onoff = not _preview
    _preview = onoff


def draw2D(mode='point',npoints=-1,zplane=0.,coords=None,func=None):
    """Enter interactive drawing mode and return the line drawing.

    See viewport.py for more details.
    This function differs in that it provides default displaying
    during the drawing operation and a button to stop the drawing operation.

    The drawing can be edited using the methods 'undo', 'clear' and 'close', which
    are presented in a combobox.
    """
    if pf.canvas.drawmode is not None:
        warning("You need to finish the previous drawing operation first!")
        return
    if func == None:
        func = highlightDrawing
    return pf.canvas.idraw(mode,npoints,zplane,func,coords,_preview)


obj_params = {}

def drawnObject(points,mode='point'):
    """Return the geometric object resulting from draw2D points"""
    minor = None
    if '_' in mode:
        mode,minor = mode.split('_')
    closed = minor=='closed'
    
    if mode == 'point':
        return points
    elif mode == 'polyline':
        return PolyLine(points,closed=closed)
    elif mode == 'curve' and points.ncoords() > 1:
        curl = obj_params.get('curl',None)
        closed = obj_params.get('closed',None)
        return BezierSpline(points,curl=curl,closed=closed)
    elif mode == 'nurbs':
        degree = obj_params.get('degree',None)
        if points.ncoords() <= degree:
            return None
        closed = obj_params.get('closed',None)
        return NurbsCurve(points,degree=degree,closed=closed)
    elif mode == 'circle' and points.ncoords() % 3 == 0:
        R,C,N = triangleCircumCircle(points.reshape(-1,3,3))
        circles = [circle(r=r,c=c,n=n) for r,c,n in zip(R,C,N)]
        if len(circles) == 1:
            return circles[0]
        else:
            return circles
    else:
        return None
    

def highlightDrawing(points,mode):
    """Highlight a temporary drawing on the canvas.

    pts is an array of points.
    """
    pf.canvas.removeHighlights()
    #print points[-1]
    PA = actors.GeomActor(Formex(points))
    PA.specular=0.0
    pf.canvas.addHighlight(PA)
    obj = drawnObject(points,mode=mode)
    if obj is not None:
        if mode == 'nurbs':
            OA = obj.actor(color=pf.canvas.settings.slcolor)
        else:
            if hasattr(obj,'toFormex'):
                F = obj.toFormex()
            else:
                F = Formex(obj)
            OA = actors.GeomActor(F,color=pf.canvas.settings.slcolor)
        OA.specular=0.0
        pf.canvas.addHighlight(OA)
    pf.canvas.update()

    
def drawPoints2D(mode,npoints=-1,zvalue=0.,coords=None):
    """Draw a 2D opbject in the xy-plane with given z-value"""
    if mode not in draw_mode_2d:
        return
    x,y,z = pf.canvas.project(0.,0.,zvalue)
    return draw2D(mode,npoints=npoints,zplane=z,coords=coords)

    
def drawObject2D(mode,npoints=-1,zvalue=0.,coords=None):
    """Draw a 2D opbject in the xy-plane with given z-value"""
    points = drawPoints2D(mode,npoints=npoints,zvalue=zvalue,coords=coords)
    return drawnObject(points,mode=mode)



def selectObject(mode=None):
    selection = objects.drawAble(like=mode+'-')
    res = widgets.Selection(
        selection.listAll(),
        'Known %ss' % mode,
        sort=True).getResult()
    # UNFINISHED

###################################

the_zvalue = 0.
    
def draw_object(mode,npoints=-1):
    print "z value = %s" % the_zvalue
    points = drawPoints2D(mode,npoints=-1,zvalue=the_zvalue)
    if points is None:
        return
    print "POINTS %s" % points
    obj = drawnObject(points,mode=mode)
    if obj is None:
        pf.canvas.removeHighlights()
        return
    print "OBJECT IS %s" % obj
    res = askItems([
        ('name',autoname[mode].peek(),{'text':'Name for storing the object'}),
        ('color','blue','color',{'text':'Color for the object'}),
        ])
    if not res:
        return
    
    name = res['name']
    color = res['color']
    if name == autoname[mode].peek():
        autoname[mode].next()
    export({name:obj})
    pf.canvas.removeHighlights()
    draw(points,color='black',flat=True)
    if mode != 'point':
        draw(obj,color=color,flat=True)
    if mode == 'nurbs':
        print "DRAWING KNOTS"
        draw(obj.knotPoints(),color=color,marksize=5)
    return name
    
            

def draw_points(npoints=-1):
    return draw_object('point',npoints=npoints)
def draw_polyline():
    return draw_object('polyline')
def draw_curve():
    global obj_params
    res = askItems([('curl',1./3.),('closed',False)])
    obj_params.update(res)
    return draw_object('curve')
def draw_nurbs():
    global obj_params
    res = askItems([('degree',3),('closed',False)])
    obj_params.update(res)
    return draw_object('nurbs')
def draw_circle():
    return draw_object('circle')


def objectName(actor):
    """Find the exported name corresponding to a canvas actor"""
    if hasattr(actor,'object'):
        obj = actor.object
        print "OBJECT",obj
        for name in pf.PF:
            print name
            print named(name)
            if named(name) is obj:
                return name
    return None
        

def splitPolyLine(c):
    """Interactively split the specified polyline"""
    pf.options.debug = 1
    XA = draw(c.coords,clear=False,bbox='last',flat=True)
    pf.canvas.pickable = [XA]
    #print "ACTORS",pf.canvas.actors
    #print "PICKABLE",pf.canvas.pickable
    k = pickPoints(filtr='single',oneshot=True)
    pf.canvas.pickable = None
    undraw(XA)
    if k.has_key(0):
        at = k[0]
        return c.split(at)
    else:
        return []
    
  
def split_curve():
    k = pickActors(filtr='single',oneshot=True)
    print k
    print k[-1]
    if not k.has_key(-1):
        return
    nr = k[-1][0]
    actor = pf.canvas.actors[nr]
    name = objectName(actor)
    print "Enter a point to split %s" % name
    c = named(name)
    cs = splitPolyLine(c)
    if len(cs) == 2:
        draw(cs[0],color='red')
        draw(cs[1],color='green')
    

_grid_data = [
    ['autosize',False],
    ['dx',1.,{'text':'Horizontal distance between grid lines'}], 
    ['dy',1.,{'text':'Vertical distance between grid lines'}], 
    ['width',100.,{'text':'Horizontal grid size'}],
    ['height',100.,{'text':'Vertical grid size'}],
    ['point',[0.,0.,0.],{'text':'Point in grid plane'}],
    ['normal',[0.,0.,1.],{'text':'Normal on the plane'}],
    ['lcolor','black','color',{'text':'Line color'}],
    ['lwidth',1.0,{'text':'Line width'}],
    ['showplane',False,{'text':'Show backplane'}],
    ['pcolor','white','color',{'text':'Backplane color'}],
    ['alpha','0.3',{'text':'Alpha transparency'}],
    ]

        
def create_grid():
    global dx,dy
    if hasattr(pf.canvas,'_grid'):
        if hasattr(pf.canvas,'_grid_data'):
            updateData(_grid_data,pf.canvas._grid_data)
    res = askItems(_grid_data)
    if res:
        pf.canvas._grid_data = res
        globals().update(res)
        
        nx = int(ceil(width/dx))
        ny = int(ceil(height/dy))
        obj = None
        if autosize:
            obj = drawable.check()
            if obj:
                bb = bbox(obj)
                nx = ny = 20
                dx = dy = bb.sizes().max() / nx * 2.
            
        ox = (-nx*dx/2.,-ny*dy/2.,0.)
        if obj:
            c = bbox(obj).center()
            ox = c + ox
            
        grid = actors.CoordPlaneActor(nx=(nx,ny,0),ox=ox,dx=(dx,dy,0.),linewidth=lwidth,linecolor=lcolor,planes=showplane,planecolor=pcolor,alpha=0.3)
        remove_grid()
        drawActor(grid)
        pf.canvas._grid = grid


def remove_grid():
    if hasattr(pf.canvas,'_grid'):
        undraw(pf.canvas._grid)
        pf.canvas._grid = None

    
def updateData(data,newdata):
    """Update the input data fields with new data values"""
    if newdata:
        for d in data:
            v = newdata.get(d[0],None)
            if v is not None:
                d[1] = v
    

# End
