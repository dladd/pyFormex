#!/usr/bin/env pyformex
# $Id$

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
from plugins.tools_menu import *

draw_mode_2d = ['point','polyline','spline','circle']
autoname = ODict([ (obj,utils.NameSequence(obj)) for obj in draw_mode_2d ])

_preview = False

def toggle_preview():
    global _preview
    res = askItems([('_preview',_preview,{'text':'Preview mode'})])
    if res:
        globals().update(res)


def draw2D(mode='point',npoints=1,zplane=0.,coords=None,func=None):
    """Enter interactive drawing mode and return the line drawing.

    See viewport.py for more details.
    This function differs in that it provides default displaying
    during the drawing operation and a button to stop the drawing operation.

    The drawing can be edited using the methods 'undo', 'clear' and 'close', which
    are presented in a combobox.
    """
    if GD.canvas.drawing_mode is not None:
        warning("You need to finish the previous drawing operation first!")
        return
    if func == None:
        func = highlightDrawing
    print "INITIAL = %s" % coords
    return GD.canvas.idraw(mode,npoints,zplane,func,coords,_preview)


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
    elif mode == 'spline' and points.ncoords() > 1:
        return BezierSpline(points,closed=closed)
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
    GD.canvas.removeHighlights()
    GD.canvas.update()
    draw(points,highlight=True,flat=True)
    objects = drawnObject(points,mode=mode)
    if objects is not None:
        draw(objects,color=GD.canvas.settings.slcolor,highlight=True,flat=True)

    
def drawPoints2D(mode,npoints=-1,zvalue=0.,coords=None):
    """Draw a 2D opbject in the xy-plane with given z-value"""
    if mode not in draw_mode_2d:
        return
    x,y,z = GD.canvas.project(0.,0.,zvalue)
    points = draw2D(mode,npoints=npoints,zplane=z,coords=coords)
    return points

    
def drawObject2D(mode,npoints=-1,zvalue=0.,coords=None):
    """Draw a 2D opbject in the xy-plane with given z-value"""
    points = drawPoints2D(mode,npoints=-1,zvalue=zvalue,coords=coords)
    return drawnObject(points,mode=mode)


###################################

_zvalue = 0.
    
def draw_object(mode,npoints=-1):
    points = drawPoints2D(mode,npoints=-1,zvalue=zvalue)
    obj = drawnObject(points,mode=mode)
    if obj is None:
        GD.canvas.removeHighlights()
        return
    res = askItems([
        ('name',autoname[mode].peek(),{'text':'Name for storing the object'}),
        ('color','blue','color',{'text':'Color for the object'}),
        ])
    if not res:
        return None
    
    name = res['name']
    color = res['color']
    if name == autoname[mode].peek():
        autoname[mode].next()
    export({name:obj})
    GD.canvas.removeHighlights()
    draw(points,color='black',flat=True)
    if mode != 'point':
        draw(obj,color=color,flat=True)
    return name
    
            

def draw_points(npoints=-1):
    return draw_object('point',npoints=npoints)
def draw_polyline():
    return draw_object('polyline')
def draw_spline():
    return draw_object('spline')
def draw_circle():
    return draw_object('circle')



def objectName(actor):
    """Find the exported name corresponding to a canvas actor"""
    if hasattr(actor,'object'):
        obj = actor.object
        print "OBJECT",obj
        for name in GD.PF:
            print name
            print named(name)
            if named(name) is obj:
                return name
    return None
        


def splitPolyLine(c):
    """Interactively split the specified polyline"""
    GD.options.debug=True
    XA = draw(c.coords,clear=False,bbox='last',flat=True)
    GD.canvas.pickable = [XA]
    #print "ACTORS",GD.canvas.actors
    #print "PICKABLE",GD.canvas.pickable
    k = pickPoints(filtr='single',oneshot=True)
    GD.canvas.pickable = None
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
    actor = GD.canvas.actors[nr]
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
    if hasattr(GD.canvas,'_grid'):
        if hasattr(GD.canvas,'_grid_data'):
            updateData(_grid_data,GD.canvas._grid_data)
    res = askItems(_grid_data)
    if res:
        GD.canvas._grid_data = res
        globals().update(res)
        
        obj = drawable.check()
        if autosize and obj:
            bb = bbox(obj)
            nx = ny = 20
            dx = dy = bb.sizes().max() / nx * 2.
        else:
            nx = int(ceil(width/dx))
            ny = int(ceil(height/dy))
            
        ox = (-nx*dx/2.,-ny*dy/2.,0.)
        if obj:
            c = bbox(obj).center()
            ox = c + ox
            
        grid = actors.CoordPlaneActor(nx=(nx,ny,0),ox=ox,dx=(dx,dy,0.),linewidth=lwidth,linecolor=lcolor,planes=showplane,planecolor=pcolor,alpha=0.3)
        remove_grid()
        drawActor(grid)
        GD.canvas._grid = grid


def remove_grid():
    if hasattr(GD.canvas,'_grid'):
        undraw(GD.canvas._grid)
        GD.canvas._grid = None

    
def updateData(data,newdata):
    """Update the input data fields with new data values"""
    if newdata:
        for d in data:
            v = newdata.get(d[0],None)
            if v is not None:
                d[1] = v
    

    
################################## Menu #############################

_menu = 'Draw'

def create_menu():
    """Create the menu."""
    MenuData = [
        ("&Set grid",create_grid),
        ("&Remove grid",remove_grid),
        ("---",None),
        ("&Toggle Preview",toggle_preview),
        ("---",None),
        ("&Draw Points",draw_points),
        ("&Draw Polyline",draw_polyline),
        ("&Draw Spline",draw_spline),
        ("&Draw Circle",draw_circle),
        ("---",None),
        ("&Split Curve",split_curve),
        ("---",None),
        ("&Reload Menu",reload_menu),
        ("&Close Menu",close_menu),
        ]
    w = widgets.Menu(_menu,items=MenuData,parent=GD.GUI.menu,before='help',tearoff=True)
    return w

def show_menu():
    """Show the menu."""
    if not GD.GUI.menu.item(_menu):
        create_menu()

def close_menu():
    """Close the menu."""
    m = GD.GUI.menu.item(_menu)
    if m :
        m.remove()


def reload_menu():
    """Reload the menu."""
    close_menu()
    import plugins
    plugins.refresh('draw2d')
    show_menu()
    setDrawOptions({'bbox':'last'})

####################################################################

if __name__ == "draw":
    # If executed as a pyformex script
    reload_menu()

# End