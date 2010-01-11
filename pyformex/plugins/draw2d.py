#!/usr/bin/env pyformex

from simple import circle
from odict import ODict
from plugins.geomtools import triangleCircumCircle
from plugins.curve import *
from plugins.tools_menu import *


def draw2D(mode ='point',npoints=1,zplane=0.,func=None):
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
    # drawing_buttons = widgets.ButtonBox('Drawing:',[('Cancel',GD.canvas.cancel_drawing),('OK',GD.canvas.accept_drawing)])
    # GD.GUI.statusbar.addWidget(drawing_buttons)
    points = GD.canvas.draw(mode,npoints,zplane,func)
    print "POINTS: %s" % points
    # GD.GUI.statusbar.removeWidget(drawing_buttons)
    return points


def highlightDrawing(points):
    """Highlight a temporary drawing on the canvas.

    pts is an array of points.
    """
    GD.canvas.removeHighlights()
    mode = GD.canvas.drawmode
    draw(points,highlight=True,flat=True)
    if mode == 'polyline':
        draw(PolyLine(points),color=GD.canvas.settings.slcolor,highlight=True,flat=True)
    elif mode == 'spline':
        if points.ncoords() > 1:
            draw(BezierSpline(points),color=GD.canvas.settings.slcolor,highlight=True,flat=True)
    elif mode == 'circle':
        print "%s POINTS" % points.ncoords()
        if points.ncoords() % 3 == 0:
            points = points.reshape(-1,3,3)
            R,C,N = triangleCircumCircle(points)
            draw([circle(r=r,c=c,n=n) for r,c,n in zip(R,C,N)],color=GD.canvas.settings.slcolor,highlight=True,flat=True)


def finalDrawing(points,mode,color):
    """Final drawing.

    pts is an array of points.
    """
    GD.canvas.removeHighlights()
    draw(points,flat=True)
    if mode == 'polyline':
        draw(PolyLine(points),mode='flat',linewidth=4,color=color,flat=True)
    elif mode == 'spline':
        if points.ncoords() > 1:
            draw(BezierSpline(points),mode='flat',linewidth=4,color=color,flat=True)
    elif mode == 'circle':
        if points.ncoords() % 3 == 0:
            points = points.reshape(-1,3,3)
            R,C,N = triangleCircumCircle(points)
            draw([circle(r=r,c=c,n=n) for r,c,n in zip(R,C,N)],mode='flat',linewidth=4,color=color,flat=True)
   


###################################

_draw_object = ['point','polyline','spline','circle']
_auto_name = ODict([ (obj,utils.NameSequence(obj)) for obj in _draw_object ])
_zvalue = 0.
    
def draw_object(mode,npoints=-1):
    if mode not in _draw_object:
        return
    x,y,z = GD.canvas.project(0.,0.,_zvalue)
    points = draw2D(mode,npoints=npoints,zplane=z)
    if len(points) == 0:
        return
    points = Coords(points)
    res = askItems([
        ('name',_auto_name[mode].peek(),{'text':'Name for storing the object'}),
        ('color','blue','color',{'text':'Color for the object'}),
        ])
    if res:
        name = res['name']
        color = res['color']
        points.specular = 0.
        if name == _auto_name[mode].peek():
            _auto_name[mode].next()
        export({name:points})
        finalDrawing(points,mode,color=color)  

def draw_points(npoints=-1):
    draw_object('point',npoints=npoints)
def draw_polyline():
    draw_object('polyline')
def draw_spline():
    draw_object('spline')
def draw_circle():
    draw_object('circle')


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
        ("&Draw Points",draw_points),
        ("&Draw Polyline",draw_polyline),
        ("&Draw Spline",draw_spline),
        ("&Draw Circle",draw_circle),
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
    show_menu()
    setDrawOptions({'bbox':'last'})

####################################################################

if __name__ == "draw":
    # If executed as a pyformex script
    reload_menu()

# End
