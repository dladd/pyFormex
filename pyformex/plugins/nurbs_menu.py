# $Id$

"""Nurbs menu

"""


from draw2d import *

from plugins.objects import *
from plugins import tools_menu
from plugins.geometry_menu import autoname 

# We subclass the DrawableObject to change its toggleAnnotation method
class NurbsObjects(DrawableObjects):
    def __init__(self):
        DrawableObjects.__init__(self,clas=NurbsCurve)

    def toggleAnnotation(self,i=0,onoff=None):
        """Toggle mesh annotations on/off.

        This functions is like DrawableObjects.toggleAnnotation but also
        updates the geometry_menu when changes are made.
        """
        DrawableObjects.toggleAnnotation(self,i,onoff)
        geometry_menu = pf.GUI.menu.item(_menu)
        toggle_menu = geometry_menu.item("toggle annotations")
        # This relies on the menu having the same items as the annotation list
        action = toggle_menu.actions()[i]
        action.setChecked(selection.hasAnnotation(i))

selection = NurbsObjects()

selection_PL = objects.DrawableObjects(clas=PolyLine)
selection_BS = objects.DrawableObjects(clas=BezierSpline)

ntoggles = len(selection.annotations)
def toggleEdgeNumbers():
    selection.toggleAnnotation(0+ntoggles)
def toggleNodeNumbers():
    print "SURFACE_MENU: %s" % selection
    selection.toggleAnnotation(1+ntoggles)
def toggleNormals():
    selection.toggleAnnotation(2+ntoggles)
def toggleAvgNormals():
    selection.toggleAnnotation(3+ntoggles)


## selection.annotations.extend([[draw_edge_numbers,False],
##                               [draw_node_numbers,False],
##                               [draw_normals,False],
##                               [draw_avg_normals,False],
##                               ])

class _options:
    color = 'blue';
    ctrl = True;
    ctrl_numbers = False;
    knots = True;
    knotsize = 6;
    knot_numbers = False;
    knot_values = False;


def drawNurbs(N):
    draw(N,color=_options.color,nolight=True)
    if _options.ctrl:
        draw(N.coords,nolight=True)
        if _options.ctrl_numbers:
            drawNumbers(N.coords,nolight=True)
    if _options.knots:
        draw(N.knotPoints(),color=_options.color,marksize=_options.knotsize,nolight=True)
        if _options.knot_numbers:
            drawNumbers(N.knotPoints(),nolight=True)
            

# Override some functions for nurbs

def createNurbsCurve(N,name=None):
    """Create a new Nurbs curve in the database.

    This will ask for a name if none is specified.
    The curve is exported under that name, drawn and selected.

    If no name is returned, the curve is not stored.
    """
    an = autoname['nurbscurve']
    drawNurbs(N)
    if name is None:
        name = an.peek()
        res = askItems([
            ('name',name,{'text':'Name for storing the object'}),
            ])
        if not res:
            return None
    
    name = res['name']
    if name == an.peek():
        an.next()
    print name
    print pf.PF
    export({name:N})
    print pf.PF
    selection.set([name])
    return name


def createInteractive():
    mode='nurbs'
    res = askItems([('degree',3),('closed',False)])
    obj_params.update(res)
    print "z value = %s" % the_zvalue
    points = drawPoints2D(mode,npoints=-1,zvalue=the_zvalue)
    if points is None:
        return
    print "POINTS %s" % points
    N = drawnObject(points,mode=mode)
    pf.canvas.removeHighlights()
    if N:
        createNurbsCurve(N,name=None)


def fromControlPolygon():
    if not selection_PL.check(single=True):
        selection_PL.ask(mode='single')

    if selection_PL.check(single=True):
        n = selection_PL.names[0]
        C = named(n)
        res = askItems([('degree',3)])
        N = NurbsCurve(C.coords,degree=res['degree'])
        createNurbsCurve(N,name=None)
        draw(C)


def fromPolyLine():
    if not selection_PL.check(single=True):
        selection_PL.ask(mode='single')

    if selection_PL.check(single=True):
        n = selection_PL.names[0]
        C = named(n)
        N = NurbsCurve(C.coords,degree=1,blended=False)
        createNurbsCurve(N,name=None)
    

################################## Menu #############################

_menu = 'Nurbs'

def create_menu(before='help'):
    """Create the menu."""
    MenuData = [
        ("&Select drawable",drawable.ask),
        ("&Set grid",create_grid),
        ("&Remove grid",remove_grid),
        ("---",None),
        ("&Toggle Preview",toggle_preview,{'checkable':True}),
        ("---",None),
        ("&Create Nurbs Curve ",[
            ("Interactive",createInteractive),
            ("From Control Polygon",fromControlPolygon),
            ("Convert PolyLine",fromPolyLine),
#            ("Convert BezierSpline",fromBezierSpline),
            ]),
        ("---",None),
        ("&Reload Menu",reload_menu),
        ("&Close Menu",close_menu),
        ("Test menu",test_menu),
        ]
    w = menu.Menu(_menu,items=MenuData,parent=pf.GUI.menu,before=before)
    return w

def show_menu(before='help'):
    """Show the menu."""
    if not pf.GUI.menu.action(_menu):
        create_menu(before=before)

def close_menu():
    """Close the menu."""
    pf.GUI.menu.removeItem(_menu)


def reload_menu():
    """Reload the menu."""
    before = pf.GUI.menu.nextitem(_menu)
    print "Menu %s was before %s" % (_menu,before)
    close_menu()
    import plugins
    plugins.refresh('draw2d')
    show_menu(before=before)
    setDrawOptions({'bbox':'last'})
    print pf.GUI.menu.actionList()

def test_menu():
    print "TEST2"
    
####################################################################

if __name__ == "draw":
    # If executed as a pyformex script
    reload_menu()

# End
