#!/usr/bin/env python pyformex.py
# $Id$

from plugins.fe import *
from plugins.properties import *
from plugins.fe_abq import *

PDB = None
parts = None
femodels = None
model = None

def deleteAll():
    global PDB,parts,femodels,model
    PDB = PropertyDB()
    parts = []
    femodels = []
    model = None
    clear()

def quad():
    """Return a unit quadrilateral Formex."""
    return Formex(mpattern('123'))

def triquad():
    """Return a triangularized unit quadrilateral Formex."""
    return Formex(mpattern('12-34'))

def rectangularMesh(b,h,nx,ny,triangles=False):
    """Create a rectangular mesh of size (b,h) with (nx,ny) divisions.

    The mesh will always comprise the domain (0,0,0)..(b,h,0).
    Default is to create a quad mesh. Set triangles True to create
    a triangular mesh.
    """
    sx,sy = float(b)/nx,float(h)/ny
    if triangles:
        base = triquad()
    else:
        base = quad()
    return base.replic2(nx,ny,1,1).scale([sx,sy,0.])

def addPart(F):
    """Add a Formex to the parts list."""
    global parts
    n = len(parts)
    F.setProp(n)
    export({'part-%s'%n:F})
    parts.append(F)
    femodels.append(F.feModel())

def drawParts():
    """Draw all parts"""
    clear()
    draw(parts)
    [ drawNumbers(p) for p in parts ]
    [ drawNumbers(Formex(fem[0]),color=red) for fem in femodels ]
    zoomAll()

def drawMerged():
    """Draw the merged parts"""
    if model is None:
        warning("You should first merge the parts!")
        return
    clear()
    draw(parts)
    draw(Formex(model.nodes))
    drawNumbers(Formex(model.nodes),color=red)
    [ drawNumbers(p) for p in parts ]
    zoomAll()


x0,y0 = 0.,0.
x1,y1 = 1.,1.
nx,ny = 4,4
eltype = 'quad'

def createPart():
    """Create a rectangular domain from user input"""
#    global F
    if model is not None:
        if ask('You have already merged the parts! I can not add new parts anymore.\nYou should first delete everything and recreate the parts.',['Delete','Cancel']) == 'Delete':
            deleteAll()
        else:
            return
    res = askItems([('x0',x0),('y0',y0),
                    ('x1',x1),('y1',y1),
                    ('nx',nx),('ny',ny),
                    ('eltype',eltype,'select',['quad','tri']),
                    ])
    if res:
        globals().update(res)
        F = rectangularMesh(x1-x0,y1-y0,nx,ny,eltype=='tri').trl([x0,y0,0])
        addPart(F)
        drawParts()


def mergeParts():
    """Merge all the parts into a Finite Element model."""
    global model
    model = Model(*mergeModels(femodels))
    drawMerged()


def getPickedNodes(K):
    """Get the list of picked nodes."""
    # This relies on drawing all parts first, then drawing the nodes
    return getPickedElems(K,len(parts))


def getPickedElems(K,p):
    """Get the list of picked elems from part p."""
    if p in K.keys():
        return K[p]
    return []


################# Functions After merging ######################

xcon = True
ycon = True

def warn():
    warning("You should first merge the parts!")

def setBoundary():
    """Pick the points with boundary condition."""
    global PDB,xcon,ycon
    if model is None:
        warn()
        return
    res = askItems([('x-constraint',xcon),('y-constraint',ycon)])
    if res:
        xcon = res['x-constraint']
        ycon = res['y-constraint']
        K = pickPoints()
        if K:
            nodeset = getPickedNodes(K)
            if len(nodeset) > 0:
                print nodeset
                print [xcon,ycon,0,0,0,0]
                PDB.nodeProp(set=nodeset,bound=[xcon,ycon,0,0,0,0])
        
xload = 0.0
yload = 0.0

def setLoad():
    """Pick the points with load condition."""
    global xload,yload
    if model is None:
        warn()
        return
    res = askItems([('x-load',xload),('y-load',yload)])
    if res:
        xload = res['x-load']
        yload = res['y-load']
        K = pickPoints()
        if K:
            nodeset = getPickedNodes(K)
            if len(nodeset) > 0:
                PDB.nodeProp(set=nodeset,cload=[xload,yload,0.,0.,0.,0.])


section = {
    'sectiontype': 'solid',
    'young_modulus': 207000,
    'poisson_ratio': 0.3,
    'thickness': 0.01,
    }


def setMaterial():
    """Set the material"""
    global section
    if model is None:
        warn()
        return
    keys = ['sectiontype','young_modulus','poisson_ratio','thickness']
    items = [ (k,section[k]) for k in keys ]
    res = askItems(items)
    if res:
        section.update(res)
        K = pickElements()
        if K:
            for k in range(len(parts)):
                e = getPickedElems(K,k) + model.celems[k]
                print k,e
                PDB.elemProp(set=e,eltype='CPS4',section=ElemSection().update(res))


def printModel():
    print "model:",model

def printDB():
    print PDB.nprop
    print PDB.eprop

#############################################################################
######### Create a menu with interactive tasks #############

def create_menu():
    """Create the FeEx menu."""
    MenuData = [
        ("&Delete All",deleteAll),
        ("&Create Part",createPart),
        ("&Show All",drawParts),
        ("---",None),
        ("&Merge Parts",mergeParts),
        ("&Show Merged Model",drawMerged),
        ("---",None),
        ("&Set boundary conditions",setBoundary),
        ("&Set loading conditions",setLoad),
        ("&Set material properties",setMaterial),
        ("---",None),
        ("&Print model",printModel),
        ("&Print property database",printDB),
        ("---",None),
        ("&Close Menu",close_menu),
        ]
    return widgets.Menu('FeEx',items=MenuData,parent=GD.gui.menu,before='help')

 
def show_menu():
    """Show the menu."""
    if not GD.gui.menu.item('FeEx'):
        create_menu()

def close_menu():
    """Close the menu."""
    m = GD.gui.menu.item('FeEx')
    if m :
        m.remove()

def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":

    # The sole intent of running this script is to create a top level
    # menu 'Hesperia'. The typical action then might be 'show_menu()'.
    # However, during development, you might want to change the menu's
    # actions will pyFormex is running, so a 'reload' action seems
    # more appropriate.
    
    reload_menu()
    deleteAll()

# End

