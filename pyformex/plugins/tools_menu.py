#!/usr/bin/env python pyformex
# $Id$

"""tools_menu.py

Graphic Tools plugin menu for pyFormex.
"""

import globaldata as GD

from gui import actors,colors
from formex import *
from gui.draw import *
from plugins import objects
from plugins.tools import *


def editFormex(F):
    """Edit a Formex"""
    print "I want to edit a Formex"
    print "%s" % F.__class__


Formex.edit = editFormex


##################### database tools ##########################

database = objects.Objects()
    

def printall():
    """Print all global variable names."""
    print listAll()
    

def printval():
    """Print selected global variables."""
    database.ask()
    database.printval()
    

def forget():
    """Forget global variables."""
    database.ask()
    database.forget()

def edit():
    """Edit a global variable."""
    database.ask(mode='single')
    F = database.check(single=True)
    if F and hasattr(F,'edit'):
        name = database[0]
        F.edit(name)
       

def editByName(name):
    pass

##################### select, read and write ##########################

selection = objects.DrawableObjects(clas=Plane)
pname = utils.NameSequence('Plane-0')

def editPlane(plane,name):
    res = askItems([('Point',list(plane.point())),
                    ('Normal',list(plane.normal())),
                    ('Name',name)],
                   caption = 'Edit Plane')
    if res:
        p = res['Point']
        n = res['Normal']
        name = res['Name']
        P = Plane(p,n)
        export({name:P})
        
Plane.edit = editPlane


def createPlane():
    res = askItems([('Point',(0.,0.,0.)),
                    ('Normal',(1.,0.,0.)),
                    ('Name',pname.next())],
                   caption = 'Create a new Plane')
    if res:
        p = res['Point']
        n = res['Normal']
        name = res['Name']
        P = Plane(p,n)
        export({name:P})
        selection.set([name])
        selection.draw()


def test():
    picked = GD.canvas.pick()
    print picked
    for p in picked:
        GD.canvas.removeActor(p)
        p.redraw(GD.canvas.rendermode,color=colors.red)
        GD.canvas.addActor(p)
    GD.canvas.update()
    


################### menu #################

def create_menu():
    """Create the Tools menu."""
    MenuData = [
        ("&Show Variables",printall),
        ("&Print Variables",printval),
        ("&Edit Variable",edit),
        ("&Forget Variables",forget),
        ("---",None),
        ("&Create Plane",createPlane),
        ("&Select Plane",selection.ask),
        ("&Draw Selection",selection.draw),
        ("&Forget Selection",selection.forget),
        ("---",None),
        ("&Test",test),
        ("---",None),
        ("&Close",close_menu),
        ]
    return widgets.Menu('Tools',items=MenuData,parent=GD.gui.menu,before='help')

    
def show_menu():
    """Show the Tools menu."""
    if not GD.gui.menu.item('Tools'):
        create_menu()


def close_menu():
    """Close the Tools menu."""
    m = GD.gui.menu.item('Tools')
    if m :
        m.remove()
    

if __name__ == "draw":
    # If executed as a pyformex script
    close_menu()
    show_menu()
    
elif __name__ == "__main__":
    print __doc__


# End
