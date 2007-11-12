#!/usr/bin/env python pyformex
# $Id$

"""tools_menu.py

Graphic Tools plugin menu for pyFormex.
"""

import globaldata as GD

from gui import actors
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

def forget():
    """Forget global variables."""
    database.ask()
    database.forget()

def edit():
    """Edit a global variable."""
    database.ask(mode='single')
    F = database.check(single=True)
    if F and hasattr(F,'edit'):
        F.edit()
       

##################### select, read and write ##########################

selection = objects.DrawableObjects(clas=Plane)

def createPlane():
    res = askItems([('Point',(0.,0.,0.)),
                    ('Normal',(1.,0.,0.)),
                    ('Name','Plane0')],
                   caption = 'Create a new Plane')
    if res:
        p = res['Point']
        n = res['Normal']
        name = res['Name']
        P = Plane(p,n)
        export({name:P})


################### menu #################

def create_menu():
    """Create the Tools menu."""
    MenuData = [
        ("&Show Variables",printall),
        ("&Edit Variable",edit),
        ("&Forget Variables",forget),
        ("---",None),
        ("&Create Plane",createPlane),
        ("&Select Plane",selection.ask),
        ("&Draw Selection",selection.draw),
        ("&Forget Selection",selection.forget),
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
