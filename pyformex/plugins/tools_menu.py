# $Id$

"""tools_menu.py

Graphic Tools plugin menu for pyFormex.
"""

import globaldata as GD

from gui import actors
from gui.draw import *
from plugins import objects
from plugins.tools import *


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

_menu = None  # protect against duplicate creation

def create_menu():
    """Create the Tools menu."""
    MenuData = [
        ("&Create Plane",createPlane),
        ("&Select Plane",selection.ask),
        ("&Draw Selection",selection.draw),
        ("&Forget Selection",selection.forget),
        ("---",None),
        ("&Close",close_menu),
        ]
    return widgets.Menu('Tools',items=MenuData,parent=GD.gui.menu,before='help')


def close_menu():
    """Close the Tools menu."""
    global _menu
    if _menu:
        _menu.remove()
    _menu = None
    
def show_menu():
    """Show the Tools menu."""
    global _menu
    if not _menu:
        _menu = create_menu()
    

if __name__ == "main":
    print __doc__


# End
