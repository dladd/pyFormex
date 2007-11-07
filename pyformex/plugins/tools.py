# $Id$


from plugins import objects
from formex import *



class Plane(object):

    def __init__(self,point,normal):
        p = Coords(point)
        n = Coords(normal)

        if p.shape != (3,1) or n.shape != (3,1):
            raise ValueError,"point or normal does not have correct shape"

        self.p = p
        self.n = n



##################### select, read and write ##########################

selection = objects.DrawableObjects(clas=Formex)

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
    """Create the Formex menu."""
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
