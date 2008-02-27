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
#from numpy import *


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


###################### Drawn Objects #############################

            
def draw_object_name(n):
    """Draw the name of an object at its center."""
    return drawText3D(named(n).center(),n)

def draw_elem_numbers(n):
    """Draw the numbers of an object's elements."""
    return drawNumbers(named(n))

def draw_bbox(n):
    """Draw the bbox of an object."""
    return drawBbox(named(n))
   

class DrawnObjects(dict):
    """A dictionary of drawn objects.
    """
    def __init__(self,*args,**kargs):
        dict.__init__(self,*args,**kargs)
        self.autodraw = False
        self.shrink = None
        self.annotations = [[draw_object_name,False],
                            [draw_elem_numbers,False],
                            [draw_bbox,False],
                            ]
        self._annotations = {}
        self._actors = []


    def draw(self,*args,**kargs):
        clear()
        print "SELECTION: %s" % self.names
        self._actors = draw(self.names,clear=False,shrink=self.shrink,*args,**kargs)
        #print self.annotations
        for i,a in enumerate(self.annotations):
            if a[1]:
                self.drawAnnotation(i)


    def ask(self,mode='multi'):
        """Interactively sets the current selection."""
        new = Objects.ask(self,mode)
        if new is not None:
            self.draw()


    def drawChanges(self):
        """Draws old and new version of a Formex with differrent colors.

        old and new can be a either Formex instances or names or lists thereof.
        old are drawn in yellow, new in the current color.
        """
        self.draw(wait=False)
        draw(self.values,color='yellow',bbox=None,clear=False,shrink=self.shrink)
 

    def undoChanges(self):
        """Undo the last changes of the values."""
        Objects.undoChanges(self)
        self.draw()


    def toggleAnnotation(self,i=0,onoff=None):
        """Toggle the display of number On or Off.

        If given, onoff is True or False. 
        If no onoff is given, this works as a toggle. 
        """
        active = self.annotations[i][1]
        #print "WAS"
        #print self.annotations
        if onoff is None:
            active = not active
        elif onoff:
            active = True
        else:
            active = False
        self.annotations[i][1] = active
        #print "BECOMES"
        #print self.annotations
        if active:
            self.drawAnnotation(i)
        else:
            self.removeAnnotation(i)
        #print self._annotations


    def drawAnnotation(self,i=0):
        """Draw some annotation for the current selection."""
        #print "DRAW %s" % i
        self._annotations[i] = [ self.annotations[i][0](n) for n in self.names ]


    def removeAnnotation(self,i=0):
        """Remove the annotation i."""
        #print "REMOVE %s" % i
        map(undraw,self._annotations[i])
        del self._annotations[i]


    def toggleNames(self):
        self.toggleAnnotation(0)
    def toggleNumbers(self):
        self.toggleAnnotation(1)
    def toggleBbox(self):
        self.toggleAnnotation(2)


    def setProperty(self,prop=None):
        """Set the property of the current selection.

        prop should be a single integer value or None.
        If None is given, a value will be asked from the user.
        If a negative value is given, the property is removed.
        If a selected object does not have a setProp method, it is ignored.
        """
        objects = self.check()
        if objects:
            if prop is None:
                res = askItems([['property',0]],
                               caption = 'Set Property Number for Selection (negative value to remove)')
                if res:
                    prop = int(res['property'])
                    if prop < 0:
                        prop = None
            for o in objects:
                if hasattr(o,'setProp'):
                    o.setProp(prop)
            self.draw()

##################### planes ##########################

planes = objects.DrawableObjects(clas=Plane)
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
        planes.set([name])
        planes.draw()
    

################# Create Selection ###################

selection = None

def set_selection(obj_type):
    global selection
    selection = pick(obj_type)

def query(mode):
    set_selection(mode)
    print report(selection)

def pick_actors():
    set_selection('actor')
def pick_elements():
    set_selection('element')
def pick_points():
    set_selection('point')
def pick_edges():
    set_selection('edge')
    
def query_actors():
    query('actor')
def query_elements():
    query('element')
def query_points():
    query('point')
def query_edges():
    query('edge')

def query_distances():
    set_selection('point')
    print reportDistances(selection)
                   

def report_selection():
    if selection is None:
        warning("You need to pick something first.")
        return
    print selection
    print report(selection)


def setprop_selection():
    """Set the property of the current selection.

    A property value is asked from the user and all items in the selection
    that have property have their value set to it.
    """
    if selection is None:
        warning("You need to pick something first.")
        return
    print selection
    res = askItems([['property',0]],
                   caption = 'Set Property Number for Selection (negative value to remove)')
    if res:
        prop = int(res['property'])
        if prop < 0:
            prop = None
        setpropCollection(selection,prop)


def grow_selection():
    if selection is None:
        warning("You need to pick something first.")
        return
    print selection
    res = askItems([('mode','node','radio',['node','edge']),
                    ('nsteps',1),
                    ],
                   caption = 'Grow method',
                   )
    if res:
        growCollection(selection,**res)
    print selection
    highlightElements(selection)


def export_selection():
    if selection is None:
        warning("You need to pick something first.")
        return
    print selection
    sel = getCollection(selection)
    if len(sel) == 0:
        warning("Nothing to export!")
        return
    options = ['list','single item']
    default = options[0]
    if len(sel) == 1:
        default = options[1]
    res = askItems([('Export with name',''),
                    ('Export as',default,'radio',options),
                    ])
    if res:
        name = res['Export with name']
        if res['Export as'] == options[0]:
            export({name:sel})
        elif res['Export as'] == options[1]:
            export(dict([ ("name-%s"%i,v) for i,v in enumerate(sel)]))


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
        ("&Select Plane",planes.ask),
        ("&Draw Selection",planes.draw),
        ("&Forget Selection",planes.forget),
        ("---",None),
        ("&Pick Actors",pick_actors),
        ("&Pick Elements",pick_elements),
        ("&Pick Points",pick_points),
        ("&Pick Edges",pick_edges),
        ("---",None),
        ('&Selection',
         [('&Create Report',report_selection),
          ('&Set Property',setprop_selection),
          ('&Grow',grow_selection),
          ('&Export',export_selection),
          ]),
        ("---",None),
        ('&Query',
         [('&Actors',query_actors),
          ('&Elements',query_elements),
          ('&Points',query_points),
          ('&Edges',query_edges),
          ('&Distances',query_distances),
          ]),
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
