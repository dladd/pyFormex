# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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

"""tools_menu.py

Graphic Tools plugin menu for pyFormex.
"""

import pyformex as GD
from gui import menu

import utils
from formex import *
from gui.draw import *
from plugins import objects
from plugins.tools import *


def editFormex(F,name):
    """Edit a Formex"""
    items = [('coords',repr(array(F.coords)),'text')]
    if F.prop is not None:
        items.append(('prop',repr(array(F.prop)),'text'))
    items.append(('eltype',F.eltype))
    res = askItems(items)
    if res:
        x = eval(res['coords'])
        p = res.get('eltype',None)
        if type(p) is str:
            p = eval(p)
        e = res['eltype']
        
        print(x)
        print(p)
        print(e)
        F.__init__(x,F.prop,e)

Formex.edit = editFormex


def command():
    """Execute an interactive command."""
    res = askItems([('command','','text')])
    if res:
        cmd = res['command']
        print("Command: %s" % cmd)
        exec(cmd)

##################### database tools ##########################

database = objects.Objects()
drawable = objects.DrawableObjects()


def printall():
    """Print all global variable names."""
    print(listAll())
    

def printval():
    """Print selected global variables."""
    database.ask()
    database.printval()


def printbbox():
    """Print selected global variables."""
    database.ask()
    database.printbbox()
    

def forget():
    """Forget global variables."""
    database.ask()
    database.forget()


def create():
    """Create a global variable."""
    ## name = ''
    ## value = ''
    ## while True:
    ##     res = askItems([('name',name),('value',value,'text')])
    ##     if res:
    ##         name = res['name']
    ##         value = res['value']
    ##         if name in database:
    ##             ans = ask("The named variable already exists! How should I proceed?",["Cancel","Rename New","Rename Old","Overwrite Old"])
    ##             if ans == "Cancel":
    ##                 return
    ##             elif ans == "Rename New":
    ##                 continue
    ##             elif ans == "Rename old":

    res = askItems([('name','')])
    if res:
        name = res['name']
        if name in database:
            warning("The variable named '%s' already exists!")
        else:
            export({name:'__initial__'})


def edit():
    """Edit a global variable."""
    database.ask(mode='single')
    F = database.check(single=True)
    if F is not None:
        name = database.names[0]
        if hasattr(F,'edit'):
            # Call specialized editor
            F.edit(name)
        else:
            # Use general editor
            res = askItems([(name,repr(F),'text')])
            if res:
                print(res)
                GD.PF.update(res)


def rename():
    """Rename a global variable."""
    database.ask(mode='single')
    F = database.check(single=True)
    if F is not None:
        oldname = database.names[0]
        res = askItems([('Name',oldname)],caption = 'Rename variable')
        if res:
            name = res['Name']
            export({name:F})
            database.forget()
            database.set(name)


def deleteAll():
    printall()
    if ack("Are you sure you want to unrecoverably delete all global variables?"):
        GD.PF = {}


def writeGeometry():
    """Write geometry to file."""
    drawable.ask()
    if drawable.check():
        filter = utils.fileDescription(['pgf','all'])
        cur = GD.cfg['workdir']
        fn = askNewFilename(cur=cur,filter=filter)
        if fn:
            drawable.writeToFile(fn)


def readGeometry():
    """Read geometry from file."""
    filter = utils.fileDescription(['pgf','all'])
    cur = GD.cfg['workdir']
    fn = askFilename(cur=cur,filter=filter)
    if fn:
        drawable.readFromFile(fn)
        drawable.draw()
        zoomAll()
        print drawable.names


def dos2unix():
    fn = askFilename(multi=True)
    print("selected %s" % fn)
    if fn:
        for f in fn:
            utils.dos2unix(f)


def unix2dos():
    fn = askFilename(multi=True)
    if fn:
        for f in fn:
            utils.unix2dos(f)

        
    
##################### planes ##########################

planes = objects.DrawableObjects(clas=Plane)
pname = utils.NameSequence('Plane-0')

def editPlane(plane,name):
    res = askItems([('Point',list(plane.point())),
                    ('Normal',list(plane.normal())),
                    ('Size',(list(plane.size()[0]),list(plane.size()[1])))],
                   caption = 'Edit Plane')
    if res:
        plane.P = res['Point']
        plane.n = res['Normal']
        plane.s = res['Size']

Plane.edit = editPlane


def createPlaneCoordsPointNormal():
    res = askItems([('Name',pname.next()),
                    ('Point',(0.,0.,0.)),
                    ('Normal',(1.,0.,0.)),
                    ('Size',((1.,1.),(1.,1.)))],
                   caption = 'Create a new Plane')
    if res:
        name = res['Name']
        p = res['Point']
        n = res['Normal']
        s = res['Size']
        P = Plane(p,n,s)
        export({name:P})
        draw(P)

    
def createPlaneCoords3Points():
    res = askItems([('Name',pname.next()),
                    ('Point 1',(0.,0.,0.)),
                    ('Point 2', (0., 1., 0.)),
                    ('Point 3', (0., 0., 1.)),
                    ('Size',((1.,1.),(1.,1.)))],
                   caption = 'Create a new Plane')
    if res:
        name = res['Name']
        p1 = res['Point 1']
        p2 = res['Point 2']
        p3 = res['Point 3']
        s = res['Size']
        pts=[p1, p2, p3]
        P = Plane(pts,size=s)
        export({name:P})
        draw(P)
    

def createPlaneVisual3Points():
    res = askItems([('Name',pname.next()),
                    ('Size',((1.,1.),(1.,1.)))],
                   caption = 'Create a new Plane')
    if res:
        name = res['Name']
        s = res['Size']
        picked = pick('point')
        pts = getCollection(picked)
        pts = asarray(pts).reshape(-1,3)
        if len(pts) == 3:
            P = Plane(pts,size=s)
            export({name:P})
            draw(P)
        else:
            warning("You have to pick exactly three points.")


################# Create Selection ###################

selection = None

def set_selection(obj_type):
    global selection
    selection = None
    selection = pick(obj_type)

def query(mode):
    set_selection(mode)
    print(report(selection))

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
    print(reportDistances(selection))

def query_angle():
    showInfo("Select two line elements.")
    set_selection('element')
    print(reportAngles(selection))


def report_selection():
    if selection is None:
        warning("You need to pick something first.")
        return
    print(selection)
    print(report(selection))


   
def setpropCollection(K,prop):
    """Set the property of a collection.

    prop should be a single non-negative integer value or None.
    If None is given, the prop attribute will be removed from the objects
    in collection even the non-selected items.
    If a selected object does not have a setProp method, it is ignored.
    """
    if K.obj_type == 'actor':
        obj = [ GD.canvas.actors[i] for i in K.get(-1,[]) ]
        for o in obj:
            if hasattr(o,'setProp'):
                o.setProp(prop)
            
    elif K.obj_type in ['element','point']:
        for k in K.keys():
            o = GD.canvas.actors[k]
            print "SETPROP ACTOR %s" % type(o)
            n =  drawable.names[k]
            print "SETPROP DRAWABLE %s" % n
            O = named(n)
            if O is not None:
                print "ALSO SETTING OBECT PROPERTIES"
            if prop is None:
                o.setProp(prop)
                if O:
                    O.setProp(prop)
            elif hasattr(o,'setProp'):
                if not hasattr(o,'prop') or o.prop is None:
                    o.setProp(0)
                print("PROP %s ?= %s" % (o.prop,O.prop))
                o.prop[K[k]] = prop
                o.setColor(o.prop)
                o.redraw(mode=GD.canvas.rendermode)
                if O and id(O.prop) != id(o.prop):
                    O.setProp(o.prop)
                print("PROP %s ?= %s" % (o.prop,O.prop))


def setprop_selection():
    """Set the property of the current selection.

    A property value is asked from the user and all items in the selection
    that have property have their value set to it.
    """
    if selection is None:
        warning("You need to pick something first.")
        return
    print(selection)
    res = askItems([['property',0]],
                   caption = 'Set Property Number for Selection (negative value to remove)')
    if res:
        prop = int(res['property'])
        if prop < 0:
            prop = None
        setpropCollection(selection,prop)
        removeHighlights()


def grow_selection():
    if selection is None:
        warning("You need to pick something first.")
        return
    print(selection)
    res = askItems([('mode','node','radio',['node','edge']),
                    ('nsteps',1),
                    ],
                   caption = 'Grow method',
                   )
    if res:
        growCollection(selection,**res)
    print(selection)
    highlightElements(selection)


def partition_selection():
    """Partition the current selection and show the result."""
    if selection is None:
        warning("You need to pick something first.")
        return
    if not selection.obj_type in ['actor','element']:
        warning("You need to pick actors or elements.")
        return
    for A in GD.canvas.actors:
        print A.atype
        if not A.atype == 'TriSurface':
            warning("Currently I can only partition TriSurfaces." )
            return
    partitionCollection(selection)
    highlightPartitions(selection)
    

def get_partition():
    """Select some partitions from the current selection and show the result."""
    if selection is None:
        warning("You need to pick something first.")
        return
    if not selection.obj_type in ['partition']:
        warning("You need to partition the selection first.")
        return
    res = askItems([['property',[1]]],
                 caption='Partition property')
    if res:
        prop = res['property']
        getPartition(selection,prop)
        highlightPartitions(selection)
    

def export_selection():
    if selection is None:
        warning("You need to pick something first.")
        return
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
            export(dict([ (name+"-%s"%i,v) for i,v in enumerate(sel)]))
     
        
################### menu #################

_menu = 'Tools'

def create_menu():
    """Create the Tools menu."""
    MenuData = [
        ('&Read Geometry File',readGeometry),
        ('&Write Geometry File',writeGeometry),
        ('-- Global Variables --',printall,dict(disabled=True)),
        ('  &List All',printall),
        ('  &Select',database.ask),
        ('  &Print Value',printval),
        ('  &Print BBox',printbbox),
        ('  &Draw',drawable.ask),
        ('  &Create',create),
        ('  &Change Value',edit),
        ('  &Rename',rename),
        ('  &Delete',forget),
        ('  &Delete All',deleteAll),
        ("---",None),
        ('&Execute pyFormex command',command),
        ("&DOS to Unix",dos2unix),
        ("&Unix to DOS",unix2dos),
        ("---",None),
        ("&Create Plane",
            [("Coordinates", 
                [("Point and normal", createPlaneCoordsPointNormal),
                ("Three points", createPlaneCoords3Points),
                ]), 
            ("Visually", 
                [("Three points", createPlaneVisual3Points),
                ]),
            ]),
        ("&Select Plane",planes.ask),
        ("&Draw Selection",planes.draw),
        ("&Forget Selection",planes.forget),
        ("---",None),
        ('&Pick',
         [("&Actors",pick_actors),
          ("&Elements",pick_elements),
          ("&Points",pick_points),
          ("&Edges",pick_edges),
          ]),
        ("&Remove Highlights",removeHighlights),
        ("---",None),
        ('&Selection',
         [('&Create Report',report_selection),
          ('&Set Property',setprop_selection),
          ('&Grow',grow_selection),
          ('&Partition',partition_selection),
          ('&Get Partition',get_partition),
          ('&Export',export_selection),
          ]),
        ("---",None),
        ('&Query',
         [('&Actors',query_actors),
          ('&Elements',query_elements),
          ('&Points',query_points),
          ('&Edges',query_edges),
          ('&Distances',query_distances),
          ('&Angle',query_angle),
          ]),
        ("---",None),
        ('&Reload',reload_menu),
        ("&Close",close_menu),
        ]
    return menu.Menu(_menu,items=MenuData,parent=GD.GUI.menu,before='help')


def show_menu():
    """Show the menu."""
    if not GD.GUI.menu.item(_menu):
        create_menu()


def close_menu():
    """Close the menu."""
    GD.GUI.menu.removeItem(_menu)


def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":

    reload_menu()

# End
