# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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
"""mesh_menu.py

Interactive menu for Mesh type objects
"""
import pyformex

import simple
from connectivity import *
from gui import menu
from gui.draw import *
from plugins import formex_menu
from plugins.objects import *
from plugins.fe import *
from elements import *
from mesh import *

##################### select, read and write ##########################

# We subclass the DrawableObject to change its toggleAnnotation method
class MeshObjects(DrawableObjects):
    def __init__(self):
        DrawableObjects.__init__(self,clas=Mesh)

    ## def toggleAnnotation(self,i=0,onoff=None):
    ##     """Toggle mesh annotations on/off.

    ##     This functions is like DrawableObjects.toggleAnnotation but also
    ##     updates the mesh_menu when changes are made.
    ##     """
    ##     DrawableObjects.toggleAnnotation(self,i,onoff)
    ##     mesh_menu = pf.GUI.menu.item(_menu)
    ##     toggle_menu = mesh_menu.item("toggle annotations")
    ##     # This relies on the menu having the same items as the annotation list
    ##     action = toggle_menu.actions()[i]
    ##     action.setChecked(selection.hasAnnotation(i))

selection = MeshObjects()


def splitProp():
    """Split the mesh based on property values"""
    from plugins import partition
    
    F = selection.check(single=True)
    if not F:
        return

    name = selection[0]
    partition.splitProp(F,name)
    

def fuseMesh():
    """Fuse the nodes of a Mesh"""
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    res = askItems([
        ('Relative Tolerance',1.e-5),
        ('Absolute Tolerance',1.e-5),
        ('Shift',0.5),
        ('Nodes per box',1)])

    if not res:
        return

    before = [ m.ncoords() for m in meshes ]
    meshes = [ m.fuse(
        rtol = res['Relative Tolerance'],
        atol = res['Absolute Tolerance'],
        shift = res['Shift'],
        nodesperbox = res['Nodes per box'],
        ) for m in meshes ]
    after = [ m.ncoords() for m in meshes ]
    print "Number of points before fusing: %s" % before
    print "Number of points after fusing: %s" % after

    names = [ "%s_fused" % n for n in selection.names ]
    export2(names,meshes)
    selection.set(names)
    clear()
    selection.draw()



def divideMesh():
    """Create a mesh by subdividing existing elements.

    """
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    eltypes = set([ m.eltype for m in meshes if m.eltype is not None])
    print "eltypes in selected meshes: %s" % eltypes
    if len(eltypes) > 1:
        warning("I can only divide meshes with the same element type\nPlease narrow your selection before trying conversion.")
        return
    if len(eltypes) == 1:
        fromtype = eltypes.pop()
    showInfo("Sorry, this function is not implemented yet!")


def convertMesh():
    """Transform the element type of the selected meshes.

    """
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    eltypes = set([ m.eltype.name() for m in meshes if m.eltype is not None])
    print "eltypes in selected meshes: %s" % eltypes
    if len(eltypes) > 1:
        warning("I can only convert meshes with the same element type\nPlease narrow your selection before trying conversion.")
        return
    if len(eltypes) == 1:
        fromtype = elementType(eltypes.pop())
        choices = ["%s -> %s" % (fromtype,to) for to in fromtype.conversions.keys()]
        if len(choices) == 0:
            warning("Sorry, can not convert a %s mesh"%fromtype)
            return
        res = askItems([
            ('_conversion',None,'vradio',{'text':'Conversion Type','choices':choices}),
            ("_compact",True),
            ('_merge',None,'hradio',{'text':"Merge Meshes",'choices':['None','Each','All']}),
            ])
        if res:
            globals().update(res)
            print "Selected conversion %s" % _conversion
            totype = _conversion.split()[-1]
            names = [ "%s_converted" % n for n in selection.names ]
            meshes = [ m.convert(totype) for m in meshes ]
            if _merge == 'Each':
                meshes = [ m.fuse() for m in meshes ]
            elif  _merge == 'All':
                print _merge
                coords,elems = mergeMeshes(meshes)
                print elems
                ## names = [ "_merged_mesh_%s" % e.nplex() for e in elems ]
                ## meshes = [ Mesh(coords,e,eltype=meshes[0].eltype) for e in elems ]
                ## print meshes[0].elems
                meshes = [ Mesh(coords,e,m.prop,m.eltype) for e,m in zip(elems,meshes) ]
            if _compact:
                print "compacting meshes"
                meshes = [ m.compact() for m in meshes ]
                
            export2(names,meshes)
            selection.set(names)
            clear()
            selection.draw()


def renumberMeshInElemsOrder():
    """Renumber the selected Meshes in elems order.

    """
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    names = selection.names
    meshes = [ M.renumber() for M in meshes ]
    export2(names,meshes)
    selection.set(names)
    clear()
    selection.draw()
    

################################## Menu #############################

_menu = 'Mesh'

def create_menu():
    """Create the menu."""
    MenuData = [
        ## ("&Convert Abaqus .inp file",convert_inp),
        ## ("&Import Converted Model",importModel),
        ## ("&Select Mesh(es)",selection.ask),
        ## ("&Draw Selection",selection.draw),
        ## ("&Forget Selection",selection.forget),
        ## ("&Convert to Formex",toFormex),
        ## ("&Convert from Formex",fromFormex),
        ("---",None),
        ("&Split on Property Value",splitProp),
        ("&Fuse Nodes",fuseMesh),
        ("&Divide Mesh",divideMesh),
        ("&Convert Mesh Eltype",convertMesh),
        ("&Renumber Mesh in Elems order",renumberMeshInElemsOrder),
        ("---",None),
        ## ("Toggle &Annotations",
        ##  [("&Name",selection.toggleNames,dict(checked=draw_object_name in selection.annotations)),
          ## ("&Element Numbers",selection.toggleNumbers,dict(checked=selection.hasNumbers())),
          ## ("&Node Numbers",selection.toggleNodeNumbers,dict(checked=selection.hasNodeNumbers())),
          ## ("&Node Marks",selection.toggleNodes,dict(checked=selection.hasNodeMarks())),
          ## ('&Bounding Box',selection.toggleBbox,dict(checked=selection.hasBbox())),
        ##   ]),
        ## ("---",None),
        ("&Reload Menu",reload_menu),
        ("&Close Menu",close_menu),
        ]
    w = menu.Menu(_menu,items=MenuData,parent=pf.GUI.menu,before='help',tearoff=False)
    return w


def show_menu():
    """Show the menu."""
    if not pf.GUI.menu.item(_menu):
        create_menu()


def close_menu():
    """Close the menu."""
    pf.GUI.menu.removeItem(_menu)


def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":

    reload_menu()

# End

