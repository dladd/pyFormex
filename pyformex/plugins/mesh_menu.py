#!/usr/bin/env pyformex
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
"""mesh_menu.py

Interactive menu for Mesh type objects

(C) 2009 Benedict Verhegghe.
"""
import pyformex
import os,sys
sys.path[:0] = ['.', os.path.dirname(__file__)]

from gui.draw import *
from plugins import objects,formex_menu
import simple
from elements import Hex8
from connectivity import *
from plugins.fe import *
from plugins.mesh import *
from gui.actors import *


##################### select, read and write ##########################

selection = objects.DrawableObjects(clas=Mesh)


setSelection = selection.set
drawSelection = selection.draw


def getParams(line):
    """Strip the parameters from a comment line"""
    s = line.split()
    d = {'mode': s.pop(0),'filename': s.pop(0)}
    d.update(dict(zip(s[::2],s[1::2])))
    return d
    

def readNodes(fil):
    """Read a set of nodes from an open mesh file"""
    a = fromfile(fil,sep=" ").reshape(-1,3)
    x = Coords(a)
    print(x.shape)
    return x


def readElems(fil,nplex):
    """Read a set of elems of plexitude nplex from an open mesh file"""
    print("Reading elements of plexitude %s" % nplex)
    e = fromfile(fil,sep=" ",dtype=Int).reshape(-1,nplex) 
    e = Connectivity(e)
    print(e.shape)
    return e

def readEsets(fil):
    """Read the eset data of type generate"""
    data = []
    for line in fil:
        s = line.strip('\n').split()
        if len(s) == 4:
            data.append(s[:1]+map(int,s[1:]))
    return data
            

def readMesh(fn):
    """Read a nodes/elems model from file.

    Returns an (x,e) tuple or None
    """
    d = {}
    GD.GUI.setBusy(True)
    fil = file(fn,'r')
    for line in fil:
        if line[0] == '#':
            line = line[1:]
        globals().update(getParams(line))
        dfil = file(filename,'r')
        if mode == 'nodes':
            d['coords'] = readNodes(dfil)
        elif mode == 'elems':
            elems = d.setdefault('elems',[])
            e = readElems(dfil,int(nplex)) - int(offset)
            elems.append(e)
        elif mode == 'esets':
            d['esets'] = readEsets(dfil)
        else:
            print("Skipping unrecognized line: %s" % line)
        dfil.close()

    GD.GUI.setBusy(False)
    fil.close()
    return d                    


def importModel(fn=None):
    """Read one or more element meshes into pyFormex.

    Models are composed of matching nodes.txt and elems.txt files.
    A single nodes fliename or a list of node file names can be specified.
    If none is given, it will be asked from the user.
    """

    if fn is None:
        fn = askFilename(".","*.mesh",multi=True)
        if not fn:
            return
    if type(fn) == str:
        fn = [fn]
        
    for f in fn:
        d = readMesh(f)
        print(type(d))
        x = d['coords']
        e = d['elems']

        modelname = os.path.basename(f).replace('.mesh','')
        export({modelname:d})
        export(dict([("%s-%d"%(modelname,i), Mesh(x,ei)) for i,ei in enumerate(e)])) 


def convert_inp(fn=None):
    """Convert an Abaqus .inp file to pyFormex .mesh.

    """
    if fn is None:
        fn = askFilename(".","*.inp",multi=True)
        if not fn:
            return

        for f in fn:
            convert_inp(f)
        return


    converter = os.path.join(GD.cfg['pyformexdir'],'bin','read_abq_inp.awk')
    dirname = os.path.dirname(fn)
    basename = os.path.basename(fn)
    cmd = 'cd %s;%s %s' % (dirname,converter,basename)
    print(cmd)
    pyformex.GUI.setBusy()
    print(utils.runCommand(cmd))
    pyformex.GUI.setBusy(False)


def toFormex(suffix=''):
    """Transform the selection to Formices.

    If a suffix is given, the Formices are stored with names equal to the
    surface names plus the suffix, else, the surface names will be used
    (and the surfaces will thus be cleared from memory).
    """
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    newnames = selection.names
    if suffix:
        newnames = [ n + suffix for n in newnames ]

    newvalues = [ named(n).toFormex() for n in newnames ]
    export2(newnames,newvalues)

    if not suffix:
        selection.clear()
    formex_menu.selection.set(newnames)
    clear()
    formex_menu.selection.draw()
    

def fromFormex(suffix=''):
    """Transform the Formex selection to TriSurfaces.

    If a suffix is given, the TriSurfaces are stored with names equal to the
    Formex names plus the suffix, else, the Formex names will be used
    (and the Formices will thus be cleared from memory).
    """
    if not formex_menu.selection.check():
        formex_menu.selection.ask()

    if not formex_menu.selection.names:
        return

    names = formex_menu.selection.names
    formices = [ named(n) for n in names ]
    if suffix:
        names = [ n + suffix for n in names ]

    t = timer.Timer()
    meshes =  dict([ (n,F.toMesh()) for n,F in zip(names,formices) if F.nplex() == 3])
    print("Converted in %s seconds" % t.seconds())
    print(meshes.keys())
    export(meshes)

    if not suffix:
        formex_menu.selection.clear()
    selection.set(meshes.keys())


def convertMesh():
    """Transform the element type of the selected meshes.

    """
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    meshes = [ named(n) for n in selection.names ]
    eltypes = set([ m.eltype for m in meshes if m.eltype is not None])
    print "eltypes in selected meshes: %s" % eltypes
    if len(eltypes) > 1:
        warning("I can only convert meshes with the same element type\nPlease narrow your selection before trying conversion.")
        return
    if len(eltypes) == 1:
        fromtype = eltypes.pop()
        choices = ["%s -> %s" % (fromtype,to) for to in mesh.from_conversions[fromtype]]
        res = askItems([\
            ("Conversion Type",None,'vradio',{'choices':choices}),
            ("Pattern",None,'hradio',{'choices':['u','d','x','r']}),
            ])
        if res:
            print "Selected conversion %s" % res["Conversion Type"]
            converter = "convert_%s" % res["Conversion Type"].replace(' -> ','_')
            print  "Selected converter: %s" % converter
            converter_func = globals().get(converter,None)
            if converter_func is None:
                warning("Sorry, I could not find the converter function!")
                return
            newnames = [ "%s_converted" % n for n in selection.names ]
            newmeshes = [ converter_func(m,res['Pattern']) for m in meshes ]
            export2(newnames,newmeshes)
    

################################## Menu #############################

_menu = 'Mesh'

def create_menu():
    """Create the menu."""
    MenuData = [
        ("&Convert Abaqus .inp file",convert_inp),
        ("&Import Converted Model",importModel),
        ("&Select Mesh(es)",selection.ask),
        ("&Draw Selection",selection.draw),
        ("&Forget Selection",selection.forget),
        ("&Convert to Formex",toFormex),
        ("&Convert from Formex",fromFormex),
        ("---",None),
        ("&Convert Mesh Type",convertMesh),
        ("---",None),
        ("&Reload Menu",reload_menu),
        ("&Close Menu",close_menu),
        ]
    w = widgets.Menu(_menu,items=MenuData,parent=GD.GUI.menu,before='help')
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
    reload(mesh)
    import mesh_menu
    reload(mesh_menu)
    show_menu()


####################################################################

if __name__ == "draw":
    # If executed as a pyformex script
    reload_menu()

# End

