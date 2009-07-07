#!/usr/bin/env pyformex
# $Id$
"""mesh_menu.py

Interactive menu for Mesh type objects

(C) 2009 Benedict Verhegghe.
"""

import os,sys
sys.path[:0] = ['.', os.path.dirname(__file__)]

from gui.draw import *
from plugins import objects
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
    print x.shape
    return x


def readElems(fil,nplex):
    """Read a set of elems of plexitude nplex from an open mesh file"""
    print "Reading elements of plexitude %s" % nplex
    e = fromfile(fil,sep=" ",dtype=Int).reshape(-1,nplex) 
    e = Connectivity(e)
    print e.shape
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
        fn = askFilename(".","*.mesh",exist=True,multi=True)
        if not fn:
            return
    if type(fn) == str:
        fn = [fn]
        
    for f in fn:
        d = readMesh(f)
        print type(d)
        x = d['coords']
        e = d['elems']

        modelname = os.path.basename(f).replace('.mesh','')
        export({modelname:d})
        export(dict([("%s-%d"%(modelname,i), Mesh(x,ei)) for i,ei in enumerate(e)])) 


def convert_inp(fn=None):
    """Convert an Abaqus .inp file to pyFormex .mesh.

    """
    if fn is None:
        fn = askFilename(".","*.inp",exist=True)
        if not fn:
            return

    converter = os.path.join(GD.cfg['pyformexdir'],'bin','read_abq_inp')
    dirname = os.path.dirname(fn)
    basename = os.path.basename(fn)
    cmd = 'cd %s;%s %s' % (dirname,converter,basename)
    print cmd
    print utils.runCommand(cmd)

################################## Menu #############################

_menu = 'Mesh'

def create_menu():
    """Create the menu."""
    MenuData = [
        ("&Convert Abaqus .inp file",convert_inp),
        ("&Import Converted Model",importModel),
        ("&Draw Selection",selection.draw),
        ("---",None),
        ("&Reload Menu",reload_menu),
        ("&Close Menu",close_menu),
        ]
    return widgets.Menu(_menu,items=MenuData,parent=GD.GUI.menu,before='help')

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
    import mesh_menu
    reload(mesh_menu)
    show_menu()


####################################################################

if __name__ == "draw":
    # If executed as a pyformex script
    reload_menu()

# End

