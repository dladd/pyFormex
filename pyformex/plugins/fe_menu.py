#!/usr/bin/env pyformex
# $Id$
"""
Finite Element Plugin Menu for pyFormex.

(C) 2009 Benedict Verhegghe.
"""

from plugins import formex_menu,surface
import simple
from elements import Hex8


######################### functions #############################

def readModel(fn):
    """Read a nodes/elems model from file.

    Returns an (x,e) tuple or None
    """
    if not os.path.exists(fn):
        error("Node file '%s' does not exist" % fn)
        return None
    efn = fn.replace('nodes','elems')
    if not os.path.exists(efn):
        error("Corresponding element file '%s' does not exist" % efn)
        return None

    print "Importing model %s" % fn
    fil = file(fn,'r')
    noffset = 0
    #noffset = int(fil.readline().split()[1])
    a = fromfile(fil,sep=" ").reshape(-1,3)
    print a.shape
    x = Coords(a)
    print x.shape
    e = fromfile(efn,sep=" ",dtype=Int).reshape(-1,3) 
    print e.shape

    # convert to numpy offset
    if noffset != 0:
        e -= noffset

    return x,e
    

def importModel(fn=None):

    if fn is None:
        fn = askFilename(".","*nodes.txt",exist=True,multi=True)
        if not fn:
            return
    if type(fn) == str:
        fn = [fn]
        
    for i,f in enumerate(fn):
        x,e = readModel(f)
        modelname = os.path.basename(f).replace('nodes.txt','')
        F = Formex(x[e],i)#,eltype='hex8')
        export({modelname:F})
        formex_menu.selection.append(modelname)
        
    formex_menu.selection.draw()


def drawModel():
    F = formex_menu.selection.check(single=True)
    if F:
        draw(F)



################################## Menu #############################

_menu = 'FeModel'

def create_menu():
    """Create the menu."""
    MenuData = [
        ("&Import Model",importModel),
        ("&Draw Model",drawModel),
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
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":

    clear()
    reset()
    smooth()
##     if GD.GUI:
##         workHere()
##         print os.getcwd()
        
    reload_menu()

# End

