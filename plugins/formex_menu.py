#!/usr/bin/env pyformex
# $Id$

"""Formex.py

Executing this script creates a Formex menu in the menubar.
"""

import globaldata as GD
from gui.draw import *
from formex import *

import commands, os, timer


selection = []
newvalues = []

##################### select, read and write ##########################


def setSelection(list):
    """Set the selection to a list of names.

    You should make sure this is a list of Formex names!
    """
    global selection
    selection = list 
    print selection
    

def setNewvalues(list):
    """Set the list of new values for the names in selection.

    You should make sure this is a list of Formices!
    """
    global newvalues
    newvalues = list 


def clearSelection():
    """Clear the current selection and newvalues."""
    global selection,newvalues
    selection = newvalues = []


def askSelection(mode=None):
    """Show the names of known formices and let the user select (one or more).

    This just returns a list of selected Formex names.
    It does not set the current selection. (see makeSelection)
    """
    return widgets.Selection(listAll(),'Known Formices',mode,sort=True,
                            selected=selection).getResult()

def makeSelection():
    """Interactively sets the current selection."""
    setSelection(askSelection('multi'))



def drawSelection():
    """Draws the current selection."""
    if selection:
        draw(selection)

        
def writeSelection():
    """Writes the currently selected Frmices to .formex files."""
    if selection:
        fn = askDirname()
        print "DIR:%s" % fn
        print os.getcwd()
        if fn:
            for name in selection:
                writeFormex(named(name),"%s.formex" % name)


def read_Formex(fn):
    GD.message("Reading file %s" % fn)
    t = timer.Timer()
    F = readFormex(fn)
    nelems,nplex = F.f.shape[0:2]
    GD.message("Read %d elems of plexitude %d in %s seconds" % (nelems,nplex, t.seconds()))
    return F


def readSelection(select=True,draw=True):
    """Read a Formex (or list) from asked file name(s).

    If select is True (default), this becomes the current selection.
    If select and draw are True (default), the selection is drawn.
    """
    types = [ 'Formex Files (*.formex)' ]
    fn = askFilename(GD.cfg['workdir'],types,exist=True,multi=True)
    if fn:
        chdir(fn[0])
        names = map(utils.projectName,fn)
        GD.gui.setBusy()
        F = map(read_Formex,fn)
        GD.gui.setBusy(False)
        Export(dict(zip(names,F)))
        if select:
            setSelection(names)
            if draw:
                drawSelection()
    return fn


################### Perform operations on Formex #######################

def drawChanges(old,new):
    """Draws old and new version of a Formex with differrent colors.

    old and new can be a either Formex instances or names or lists thereof.
    old are drawn in yellow, new in the current color.
    """
    draw(old,color='yellow',wait=False)
    draw(new)


def keepChanges():
    """Draws old and new versions and replaces ol by new after confirmation.

    The current versions of the selection and new versions are displayed
    together, with different colors. Then the user is asked for confirmation
    to replace the current with the new versions.
    Finally, the (replaced or not) current selection is drawn.

    Before using this function, selection should be set to a list of names,
    and newvalues should be set to an equally long list of Formices.
    """
    drawChanges(selection,newvalues)
    if ack("Keep the changes to the displayed Formices?"):
        Export(dict(zip(selection,newvalues)))
    clear()
    drawSelection()
    

def translateSelection():
    """Translate the selection."""
    if not selection:
        return
    itemlist = [ [ 'direction',0], ['distance','1.0'] ] 
    res,accept = widgets.inputDialog(itemlist,'Translation Parameters').getResult()
    if accept:
        dir = int(res[0][1])
        dist = float(res[1][1])
        oldF = map(named,selection)
        setNewvalues([ F.translate(dir,dist) for F in oldF ])
        keepChanges()


def centerSelection():
    """Center the selection."""
    if not selection:
        return
    oldF = map(named,selection)
    setNewvalues([ F.translate(-F.center()) for F in oldF ])
    keepChanges()


def rotateSelection():
    """Rotate the selection."""
    if not selection:
        return
    itemlist = [ [ 'axis',0], ['angle','0.0'] ] 
    res,accept = widgets.inputDialog(itemlist,'Rotation Parameters').process()
    if accept:
        axis = int(res[0][1])
        angle = float(res[1][1])
        oldF = map(named,selection)
        setNewvalues([ F.rotate(angle,axis) for F in oldF ])
        keepChanges()


def combineSelection():
    """Rotate the selection."""
    if not selection:
        return
    oldF = map(named,selection)
    plexitude = array([ F.nplex() for F in oldF ])
    if plexitude.min() == plexitude.max():
        res,accept = widgets.inputDialog(itemlist,'Name for the concatenation').process()
        if accept:
            F = Formex.concatenate(oldF)
            ask
        
def clipSelection():
    """Clip the stl model."""
    global F
    itemlist = [['axis',0],['begin',0.0],['end',1.0]]
    res,accept = widgets.inputDialog(itemlist,'Clipping Parameters').process()
    if accept:
        Flist = byName(selection)
        bb = bbox(Flist)
        axis = int(res[0][1])
        xmi = bb[0][axis]
        xma = bb[1][axis]
        dx = xma-xmi
        xc1 = xmi + float(res[1][1]) * dx
        xc2 = xmi + float(res[2][1]) * dx
        for F in Flist:
            w = F.test(dir=axis,min=xc1,max=xc2)
            oldF = F.cclip(w)
            F = F.clip(w)
            drawChanges(F,oldF)



################### menu #################

_menu = None

def create_menu():
    """Create the Formex menu."""
    menu = widgets.Menu('Formex')
    MenuData = [
#        ("&List Formices",formex_list),
        ("&Select",makeSelection),
        ("&Draw Selection",drawSelection),
        ("&Save Selection in Formex file",writeSelection),
        ("&Read Formex Files",readSelection),
        ("&Translate Selection",translateSelection),
        ("&Center Selection",centerSelection),
        ("&Rotate Selection",rotateSelection),
        ("&Clip Selection",clipSelection),
        ("&Close",close_menu),
        ]
    menu.addItems(MenuData)
    return menu

def close_menu():
    """Close the Formex menu."""
    global _menu
    if _menu:
        _menu.close()
    _menu = None
    
def show_menu():
    """Show the Formex menu."""
    global _menu
    if not _menu:
        _menu = create_menu()
    

if __name__ == "main":
    print __doc__

# End

