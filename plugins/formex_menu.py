#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Sat Mar 10 20:05:55 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##

"""Formex.py

Executing this script creates a Formex menu in the menubar.
"""

import globaldata as GD
from gui.draw import *
from formex import *
from plugins.partition import *

import commands, os, timer

# The selection should only be set by setSelection()!!
selection = []  # a list of names of the currently selected Formices
oldvalues = []  # a list of Formex instances corresponding to the selection
                # BEFORE the last transformation  

##################### select, read and write ##########################


def setSelection(list):
    """Set the selection to a list of names.

    You should make sure this is a list of Formex names!
    This will also set oldvalues to the corresponding Formices.
    """
    global selection
    selection = list
    oldvalues = map(named,list)
    print selection


def clearSelection():
    """Clear the current selection."""
    setSelection([])


## def saveSelection():
##     """Save the current values of selection so that changes can be undone."""
##     setSelection(selection)


def changeSelection(newvalues):
    """Replace the current values of selection by new ones."""
    Export(dict(zip(selection,newvalues)))


def checkSelection(single=False):
    """Check that we have a current selection.

    Returns the list of Formices corresponding to the current selection.
    If single==True, the selection should hold exactly one Formex name and
    a single Formex instance is returned.
    If there is no selection, or more than one in case of single==True,
    an error message is displayed and an empty list is returned.
    """
    if not selection:
        warning("No Formex selected")
        return []
    if single and len(selection) > 1:
        warning("You should select exactly one Formex")
        return []
    if single:
        return named(selection[0])
    else:
        return map(named,selection)
    

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


#################### Read/Write Formex File ##################################


def writeSelection():
    """Writes the currently selected Formices to .formex files."""
    if len(selection) == 1:
        name = selection[0]
        fn = askFilename(GD.cfg['workdir'],file="%s.formex" % name,
                         filter=['(*.formex)','*'],exist=False)
        if fn:
            print "Writing Formex '%s' to file '%s'" % (name,fn)
            chdir(fn)
            named(name).write(fn)


def read_Formex(fn):
    GD.message("Reading file %s" % fn)
    t = timer.Timer()
    F = Formex.read(fn)
    nelems,nplex = F.f.shape[0:2]
    GD.message("Read %d elems of plexitude %d in %s seconds" % (nelems,nplex, t.seconds()))
    return F


def readSelection(select=True,draw=True):
    """Read a Formex (or list) from asked file name(s).

    If select is True (default), this becomes the current selection.
    If select and draw are True (default), the selection is drawn.
    """
    types = [ 'Formex Files (*.formex)', 'All Files (*)' ]
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


def undoChanges():
    """Undo the changes of the last transformation.

    The current versions of the selection are set back to the values prior
    to the last transformation.
    """
    changeSelection(oldvalues)
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
    FL = checkSelection()
    if not FL:
        return
##     itemlist = [ [ 'axis',0], ['angle','0.0'] ] 
##     res,accept = widgets.inputDialog(itemlist,'Rotation Parameters').getResult()
##     if accept:
##         axis = int(res[0][1])
##         angle = float(res[1][1])
##         oldF = map(named,selection)
##         setNewvalues([ F.rotate(angle,axis) for F in oldF ])
##         keepChanges()
    res = askItems([['axis',0],['angle','0.0']])
    if res:
        axis = int(res['axis'])
        angle = float(res['angle'])
        changeSelection([ F.rotate(angle,axis) for F in FL ])


def combineSelection():
    """Rotate the selection."""
    if not selection:
        return
    oldF = map(named,selection)
    plexitude = array([ F.nplex() for F in oldF ])
    if plexitude.min() == plexitude.max():
        res,accept = widgets.inputDialog(itemlist,'Name for the concatenation').getResult()
        if accept:
            F = Formex.concatenate(oldF)
            info("This is not implemented yet!")
        
def clipSelection():
    """Clip the stl model."""
    if not selection:
        return
    itemlist = [['axis',0],['begin',0.0],['end',1.0]]
    res,accept = widgets.inputDialog(itemlist,'Clipping Parameters').getResult()
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
    

def partitionSelection():
    """Partition the selection."""
    F = checkSelection(single=True)
    if not F:
        return

    name = selection[0]
    GD.message("Partitioning Formex '%s'" % name)
    cuts = partition(F)
    print array(cuts)
    

def createParts():
    """Create parts of the current selection, based on property values."""
    F = checkSelection(single=True)
    if not F:
        return

    name = selection[0]
    splitProp(F,name)


################### menu #################

_menu = None

def create_menu():
    """Create the Formex menu."""
    menu = widgets.Menu('Formex')
    MenuData = [
#        ("&List Formices",formex_list),
        ("&Select",makeSelection),
        ("&Draw Selection",drawSelection),
        ("&Save Selection to file(s)",writeSelection),
        ("&Read Formex Files",readSelection),
        ("&Translate Selection",translateSelection),
        ("&Center Selection",centerSelection),
        ("&Rotate Selection",rotateSelection),
        ("&Clip Selection",clipSelection),
        ("&Partition Selection",partitionSelection),
        ("&Create Parts",createParts),
        ("&Undo Last Changes",undoChanges),
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

