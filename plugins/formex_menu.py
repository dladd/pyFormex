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
    global selection,oldvalues
    selection = list
    oldvalues = map(named,selection)


def clearSelection():
    """Clear the current selection."""
    setSelection([])


## def saveSelection():
##     """Save the current values of selection so that changes can be undone."""
##     setSelection(selection)


def changeSelection(newvalues):
    """Replace the current values of selection by new ones."""
    global oldvalues
    oldvalues = map(named,selection)
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
    drawSelection()


def drawSelection():
    """Draws the current selection."""
    clear()
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



################### Change attributes of Formex #######################

def setProperty():
    """Set the property of the current selection.

    If the user gives a negative value, the property is removed.
    """
    FL = checkSelection()
    if FL:
        res = askItems([['property',0]],
                       caption = 'Set Property Number of Selection')
        if res:
            p = int(res['property'])
            if p < 0:
                p = None
            for F in FL:
                F.setProp(p)
            drawSelection()


def forgetSelection():
    if selection:
        forget(selection)


################### Perform operations on Formex #######################

def drawChanges():
    """Draws old and new version of a Formex with differrent colors.

    old and new can be a either Formex instances or names or lists thereof.
    old are drawn in yellow, new in the current color.
    """
    clear()
    draw(selection,wait=False)
    draw(oldvalues,color='yellow',bbox=None)


def undoChanges():
    """Undo the changes of the last transformation.

    The current versions of the selection are set back to the values prior
    to the last transformation.
    """
    changeSelection(oldvalues)
    drawSelection()
    

def translateSelection():
    """Translate the selection."""
    FL = checkSelection()
    if FL:
        res = askItems([['direction',0],['distance','1.0']],
                       caption = 'Translation Parameters')
        if res:
            dir = int(res['direction'])
            dist = float(res['distance'])
            changeSelection([ F.translate(dir,dist) for F in FL ])
            drawChanges()


def centerSelection():
    """Center the selection."""
    FL = checkSelection()
    if FL:
        changeSelection([ F.translate(-F.center()) for F in FL ])
        drawChanges()


def rotateSelection():
    """Rotate the selection."""
    FL = checkSelection()
    if FL:
        res = askItems([['axis',2],['angle','90.0']])
        if res:
            axis = int(res['axis'])
            angle = float(res['angle'])
            changeSelection([ F.rotate(angle,axis) for F in FL ])
            drawChanges()
            
        
def clipSelection():
    """Clip the selection."""
    FL = checkSelection()
    if FL:
        res = askItems([['axis',0],['begin',0.0],['end',1.0]],caption='Clipping Parameters')
        if res:
            bb = bbox(FL)
            axis = int(res['axis'])
            xmi = bb[0][axis]
            xma = bb[1][axis]
            dx = xma-xmi
            xc1 = xmi + float(res['begin']) * dx
            xc2 = xmi + float(res['end']) * dx
            changeSelection([ F.clip(F.test(dir=axis,min=xc1,max=xc2)) for F in FL ])
            drawChanges()


def concatenateSelection():
    """Concatenate the selection."""
    FL = checkSelection()
    if FL:
        plexitude = array([ F.nplex() for F in FL ])
        if plexitude.min() == plexitude.max():
            res = askItems([['name','combined']],'Name for the concatenation')
            if res:
                name = res['name']
                Export({name:Formex.concatenate(FL)})
                setSelection([name])
                drawSelection()
        else:
            warning('You can only concatenate Formices with the same plexitude!')
    

def partitionSelection():
    """Partition the selection."""
    F = checkSelection(single=True)
    if not F:
        return

    name = selection[0]
    GD.message("Partitioning Formex '%s'" % name)
    cuts = partition(F)
    GD.message("Subsequent cutting planes: %s" % cuts)
    

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
#        ("&Draw Changes",drawChanges),
        ("&Save Selection to file(s)",writeSelection),
        ("&Read Formex Files",readSelection),
        ("---",None),
        ("&Set Property",setProperty),
        ("&Forget ",forgetSelection),
        ("&Translate Selection",translateSelection),
        ("&Center Selection",centerSelection),
        ("&Rotate Selection",rotateSelection),
        ("&Clip Selection",clipSelection),
        ("&Concatenate Selection",concatenateSelection),
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

