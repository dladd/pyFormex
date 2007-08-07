#!/usr/bin/env python pyformex.py
# $Id$
##
## This file is part of pyFormex 0.5 Release Mon Jul 30 13:38:48 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""formex_menu.py

This is a pyFormex plugin. It is not intended to be executed as a script,
but to be loaded as a plugin.
"""

import globaldata as GD
from gui.draw import *
from formex import *
from plugins import stl,inertia,partition,sectionize


import commands, os, timer

# The selection should only be set by setSelection()!!
selection = []  # a list of names of the currently selected Formices
oldvalues = []  # a list of Formex instances corresponding to the selection
                # BEFORE the last transformation
data = {} # a dict with global data


# If True, this makes element numbers to be displayed by drawSelection
show_numbers = False
shown_numbers = []  # The collection of numbers


def toggleNumbers(value=None):
    """Toggle the display of number On or Off.

    If given, value is True or False. 
    If no value is given, this works as a toggle. 
    """
    global show_numbers
    if value is None:
        show_numbers = not show_numbers
    elif value:
        show_numbers = True
    else:
        show_numbers = False
    if show_numbers:
        showSelectionNumbers()
    else:
        removeSelectionNumbers()


def showSelectionNumbers():
    """Draw the nubers for the current selection."""
    global shown_numbers
    for F in checkSelection(warn=False):
        shown_numbers.append(drawNumbers(F))


def removeSelectionNumbers():
    """Remove (all) the element numbers."""
    map(undraw,shown_numbers)


##################### select, read and write ##########################

def setSelection(namelist):
    """Set the selection to a list of names.

    You should make sure this is a list of Formex names!
    This will also set oldvalues to the corresponding Formices.
    """
    global selection,oldvalues
    if type(namelist) == str:
        namelist = [ namelist ]
    selection = namelist
    oldvalues = map(named,selection)


def clearSelection():
    """Clear the current selection."""
    setSelection([])


def changeSelection(newvalues):
    """Replace the current values of selection by new ones."""
    global oldvalues
    oldvalues = map(named,selection)
    export(dict(zip(selection,newvalues)))


def checkSelection(single=False,warn=True):
    """Check that we have a current selection.

    Returns the list of Formices corresponding to the current selection.
    If single==True, the selection should hold exactly one Formex name and
    a single Formex instance is returned.
    If there is no selection, or more than one in case of single==True,
    an error message is displayed and an empty list is returned.
    """
    if not selection:
        if warn:
            warning("No Formex selected")
        return []
    if single and len(selection) > 1:
        if warn:
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


def drawSelection(*args,**kargs):
    """Draws the current selection.

    Any arguments are passed to draw()"""
    clear()
    if selection:
        draw(selection,*args,**kargs)
        if show_numbers:
            showSelectionNumbers()
            

#################### Read/Write Formex File ##################################


def writeSelection():
    """Writes the currently selected Formices to .formex files."""
    if len(selection) == 1:
        name = selection[0]
        fn = askFilename(GD.cfg['workdir'],file="%s.formex" % name,
                         filter=['(*.formex)','*'],exist=False)
        if fn:
            print "Writing Formex '%s' to file '%s'" % (name,fn)
            print named(name).bbox()
            chdir(fn)
            named(name).write(fn)


def writeSelectionSTL():
    """Writes the currently selected Formices to .stl files."""
    if len(selection) == 1:
        name = selection[0]
        fn = askFilename(GD.cfg['workdir'],file="%s.stl" % name,
                         filter=['(*.stl)','*'],exist=False)
        if fn:
            print "Writing Formex '%s' to file '%s'" % (name,fn)
            print named(name).bbox()
            chdir(fn)
            stl.write_stla(fn,named(name).f)


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
    #print fn
    #return fn
    if fn:
        chdir(fn[0])
        names = map(utils.projectName,fn)
        GD.gui.setBusy()
        F = map(read_Formex,fn)
        GD.gui.setBusy(False)
        export(dict(zip(names,F)))
        if select:
            print "Got selection %s" % str(names)
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


def printBbox():
    """Print the bbox of the current selection."""
    FL = checkSelection()
    if FL:
        GD.message("Bbox of selection: %s" % bbox(FL))


def unitAxes():
    """Create a set of three axes."""
    Hx = Formex(pattern('1'),5).translate([-0.5,0.0,0.0])
    Hy = Hx.rotate(90)
    Hz = Hx.rotate(-90,1)
    Hx.setProp(4)
    Hy.setProp(5)
    Hz.setProp(6)
    return Formex.concatenate([Hx,Hy,Hz])    


def showPrincipal():
    """Show the principal axes."""
    global data
    F = checkSelection(single=True)
    if not F:
        return
    # compute the axes
    C,I = inertia.inertia(F.f)
    GD.message("Center of gravity: %s" % C)
    GD.message("Inertia tensor: %s" % I)
    Iprin,Iaxes = inertia.principal(I)
    GD.message("Principal Values: %s" % Iprin)
    GD.message("Principal Directions: %s" % Iaxes)
    data['principal'] = (C,I,Iprin,Iaxes)
    # now display the axes
    siz = F.size()
    H = unitAxes().scale(1.1*siz).affine(Iaxes.transpose(),C)
    A = 0.1*siz * Iaxes.transpose()
    G = Formex([[C,C+Ax] for Ax in A],3)
    draw([G,H])
    export({'principalAxes':H})
    return data['principal']


def rotatePrincipal():
    """Rotate the selection according to the last shown principal axes."""
    global data
    if not data.has_key('principal'):
        showPrincipal() 
    FL = checkSelection()
    if FL:
        ctr = data['principal'][0]
        rot = data['principal'][3]
        changeSelection([ F.trl(-ctr).rot(rot).trl(ctr) for F in FL ])
        drawChanges()



################### Perform operations on Formex #######################

def drawChanges():
    """Draws old and new version of a Formex with differrent colors.

    old and new can be a either Formex instances or names or lists thereof.
    old are drawn in yellow, new in the current color.
    """
    clear()
    drawSelection(wait=False)
    draw(oldvalues,color='yellow',bbox=None,alpha=0.5)


def undoChanges():
    """Undo the changes of the last transformation.

    The current versions of the selection are set back to the values prior
    to the last transformation.
    """
    changeSelection(oldvalues)
    drawSelection()
    

def scaleSelection():
    """Scale the selection."""
    FL = checkSelection()
    if FL:
        res = askItems([['scale',1.0]],
                       caption = 'Scale Factor')
        if res:
            scale = float(res['scale'])
            changeSelection([ F.scale(scale) for F in FL ])
            drawChanges()

            
def scale3Selection():
    """Scale the selection with 3 scale values."""
    FL = checkSelection()
    if FL:
        res = askItems([['x-scale',1.0],['y-scale',1.0],['z-scale',1.0]],
                       caption = 'Scaling Factors')
        if res:
            scale = map(float,[res['%c-scale'%c] for c in 'xyz'])
            changeSelection([ F.scale(scale) for F in FL ])
            drawChanges()


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


def rotateAround():
    """Rotate the selection."""
    FL = checkSelection()
    if FL:
        res = askItems([['axis',2],['angle','90.0'],['around','[0.0,0.0,0.0]']])
        if res:
            axis = int(res['axis'])
            angle = float(res['angle'])
            around = eval(res['around'])
            GD.debug('around = %s'%around)
            changeSelection([ F.rotate(angle,axis,around) for F in FL ])
            drawChanges()

def rollAxes():
    """Rotate the selection."""
    FL = checkSelection()
    if FL:
        changeSelection([ F.rollaxes() for F in FL ])
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
                export({name:Formex.concatenate(FL)})
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
    cuts = partition.partition(F)
    GD.message("Subsequent cutting planes: %s" % cuts)
    if ack('Save cutting plane data?'):
        types = [ 'Text Files (*.txt)', 'All Files (*)' ]
        fn = askFilename(GD.cfg['workdir'],types,exist=False)
        if fn:
            chdir(fn)
            fil = file(fn,'w')
            fil.write("%s\n" % cuts)
            fil.close()
    

def createParts():
    """Create parts of the current selection, based on property values."""
    F = checkSelection(single=True)
    if not F:
        return

    name = selection[0]
    partition.splitProp(F,name)


def sectionizeSelection():
    """Sectionize the selection."""
    F = checkSelection(single=True)
    if not F:
        return

    name = selection[0]
    GD.message("Sectionizing Formex '%s'" % name)
    ns,th,segments = sectionize.createSegments(F)
    if not ns:
        return
    
    sections,ctr,diam = sectionize.sectionize(F,segments,th)
    #GD.message("Centers: %s" % ctr)
    #GD.message("Diameters: %s" % diam)
    if ack('Save section data?'):
        types = [ 'Text Files (*.txt)', 'All Files (*)' ]
        fn = askFilename(GD.cfg['workdir'],types,exist=False)
        if fn:
            chdir(fn)
            fil = file(fn,'w')
            fil.write("%s\n" % ctr)
            fil.write("%s\n" % diam)
            fil.close()
    if ack('Draw circles?'):
        circles = sectionize.drawCircles(sections,ctr,diam)
        ctrline = sectionize.connectPoints(ctr)
        if ack('Draw circles on Formex ?'):
            sectionize.drawAllCircles(F,circles)
        circles = Formex.concatenate(circles)
        circles.setProp(3)
        ctrline.setProp(1)
        draw(ctrline,color='red')
        export({'circles':circles,'ctrline':ctrline,'flypath':ctrline})
        if ack('Fly through the Formex ?'):
            flyAlong(ctrline)
##        if ack('Fly through in smooth mode ?'):
##            smooth()
##            flytruCircles(ctr)
    drawSelection()


def flyThru():
    """Fly through the structure along the flypath."""
    path = named('flypath')
    if path:
        flyAlong(path)
    else:
        warning("You have to define a flypath first!")
        

################### menu #################

_menu = None  # protect against duplicate creation

def create_menu():
    """Create the Formex menu."""
    MenuData = [
#        ("&List Formices",formex_list),
        ("&Select",makeSelection),
        ("&Draw Selection",drawSelection),
        ('&Print Bbox',printBbox),
        ('&List Formices',printall),
#        ("&Draw Changes",drawChanges),
        ("&Save Selection as Formex",writeSelection),
        ("&Save Selection as STL File",writeSelectionSTL),
        ("&Read Formex Files",readSelection),
        ("---",None),
        ("&Set Property",setProperty),
        ("&Toggle Numbers",toggleNumbers),
        ("&Forget ",forgetSelection),
        ("&Undo Last Changes",undoChanges),
        ("---",None),
        ("&Transform",
         [("&Scale Selection",scaleSelection),
          ("&Non-uniformly Scale Selection",scale3Selection),
          ("&Translate Selection",translateSelection),
          ("&Center Selection",centerSelection),
          ("&Rotate Selection",rotateSelection),
          ("&Rotate Selection Around",rotateAround),
          ("&Roll Axes",rollAxes),
          ("&Clip Selection",clipSelection),
          ]),
        ("---",None),
        ("Show &Principal Axes",showPrincipal),
        ("Rotate to &Principal Axes",rotatePrincipal),
        ("---",None),
        ("&Concatenate Selection",concatenateSelection),
        ("&Partition Selection",partitionSelection),
        ("&Create Parts",createParts),
        ("&Sectionize Selection",sectionizeSelection),
        ("---",None),
        ("&Fly",flyThru),
        ("---",None),
        ("&Close",close_menu),
        ]
    return widgets.Menu('Formex',items=MenuData,parent=GD.gui.menu,before='help')


def close_menu():
    """Close the Formex menu."""
    global _menu
    if _menu:
        _menu.remove()
    _menu = None
    
def show_menu():
    """Show the Formex menu."""
    global _menu
    if not _menu:
        _menu = create_menu()
    

if __name__ == "main":
    print __doc__

# End

