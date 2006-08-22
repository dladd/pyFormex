#!/usr/bin/env python
# $Id $

# This file is intended to disappear, after its contents has been
# moved to a more appropriate place.

import globaldata as GD
import gui
import draw
import widgets
import canvas
import help

import sys,time,os,string
import qt
import qtgl

    

def askConfigPreferences(items,section=None):
    """Ask preferences stored in config variables.

    Items in list should only be keys. The current values are retrieved
    from the config.
    A config section name should be specified if the items are not in the
    top config level.
    """
    if section:
        store = GD.cfg[section]
    else:
        store = GD.cfg
    # insert current values
    for it in items:
        it.insert(1,store.setdefault(it[0],''))
    res,accept = widgets.inputDialog(items,'Config Dialog').process()
    if accept:
        GD.prefsChanged = True
        #print "ACCEPTED following values:"
        for r in res:
            #print r
            store[r[0]] = eval(r[1])
    #print GD.cfg


def newaskConfigPreferences(items,store):
    """Ask preferences stored in config variables.

    Items in list should only be keys. The current values are retrieved
    from the config.
    A config section name should be specified if the items are not in the
    top config level.
    """
    if not store:
        store = GD.cfg
    itemlist = [ [ i,store.setdefault(i,'') ] for i in items ]
    res,accept = widgets.inputDialog(itemlist,'Config Dialog').process()
    if accept:
        #print "ACCEPTED following values:"
        for r in res:
            #print r
            store[r[0]] = eval(r[1])


def prefHelp():
    askConfigPreferences([['viewer'],['homepage'],['history'],['bookmarks']],'help')

def prefDrawtimeout():
    askConfigPreferences([['drawtimeout','int']])


def prefBGcolor():
    col = qt.QColorDialog.getColor(qt.QColor(GD.cfg.setdefault('bgcolor','')))
    if col.isValid():
        GD.prefsChanged = True
        GD.cfg['bgcolor'] = str(col.name()) # convert qstring to Python string!
        draw.bgcolor(col)


def prefLinewidth():
    askConfigPreferences([['linewidth']])
    draw.linewidth(GD.cfg['linewidth'])

def prefSize():
    GD.gui.resize(800,600)
    
def prefCanvasSize():
    res = draw.askItems([['w',GD.canvas.width()],['h',GD.canvas.height()]])
    GD.canvas.resize(int(res['w']),int(res['h']))
        
    
def prefRender():
    askConfigPreferences([['specular'], ['shininess']],'render')

##def prefLight0():
##    askConfigPreferences([['light0']],'render')
##    draw.smooth()

##def prefLight1():
##    askConfigPreferences([['light1']],'render')
##    draw.smooth()

def prefLight(light=0):
    store = GD.cfg.render["light%d" % light]
    keys = [ 'ambient', 'diffuse', 'specular', 'position' ]
    newaskConfigPreferences(keys,store)

def prefLight0():
    prefLight(0)
    draw.smooth()

def prefLight1():
    prefLight(1)
    draw.smooth()
    

# Examples Menu
def insertExampleMenu():
    """Insert the examples menu in the menudata.

    Examples are all the .py files in the subdirectory examples,
    provided there name does not start with a '.' or '_' and
    their first line ends with 'pyformex'
    """
    global example
    dir = GD.cfg.exampledir
    if not os.path.isdir(dir):
        return
    example = filter(lambda s:s[-3:]==".py" and s[0]!='.' and s[0]!='_',os.listdir(dir))
    example = filter(lambda s:file(os.path.join(GD.cfg.exampledir,s)).readlines()[0].strip().endswith('pyformex'),example)
    example.sort()
    vm = ("Popup","&Examples",[
        ("VAction","&%s"%os.path.splitext(t)[0],("runExample",i)) for i,t in enumerate(example)
        ])
    nEx = len(vm[2])
    vm[2].append(("VAction","Run All Examples",("runExamples",nEx)))
    MenuData.insert(4,vm)

def runExample(i):
    """Run example i from the list of found examples."""
    global example
    gui.setcurfile(os.path.join(GD.cfg.exampledir,example[i]))
    play()

def runExamples(n):
    """Run the first n examples."""
    for i in range(n):
        runExample(i)



###################### Actions #############################################
# Actions are just python functions, preferably without arguments
# Actions are often used as slots, which are triggered by signals,
#   e.g. by clicking a menu item or a tool button.
# Since signals have no arguments:
# Can we use python functions with arguments as actions ???
# - In menus we can have the menuitems send an integer id.
# - For other cases (like toolbuttons), we can subclass QAction and let it send
#   a signal with appropriate arguments 


def NotImplemented():
    draw.warning("This option has not been implemented yet!")

#####################################################################
# Opening, Playing and Saving pyformex scripts

save = NotImplemented
saveAs = NotImplemented

def editor():
    if GD.gui.editor:
        print "Close editor"
        GD.gui.closeEditor()
    else:
        print "Open editor"
        GD.gui.showEditor()

############################################################################

# JUST TESTING:
def userView(i=1):
    if i==1:
        frontView()
    else:
        isoView()

     

def localAxes():
    GD.cfg.gui['localaxes'] = True 
def globalAxes():
    GD.cfg.gui['localaxes'] = False 


#### End
