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
    

def AddMenuItems(menu, items=[]):
    """Add a list of items to a menu.

    Each item is a tuple of three strings : Type, Text, Value.
    Type specifies the menu item type and must be one of
    'Sep', 'Popup', 'Action', 'VAction', 'QAction'.
    
    'Sep' is a separator item. Its Text and Value fields are not used.
    
    For the other types, Text is the string that will be displayed in the
    menu. It can include a '&' character to flag the hotkey.
    
    'Popup' is a popup submenu item. Its value should be an item list,
    defining the menu to pop up when activated.
    
    'Action' is an active item. Its value is a python function that will be
    executed when the item is activated. It should be a global function
    without arguments.
    
    'VAction' is an active item where the value is a tuple of an function
    and an integer argument. When activated, the function will be executed
    with the specified argument. With 'Vaction', you can bind multiple
    menu items to the same function.

    'QAction' signals that the value is a qt QAction. It is advisable to
    construct a QAction if you have to use the sameaction in more than one
    place of your program. With this type Text may be None, if it was already
    set in the QAction itself.
    """
    for key,txt,val in items:
        if key == "Sep":
            menu.insertSeparator()
        elif key == "Popup":
            pop = qt.QPopupMenu(menu,txt)
            AddMenuItems(pop,val)
            menu.insertItem(txt,pop)
        elif key == "Action":
            menu.insertItem(txt,eval(val))
        elif key == "VAction":
            id = menu.insertItem(txt,eval(val[0]))
            menu.setItemParameter(id,val[1])
        elif key == "SAction":
            menu.insertItem(txt,val)
        elif key == "QAction":
            if txt:
                val.setProperty("menuText",qt.QVariant(txt))
            val.addTo(menu)
        else:
            raise RuntimeError, "Invalid key %s in menu item"%key
#
# Using ("SAction","text",foo) is almost equivalent to
# using ("Action","text","foo"), but the latter allows for function
# that have not been defined yet in this scope!
#
MenuData = [
    ("Popup","&File",[
        ("Action","&Open","openFile"),
        ("Action","&Play","play"),
        ("Action","&Edit","edit"),
#        ("Action","&Save","save"),
#        ("Action","Save &As","saveAs"),
        ("Sep",None,None),
        ("Action","Save &Image","saveImage"),
        ("Action","Toggle &MultiSave","multiSave"),
        ("Sep",None,None),
        ("Action","E&xit","draw.exit"), ]),
    ("Popup","&Settings",[
#        ("Action","&Preferences","preferences"), 
        ("Action","Show &Triade","draw.drawTriade"), 
        ("Action","&Drawwait Timeout","prefDrawtimeout"), 
        ("Action","&Background Color","prefBGcolor"), 
        ("Action","Line&Width","prefLinewidth"), 
        ("Action","&Canvas Size","prefCanvasSize"), 
        ("Action","&LocalAxes","localAxes"),
        ("Action","&GlobalAxes","globalAxes"),
        ("Action","&Wireframe","draw.wireframe"),
        ("Action","&Flat","draw.flat"),
        ("Action","&Smooth","draw.smooth"),
        ("Action","&Render","prefRender"),
        ("Action","&Light0","prefLight0"),
        ("Action","&Light1","prefLight1"),
        ("Action","&Help","prefHelp"),
        ("Action","&Save Preferences","draw.savePreferences"), ]),
    ("Popup","&Camera",[
        ("Action","&Zoom In","zoomIn"), 
        ("Action","&Zoom Out","zoomOut"), 
        ("Action","&Dolly In","dollyIn"), 
        ("Action","&Dolly Out","dollyOut"), 
        ("Action","Pan &Right","transRight"), 
        ("Action","Pan &Left","transLeft"), 
        ("Action","Pan &Up","transUp"),
        ("Action","Pan &Down","transDown"),
        ("Action","Rotate &Right","rotRight"),
        ("Action","Rotate &Left","rotLeft"),
        ("Action","Rotate &Up","rotUp"),
        ("Action","Rotate &Down","rotDown"),  ]),
    ("Popup","&Actions",[
        ("Action","&Step","draw.step"),
        ("Action","&Continue","draw.fforward"), 
        ("Action","&Clear","draw.clear"),
        ("Action","&Redraw","draw.redraw"),
        ("Action","&DrawSelected","draw.drawSelected"),
        ("Action","&ListFormices","draw.printall"),
        ("Action","&PrintGlobals","draw.printglobals"),  ]),
    ("Popup","&Help",[
        ("Action","&Help","dohelp"),
        ("Action","&About","help.about"), 
        ("Action","&Warning","help.testwarning"), ]) ]

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

def addDefaultMenu(menu):
    """Add the default menu structure to the GUI."""
    AddMenuItems(menu,MenuData)

def runExample(i):
    """Run example i from the list of found examples."""
    global example
    gui.setcurfile(os.path.join(GD.cfg.exampledir,example[i]))
    play()

def runExamples(n):
    """Run the first n examples."""
    for i in range(n):
        runExample(i)

# add action buttons to toolbar
def addActionButtons(toolbar):
    global action
    action = {}
    dir = GD.cfg.icondir
    buttons = [ [ "Play", "next.xbm", play, False ],
                [ "Step", "nextstop.xbm", draw.step, False ],
                [ "Continue", "ff.xbm", draw.fforward, False ],
              ]
    for b in buttons:
        a = qt.QAction(b[0],qt.QIconSet(qt.QPixmap(os.path.join(dir,b[1]))),b[1],0,toolbar)
        qt.QObject.connect(a,qt.SIGNAL("activated()"),b[2])
        a.addTo(toolbar)
        a.setEnabled(b[3])
        action[b[0]] = a
    return action

# add camera buttons to toolbar (repeating)
def addCameraButtons(toolbar):
    dir = GD.cfg['icondir']
    buttons = [ [ "Rotate left", "rotleft.xbm", rotLeft ],
                [ "Rotate right", "rotright.xbm", rotRight ],
                [ "Rotate up", "rotup.xbm", rotUp ],
                [ "Rotate down", "rotdown.xbm", rotDown ],
                [ "Twist left", "twistleft.xbm", twistLeft ],
                [ "Twist right", "twistright.xbm", twistRight ],
                [ "Translate left", "left.xbm", transLeft ],
                [ "Translate right", "right.xbm", transRight ],
                [ "Translate down", "down.xbm", transDown ],
                [ "Translate up", "up.xbm", transUp ],
                [ "Zoom In", "zoomin.xbm", zoomIn ],
                [ "Zoom Out", "zoomout.xbm", zoomOut ],  ]
    for b in buttons:
        w = qt.QToolButton(qt.QIconSet(qt.QPixmap(os.path.join(dir,b[1]))),b[0],"",b[2],toolbar)
        w.setAutoRepeat(True)



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


##def openFile():
##    """Open a file and set it as the current file"""
##    dir = GD.cfg.get('workdir',".")
##    fs = widgets.FileSelectionDialog(dir,"pyformex scripts (*.frm *.py)")
##    fn = fs.getFilename()
##    if fn:
##        GD.cfg['workdir'] = os.path.dirname(fn)
##        gui.setcurfile(fn)

def openFile():
    """Open a file and set it as the current file"""
    cur = GD.cfg.get('curfile',GD.cfg.get('workdir','.'))
    fn = qt.QFileDialog.getSaveFileName(
        cur,"pyformex scripts (*.frm *.py)",None,"Open file dialog",
        "Choose a file to open (New or Existing)" )
    if fn:
        fn = str(fn)
        GD.cfg['workdir'] = os.path.dirname(fn)
        gui.setcurfile(fn)
        
def edit():
    """Load the current file in the editor.

    This only works if the editor was set in the configuration.
    The author uses 'gnuclient' to load the files in a running copy
    of xemacs.
    """
    if GD.cfg['edit']:
        cmd = GD.cfg['edit']
        pid = os.spawnlp(os.P_NOWAIT, cmd, cmd, GD.cfg['curfile'])
        draw.log("Spawned %d" % pid)


def play():
    """Play the current file.

    This only does something if the current file is a pyFormex script.
    """
    if GD.canPlay:
        draw.play(GD.cfg['curfile'])

    
def saveImage():
    """Save the current rendering in image format.

    This function will open a file selection dialog, and if a valid
    file is returned, the current OpenGL rendering will be saved to it.
    """
    global canvas
    dir = GD.cfg.get('workdir',".")
    fs = widgets.FileSelectionDialog(dir,pattern="Images (*.png *.jpg)",mode=qt.QFileDialog.AnyFile)
    fn = fs.getFilename()
    if fn:
        GD.cfg['workdir'] = os.path.dirname(fn)
        draw.saveImage(fn,verbose=True)

def multiSave():
    """Save a sequence of images.

    If the filename supplied has a trailing numeric part, subsequent images
    will be numbered continuing from this number. Otherwise a numeric part
    -000, -001, will be added to the filename.
    """
    if draw.multisave:
        fn = None
    else:
        dir = GD.cfg.get('workdir',".")
        fs = widgets.FileSelectionDialog(dir,pattern="Images (*.png *.jpg)",mode=qt.QFileDialog.AnyFile)
        fn = fs.getFilename()
    draw.saveMulti(fn,verbose=True)


############################################################################
            

def zoomIn():
    global canvas
    GD.canvas.zoom(1./GD.cfg.gui['zoomfactor'])
    GD.canvas.update()
def zoomOut():
    global canvas
    GD.canvas.zoom(GD.cfg.gui['zoomfactor'])
    GD.canvas.update()
##def panRight():
##    global canvas,config
##    canvas.camera.pan(+5)
##    canvas.update()   
##def panLeft():
##    global canvas,config
##    canvas.camera.pan(-5)
##    canvas.update()   
##def panUp():
##    global canvas,config
##    canvas.camera.pan(+5,0)
##    canvas.update()   
##def panDown():
##    global canvas,config
##    canvas.camera.pan(-5,0)
##    canvas.update()   
def rotRight():
    global canvas
    GD.canvas.camera.rotate(+GD.cfg.gui['rotfactor'],0,1,0)
    GD.canvas.update()   
def rotLeft():
    global canvas
    GD.canvas.camera.rotate(-GD.cfg.gui['rotfactor'],0,1,0)
    GD.canvas.update()   
def rotUp():
    global canvas
    GD.canvas.camera.rotate(-GD.cfg.gui['rotfactor'],1,0,0)
    GD.canvas.update()   
def rotDown():
    global canvas
    GD.canvas.camera.rotate(+GD.cfg.gui['rotfactor'],1,0,0)
    GD.canvas.update()   
def twistLeft():
    global canvas
    GD.canvas.camera.rotate(+GD.cfg.gui['rotfactor'],0,0,1)
    GD.canvas.update()   
def twistRight():
    global canvas
    GD.canvas.camera.rotate(-GD.cfg.gui['rotfactor'],0,0,1)
    GD.canvas.update()   
def transLeft():
    global canvas
    GD.canvas.camera.translate(-GD.cfg.gui['panfactor'],0,0,GD.cfg.gui['localaxes'])
    GD.canvas.update()   
def transRight():
    global canvas
    GD.canvas.camera.translate(GD.cfg.gui['panfactor'],0,0,GD.cfg.gui['localaxes'])
    GD.canvas.update()   
def transDown():
    global canvas
    GD.canvas.camera.translate(0,-GD.cfg.gui['panfactor'],0,GD.cfg.gui['localaxes'])
    GD.canvas.update()   
def transUp():
    global canvas
    GD.canvas.camera.translate(0,GD.cfg.gui['panfactor'],0,GD.cfg.gui['localaxes'])
    GD.canvas.update()   
def dollyIn():
    global canvas
    GD.canvas.camera.dolly(1./GD.cfg.gui['zoomfactor'])
    GD.canvas.update()   
def dollyOut():
    global canvas
    GD.canvas.camera.dolly(GD.cfg.gui['zoomfactor'])
    GD.canvas.update()   

def frontView():
    view("front");
def backView():
    view("back");
def leftView():
    view("left");
def rightView():
    view("right");
def topView():
    view("top");
def bottomView():
    view("bottom");
def isoView():
    view("iso");
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


## We need this because the menufunctions take an argument and the
## help.help function has a default argument
##
def dohelp():
    help.help()


#### End
