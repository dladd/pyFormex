#!/usr/bin/env python
# $Id $

import globaldata as GD
import gui
from utils import *
import draw
import widgets
import canvas

import sys,time,os,string,commands
import qt
import qtgl

def askConfigPreferences(items,section=None):
    """Ask preferences stored in config variables.

    Items in list should not have the value.
    """
    # insert current values
    for it in items:
        it.insert(1,GD.cfg.setdefault(it[0],''))
    print "Asking Prefs ",items
    res = widgets.ConfigDialog(items).process()
    for r in res:
        print r
        GD.cfg[r[0]] = r[1]
    print GD.cfg

def prefDrawtimeout():
    askConfigPreferences([['drawtimeout','int']])

def prefBGcolor():
    #askConfigPreferences([['bgcolor']])
    #draw.bgcolor(GD.cfg['bgcolor'])
    col = qt.QColorDialog.getColor(qt.QColor(GD.cfg.setdefault('bgcolor','')))
    if col.isValid():
        GD.cfg['bgcolor'] = col.name()
        draw.bgcolor(col)
        
def prefLinewidth():
    askConfigPreferences([['linewidth']])
    draw.linewidth(GD.cfg['linewidth'])

def prefSize():
    GD.gui.resize(800,600)

def preferences():
    test = [["item1","value1"],
            ["item2",""],
            ["an item with a long name","and a very long value for this item, just to test the scrolling"]]
    res = widgets.ConfigDialog(test).process()
    print res

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
        ("Action","&Edit","editor"),
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
        ("Action","&Canvas Size","prefSize"), 
        ("Action","&LocalAxes","localAxes"),
        ("Action","&GlobalAxes","globalAxes"),
        ("Action","&Wireframe","draw.wireframe"),
        ("Action","&Flat","draw.flat"),
        ("Action","&Smooth","draw.smooth"), ]),
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
        ("Action","&ListAll","draw.listall"),
#        ("Action","&Print","printit"),
#        ("Action","&Bbox","printbbox"),
        ("Action","&Globals","draw.printglobals"),  ]),
    ("Popup","&Help",[
        ("Action","&Help","showHelp"),
        ("Action","&About","about"), 
        ("Action","&Warning","testwarning"), ]) ]

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
    draw.playFile(os.path.join(GD.cfg.exampledir,example[i]))

def runExamples(n):
    """Run the first n examples."""
    for i in range(n):
        runExample(i)

# add action buttons to toolbar
def addActionButtons(toolbar):
    global action
    action = {}
    dir = GD.cfg.icondir
    buttons = [ [ "Step", "next.xbm", draw.step, False ],
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

# open a pyformex script
def openFile():
    dir = GD.cfg.get('workdir',".")
    fs = widgets.FileSelectionDialog(dir,"pyformex scripts (*.frm *.py)")
    fn = fs.getFilename()
    if fn:
        GD.cfg['workdir'] = os.path.dirname(fn)
        GD.cfg['curfile'] = fn
        if GD.cfg['edit']:
            cmd = GD.cfg['edit']
            print cmd
            print fn
            pid = os.spawnlp(os.P_NOWAIT, cmd, cmd, fn)
            print "Spawned %d" % pid

# play the current file
def play():
    if GD.cfg['curfile']:
        draw.playFile(GD.cfg['curfile'])

    

multisave = None

def saveNext():
    global multisave
    name,nr,fmt = multisave
    nr += 1
    multisave = [ name,nr,fmt ]
    GD.canvas.save(name % nr,fmt)


def multiSave():
    """Save a sequence of images.

    If the filename supplied has a trailing numeric part, subsequent images
    will be numbered continuing from this number. Otherwise a numeric part
    -000, -001, will be added to the filename.
    """
    global canvas,multisave
    if multisave:
        print "Stop auto mode"
        qt.QObject.disconnect(GD.canvas,qt.PYSIGNAL("save"),saveNext)
        multisave = None
        return
    
    fs = widgets.FileSelectionDialog(pattern="Images (*.png *.jpg)",mode=qt.QFileDialog.AnyFile)
    fn = fs.getFilename()
    if fn:
        print "Start auto mode"
        name,ext = os.path.splitext(fn)
        fmt = imageFormatFromExt(ext)
        print name,ext,fmt
        if fmt in qt.QImage.outputFormats():
            name,number = splitEndDigits(name)
            if len(number) > 0:
                nr = int(number)
                name += "%%0%dd" % len(number)
            else:
                nr = 0
                name += "-%03d"
            if len(ext) == 0:
                ext = '.%s' % fmt.lower()
            name += ext
            draw.warning("Each time you hit the 'S' key,\nthe image will be saved to the next number.")
            qt.QObject.connect(GD.canvas,qt.PYSIGNAL("save"),saveNext)
            multisave = [ name,nr,fmt ]
        else:
            draw.warning("Sorry, can not save in %s format!\n"
                    "Suggest you use PNG format ;)"%fmt)


############################################################################
# online help system

global help
help = None
def showHelp():
    """Start up the help browser"""
    global help
    from helpviewer import HelpViewer
    print "help = ",help
    if help == None:
        dir = os.path.join(GD.cfg['docdir'],"html")
        home = os.path.join(dir,"formex.html")
        print "Help file = ",home
        help = HelpViewer(home, dir,bookfile=GD.cfg['helpbookmarks'])
        help.setCaption("pyFormex - Helpviewer")
        help.setAbout("pyFormex Help", \
                  "This is the pyFormex HelpViewer.<p>It was modeled after the HelpViewer example " \
                  "from the Qt documentation.</p>")
        #help.resize(800,600)
        help.connect(help,qt.SIGNAL("destroyed()"),closeHelp)
    help.show()

def closeHelp():
    """Close the help browser"""
    global help
    help = None
        

def about():
    about = qt.QMessageBox()
    about.about(about,"About pyFormex",
        GD.Version+"\n\n"
        "pyFormex is a python implementation of Formex algebra\n\n"
        "http://pyformex.berlios.de\n\n"
        "Copyright 2004 Benedict Verhegghe\n"
        "Distributed under the GNU General Public License.\n\n"
        "For help or information, mailto benedict.verhegghe@ugent.be\n" )

def testwarning():
    draw.warning("Smoking may be hazardous to your health!\nWindows is a virus!")

    
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
        ext = os.path.splitext(fn)[1]
        fmt = imageFormatFromExt(ext)
        if fmt in GD.image_formats_qt + GD.image_formats_gl2ps:
            if len(ext) == 0:
                ext = '.%s' % fmt.lower()
                fn += ext
            if fmt == 'TEX':
                draw.warning("This will only write a LaTeX fragment to include the image\n%s\nYou still have to create the .EPS format image separately.\n"
                             % fn.replace(ext,'.eps'))
            GD.canvas.save(fn,fmt)
        else:
            draw.warning("Sorry, can not save in %s format!\n"
                    "Suggest you use PNG format ;)"%fmt)

def zoomIn():
    global canvas
    GD.canvas.zoom(1./GD.cfg['zoomfactor'])
    GD.canvas.update()
def zoomOut():
    global canvas
    GD.canvas.zoom(GD.cfg['zoomfactor'])
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
    GD.canvas.camera.rotate(+GD.cfg['rotfactor'],0,1,0)
    GD.canvas.update()   
def rotLeft():
    global canvas
    GD.canvas.camera.rotate(-GD.cfg['rotfactor'],0,1,0)
    GD.canvas.update()   
def rotUp():
    global canvas
    GD.canvas.camera.rotate(-GD.cfg['rotfactor'],1,0,0)
    GD.canvas.update()   
def rotDown():
    global canvas
    GD.canvas.camera.rotate(+GD.cfg['rotfactor'],1,0,0)
    GD.canvas.update()   
def twistLeft():
    global canvas
    GD.canvas.camera.rotate(+GD.cfg['rotfactor'],0,0,1)
    GD.canvas.update()   
def twistRight():
    global canvas
    GD.canvas.camera.rotate(-GD.cfg['rotfactor'],0,0,1)
    GD.canvas.update()   
def transLeft():
    global canvas
    GD.canvas.camera.translate(-GD.cfg['panfactor'],0,0,GD.cfg['localaxes'])
    GD.canvas.update()   
def transRight():
    global canvas
    GD.canvas.camera.translate(GD.cfg['panfactor'],0,0,GD.cfg['localaxes'])
    GD.canvas.update()   
def transDown():
    global canvas
    GD.canvas.camera.translate(0,-GD.cfg['panfactor'],0,GD.cfg['localaxes'])
    GD.canvas.update()   
def transUp():
    global canvas
    GD.canvas.camera.translate(0,GD.cfg['panfactor'],0,GD.cfg['localaxes'])
    GD.canvas.update()   
def dollyIn():
    global canvas
    GD.canvas.camera.dolly(1./GD.cfg['zoomfactor'])
    GD.canvas.update()   
def dollyOut():
    global canvas
    GD.canvas.camera.dolly(GD.cfg['zoomfactor'])
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
    GD.cfg['localaxes'] = True 
def globalAxes():
    GD.cfg['localaxes'] = False 


def system(cmdline,result='output'):
    if result == 'status':
        return os.system(cmdline)
    elif result == 'output':
        return commands.getoutput(cmdline)
    elif result == 'both':
        return commands.getstatusoutput(cmdline)


#### End
