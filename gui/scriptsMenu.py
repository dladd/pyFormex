#!/usr/bin/env python
# $Id $
"""Menu with pyFormex scripts."""

import os
import globaldata as GD
import menu, utils, gui, fileMenu
from PyQt4 import QtCore, QtGui


class ScriptsMenu(QtGui.QMenu):
    """A menu of pyFormex scripts in a directory."""
    
    def __init__(self,dir):
        """Create a menu with files in dir. 
        
        By default, files from dir are only included if:
          - the name ends with '.py'
          - the name does not start with '.' or '_'
          - the file is recognized as a pyFormex script by isPyFormex()
        
        An option to reload the directory is always included.
        """
        QtGui.QMenu.__init__(self,'Examples')
        self.dir = dir
        files = os.listdir(dir)
        filter1 = lambda s:s[-3:]==".py" and s[0]!='.' and s[0]!='_'
        filter2 = lambda s:utils.isPyFormex(os.path.join(dir,s))
        files = filter(filter1,files)
        files = filter(filter2,files)
        files.sort()
        if GD.options.debug:
            print "Found Examples in %s" % dir
            print files
        for f in files:
            self.addAction(os.path.splitext(f)[0])
        self.connect(self,QtCore.SIGNAL("triggered(QAction*)"),self.run)
    

    def run(self,action):
        """Run the selected example."""
        selected = os.path.join(GD.cfg.exampledir,str(action.text()))+'.py'
        print selected
        gui.setcurfile(selected)
        fileMenu.play()
        
        
### Examples Menu
##def insertExampleMenu(dir,menu,pos):
##    """Insert an examples menu before pos in the given menu.

##    Examples are all the .py files in the subdirectory examples,
##    provided there name does not start with a '.' or '_' and
##    their first line ends with 'pyformex'
##    """
##    if not os.path.isdir(dir):
##        return None
##    example = filter(lambda s:s[-3:]==".py" and s[0]!='.' and s[0]!='_',os.listdir(dir))
##    example = filter(lambda s:utils.isPyFormex(os.path.join(GD.cfg.exampledir,s)),example)
##    example.sort()
##    if GD.options.debug:
##        print "Found Examples in %s", %dir
##        print example
##    vm = ("Popup","&Examples",[
##        ("VAction","&%s"%os.path.splitext(t)[0],("runExample",i)) for i,t in enumerate(example)
##        ])
##    nEx = len(vm[2])
##    vm[2].append(("VAction","Run All Examples",("runExamples",nEx)))
##    MenuData.insert(4,vm)

##def runExample(i):
##    """Run example i from the list of found examples."""
##    global example
##    gui.setcurfile(os.path.join(GD.cfg.exampledir,example[i]))
##    play()

##def runExamples(n):
##    """Run the first n examples."""
##    for i in range(n):
##        runExample(i)

##    for key,txt,val in items:
##        if key == "Sep":
##            menu.addSeparator()
##        elif key == "Popup":
##            pop = QtGui.QMenu(txt,menu)
##            addMenuItems(pop,val)
##            menu.addMenu(pop)
##        elif key == "Action":
##            menu.addAction(txt,eval(val))
