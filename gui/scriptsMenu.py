#!/usr/bin/env python
# $Id$
"""Menu with pyFormex scripts."""

import os
import globaldata as GD
import menu, utils, fileMenu
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
        selected = os.path.join(self.dir,str(action.text()))+'.py'
        if GD.options.debug:
            print "Playing ",selected
        GD.gui.setcurfile(selected)
        fileMenu.play()
        
