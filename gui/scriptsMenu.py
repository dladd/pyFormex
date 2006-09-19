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
        self.load(dir)
        

    def load(self,dir):
        self.dir = dir
        files = os.listdir(dir)
        filter1 = lambda s:s[-3:]==".py" and s[0]!='.' and s[0]!='_'
        filter2 = lambda s:utils.isPyFormex(os.path.join(dir,s))
        files = filter(filter1,files)
        files = filter(filter2,files)
        files.sort()
        self.files = map(lambda s:s[:-3],files)
        if GD.options.debug:
            print "Found Examples in %s" % dir
            print self.files
        for f in self.files:
            self.addAction(f)
        self.connect(self,QtCore.SIGNAL("triggered(QAction*)"),self.run)
        self.addSeparator()
        self.addAction('Run next example',self.runNext)
        self.addAction('Run all examples',self.runAll)
        self.addAction('Reload examples',self.reLoad)
        self.current = ""
        

    def run(self,action):
        """Run the selected example."""
        script = str(action.text())
        if script in self.files:
            self.runScript(script)
    

    def runScript(self,filename):
        """Run the specified example."""
        self.current = filename
        selected = os.path.join(self.dir,filename+'.py')
        GD.debug("Playing script %s" % selected)
        GD.gui.setcurfile(selected)
        fileMenu.play()
        

    def runAll(self):
        """Run all examples."""
        print "Running all examples"
        for f in self.files:
            self.runScript(f)


    def runNext(self):
        try:
            i = self.files.index(self.current) + 1
        except ValueError:
            i = 0
        print "This is example %s out of %s" % (i,len(self.files))
        if i < len(self.files):
            self.runScript(self.files[i])


    def reLoad(self):
        pass
    
