#!/usr/bin/env python
# $Id$
"""Menu with pyFormex scripts."""

import os
import globaldata as GD
import menu, utils, fileMenu
from PyQt4 import QtCore, QtGui


class ScriptsMenu(QtGui.QMenu):
    """A menu of pyFormex scripts in a directory."""
    
    def __init__(self,title,dir):
        """Create a menu with files in dir. 
        
        By default, files from dir are only included if:
          - the name ends with '.py'
          - the name does not start with '.' or '_'
          - the file is recognized as a pyFormex script by isPyFormex()
        
        An option to reload the directory is always included.
        """
        QtGui.QMenu.__init__(self,title)
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
            print "Found Scripts in %s" % dir
            print self.files
        for f in self.files:
            self.addAction(f)
        self.connect(self,QtCore.SIGNAL("triggered(QAction*)"),self.run)
        self.addSeparator()
        self.addAction('Run next script',self.runNext)
        self.addAction('Run all following scripts',self.runAllNext)
        self.addAction('Run all scripts',self.runAll)
        self.addAction('Reload scripts',self.reLoad)
        self.current = ""
        

    def run(self,action):
        """Run the selected script."""
        script = str(action.text())
        if script in self.files:
            self.runScript(script)
    

    def runScript(self,filename):
        """Run the specified script."""
        self.current = filename
        selected = os.path.join(self.dir,filename+'.py')
        GD.debug("Playing script %s" % selected)
        GD.gui.setcurfile(selected)
        fileMenu.play()
        

    def runAll(self):
        """Run all scripts."""
        print "Running all scripts"
        for f in self.files:
            self.runScript(f)


    def runNext(self):
        try:
            i = self.files.index(self.current) + 1
        except ValueError:
            i = 0
        print "This is script %s out of %s" % (i,len(self.files))
        if i < len(self.files):
            self.runScript(self.files[i])


    def runAllNext(self):
        try:
            i = self.files.index(self.current)
        except ValueError:
            i = 0
        print "Running scripts %s-%s" % (i,len(self.files))
        for f in self.files[i:]:
            self.runScript(f)


    def reLoad(self):
        self.clear()
        self.load(self.dir)
        
    
