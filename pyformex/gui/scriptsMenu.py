#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.5 Release Fri Aug 10 12:04:07 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Menu with pyFormex scripts."""

import os
import globaldata as GD

from PyQt4 import QtCore, QtGui

import utils
import draw
    

class ScriptsMenu(QtGui.QMenu):
    """A menu of pyFormex scripts in a directory or list."""
    
    def __init__(self,title,dir=None,files=None,max=0,autoplay=False):
        """Create a menu with files in dir.

        If dir is a directory, all files in that directory are used.
        If dir is a list, this list of files is used
        By default, files from dir are only included if:
          - the name ends with '.py'
          - the name does not start with '.' or '_'
          - the file is recognized as a pyFormex script by isPyFormex()
        
        An option to reload the directory is always included.
        """
        QtGui.QMenu.__init__(self,title)
        self.dir = dir
        self.files = files
        self.max = max
        self.autoplay = autoplay
        self.load()
        

    def load(self):
        if self.dir:
            files = os.listdir(self.dir)
        else:
            files = self.files
        filter1 = lambda s:s[-3:]==".py" and s[0]!='.' and s[0]!='_'
        if self.dir:
            filter2 = lambda s:utils.isPyFormex(os.path.join(self.dir,s))
        else:
            filter2 = utils.isPyFormex
        files = filter(filter1,files)
        files = filter(filter2,files)
        if self.dir:
            files.sort()
            files = map(lambda s:s[:-3],files)
        else:
            if self.max > 0 and len(files) > self.max:
                files = files[:self.max]
        self.files = files
        if GD.options.debug:
            print "Found Scripts in %s" % self.dir
            print self.files
        self.actions = [ self.addAction(f) for f in self.files ]           
        self.connect(self,QtCore.SIGNAL("triggered(QAction*)"),self.run)
        if self.dir:
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
        if self.dir:
            selected = os.path.join(self.dir,filename+'.py')
        else:
            selected = filename
        GD.debug("Playing script %s" % selected)
        GD.gui.setcurfile(selected)
        if self.autoplay:
            draw.play()
        

    def runAll(self):
        """Run all scripts."""
        self.current = ""
        self.runAllNext()


    def runNext(self):
        try:
            i = self.files.index(self.current) + 1
        except ValueError:
            i = 0
        GD.debug("This is script %s out of %s" % (i,len(self.files)))
        if i < len(self.files):
            self.runScript(self.files[i])


    def runAllNext(self):
        try:
            i = self.files.index(self.current)
        except ValueError:
            i = 0
        GD.debug("Running scripts %s-%s" % (i,len(self.files)))
        GD.gui.actions['Stop'].setEnabled(True)
        for f in self.files[i:]:
            self.runScript(f)
            GD.debug("draw.exitrequested == %s" % draw.exitrequested)
            if draw.exitrequested:
                break
        GD.gui.actions['Stop'].setEnabled(False)
        GD.debug("Exiting runAllNext")


    def reLoad(self):
        """Reload the scripts from dir."""
        if self.dir:
            self.clear()
            self.load()


    def add(self,filename):
        """Add a new filename to the front of the menu."""
        files = self.files
        if filename in files:
            files.remove(filename)
        files[0:0] = [ filename ]
        if self.max > 0 and len(files) > self.max:
            files = files[:self.max]
        while len(self.actions) < len(files):
            self.actions.append(self.addAction(filename))
        for a,f in zip(self.actions,self.files):
            a.setText(f)
        
