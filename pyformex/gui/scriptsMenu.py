#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Menu with pyFormex scripts."""

import globaldata as GD

from PyQt4 import QtCore, QtGui

import utils
import draw
import os,random
    

class ScriptsMenu(QtGui.QMenu):
    """A menu of pyFormex scripts in a directory or list."""
    
    def __init__(self,title,files,max=0,autoplay=False,recursive=True):
        """Create a menu with pyFormex scripts to play.

        files is either a list of files (including path) to insert in the
        menu, or a directory path from which the files will be autoloaded.
        
        If it is a list of filenames, their paths may be different. This is
        e.g. used to create a history of previously played scripts
        
        If files is a directory path, a cascading menu of all pyFormex scripts
        in that directory and subdirectories will be created.

        By default, files are only included if:
          - the name ends with '.py'
          - the name does not start with '.' or '_'
          - the file is recognized as a pyFormex script by isPyFormex()

        If files is a directory path, extra options will be included to
        play all scripts, the next script or all following scripts or to
        reload the scripts from that directory.

        If files is a list, a maximum number of items in the list may be
        specified. If it is > 0, no more than max scripts will be allowed.
        New ones are added on top, while bottom ones will drop off.

        Selecting a menu item will make the corresponding file the current
        script and, if autoplay was set True, the script is executed.
        
        The menu will also contain some extra options:
        - execute all files
        - execute current and all following files
        - close the menu
        - reload the menu
        """
        QtGui.QMenu.__init__(self,title)
        self.dir = None
        self.files = None
        if type(files) == str:
            self.dir = files
        else:
            self.files = files
        self.max = max
        self.autoplay = autoplay
        self.recursive = recursive
        self.menus = []
        self.load()


    def loadSubmenus(self,dirs):
        filter2 = lambda s:os.path.isdir(os.path.join(self.dir,s))
        dirs = filter(filter2,dirs)
        dirs.sort()
        for d in dirs:
            m = ScriptsMenu(d,os.path.join(self.dir,d),autoplay=self.autoplay,recursive=self.recursive)
            self.addMenu(m)
            self.menus.append(m)
            

    def load(self):
        if self.dir:
            files = os.listdir(self.dir)
        else:
            files = self.files
        filter1 = lambda s: s[0]!='.' and s[0]!='_'
        files = filter(filter1,files)
        
        filter1 = lambda s: s[-3:]==".py"
        if self.dir:
            if self.recursive:
                self.loadSubmenus(files)
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
            self.addAction('Run a random script',self.runRandom)
            self.addAction('Run all in random order',self.runAllRandom)
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


    def runNext(self):
        """Run the next script."""
        try:
            i = self.files.index(self.current) + 1
        except ValueError:
            i = 0
        GD.debug("This is script %s out of %s" % (i,len(self.files)))
        if i < len(self.files):
            self.runScript(self.files[i])


    def runAllNext(self):
        """Run the current and all following scripts."""
        try:
            i = self.files.index(self.current)
        except ValueError:
            i = 0
        GD.debug("Running scripts %s-%s" % (i,len(self.files)))
        self.runAllFiles(self.files[i:])
        GD.debug("Exiting runAllNext")
        

    def runAll(self):
        """Run all scripts."""
        GD.debug("Playing all scripts in order")
        self.runAllFile(self.files)
        GD.debug("Finished playing all scripts")


    ### THIS should be moved to a playAll function in draw/script module
    def runAllFiles(self,files,randomize=False):
        """Run all the scripts in given list."""
        GD.gui.actions['Stop'].setEnabled(True)
        if randomize:
            random.shuffle(files)
        for f in files:
            draw.layout(1)
            self.runScript(f)
            #GD.debug("draw.exitrequested == %s" % draw.exitrequested)
            if draw.exitrequested:
                break
        GD.gui.actions['Stop'].setEnabled(False)


    def runRandom(self):
        """Run a random script."""
        i = random.randint(0,len(self.files)-1)
        self.runScript(self.files[i])


    def runAllRandom(self):
        """Run all scripts in a random order."""
        GD.debug("Playing all scripts in random order")
        self.runAllFiles(self.files,randomize=True)
        GD.debug("Finished playing all scripts")
                       

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


# End
