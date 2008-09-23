#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Menu with pyFormex scripts."""

import pyformex as GD

from PyQt4 import QtCore, QtGui

import utils
import draw
import os,random
    
catname = 'scripts.cat'

def extractKeyword(s):
    """Extract a keyword =value pair from a string.

    If the string s is of the form
      keyword = value
    a tuple (keyword,value) is returned; else None.
    """
    i = s.find('=')
    if i >= 0:
        try:
            key = s[:i].strip()
            if len(key) > 0:
                return key, eval(s[i+1:].strip())
        except:
            GD.debug("Error processing keywords %s" % s.strip('\n'))
            pass
    return None

    
def scriptKeywords(fn,keyw=None):
    """Read the script keywords from a script file.

    fn is the full path name of a pyFormex script file.
    keyw is an optional list of keywords.
    
    Script keywords are written in the form
      key = value
    in the docstring of the script.
    The docstring is the first non-indented multiline string of the file.
    A multiline string is a string delimited by triple double-quotes.
    Matching lines are placed in a dictionary which becomes the return value.
    
    If a list of keywords is given, the return dictionary will only contain
    the matching values.
    """
    fil = file(fn,'r')
    keys = {}
    ok = False
    for line in fil:
        if not ok and line.startswith('"""'):
            ok = True
            line = line[3:]
        if ok:
            i = line.find('"""')
            if i >= 0:
                line = line[:i]
            pair = extractKeyword(line)
            if pair:
                keys.update((pair,))
            if i >= 0:
                return keys
    return keys


def sortSets(d):
    """Turn the set values in d into sorted lists."""
    for k in d:
        if type(d[k]) == set:
            d[k] = list(d[k])
            d[k].sort()
  


class ScriptsMenu(QtGui.QMenu):
    """A menu of pyFormex scripts in a directory or list."""
    
    def __init__(self,title,dir=None,files=None,ext=None,recursive=None,max=0,autoplay=False,toplevel=True):
        """Create a menu with pyFormex scripts to play.

        dir, files, ext determine the list of script names and files in the
        menu. When all three are specified, files is a list of script file
        names whose full path names are given by dir/filename+ext.

        If no dir is specified, files should be full path names of the scripts
        (possibly only to be extended by a specified extension). Default
        extension is ''.

        If a dir is specified, files should all be relative to that dir.
        If no files are given, all files in that directory whose names end
        with the specifiied extension AND do no start with either '.' or '_'
        will be selected.
        If no extension is given, it defaults to '.py'.
        If files are given including the extension, specify '' to suppress
        the default extension.
        
        Commonly used initialisations are:
        - dir=path: to specify all pyFormex scripts in that directory
        - files=[list of full path names], recursive=False: to specify a list
           of files spread over random directories (e.g. to create a history
           menu of previously played scripts).
    
        If recursive is True (default if dir is specified), a cascading menu
        of all pyFormex scripts in that directory and subdirectories will be
        created.

        By default, files are only included if:
        - the name ends with '.py'
        - the name does not start with '.' or '_'
        - the file is recognized as a pyFormex script by isPyFormex()

        If files is a list, a maximum number of items in the list may be
        specified. If it is > 0, no more than max scripts will be allowed.
        New ones are added on top, while bottom ones will drop off.

        With the resulting files, a menu is created. Selecting a menu item
        will make the corresponding file the current script and, if autoplay
        was set True, the script is executed.

        If only a directory path was specified, extra options will be included
        in the menu:
        - execute all files
        - execute current and all following files
        - execute a random script
        - execute all files in random order
        If the menu is a toplevel, it will furthermore have the extra options
        - close the menu
        - reload the menu
        - classify the scripts according to keywords
        """
        QtGui.QMenu.__init__(self,title)
        self.dir = dir
        self.files = files
        if self.dir is None and self.files is None:
            raise ValueError,"At lest one of 'dir' or 'files' must be set."
        if ext is None:
            if self.dir is None:
                ext = ''
            else:
                ext = '.py'
        self.ext = ext
        if recursive is None:
            recursive = True
        self.recursive = recursive and self.dir is not None
        self.toplevel = toplevel
        self.max = max
        self.autoplay = autoplay
        self.menus = []
        self.load()
        

    def fileName(self,scriptname):
        """Return the full pathname for a scriptname."""
        fn = scriptname + self.ext
        if self.dir:
            return os.path.join(self.dir,fn)
        else:
            return fn


    def loadSubmenus(self,dirs=[]):
        if not dirs:
            dirs = os.listdir(self.dir)
        filtr = lambda s:os.path.isdir(os.path.join(self.dir,s))
        dirs = filter(filtr,dirs)
        filtr = lambda s: s[0]!='.' and s[0]!='_'
        dirs = filter(filtr,dirs)
        dirs.sort()
        for d in dirs:
            m = ScriptsMenu(d,os.path.join(self.dir,d),autoplay=self.autoplay,recursive=self.recursive)
            self.addMenu(m)
            self.menus.append(m)
            

    def getFiles(self):
        """Get a list of scripts in self.dir"""
        files = os.listdir(self.dir)
        filtr = lambda s: s[0]!='.' and s[0]!='_'
        files = filter(filtr,files)
        if self.ext:
            filtr = lambda s: s.endswith(self.ext)
            files = filter(filtr,files)
            n = len(self.ext)
            files = [ f[:-n] for f in files ]

        filtr = lambda s:utils.isPyFormex(self.fileName(s))
        files = filter(filtr,files)

        if self.max > 0 and len(files) > self.max:
            files = files[:self.max]

        files.sort()
        return files

 
    def filterFiles(self,files):
        """Filter a list of scripts"""
        filtr = lambda s:utils.isPyFormex(self.fileName(s))
        files = filter(filtr,files)

        if self.max > 0 and len(files) > self.max:
            files = files[:self.max]

        return files
      

    def loadFiles(self,files=None):
        """Load the script files in this menu"""
        if files is None:
            files = self.getFiles()

        self.files = self.filterFiles(files)
        
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
        self.current = ""


    def loadCatalog(self):
        catfile = os.path.join(self.dir,catname)
        if os.path.exists(catfile):
            execfile(catfile,globals())
            for k in kat:
                if k == 'all':
                    files = col[k]
                else:
                    files = []
                mk = ScriptsMenu(k.capitalize(),dir=self.dir,files=files,recursive=False,toplevel=False,autoplay=self.autoplay)
                for i in cat[k]:
                    ki = '%s/%s' % (k,i)
                    mi = ScriptsMenu(i.capitalize(),dir=self.dir,files=col[ki],recursive=False,toplevel=False,autoplay=self.autoplay)
                    mk.addMenu(mi)
                    mk.menus.append(mi)
                self.addMenu(mk)
                self.menus.append(mk)
            self.files = []
            return True
        return False


    def load(self):
        if self.dir is None:
            self.loadFiles(self.files)

        else:
            if self.files is None:
                self.loadCatalog()

            if self.recursive:
                self.loadSubmenus()

            if self.files or self.files is None:
                self.loadFiles(self.files)

            if self.toplevel:
                self.addAction('Classify scripts',self._classify)
                self.addAction('Remove catalog',self._unclassify)
                self.addAction('Reload scripts',self.reload)



    def run(self,action):
        """Run the selected script."""
        script = str(action.text())
        if script in self.files:
            self.runScript(script)
    

    def runScript(self,filename):
        """Run the specified script."""
        self.current = filename
        selected = self.fileName(filename)
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
        self.runAllFiles(self.files)
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
                       

    def reload(self):
        """Reload the scripts from dir.

        This is only available if a directory path was specified and
        no files.
        """
        GD.debug("RELOADING THIS MENU")
        if self.dir:
            self.clear()
            self.menus = []
            self.files = None
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


    def classify(self):
        """Classify the files in submenus according to keywords."""
        kat = ['all','level','topics','techniques']
        cat = dict([ (k,set()) for k in kat])
        cat['level'] = [ 'beginner', 'normal', 'advanced' ]
        col = {'all':set()}
        for f in self.filterFiles(self.getFiles()):
            col['all'].update([f])
            fn = self.fileName(f)
            d = scriptKeywords(fn)
            for k,v in d.items():
                if not k in kat:
                    GD.debug("Skipping unknown keyword %s in script %s" % (k,fn))
                    continue
                if k == 'level':
                    v = [v]
                else:
                    cat[k].update(v)
                for i in v:
                    ki = '%s/%s' % (k,i)
                    if not ki in col.keys():
                        col[ki] = set()
                    col[ki].update([f])

        sortSets(cat)
        sortSets(col)
            
        return kat,cat,col


    def _classify(self):
        """Classify, symlink and reload the scripts"""
        if self.dir:
            f = os.path.join(self.dir,catname)
            s = "kat = %r\ncat = %r\ncol = %r\n" % self.classify()
            file(f,'w').writelines(s)
            self.reload()


    def _unclassify(self):
        """Remove the catalog and reload the scripts unclassified"""
        if self.dir:
            f = os.path.join(self.dir,catname)
            if os.path.exists(f):
                os.remove(f)
                self.reload()
            
# End
