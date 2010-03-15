#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""Menu with pyFormex scripts."""

import pyformex as GD

from PyQt4 import QtCore, QtGui

import utils
import draw
import menu
import os,random
from gettext import gettext as _
    
catname = 'scripts.cat'

def extractKeyword(s):
    """Extract a ``keyword = value`` pair from a string.

    If the input string `s` is of the form ``keyword = value``
    a tuple (keyword,value) is returned, else None.
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

    - `fn`: the full path name of a pyFormex script file.
    - `keyw`: an optional list of keywords.
    
    Script keywords are written in the form::

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
    """Turn the set values in d into sorted lists.

    - `d`: a Python dictionary

    All the values in the dictionary are checked. Those that are of type
    `set` are converted to a sorted list.
    """
    for k in d:
        if type(d[k]) == set:
            d[k] = list(d[k])
            d[k].sort()
  


class ScriptMenu(QtGui.QMenu):
    """A menu of pyFormex scripts in a directory or list.

    This class creates a menu of pyFormex scripts collected from a directory
    or specified in a list. It is e.g. used in the pyFormex GUI to create
    the examples menu, and for the scripts history. The pyFormex scripts
    can then be executed from the menu. The user may use this class to add
    his own scripts into the pyFormex GUI.

    Only files that are recognized by :func:`utils.isPyFormex()` as being
    pyFormex scripts will be added to the menu. 

    The constructor takes the following arguments:

    - `title`: the top level label for the menu
    - `files`: a list of file names of pyFormex scripts. If no `dir` nor `ext`
      arguments are given, these should be the full path names to the script
      files. If omitted, all files in the directory `dir` whose name is ending
      with `ext` *and do not start with either '.' or '_'*, will be selected.
    - `dir`: an optional directory path. If given, it will be prepended to
      each file name in `files` and `recursive` will be True by default.
    - `ext`: an extension to be added to each filename. If `dir` was specified,
      the default extension is '.py'. If no `dir` was specified, the default
      extension is an empty string.
    - `recursive`: if True, a cascading menu of all pyFormex scripts in the
      directory and below will be constructed.
    - `max`: if specified, the list of files will be truncated to this number
      of items. Adding more files to the menu will then be done at the top and  
      the surplus number of files will be dropped from the bottom of the list.

    The defaults were thus chosen to be convenient for the two most frequent
    uses of this class::

      ScriptMenu('My Scripts',dir="/path/to/my/sciptsdir")

    creates a menu will all pyFormex scripts in the specified path and its
    subdirectories.

    ::

      ScriptMenu('History',files=["/my/script1.py","/some/other/script.pye"],recursive=False)

    is typically used to create a history menu of previously visited files

    With the resulting file list, a menu is created. Selecting a menu item
    will make the corresponding file the current script and unless the
    `autoplay` configuration variable was set to False, the script is executed.

    If the file specification was done by a directory path only,
    some extra options will be included in the menu.
    They are fairly self-explaining and mainly intended for the pyFormex
    developers, in order to test the functionality by running a sets of
    example scripts:

    - :menuselection:`Run next script`
    - :menuselection:`Run all following scripts`
    - :menuselection:`Run all scripts`
    - :menuselection:`Run a random script`
    - :menuselection:`Run all in random order`


    Furthermore, if the menu is a toplevel one, it will have the following
    extra options:

    - :menuselection:`Classify scripts`
    - :menuselection:`Remove catalog`
    - :menuselection:`Reload scripts`

    The first option uses the keyword specifications in the scripts docstring
    to make a classification of the scripts according to keywords.
    See the :func:`scriptKeywords()` function for more info. The second option
    removes the classification. Both options are especially useful for the
    pyFormex examples.

    The last option reloads a ScriptMenu. This can be used to update the menu
    when you created a new script file.
    """
    
    def __init__(self,title,dir=None,files=None,ext=None,recursive=None,max=0,autoplay=False,toplevel=True):
        """Create a menu with pyFormex scripts to play."""
        QtGui.QMenu.__init__(self,title)
        self.dir = dir
        self.files = files
        if self.dir is None and self.files is None:
            raise ValueError,"At least one of 'dir' or 'files' must be set."
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
            m = ScriptMenu(d,os.path.join(self.dir,d),autoplay=self.autoplay,recursive=self.recursive)
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
            print("Found Scripts in %s" % self.dir)
            print(self.files)
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
                mk = ScriptMenu(k.capitalize(),dir=self.dir,files=files,recursive=False,toplevel=False,autoplay=self.autoplay)
                for i in cat[k]:
                    ki = '%s/%s' % (k,i)
                    mi = ScriptMenu(i.capitalize(),dir=self.dir,files=col[ki],recursive=False,toplevel=False,autoplay=self.autoplay)
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
        GD.GUI.setcurfile(selected)
        if self.autoplay:
            GD.debug("Drawing Options: %s" % GD.canvas.options)
            draw.reset()
            draw.play()


    def runNext(self):
        """Run the next script."""
        try:
            i = self.files.index(self.current) + 1
        except ValueError:
            i = 0
            print("You should first run a script from the menu, to define the next")
            return
        GD.debug("This is script %s out of %s" % (i,len(self.files)))
        if i < len(self.files):
            self.runScript(self.files[i])


    def runAllNext(self):
        """Run the current and all following scripts."""
        try:
            i = self.files.index(self.current)
        except ValueError:
            i = 0
            print("You should first run a script from the menu, to define the following")
            return
        GD.debug("Running scripts %s-%s" % (i,len(self.files)))
        self.runAllFiles(self.files[i:])
        GD.debug("Exiting runAllNext")
        

    def runAll(self):
        """Run all scripts."""
        GD.debug("Playing all scripts in order")
        self.runAllFiles(self.files)
        GD.debug("Finished playing all scripts")


    ### THIS should be moved to a playAll function in draw/script module
    def runAllFiles(self,files,randomize=False,pause=0.):
        """Run all the scripts in given list."""
        GD.GUI.actions['Stop'].setEnabled(True)
        if randomize:
            random.shuffle(files)
        for f in files:
            draw.layout(1)
            self.runScript(f)
            #GD.debug("draw.exitrequested == %s" % draw.exitrequested)
            if draw.exitrequested:
                break
            if pause > 0.:
                sleep(pause)
        GD.GUI.actions['Stop'].setEnabled(False)


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

############### The pyFormex Script menu ############################

from prefMenu import setScriptDirs

def createScriptMenu(parent=None,before=None):
    """Create the menu(s) with pyFormex scripts

    This creates a menu with all examples distributed with pyFormex.
    By default, this menu is put in the top menu bar with menu label 'Examples'.

    The user can add his own script directories through the configuration
    settings. In that case the 'Examples' menu and menus for all the
    configured script paths will be gathered in a top level popup menu labeled
    'Scripts'.

    The menu will be placed in the top menu bar before the specified item.
    If a menu item named 'Examples' or 'Scripts' already exists, it is
    replaced.
    """
    from odict import ODict
    scriptmenu = menu.Menu('&Scripts',parent=parent,before=before)
    scriptmenu.menuitems = ODict()
    # Create a copy to leave the cfg unchanged!
    scriptdirs = [] + GD.cfg['scriptdirs']
    # Fill in missing default locations : this enables the user
    # to keep the pyFormex installed examples in his config
    knownscriptdirs = { 'examples': GD.cfg['examplesdir'] }
    for i,item in enumerate(scriptdirs):
        if type(item[0]) is str and not item[1] and item[0].lower() in knownscriptdirs:
            scriptdirs[i] = (item[0].capitalize(),knownscriptdirs[item[0].lower()])

    for txt,dirname in scriptdirs:
        GD.debug("Loading script dir %s" % dirname)
        if os.path.exists(dirname):
            m = ScriptMenu(txt,dir=dirname,autoplay=True)
            scriptmenu.insert_menu(m)
            txt = utils.strNorm(txt)
            scriptmenu.menuitems[txt] = m

    scriptmenu.insertItems([
        ('---',None),
        (_('&Configure Script Paths'),setScriptDirs),
        (_('&Reload Script Menu'),reloadScriptMenu),
        ])
    
    return scriptmenu


def reloadScriptMenu():
    menu = GD.GUI.menu.item('scripts')
    if menu is not None:
        before = GD.GUI.menu.nextitem('scripts')
        GD.GUI.menu.removeItem('scripts')
        newmenu = createScriptMenu(GD.GUI.menu,before)
 
    
# End
