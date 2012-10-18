# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Menu with pyFormex apps.

"""
from __future__ import print_function

import pyformex as pf
import apps

from PyQt4 import QtCore, QtGui

import utils,olist
import script,draw
import menu
import os,random
from gettext import gettext as _
    
catname = 'apps.cat'


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
 

def classify(appdir,pkg):
    """Classify the files in submenus according to keywords."""
    kat = ['status','level','topics','techniques','all']
    cat = dict([ (k,set()) for k in kat])
    cat['status'] = [ 'failed', 'checked', 'unchecked' ]
    cat['level'] = [ 'beginner', 'normal', 'advanced' ]
    col = {'all':set()}

    class failed(object):
        _status = 'failed'

    
    for appname in apps.detect(appdir):
        
        col['all'].update([appname])
        try:
            app = apps.load(pkg+'.'+appname)
        except:
            app = failed
            pf.debug("Failed to load app '%s'" % (pkg+'.'+appname),pf.DEBUG.APPS)
        for k in kat:
            if hasattr(app,'_'+k):
                v = getattr(app,'_'+k)
                if type(v) is list:
                    v = [ vi.lower() for vi in v ]
                else:
                    v = v.lower()
                if k in ['status','level']:
                    v = [v]
                else:
                    cat[k].update(v)
                for i in v:
                    ki = '%s/%s' % (k,i)
                    if not ki in col.keys():
                        col[ki] = set()
                    col[ki].update([appname])

    sortSets(cat)
    sortSets(col)

    return kat,cat,col


def splitAlpha(strings,n):
    """Split a series of strings in alphabetic collections.

    The strings are split over a series of bins in alphabetical order.
    Each bin can contain strings starting with multiple successive
    characters, but not more than n items. Items starting with the same
    character are always in the same bin. If any starting character
    occurs more than n times, the maximum will be exceeded. 

    - `files`: a list of strings start with an upper case letter ('A'-'Z')
    - `n`: the desired maximum number of items in a bin.

    Returns: a tuple of

    - `labels`: a list of strings specifying the range of start characters
      (or the single start character) for the bins
    - `groups`: a list with the contents of the bins. Each item is a list
      of sorted strings starting with one of the characters in the
      corresponding label 
    """
    from arraytools import multiplicity
    strings = sorted(strings)
    #print(strings)
    mult,bins = multiplicity([ord(f[0]) for f in strings ])
    #print(mult)
    #print([chr(b) for b in bins])
    count = dict(zip(bins,mult))
    cat = []
    grp = []

    def accept(i,j,mtot):
        if i == j:
            cat.append(chr(i))
        else:
            cat.append('%c-%c' % (chr(i),chr(j)))
        grp.append(strings[:mtot])
        del strings[:mtot]
            
    j = i = ord('A')
    mtot = count.get(i,0)
    while j < ord('Z'):
        if mtot > n:
            accept(i,j,mtot)
            j = i = i+1
            mtot = count.get(i,0)
        else:
            mj = count.get(j+1,0)
            if mtot+mj > n:
                accept(i,j,mtot)
                j = i = j+1
                mtot = mj
            else:
                j += 1
                mtot += mj

    accept(i,j,mtot)
    
    #print(cat,grp)
    #print([len(g) for g in grp])
    return cat,grp


class AppMenu(QtGui.QMenu):
    """A menu of pyFormex applications in a directory or list.

    This class creates a menu of pyFormex applications collected from a
    directory or specified as a list of modules.
    It is used in the pyFormex GUI to create
    the examples menu, and for the apps history. The pyFormex apps
    can then be run from the menu or from the button toolbar.
    The user may use this class to add his own apps into the pyFormex GUI.

    Apps are simply Python modules that have a 'run' function.
    Only these modules will be added to the menu. 

    The constructor takes the following arguments:

    - `title`: the top level label for the menu
    - `apps`: a list of which have a 'run' function. If omitted, a
      list of apps will be constructed from the directory path `dir`.
      Python files in the directory `dir` whose name does not start with
      either '.' or '_'*, will be selected.
    - `dir`: an optional directory path. If specified, all Python files in
      `dir` that do not start with either '.' or '_'*, will be loaded and
      the corresponding modules will be added to prepended to
      each file name in `files` and `recursive` will be True by default.
    - `recursive`: if True, a cascading menu of all pyFormex scripts in the
      directory and below will be constructed.
    - `max`: if specified, the list of files will be truncated to this number
      of items. Adding more files to the menu will then be done at the top and  
      the surplus number of files will be dropped from the bottom of the list.

    The defaults were thus chosen to be convenient for the two most frequent
    uses of this class::

      AppMenu('My Apps',dir="/path/to/my/appsdir")

    creates a menu will all pyFormex apps in the specified path and its
    subdirectories.

    ::

      ScriptMenu('History',files=["/my/script1.py","/some/other/script.pye"],recursive=False)

    is typically used to create a history menu of previously visited files.

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
    
    def __init__(self,title,dir=None,files=None,recursive=None,max=0,autoplay=False,toplevel=True):
        """Create a menu with pyFormex apps to play."""
        QtGui.QMenu.__init__(self,title)
        self.dir = dir
        self.files = files
        if self.dir is None and self.files is None:
            raise ValueError,"At least one of 'dir' or 'files' must be set."
        if recursive is None:
            recursive = True
        self.recursive = recursive and self.dir is not None
        self.toplevel = toplevel
        self.max = max
        self.autoplay = autoplay
        self.menus = []
        if self.dir:
            self.pkg = os.path.basename(self.dir)
        else:
            self.pkg = None
        self.load()


    def loadSubmenus(self,dirs=[]):
        if not dirs:
            dirs = os.listdir(self.dir)
        filtr = lambda s:os.path.isdir(os.path.join(self.dir,s))
        dirs = filter(filtr,dirs)
        filtr = lambda s: s[0]!='.' and s[0]!='_'
        dirs = filter(filtr,dirs)
        dirs.sort()
        for d in dirs:
            m = AppMenu(d,os.path.join(self.dir,d),autoplay=self.autoplay,recursive=self.recursive)
            self.addMenu(m)
            self.menus.append(m)
      

    def loadFiles(self,files=None):
        """Load the app files in this menu"""
        if files is None:
            files = apps.detect(self.dir)

        if self.max > 0 and len(files) > self.max:
            files = files[:self.max]

        self.files = files

        pf.debug("Found Apps in %s\n%s" % (self.dir,self.files),pf.DEBUG.INFO)
            
        self.actions = [ self.addAction(f) for f in self.files ]           
        self.connect(self,QtCore.SIGNAL("triggered(QAction*)"),self.run)
        
        if self.dir:
            self.addSeparator()
            self.addAction('Run next app',self.runNext)
            self.addAction('Run all following apps',self.runAllNext)
            self.addAction('Run all apps',self.runAll)
            self.addAction('Run a random app',self.runRandom)
            self.addAction('Run all in random order',self.runAllRandom)
        self.current = ""


    def loadCatalog(self):
        n = 20
        catfile = os.path.join(self.dir,catname)
        if os.path.exists(catfile):
            pf.execFile(catfile,globals())
            for k in kat:
                if k == 'all':
                    files = col[k]
                    if len(files) > n:
                        # Create submenus per character class
                        lbl,grp = splitAlpha(files,n)
                        cat[k] = lbl
                        for l,g in zip(lbl,grp):
                            col['all/'+l] = g
                        #print(col)
                        files = []
                    else:
                        # Keep all in same menu
                        pass
                else:
                    files = []
                mk = AppMenu(k.capitalize(),dir=self.dir,files=files,recursive=False,toplevel=False,autoplay=self.autoplay)
                for i in cat[k]:
                    if '-' in i:
                        # alpha label like A-B
                        lbl = i
                    else:
                        # string from catalog file
                        lbl = i.capitalize()
                    ki = '%s/%s' % (k,i)
                    mi = AppMenu(lbl,dir=self.dir,files=col.get(ki,[]),recursive=False,toplevel=False,autoplay=self.autoplay)
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
                self.addAction('Classify apps',self._classify)
                self.addAction('Remove catalog',self._unclassify)
                self.addAction('Reload apps',self.reload)



    def fullAppName(self,app):
        if self.pkg:
            return "%s.%s" % (self.pkg,app)
        else:
            return app


    def run(self,action):
        """Run the selected app.

        This function is executed when the menu item is selected.
        """
        app = str(action.text())
        if app in self.files:
            self.runApp(app,play=self.autoplay)
    

    def runApp(self,app,play=True):
        """Set/Run the specified app.

        Set the specified app as the current app,
        and run it if play==True.
        """
        self.current = app
        if play:
            appname = self.fullAppName(app)
            pf.debug("Running application %s" % appname,pf.DEBUG.APPS|pf.DEBUG.MENU)
            script.runAny(appname)
        

    def runMany(self,seq):
        """Run a sequence of apps.

        The sequence is specified as a list of indices in the self.fiels list.
        """
        from gui.draw import layout
        for i in seq:
            layout(1)
            self.runApp(self.files[i])
       

    def runNext(self):
        """Run the next app, or the first if none was played yet."""
        try:
            i = self.files.index(self.current) + 1
        except ValueError:
            i = 0
        self.runMany([i])


    def runRandom(self):
        """Run a random script."""
        i = random.randint(0,len(self.files)-1)
        self.runMany([i])


    def runAll(self,first=0,last=None):
        """Run all apps in the range [first,last].

        If last is None, the length of the file list is used.
        Notice that the range is Python style.
        """
        if last is None:
            last = len(self.files)
        self.runMany(range(first,last))


    def runAllNext(self):
        """Run all the next apps, starting with the current one.

        If there is no current, start  from the first.
        """
        try:
            i = self.files.index(self.current)
        except ValueError:
            i = 0
        self.runMany(range(i,len(self.files)))


    def runAllRandom(self):
        """Run all scripts in a random order."""
        order = range(len(self.files))
        random.shuffle(order)
        self.runMany(order)


    def reload(self):
        """Reload the scripts from dir.

        This is only available if a directory path was specified and
        no files.
        """
        pf.debug("Reloading this menu",pf.DEBUG.APPS)
        if self.dir:
            self.clear()
            self.menus = []
            self.files = None
            self.load()


    def add(self,name,strict=True):
        """Add a new filename to the front of the menu.

        By default, only legal pyFormex apps can be added.
        """
        if strict:
            appname = self.fullAppName(name)
            app = apps.load(appname)
            if app is None:
                print("%s is NO MODULE!" % appname)
                return
            
        files = self.files
        olist.toFront(files,name)
        if self.max > 0 and len(files) > self.max:
            files = files[:self.max]
        while len(self.actions) < len(files):
            self.actions.append(self.addAction(name))
        for a,f in zip(self.actions,self.files):
            a.setText(f)


    def _classify(self):
        """Classify, symlink and reload the scripts"""
        pf.debug("Classifying scripts",pf.DEBUG.APPS)
        if self.dir:
            f = os.path.join(self.dir,catname)
            kat,cat,col = classify(self.dir,self.pkg)
            s = "kat = %r\ncat = %r\ncol = %r\n" % (kat,cat,col)
            pf.debug("Writing catalog %s" % s,pf.DEBUG.APPS)
            open(f,'w').writelines(s)
            self.reload()


    def _unclassify(self):
        """Remove the catalog and reload the scripts unclassified"""
        if self.dir:
            f = os.path.join(self.dir,catname)
            if os.path.exists(f):
                os.remove(f)
                self.reload()


############### The pyFormex App menu ############################

from prefMenu import setDirs

 
def createMenu(parent=None,before=None):
    """Create the menu(s) with pyFormex apps

    This creates a menu with all examples distributed with pyFormex.
    By default, this menu is put in the top menu bar with menu label 'Examples'.

    The user can add his own app directories through the configuration
    settings. In that case the 'Examples' menu and menus for all the
    configured app paths will be gathered in a top level popup menu labeled
    'Apps'.

    The menu will be placed in the top menu bar before the specified item.
    If a menu item named 'Examples' or 'Apps' already exists, it is
    replaced.
    """
    import sys
    from odict import ODict
    appmenu = menu.Menu('&Apps',parent=parent,before=before)
    appmenu.menuitems = ODict()

    
    # Go ahead and load the apps
    for d in pf.appdirs:
        pf.debug("Loading %s" % d,pf.DEBUG.MENU)
        m = AppMenu(d.name,dir=d.path,autoplay=True)
        appmenu.insert_menu(m)
        appmenu.menuitems[utils.strNorm(d.name)] = m

    if pf.cfg.get('gui/history_in_main_menu',False):
        before = pf.GUI.menu.item('help')
        pf.GUI.menu.insertMenu(before,pf.GUI.history)
    else:
        filemenu = pf.GUI.menu.item('file')
        before = filemenu.item('---1')
        filemenu.insertMenu(before,pf.GUI.history)
 
    hist = AppMenu('Last Run',files=pf.cfg['gui/apphistory'],max=pf.cfg['gui/history_max'])
    
    appmenu.insertItems([
        ('---',None),
        (_('&Configure App Paths'),setDirs,{'data':'appdirs'}),
        (_('&List loaded Apps'),script.printLoadedApps),
        (_('&Unload Current App'),menu.unloadCurrentApp),
        (_('&Reload App Menu'),reloadMenu,{'data':'apps'}),
        ])

    appmenu.insertMenu(appmenu.item('---'),hist)
    pf.GUI.apphistory = hist
    
    return appmenu


def reloadMenu(name='apps'):
    """Reload the named menu."""
    menu = pf.GUI.menu.item(name)
    if menu is not None:
        before = pf.GUI.menu.nextitem(name)
        pf.GUI.menu.removeItem(name)
        # reset pf.appdirs, we may have configuration changes
        import apps
        apps.setAppDirs()
        newmenu = createMenu(pf.GUI.menu,before)
 

def create_app_menu(parent=None,before=None): 	 
    loadmenu = menu.Menu('&Run Applications',parent=parent,before=before)
    loadactions = menu.ActionList(function=apps.run,menu=loadmenu) 	 
    for name in apps._available_apps:
        descr = name.capitalize().replace('_',' ')
        loadactions.add(name,icon=None,text=descr)
        
    return loadactions
    
# End
