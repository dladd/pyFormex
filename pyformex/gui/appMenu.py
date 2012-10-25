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
            d[k] = sorted(d[k])


def classify(appdir,pkg,nmax=0):
    """Classify the files in submenus according to keywords.

    """
    class failed(object):
        """A class to allow failing examples in the catalog"""
        _status = 'failed'

    all_apps = sorted(apps.detect(appdir))
    kat = ['status','level','topics','techniques','all']
    cat = dict([ (k,set()) for k in kat])
    cat['status'] = [ 'failed', 'checked', 'unchecked' ]
    cat['level'] = [ 'beginner', 'normal', 'advanced' ]
    col = {}

    if nmax > 9: # Do not exagerate!
        # split the full collection in alphabetical groups of length nmax
        lbl,grp = splitAlpha(all_apps,nmax)
        cat['all'] = lbl
        for l,g in zip(lbl,grp):
            col['all/'+l] = g
                            
    for i,appname in enumerate(all_apps):
        
        #col['all'].update([appname])
        try:
            app = apps.load(pkg+'.'+appname)
        except:
            app = failed
            print("Failed to load app '%s'" % (pkg+'.'+appname))
            
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

    return all_apps,kat,cat,col


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
    mult,bins = multiplicity([ord(f[0]) for f in strings ])
    print(bins)
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

    def group(fromchar,tochar):
        j = i = ord(fromchar)
        mtot = count.get(i,0)
        while j < ord(tochar):
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

    group('A','Z')
    group('a','z')
    
    return cat,grp


class AppMenu(QtGui.QMenu):
    """A menu of pyFormex applications in a directory or list.

    This class creates a menu of pyFormex applications or scripts
    collected from a directory or specified as a list of modules.
    It is used in the pyFormex GUI to create
    the examples menu, and for the apps history. The pyFormex apps
    can then be run from the menu or from the button toolbar.
    The user may use this class to add his own apps/scripts
    into the pyFormex GUI.

    Apps are simply Python modules that have a 'run' function.
    Only these modules will be added to the menu. 
    Only files that are recognized by :func:`utils.is_pyFormex()` as being
    pyFormex scripts will be added to the menu. 

    The constructor takes the following arguments:

    - `title`: the top level label for the menu
    - `dir`: an optional directory path. If specified, and no `files` argument
      is specified, all Python files in `dir` that do not start with either
      '.' or '_'*, will be considered for inclusion in the menu.
      If mode=='app', they will only be included if
      they can be loaded as a module. If mode=='script', they will only be
      included if they are considered a pyFormex script by utils.is_pyFormex.
      If `files` is specified, `dir` will just be prepended to each file in
      the list.
    - `files`: an explicit list of file names of pyFormex scripts.
      If no `dir` nor `ext` arguments are given, these should be the full path
      names to the script files. Otherwise, `dir` is prepended and `ext` is
      appended to each filename.
    - `ext`: an extension to be added to each filename. If `dir` was specified,
      the default extension is '.py'. If no `dir` was specified, the default
      extension is an empty string.
    - `recursive`: if True, a cascading menu of all pyFormex scripts in the
      directory and below will be constructed. If only `dir` and no `files`
      are specified, the default is True
    - `max`: if specified, the list of files will be truncated to this number
      of items. Adding more files to the menu will then be done at the top and  
      the surplus number of files will be dropped from the bottom of the list.

    The defaults were thus chosen to be convenient for the three most frequent
    uses of this class::

      AppMenu('My Apps',dir="/path/to/my/appsdir")

    creates a menu with all pyFormex apps in the specified path and its
    subdirectories.

    ::

      ApptMenu('My Scripts',dir="/path/to/my/sciptsdir",mode='scripts')

    creates a menu with all pyFormex scripts in the specified path and its
    subdirectories.

    ::

      AppMenu('History',files=["/my/script1.py","/some/other/script.pye"],mode='script',recursive=False)

    is typically used to create a history menu of previously visited script
    files.

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
    
    def __init__(self,title,dir=None,files=None,mode='app',ext=None,recursive=None,max=0,autoplay=False,toplevel=True):
        """Create a menu with pyFormex apps/scripts to play."""
        QtGui.QMenu.__init__(self,title)
        self.dir = dir
        self.files = files
        if self.dir is None and self.files is None:
            raise ValueError,"At least one of 'dir' or 'files' must be set."
        self.mode = mode
        if ext is None and self.mode != 'app':
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
        if self.dir and self.mode == 'app':
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
            m = AppMenu(d,os.path.join(self.dir,d),mode=self.mode,ext=self.ext,autoplay=self.autoplay,recursive=self.recursive)
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

        files = self.filterFiles(files)
        ## filtr = lambda s:utils.is_pyFormex(self.fileName(s))
        ## files = filter(filtr,files)

        ## if self.max > 0 and len(files) > self.max:
        ##     files = files[:self.max]

        files.sort()
        return files

 
    def filterFiles(self,files):
        """Filter a list of scripts"""
        filtr = lambda s:utils.is_pyFormex(self.fileName(s))
        files = filter(filtr,files)

        if self.max > 0 and len(files) > self.max:
            files = files[:self.max]

        return files
      

    def loadFiles(self,files=None):
        """Load the app/script files in this menu"""
        if files is None:
            if self.mode == 'app':
                files = apps.detect(self.dir)
            else:
                files = self.getFiles()

        if self.mode != 'app':
            files = self.filterFiles(files)

        if self.max > 0 and len(files) > self.max:
            files = files[:self.max]

        self.files = files

        pf.debug("Found %ss in %s\n%s" % (self.mode.capitalize(),self.dir,self.files),pf.DEBUG.INFO)
            
        self.actions = [ self.addAction(f) for f in self.files ]           
        self.connect(self,QtCore.SIGNAL("triggered(QAction*)"),self.run)

# BV: Removed the runall options, since these were only introduced
#     for testing, and should not be in release 1.0
#
        if self.dir:
            self.addSeparator()
#            self.addAction('Run next app',self.runNext)
#            self.addAction('Run all following apps',self.runAllNext)
#            self.addAction('Run all apps',self.runAll)
#            self.addAction('Run a random app',self.runRandom)
#            self.addAction('Run all in random order',self.runAllRandom)
        self.current = ""


    def loadCatalog(self):
        catfile = os.path.join(self.dir,catname)
        if os.path.exists(catfile):
            pf.execFile(catfile,globals())
            for k in kat:
                if k == 'all_apps' and self.mode != 'app':
                    files = col[k]
                else:
                    files = []
                mk = AppMenu(k.capitalize(),dir=self.dir,files=files,mode=self.mode,recursive=False,toplevel=False,autoplay=self.autoplay)
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


    def fileName(self,script):
        """Return the full pathname for a script."""
        fn = script + self.ext
        if self.dir:
            return os.path.join(self.dir,fn)
        else:
            return fn


    def fullAppName(self,app):
        """Return the pkg.module name for an app."""
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
        if self.mode != 'app':
            pf.GUI.setcurfile(app)
        if play:
            if self.mode == 'app':
                appname = self.fullAppName(app)
            else:
                appname = self.fileName(app)
            pf.debug("Running application %s" % appname,pf.DEBUG.APPS|pf.DEBUG.MENU)
            script.runAny(appname)
        

    def runMany(self,seq):
        """Run a sequence of apps.

        The sequence is specified as a list of indices in the self.fiels list.
        """
        from gui.draw import layout,reset
        for i in seq:
            layout(1)
            reset()
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


    def runAllAtOnce(self,recursive=True):
        """Run all examples in the appmenu

        If recursive is True (default), alsot the apps in the
        submenus are executed.
        """
        self.runAll()
        for m in self.menus:
            m.runAllAtOnce(recursive=recursive)


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
            if self.mode == 'app':
                appname = self.fullAppName(name)
                app = apps.load(appname)
                if app is None:
                    print("%s is NO MODULE!" % appname)
                    return
            else:
                if not utils.is_pyFormex(name):
                    return
        files = self.files
        olist.toFront(files,name)
        if self.max > 0 and len(files) > self.max:
            files = files[:self.max]
        while len(self.actions) < len(files):
            self.actions.append(self.addAction(name))
        for a,f in zip(self.actions,self.files):
            a.setText(f)


    def _classify(self,nmax=20):
        """Classify, symlink and reload the scripts"""
        pf.debug("Classifying scripts",pf.DEBUG.APPS)
        if self.dir:
            f = os.path.join(self.dir,catname)
            all_apps,kat,cat,col = classify(self.dir,self.pkg,nmax)
            s = "all_apps = %r\nkat = %r\ncat = %r\ncol = %r\n" % (all_apps,kat,cat,col)
            open(f,'w').writelines(s)
            print("Created catalog %s" % f)
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

 
def createMenu(parent=None,before=None,mode='app'):
    if mode == 'app':
        return createAppMenu(parent,before)
    else:
        return createScriptMenu(parent,before)

 
def createAppMenu(parent=None,before=None):
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
        (_('&Reload App Menu'),reloadMenu,{'data':'app'}),
        ])

    appmenu.insertMenu(appmenu.item('---'),hist)
    pf.GUI.apphistory = hist
    
    return appmenu


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
    appmenu = menu.Menu('&Scripts',parent=parent,before=before)
    appmenu.menuitems = ODict()
    # Create a copy to leave the cfg unchanged!
    scriptdirs = [] + pf.cfg['scriptdirs']
    # Fill in missing default locations : this enables the user
    # to keep the pyFormex installed examples in his config
    knownscriptdirs = { 'examples': pf.cfg['examplesdir'] }
    for i,item in enumerate(scriptdirs):
        if type(item[0]) is str and not item[1] and item[0].lower() in knownscriptdirs:
            scriptdirs[i] = (item[0].capitalize(),knownscriptdirs[item[0].lower()])

    for txt,dirname in scriptdirs:
        pf.debug("Loading script dir %s" % dirname,pf.DEBUG.SCRIPT)
        if os.path.exists(dirname):
            m = AppMenu(txt,dir=dirname,mode='script',autoplay=True)
            appmenu.insert_menu(m)
            txt = utils.strNorm(txt)
            appmenu.menuitems[txt] = m

    appmenu.insertItems([
        ('---',None),
        (_('&Configure Script Paths'),setDirs,{'data':'scriptdirs'}),
        (_('&Reload Script Menu'),reloadMenu,{'data':'script'}),
        ])
    
    return appmenu


def reloadMenu(mode='app'):
    """Reload the named menu."""
    name = mode+'s'
    menu = pf.GUI.menu.item(name)
    if menu is not None:
        before = pf.GUI.menu.nextitem(name)
        pf.GUI.menu.removeItem(name)
        if mode == 'app':
            # reset pf.appdirs, we may have configuration changes
            import apps
            apps.setAppDirs()
        newmenu = createMenu(pf.GUI.menu,before,mode=mode)



# End
