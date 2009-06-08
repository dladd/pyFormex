# $Id$
##
##  This file is part of pyFormex 0.8 Release Mon Jun  8 11:56:55 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""Functions from the File menu."""

import os,shutil
import pyformex as GD
import widgets
import utils
import project
import draw
from script import processArgs
import image
from plugins import surface_menu,formex_menu,tools_menu,postproc_menu

from gettext import gettext as _


##################### handle project files ##########################

the_project = None


def createProject(create=True,compression=0,addGlobals=None):
    """Open a file selection dialog and let the user select a project.

    The default will let the user create new project files as well as open
    existing ones.
    Use create=False or the convenience function openProject to only accept
    existing project files.

    If a compression level (1..9) is given, the contents will be compressed,
    resulting in much smaller project files at the cost of  

    Only one pyFormex project can be open at any time. The open project
    owns all the global data created and exported by any script.
    If addGlobals is None, the user is asked whether the current globals
    should be added to the project. Set True or False to force or reject
    the adding without asking.
    """
    global the_project

    # ask filename from user
    if the_project is None:
        cur = GD.cfg.get('workdir','.')
    else:
        options = ['Cancel','Close without saving','Save and Close']
        ans = draw.ask("Another project is still open. Shall I close it first?",
                    options)
        if ans == 'Cancel':
            return
        if ans == options[2]:
            the_project.save()
        cur = the_project.filename
    typ = [ 'pyFormex projects (*.pyf)', 'All files (*)' ]
    res = widgets.ProjectSelection(cur,typ,exist=not create).getResult()
    if res is None:
        # user canceled
        return

    fn = res.fn
    if not fn.endswith('.pyf'):
        fn += '.pyf'
    legacy = res.leg
    compression = res.cpr
    print fn,legacy,compression

    if create and os.path.exists(fn):
        res = draw.ask("The project file '%s' already exists\nShall I delete the contents or add to it?" % fn,['Delete','Add','Cancel'])
        if res == 'Cancel':
            return
        if res == 'Add':
            create = False
    GD.message("Opening project %s" % fn)
    
    if GD.PF:
        GD.message("Exported symbols: %s" % GD.PF.keys())
        if addGlobals is None:
            res = draw.ask("pyFormex already contains exported symbols.\nShall I delete them or add them to your project?",['Delete','Add','Cancel'])
            if res == 'Cancel':
                # ESCAPE FROM CREATING THE PROJECT
                return

            addGlobals = res == 'Add'

    # OK, we have all data, now create/open the project
    GD.GUI.setBusy()
    the_project = project.Project(fn,create=create,signature = GD.Version,compression=compression,legacy=legacy)
    GD.GUI.setBusy(False)
    if GD.PF and addGlobals:
        the_project.update(GD.PF)
    GD.PF = the_project
    GD.GUI.setcurproj(fn)
    GD.cfg['workdir'] = os.path.dirname(fn)
    GD .message("Project contents: %s" % the_project.keys())
    if hasattr(the_project,'autofile') and draw.ack("The project has an autofile attribute: %s\nShall I execute this script?" % the_project.autofile):
        processArgs([the_project.autofile])


def openProject():
    """Create a new project file, uncompressed by default.

    If a compression level (1..9) is specified, the stored data will be
    gzipped. This may substantially reduce the size of large databases.

    This internal compression is easier to the user than externally
    compressing the resulting files.
    """
    createProject(create=False)
 

def setAutoFile():
    """Set the current script as autoScriptFile in the project"""
    global the_project
    if the_project is not None and GD.cfg['curfile'] and GD.GUI.canPlay:
        the_project.autofile = GD.cfg['curfile']
            

def saveProject():
    if the_project is not None:
        GD.message("Project contents: %s" % the_project.keys())
        GD.GUI.setBusy()
        the_project.save()
        GD.GUI.setBusy(False)


def saveAsProject():
    if the_project is not None:
        closeProjectWithoutSaving()
        createProject(addGlobals=True)
        saveProject()


def closeProjectWithoutSaving():
    """Close the current project without saving it."""
    closeProject(False)
    

def closeProject(save=True):
    """Close the current project, saving it by default."""
    global the_project
    if the_project is not None:
        GD.message("Closing project %s" % the_project.filename)
        if save:
            saveProject()
        # The following is needed to copy the globals to a new dictionary
        # before destroying the project
        GD.PF = {}
        GD.PF.update(the_project)
        GD.GUI.setcurproj('None')
    the_project = None


def askCloseProject():
    if the_project is not None:
        choices = ['Exit without saving','SaveAs and Exit','Save and Exit']
        res = draw.ask("You have an unsaved open project: %s\nWhat do you want me to do?"%the_project.filename,choices,default=2)
        res = choices.index(res)
        if res == 1:
            saveAsProject()
        elif res == 2:
            saveProject()


##################### handle script files ##########################

def openScript(fn=None,exist=True,create=False):
    """Open a pyFormex script and set it as the current script.

    If no filename is specified, a file selection dialog is started to select
    an existing script, or allow to create a new file if exist is False.

    If the file exists and is a pyFormex script, it is set ready to execute.

    If create is True, a default pyFormex script template will be written
    to the file, overwriting the contents if the file existed. Then, the
    script is loaded into the editor.

    We encourage the use of createScript() to create new scripts and leave
    openScript() to open existing scripts.
    """
    if fn is None:
        cur = GD.cfg.get('curfile',GD.cfg.get('workdir','.'))
        typ = "pyFormex scripts (*.py)"
        fn = widgets.FileSelection(cur,typ,exist=exist).getFilename()
    if fn:
        if create:
            if not exist and os.path.exists(fn) and not draw.ack("The file %s already exists.\n Are you sure you want to overwrite it?" % fn):
                return None
            template = GD.cfg['scripttemplate']
            if (os.path.exists(template)):
                shutil.copyfile(template,fn)
        GD.cfg['workdir'] = os.path.dirname(fn)
        GD.GUI.setcurfile(fn)
        GD.GUI.history.add(fn)
        if create:
            editScript(fn)
    return fn

      
def createScript(fn=None):
    return openScript(fn,exist=False,create=True)

    
def editScript(fn=None):
    """Load the current file in the editor.

    This only works if the editor was set in the configuration.
    The author uses 'emacsclient' to load the files in a running copy
    of Emacs.
    If a filename is specified, that file is loaded instead.
    """
    if GD.cfg['editor']:
        if fn is None:
            fn = GD.cfg['curfile']
        pid = utils.spawn('%s %s' % (GD.cfg['editor'],fn))
    else:
        draw.warning('No known editor was found or configured')

    
def saveImage(multi=False):
    """Save an image to file.

    This will show the Save Image dialog, with the multisave mode checked if
    multi = True. Then, depending on the user's selection, it will either:
     - save the current Canvas/Window to file
     - start the multisave/autosave mode
     - do nothing
    """
    pat = map(utils.fileDescription, ['img','icon','all'])  
    dia = widgets.SaveImageDialog(GD.cfg['workdir'],pat,multi=multi)
    opt = dia.getResult()
    if opt:
        GD.cfg['workdir'] = os.path.dirname(opt.fn)
        image.save(filename=opt.fn,
                   window=opt.wi,
                   multi=opt.mu,
                   hotkey=opt.hk,
                   autosave=opt.au,
                   border=opt.bo,
                   rootcrop=opt.rc
                   )
def saveIcon():
    """Save an image as icon.

    This will show the Save Image dialog, with the multisave mode off and
    asking for an icon file name. Then save the current rendering to that file.
    """
    ## We should create a specialized input dialog, asking also for the size 
    fn = draw.askFilename(filter=utils.fileDescription('icon'))
    if fn:
        image.saveIcon(fn,size=32)

    
def startMultiSave():
    """Start/change multisave mode."""
    saveImage(True)


def stopMultiSave():
    """Stop multisave mode."""
    image.save()

from imageViewer import ImageViewer
viewer = None
def showImage():
    """Display an image file."""
    global viewer
    fn = draw.askFilename(filter=utils.fileDescription('img'),multi=False,exist=True)
    if fn:
        viewer = ImageViewer(GD.app,fn)
        viewer.show()
        
    


def setOptions():
    options = ['test','debug','uselib','safelib','fastencode']
    options = [ o for o in options if hasattr(GD.options,o) ]
    items = [ (o,getattr(GD.options,o)) for o in options ]
    res = draw.askItems(items)
    if res:
        for o in options:
            setattr(GD.options,o,res[o])


MenuData = [
    (_('&Start new project'),createProject),
    (_('&Open existing project'),openProject),
    (_('&Set current script as AutoFile'),setAutoFile),
    (_('&Save project'),saveProject),
    (_('&Save project As'),saveAsProject),
    (_('&Close project without saving'),closeProjectWithoutSaving),
    (_('&Save and Close project'),closeProject),
    ('---',None),
    (_('&Create new script'),createScript),
    (_('&Open existing script'),openScript),
    (_('&Play script'),draw.play),
    (_('&Edit script'),editScript),
    (_('&Change workdir'),draw.askDirname),
    (_('---1'),None),
    (_('&Save Image'),saveImage),
    (_('Start &MultiSave'),startMultiSave),
    (_('Save &Next Image'),image.saveNext),
    (_('Create &Movie'),image.createMovie),
    (_('&Stop MultiSave'),stopMultiSave),
    (_('&Save as Icon'),saveIcon),
    (_('&Show Image'),showImage),
    (_('---2'),None),
    (_('Load &Plugins'),[
        (_('Surface menu'),surface_menu.show_menu),
        (_('Formex menu'),formex_menu.show_menu),
        (_('Tools menu'),tools_menu.show_menu),
        (_('Postproc menu'),postproc_menu.show_menu),
        ]),
    (_('&Options'),setOptions),
    (_('---3'),None),
    (_('E&xit'),draw.closeGui),
]

#onExit(closeProject)

# End
