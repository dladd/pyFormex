# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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
"""Functions from the File menu."""

import os,shutil
import pyformex as pf
import widgets
import utils
import project
import draw
from script import processArgs,play
import image
import plugins

from gettext import gettext as _
from prefMenu import updateSettings


##################### handle project files ##########################

the_project = None


def createProject(create=True,compression=0,addGlobals=None,makeDefault=True):
    """Open a file selection dialog and let the user select a project.

    The default will let the user create new project files as well as open
    existing ones.
    Use create=False or the convenience function openProject to only accept
    existing project files.

    If a compression level (1..9) is given, the contents will be compressed,
    resulting in much smaller project files at the cost of  

    Only one pyFormex project can be open at any time. The open project
    owns all the global data created and exported by any script.

    If makeDefault is True, an already open project will be closed and
    the opened project becomes the current project.
    If makeDefault is False, the project data are imported into pf.PF
    and the current project does not change. This means that if a project was
    open, the imported data will be added to it.
    
    If addGlobals is None, the user is asked whether the current globals
    should be added to the project. Set True or False to force or reject
    the adding without asking.
    """
    global the_project

    # ask filename from user
    if the_project is None:
        cur = pf.cfg.get('workdir','.')
    else:
        if makeDefault:
            options = ['Cancel','Close without saving','Save and Close']
            ans = draw.ask("Another project is still open. Shall I close it first?",
                        options)
            if ans == 'Cancel':
                return
            if ans == options[2]:
                the_project.save()
        cur = the_project.filename
    typ = utils.fileDescription(['pyf','all'])
    res = widgets.ProjectSelection(cur,typ,exist=not create).getResult()
    if res is None:
        # user canceled
        return

    fn = res.fn
    if not fn.endswith('.pyf'):
        fn += '.pyf'
    legacy = res.leg
    ignoresig = res.sig
    compression = res.cpr
    #print(fn,legacy,compression)

    if create and os.path.exists(fn):
        res = draw.ask("The project file '%s' already exists\nShall I delete the contents or add to it?" % fn,['Delete','Add','Cancel'])
        if res == 'Cancel':
            return
        if res == 'Add':
            create = False
    pf.message("Opening project %s" % fn)
    
    if pf.PF:
        pf.message("Exported symbols: %s" % pf.PF.keys())
        if addGlobals is None:
            res = draw.ask("pyFormex already contains exported symbols.\nShall I delete them or add them to your project?",['Delete','Add','Cancel'])
            if res == 'Cancel':
                # ESCAPE FROM CREATING THE PROJECT
                return

            addGlobals = res == 'Add'

    # OK, we have all data, now create/open the project
        
    updateSettings({'workdir':os.path.dirname(fn)},save=True)
    sig = pf.Version[:pf.Version.rfind('-')]
    if ignoresig:
        sig = ''

    proj = _open_project(fn,create,sig,compression,legacy)
        
    pf.message("Project contents: %s" % proj.keys())
    
    if hasattr(proj,'_autoscript_'):
        _ignore = "Ignore it!"
        _show = "Show it"
        _edit = "Load it in the editor"
        _exec = "Execute it"
        res = draw.ask("There is an autoscript stored inside the project.\nIf you received this project file from an untrusted source, you should probably not execute it.",[_ignore,_show,_edit,_exec])
        if res == _show:
            res = draw.showText(proj._autoscript_)#,actions=[_ignore,_edit,_show])
            return
        if res == _exec:
            draw.playScript(proj._autoscript_)
        elif res == _edit:
            fn = "_autoscript_.py"
            draw.checkWorkdir()
            f = file(fn,'w')
            f.write(proj._autoscript_)
            f.close()
            openScript(fn)
            editScript(fn)

    if hasattr(proj,'autofile') and draw.ack("The project has an autofile attribute: %s\nShall I execute this script?" % proj.autofile):
        processArgs([proj.autofile])

    if makeDefault:
        the_project = proj
        if pf.PF and addGlobals:
            the_project.update(pf.PF)
        pf.PF = the_project
        pf.GUI.setcurproj(fn)

    else:
        # Just import the data into current project
        pf.PF.update(proj)

    pf.message("Exported symbols: %s" % pf.PF.keys())


def openProject():
    """Open an existing project.

    Ask the user to select an existing project file, and then open it.
    """
    createProject(create=False)


def importProject():
    """Import an existing project.

    Ask the user to select an existing project file, and then import
    its data into the current project.
    """
    createProject(create=False,addGlobals=False,makeDefault=False)


def _open_project(fn,create,signature,compression,legacy):
    """Open a project in the GUI

    This is a low level function not intended for the user.
    It is equivalent to creating a project instance, but has
    exception trapping
    """
    # Loading the project may take a long while; attent user
    pf.GUI.setBusy()
    try:
        proj = project.Project(fn,create,signature,compression,legacy)
    except:
        proj = None
        raise
    finally:
        pf.GUI.setBusy(False)
    return proj
    

def setAutoScript():
    """Set the current script as autoScript in the project"""
    global the_project
    if the_project is not None and pf.cfg['curfile'] and pf.GUI.canPlay:
        the_project._autoscript_ = file(pf.cfg['curfile']).read()
 

def setAutoFile():
    """Set the current script as autoScriptFile in the project"""
    global the_project
    if the_project is not None and pf.cfg['curfile'] and pf.GUI.canPlay:
        the_project.autofile = pf.cfg['curfile']
            
def removeAutoScript():
    global the_project
    delattr(the_project,'_autoscript_')
            
def removeAutoFile():
    global the_project
    delattr(the_project,'autofile')

def saveProject():
    if the_project is not None:
        pf.message("Project contents: %s" % the_project.keys())
        pf.GUI.setBusy()
        the_project.save()
        pf.GUI.setBusy(False)


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
        pf.message("Closing project %s" % the_project.filename)
        if save:
            saveProject()
        # The following is needed to copy the globals to a new dictionary
        # before destroying the project
        pf.PF = {}
        pf.PF.update(the_project)
        pf.GUI.setcurproj('None')
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
        cur = pf.cfg['curfile']
        if cur is None:
            cur = pf.cfg['workdir']
        if cur is None:
            cur  = '.'
        typ = utils.fileDescription('pyformex')
        fn = widgets.FileSelection(cur,typ,exist=exist).getFilename()
    if fn:
        if create:
            if not exist and os.path.exists(fn) and not draw.ack("The file %s already exists.\n Are you sure you want to overwrite it?" % fn):
                return None
            template = pf.cfg['scripttemplate']
            if (os.path.exists(template)):
                shutil.copyfile(template,fn)
        updateSettings({'workdir':os.path.dirname(fn)},save=True)
        pf.GUI.setcurfile(fn)
        pf.GUI.history.add(fn)
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
    if pf.cfg['editor']:
        if fn is None:
            fn = pf.cfg['curfile']
        pid = utils.spawn('%s %s' % (pf.cfg['editor'],fn))
    else:
        draw.warning('No known editor was found or configured')

##################### other functions ##########################

    
def saveImage(multi=False):
    """Save an image to file.

    This will show the Save Image dialog, with the multisave mode checked if
    multi = True. Then, depending on the user's selection, it will either:
     - save the current Canvas/Window to file
     - start the multisave/autosave mode
     - do nothing
    """
    pat = map(utils.fileDescription, ['img','icon','all'])  
    dia = widgets.SaveImageDialog(pf.cfg['workdir'],pat,multi=multi)
    opt = dia.getResult()
    print opt
    if opt:
        if opt.fm == 'From Extension':
            opt.fm = None
        if opt.qu < 0:
            opt.qu = -1
        updateSettings({'workdir':os.path.dirname(opt.fn)},save=True)
        image.save(filename=opt.fn,
                   format=opt.fm,
                   quality=opt.qu,
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
    fn = draw.askNewFilename(filter=utils.fileDescription('icon'))
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
    fn = draw.askFilename(filter=utils.fileDescription('img'))
    if fn:
        viewer = ImageViewer(pf.app,fn)
        viewer.show()


MenuData = [
    (_('&Start new project'),createProject),
    (_('&Open existing project'),openProject),
    (_('&Import a project'),importProject),
    (_('&Set current script as AutoScript'),setAutoScript),
    (_('&Remove the AutoScript'),removeAutoScript),
    (_('&Set current script as AutoFile'),setAutoFile),
    (_('&Remove the AutoFile'),removeAutoFile),
    (_('&Save project'),saveProject),
    (_('&Save project As'),saveAsProject),
    (_('&Close project without saving'),closeProjectWithoutSaving),
    (_('&Save and Close project'),closeProject),
    ('---',None),
    (_('&Create new script'),createScript),
    (_('&Open existing script'),openScript),
    (_('&Play script'),play),
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
    (_('E&xit'),draw.closeGui),
]


#onExit(closeProject)

# End
