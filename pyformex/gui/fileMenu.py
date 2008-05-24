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
"""Functions from the File menu."""

import os
import globaldata as GD
import widgets
import draw
import utils
import image
#from plugins import surface_menu,formex_menu,tools_menu

from plugins import project
from gettext import gettext as _



##################### handle project files ##########################

the_project = None
the_project_saved = False

def createProject():
    openProject(exist=False)

def openProject(exist=True):
    """Open a file selection dialog and let the user select a project.

    The default only accepts existing project files.
    Use createProject() to accept new file names.
    """
    global the_project,the_project_saved
    if the_project is None:
        cur = GD.cfg.get('workdir','.')
    else:
        if draw.ask("Another project is still open. Shall I close it first?",
                    ['Close','Cancel']) == 'Cancel':
            return
        cur = the_project.filename
    typ = [ 'pyFormex projects (*.pyf)', 'All files (*)' ]
    fn = widgets.FileSelection(cur,typ,exist=exist).getFilename()
    if fn:
        if not fn.endswith('.pyf'):
            fn += '.pyf'
        if not exist and os.path.exists(fn):
            res = draw.ask("The project file '%s' already exists\nShall I delete the contents or add to it?" % fn,['Delete','Add','Cancel'])
            if res == 'Cancel':
                return
            if res == 'Add':
                exist = True
        if GD.PF:
            GD.message("Exportd symbols: %s" % GD.PF.keys())
            res = draw.ask("pyFormex already contains exported symbols.\nShall I add them to your project?",['Delete','Add','Cancel'])
            if res == 'Cancel':
                return
            if res == 'Delete':
                GD.PF = {}
        GD.message("Opening project %s" % fn)
        the_project = project.Project(fn,create=not exist)
        the_project_saved = False
        if GD.PF:
            the_project.update(GD.PF)
        GD.PF = the_project
        GD.gui.setcurproj(fn)
        GD .message("Project contents: %s" % the_project.keys())

def saveProject():
    global the_project,the_project_saved
    if the_project is not None:
        the_project.save()
        the_project_saved = True

def closeProject():
    global the_project,the_project_saved
    if not (the_project_saved or the_project is None):
        GD.message("Closing project %s" % the_project.filename)
        GD .message("Project contents: %s" % the_project.keys())
        the_project.save()
        GD.PF = {}
        GD.PF.update(the_project)
        GD.gui.setcurproj('None')
    the_project = None


##################### handle script files ##########################


def createScript():
    openScript(False)

def openScript(exist=True):
    """Open a file selection dialog and set the selection as the current file.

    The default only accepts existing files. Use newFile() to accept new files.
    """
    cur = GD.cfg.get('curfile',GD.cfg.get('workdir','.'))
    typ = "pyFormex scripts (*.py)"
    fn = widgets.FileSelection(cur,typ,exist=exist).getFilename()
    if fn:
        GD.cfg['workdir'] = os.path.dirname(fn)
        GD.gui.setcurfile(fn)
        GD.gui.history.add(fn)
      
        
def edit():
    """Load the current file in the editor.

    This only works if the editor was set in the configuration.
    The author uses 'gnuclient' to load the files in a running copy
    of (X)Emacs.
    """
    if GD.cfg['editor']:
        pid = utils.spawn('%s %s' % (GD.cfg['editor'],GD.cfg['curfile']))
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
        image.saveImage(filename=opt.fn,
                        window=opt.wi,
                        multi=opt.mu,
                        hotkey=opt.hk,
                        autosave=opt.as,
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
    image.saveImage()


# End
