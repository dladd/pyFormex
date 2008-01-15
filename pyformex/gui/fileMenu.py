# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
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

from plugins import project



##################### handle project files ##########################

the_project = None

def createProject():
    openProject(False)

def openProject(exist=True):
    """Open a file selection dialog and let the suer select a project.

    The default only accepts existing project files.
    Use createProject() to accept new file names.
    """
    global the_project
    if the_project is None:
        cur = GD.cfg.get('workdir','.')
    else:
        cur = the_project.filename
    typ = [ 'pyFormex projects (*.pyf)', 'All files (*)' ]
    fn = widgets.FileSelection(cur,typ,exist=exist).getFilename()
    if fn:
        if not fn.endswith('.pyf'):
            fn += '.pyf'
        GD.message("Opening project %s" % fn)
        the_project = project.Project(fn)
        GD.PF = the_project
        GD.gui.setcurproj(fn)

def saveProject():
    if the_project is not None:
        the_project.save()

def closeProject():
    global the_project
    if the_project is not None:
        GD.message("Closing project %s" % the_project.filename)
        the_project.save()
        GD.PF = {}
        GD.PF.update(the_project)
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

    
def startMultiSave():
    """Start/change multisave mode."""
    saveImage(True)


def stopMultiSave():
    """Stop multisave mode."""
    image.saveImage()

# End
