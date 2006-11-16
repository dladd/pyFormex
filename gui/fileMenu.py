#!/usr/bin/env python
# $Id$
"""Functions from the File menu."""

import os
import globaldata as GD
import widgets
import gui
import draw
import utils


def newFile():
    return openFile(False)


def openFile(exist=True):
    """Open a file selection dialog and set the selection as the current file.

    The default only accepts existing files. Use newFile() to accept new files.
    """
    cur = GD.cfg.get('curfile',GD.cfg.get('workdir','.'))
    print cur,os.path.dirname(cur)
    fs = widgets.FileSelection(cur,"pyformex scripts (*.frm *.py)",exist=exist)
    fn = fs.getFilename()
    if fn:
        GD.cfg['workdir'] = os.path.dirname(fn)
        GD.gui.setcurfile(fn)

        
def edit():
    """Load the current file in the editor.

    This only works if the editor was set in the configuration.
    The author uses 'gnuclient' to load the files in a running copy
    of xemacs.
    """
    if GD.cfg['editor']:
        pid = utils.spawn(GD.cfg['editor'] % GD.cfg['curfile'])


play = draw.play
    
def saveImage(multi=False):
    """Save an image to file.

    This will show the Save Image dialog, with the multisave mode checked if
    multi = True. Then, depending on the user's selection, it will either:
     - save the current Canvas/Window to file
     - start the multisave/autosave mode
     - do nothing
    """
    dia = widgets.SaveImageDialog(GD.cfg['workdir'],["Images (*.png)","Images (*.png *.jpg *.eps)"],multi=multi)
    fn,window,multi,hotkey,auto = dia.getResult()
    if fn:
        GD.cfg['workdir'] = os.path.dirname(fn)
        draw.saveImage(fn,window,multi,hotkey,auto)

    
def startMultiSave():
    """Start/change multisave mode."""
    saveImage(True)


def stopMultiSave():
    """Stop multisave mode."""
    draw.saveImage()

# End
