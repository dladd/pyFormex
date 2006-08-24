#!/usr/bin/env python
# $Id$
"""Functions from the File menu."""

import os
import globaldata as GD
import widgets
import gui
import draw


def newFile():
    return openFile(False)


def openFile(exist=True):
    """Open a file selection dialog and set the selection as the current file.

    The default only accepts existing files. Use newFile() to accept new files.
    """
    cur = GD.cfg.get('curfile',GD.cfg.get('workdir','.'))
    fs = widgets.FileSelection(cur,"pyformex scripts (*.frm *.py)",exist=exist)
    fn = fs.getFilename()
    if fn:
        GD.cfg['workdir'] = os.path.dirname(fn)
        gui.setcurfile(fn)

        
def edit():
    """Load the current file in the editor.

    This only works if the editor was set in the configuration.
    The author uses 'gnuclient' to load the files in a running copy
    of xemacs.
    """
    if GD.cfg['edit']:
        cmd = GD.cfg['edit']
        pid = os.spawnlp(os.P_NOWAIT, cmd, cmd, GD.cfg['curfile'])
        draw.log("Spawned %d" % pid)


play = draw.play
    
def saveImage():
    """Save the current rendering in image format.

    This function will open a file selection dialog, and if a valid
    file is returned, the current OpenGL rendering will be saved to it.
    """
    global canvas
    dir = GD.cfg.get('workdir',".")
    fs = widgets.FileSelection(dir,pattern="Images (*.png *.jpg *.eps)")
    fn = fs.getFilename()
    if fn:
        GD.cfg['workdir'] = os.path.dirname(fn)
        print "Will now save image"
        draw.saveImage(fn,verbose=True)

def multiSave():
    """Save a sequence of images.

    If the filename supplied has a trailing numeric part, subsequent images
    will be numbered continuing from this number. Otherwise a numeric part
    -000, -001, will be added to the filename.
    """
    if draw.multisave:
        fn = None
    else:
        dir = GD.cfg.get('workdir',".")
        fs = widgets.FileSelection(dir,pattern="Images (*.png *.jpg)")
        fn = fs.getFilename()
    draw.saveMulti(fn,verbose=True)

