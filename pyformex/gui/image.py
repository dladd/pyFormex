#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
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
"""Functions for saving renderings to image files."""

__all__ = [ 'save', 'saveNext', 'saveIcon' ]

import pyformex as GD

from OpenGL import GL
from PyQt4 import QtCore,QtGui
import utils
import os

# Find interesting supporting software
utils.hasExternal('ImageMagick')


# The image formats recognized by pyFormex
image_formats_qt = []
image_formats_qtr = []
image_formats_gl2ps = []
image_formats_fromeps = []

# global parameters for multisave mode
multisave = None 

# Set some globals
GD.debug("LOADING IMAGE FORMATS")
image_formats_qt = map(str,QtGui.QImageWriter.supportedImageFormats())
image_formats_qtr = map(str,QtGui.QImageReader.supportedImageFormats())
if GD.cfg.get('imagesfromeps',False):
    GD.image_formats_qt = []
if GD.options.debug:
    print "Qt image types for saving: ",image_formats_qt
    print "Qt image types for input: ",image_formats_qtr
    print "gl2ps image types:",image_formats_gl2ps
    print "image types converted from EPS:",image_formats_fromeps
 
def imageFormats():
    """Return a list of the valid image formats.

    image formats are lower case strings as 'png', 'gif', 'ppm', 'eps', etc.
    The available image formats are derived from the installed software.
    """
    return image_formats_qt + \
           image_formats_gl2ps + \
           image_formats_fromeps


def checkImageFormat(fmt,verbose=False):
    """Checks image format; if verbose, warn if it is not.

    Returns the image format, or None if it is not OK.
    """
    GD.debug("Format requested: %s" % fmt)
    GD.debug("Formats available: %s" % imageFormats())
    if fmt in imageFormats():
        if fmt == 'tex' and verbose:
            GD.warning("This will only write a LaTeX fragment to include the 'eps' image\nYou have to create the .eps image file separately.\n")
        return fmt
    else:
        if verbose:
            error("Sorry, can not save in %s format!\n"
                  "I suggest you use 'png' format ;)"%fmt)
        return None


def imageFormatFromExt(ext):
    """Determine the image format from an extension.

    The extension may or may not have an initial dot and may be in upper or
    lower case. The format is equal to the extension characters in lower case.
    If the supplied extension is empty, the default format 'png' is returned.
    """
    if len(ext) > 0:
        if ext[0] == '.':
            ext = ext[1:]
        fmt = ext.lower()
    else:
        fmt = 'png'
    return fmt


##### LOW LEVEL FUNCTIONS ##########

def save_canvas(canvas,fn,fmt='png',options=None):
    """Save the rendering on canvas as an image file.

    canvas specifies the qtcanvas rendering window.
    fn is the name of the file
    fmt is the image file format
    """
    # make sure we have the current content displayed (on top)
    canvas.makeCurrent()
    canvas.raise_()
    canvas.display()
    GD.app.processEvents()
    size = canvas.size()
    w = int(size.width())
    h = int(size.height())
    GD.debug("Saving image with current size %sx%s" % (w,h))
    
    if fmt in image_formats_qt:
        GD.debug("Image format can be saved by Qt")
        GL.glFlush()
        qim = canvas.grabFrameBuffer()
        if qim.save(fn,fmt):
            sta = 0
        else:
            sta = 1

    elif fmt in image_formats_gl2ps:
        GD.debug("Image format can be saved by gl2ps")
        sta = save_PS(canvas,fn,fmt)

    elif fmt in image_formats_fromeps:
        GD.debug("Image format can be converted from eps")
        fneps = os.path.splitext(fn)[0] + '.eps'
        delete = not os.path.exists(fneps)
        save_PS(canvas,fneps,'eps')
        if os.path.exists(fneps):
            cmd = 'pstopnm -portrait -stdout %s' % fneps
            if fmt != 'ppm':
                cmd += '| pnmto%s > %s' % (fmt,fn)
            utils.runCommand(cmd)
            if delete:
                os.remove(fneps)

    return sta


if utils.hasModule('gl2ps'):

    import gl2ps

    _producer = GD.Version + ' (http://pyformex.berlios.de)'
    _gl2ps_types = { 'ps':gl2ps.GL2PS_PS, 'eps':gl2ps.GL2PS_EPS,
                     'pdf':gl2ps.GL2PS_PDF, 'tex':gl2ps.GL2PS_TEX }
    image_formats_gl2ps = _gl2ps_types.keys()
    image_formats_fromeps = [ 'ppm', 'png', 'jpeg', 'rast', 'tiff',
                                 'xwd', 'y4m' ]

    def save_PS(canvas,filename,filetype=None,title='',producer='',
               viewport=None):
        """ Export OpenGL rendering to PostScript/PDF/TeX format.

        Exporting OpenGL renderings to PostScript is based on the PS2GL
        library by Christophe Geuzaine (http://geuz.org/gl2ps/), linked
        to Python by Toby Whites's wrapper
        (http://www.esc.cam.ac.uk/~twhi03/software/python-gl2ps-1.1.2.tar.gz)

        This function is only defined if the gl2ps module is found.

        The filetype should be one of 'ps', 'eps', 'pdf' or 'tex'.
        If not specified, the type is derived from the file extension.
        In case of the 'tex' filetype, two files are written: one with
        the .tex extension, and one with .eps extension.
        """
        fp = file(filename, "wb")
        if filetype:
            filetype = _gl2ps_types[filetype]
        else:
            s = filename.lower()
            for ext in _gl2ps_types.keys():
                if s.endswith('.'+ext):
                    filetype = _gl2ps_types[ext]
                    break
            if not filetype:
                filetype = gl2ps.GL2PS_EPS
        if not title:
            title = filename
        if not viewport:
            viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        bufsize = 0
        state = gl2ps.GL2PS_OVERFLOW
        opts = gl2ps.GL2PS_SILENT | gl2ps.GL2PS_SIMPLE_LINE_OFFSET | gl2ps.GL2PS_USE_CURRENT_VIEWPORT
        ##| gl2ps.GL2PS_NO_BLENDING | gl2ps.GL2PS_OCCLUSION_CULL | gl2ps.GL2PS_BEST_ROOT
        ##color = GL[[0.,0.,0.,0.]]
        print "VIEWPORT %s" % str(viewport)
        print fp
        viewport=None
        while state == gl2ps.GL2PS_OVERFLOW:
            bufsize += 1024*1024
            gl2ps.gl2psBeginPage(title, _producer, viewport, filetype,
                                 gl2ps.GL2PS_BSP_SORT, opts, GL.GL_RGBA,
                                 0, None, 0, 0, 0, bufsize, fp, '')
            canvas.display()
            GL.glFinish()
            state = gl2ps.gl2psEndPage()
        fp.close()
        return 0



def save_window(filename,format,windowname=None):
    """Save a window as an image file.

    This function needs a filename AND format.
    If a window is specified, the named window is saved.
    Else, the main pyFormex window is saved.
    """
    if windowname is None:
        windowname = GD.GUI.windowTitle()
    GD.GUI.raise_()
    GD.GUI.repaint()
    GD.GUI.toolbar.repaint()
    GD.GUI.update()
    GD.canvas.makeCurrent()
    GD.canvas.raise_()
    GD.canvas.update()
    GD.app.processEvents()
    cmd = 'import -window "%s" %s:%s' % (windowname,format,filename)
    sta,out = utils.runCommand(cmd)
    return sta


def save_main_window(filename,format,border=False):
    """Save the main pyFormex window as an image file.

    This function needs a filename AND format.
    This is an alternative for save_window, by grabbin it from the root
    window, using save_rect.
    This allows us to grab the border as well.
    """
    GD.GUI.repaint()
    GD.GUI.toolbar.repaint()
    GD.GUI.update()
    GD.canvas.update()
    GD.app.processEvents()
    if border:
        geom = GD.GUI.frameGeometry()
    else:
        geom = GD.GUI.geometry()
    x,y,w,h = geom.getRect()
    return save_rect(x,y,w,h,filename,format)


def save_rect(x,y,w,h,filename,format):
    """Save a rectangular part of the screen to a an image file."""
    cmd = 'import -window root -crop "%sx%s+%s+%s" %s:%s' % (w,h,x,y,format,filename)
    sta,out = utils.runCommand(cmd)
    return sta


#### USER FUNCTIONS ################

def save(filename=None,window=False,multi=False,hotkey=True,autosave=False,border=False,rootcrop=False,format=None,verbose=False):
    """Saves an image to file or Starts/stops multisave maode.

    With a filename and multi==False (default), the current viewport rendering
    is saved to the named file.

    With a filename and multi==True, multisave mode is started.
    Without a filename, multisave mode is turned off.
    Two subsequent calls starting multisave mode without an intermediate call
    to turn it off, do not cause an error. The first multisave mode will
    implicitely be ended before starting the second.

    In multisave mode, each call to saveNext() will save an image to the
    next generated file name.
    Filenames are generated by incrementing a numeric part of the name.
    If the supplied filename (after removing the extension) has a trailing
    numeric part, subsequent images will be numbered continuing from this
    number. Otherwise a numeric part '-000' will be added to the filename.
    
    If window is True, the full pyFormex window is saved.
    If window and border are True, the window decorations will be included.
    If window is False, only the current canvas viewport is saved.

    If hotkey is True, a new image will be saved by hitting the 'S' key.
    If autosave is True, a new image will be saved on each execution of
    the 'draw' function.
    If neither hotkey nor autosave are True, images can only be saved by
    executing the saveNext() function from a script.

    If no format is specified, it is derived from the filename extension.
    fmt should be one of the valid formats as returned by imageFormats()
  
    If verbose=True, error/warnings are activated. This is usually done when
    this function is called from the GUI.
    
    """
    global multisave

    # Leave multisave mode if no filename or starting new multisave mode
    if multisave and (filename is None or multi):
        GD.message("Leave multisave mode")
        QtCore.QObject.disconnect(GD.GUI,QtCore.SIGNAL("Save"),saveNext)
        multisave = None

    if filename is None:
        return

    #chdir(filename)
    name,ext = os.path.splitext(filename)
    # Get/Check format
    if not format: # is None:
        format = checkImageFormat(imageFormatFromExt(ext))
    if not format:
        return

    if multi: # Start multisave mode
        names = utils.NameSequence(name,ext)
        if os.path.exists(names.peek()):
            next = names.next()
        GD.message("Start multisave mode to files: %s (%s)" % (names.name,format))
        #print hotkey
        if hotkey:
             QtCore.QObject.connect(GD.GUI,QtCore.SIGNAL("Save"),saveNext)
             if verbose:
                 GD.warning("Each time you hit the '%s' key,\nthe image will be saved to the next number." % GD.cfg['keys/save'])
        multisave = (names,format,window,border,hotkey,autosave,rootcrop)
        print "MULTISAVE %s "% str(multisave)
        return multisave is None

    else: # Save the image
        if window:
            if rootcrop:
                sta = save_main_window(filename,format,border=border)
            else:
                sta = save_window(filename,format)
        else:
            sta = save_canvas(GD.canvas,filename,format)
        if sta:
            GD.debug("Error while saving image %s" % filename)
        else:
            GD.message("Image file %s written" % filename)
        return


# Keep the old name for compatibility
saveImage = save

    
def saveNext():
    """In multisave mode, saves the next image.

    This is a quiet function that does nothing if multisave was not activated.
    It can thus safely be called on regular places in scripts where one would
    like to have a saved image and then either activate the multisave mode
    or not.
    """
    if multisave:
        names,format,window,border,hotkey,autosave,rootcrop = multisave
        name = names.next()
        save(name,window,False,hotkey,autosave,border,rootcrop,format,False)


def saveIcon(fn,size=32):
    """Save the current rendering as an icon."""
    savew,saveh = GD.canvas.width(),GD.canvas.height()
    GD.canvas.resize(size,size)
    if not fn.endswith('.xpm'):
        fn += '.xpm'
    save(fn)
    GD.canvas.resize(savew,saveh)


def autoSaveOn():
    """Returns True if autosave multisave mode is currently on.

    Use this function instead of directly accessing the autosave variable.
    """
    return multisave and multisave[-2]


def createMovie():
    """Create a movie from a saved sequence of images."""
    if not multisave:
        GD.warning('You need to start multisave mode first!')
        return

    names,format,window,border,hotkey,autosave,rootcrop = multisave
    glob = names.glob()
    if glob.split('.')[-1] != 'jpg':
        GD.warning("Currently you need to save in 'jpg' format to create movies")
        return
    
    #cmd = "mencoder -ovc lavc -fps 5 -o output.avi %s" % names.glob()
    cmd = "ffmpeg -r 1 -i %s output.mp4" % names.glob()
    GD.debug(cmd)
    utils.runCommand(cmd)


def saveMovie(filename,format,windowname=None):
    """Create a movie from the pyFormex window."""
    if windowname is None:
        windowname = GD.GUI.windowTitle()
    GD.GUI.raise_()
    GD.GUI.repaint()
    GD.GUI.toolbar.repaint()
    GD.GUI.update()
    GD.canvas.makeCurrent()
    GD.canvas.raise_()
    GD.canvas.update()
    GD.app.processEvents()
    windowid = windowname
    cmd = "xvidcap --fps 5 --window %s --file %s" % (windowid,filename)
    GD.debug(cmd)
    #sta,out = utils.runCommand(cmd)
    return sta


### End
