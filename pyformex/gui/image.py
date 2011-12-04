#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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
"""Saving OpenGL renderings to image files.

This module defines some functions that can be used to save the
OpenGL rendering and the pyFormex GUI to image files. There are even
provisions for automatic saving to a series of files and creating
a movie from these images.
"""


import pyformex as pf

from OpenGL import GL
from PyQt4 import QtCore,QtGui,QtOpenGL
import utils
import os


# The image formats recognized by pyFormex
image_formats_qt = []
image_formats_qtr = []
image_formats_gl2ps = []
image_formats_fromeps = []

def imageFormats(type='w'):
    if type == 'r':
        return image_formats_qtr
    else:
        return image_formats_qt + image_formats_gl2ps + image_formats_fromeps
        

# global parameters for multisave mode
multisave = None

# imported module
gl2ps = None

def initialize():
    """Initialize the image module."""
    global image_formats_qt,image_formats_qtr,image_formats_gl2ps,image_formats_fromeps,gl2ps,_producer,_gl2ps_types
    
    # Find interesting supporting software
    utils.hasExternal('ImageMagick')
    # Set some globals
    pf.debug("LOADING IMAGE FORMATS")
    image_formats_qt = map(str,QtGui.QImageWriter.supportedImageFormats())
    image_formats_qtr = map(str,QtGui.QImageReader.supportedImageFormats())
    ## if pf.cfg.get('imagesfromeps',False):
    ##     pf.image_formats_qt = []

    if utils.hasModule('gl2ps'):

        import gl2ps

        _producer = pf.Version + ' (%s)' % pf.cfg.get('help/website','')
        _gl2ps_types = {
            'ps':gl2ps.GL2PS_PS,
            'eps':gl2ps.GL2PS_EPS,
            'tex':gl2ps.GL2PS_TEX,
            'pdf':gl2ps.GL2PS_PDF,
            }
        if utils.checkVersion('gl2ps','1.03') >= 0:
            _gl2ps_types.update({
                'svg':gl2ps.GL2PS_SVG,
                'pgf':gl2ps.GL2PS_PGF,
                })
        image_formats_gl2ps = _gl2ps_types.keys()
        image_formats_fromeps = [ 'ppm', 'png', 'jpeg', 'rast', 'tiff',
                                     'xwd', 'y4m' ]
    pf.debug("""
Qt image types for saving: %s
Qt image types for input: %s
gl2ps image types: %s
image types converted from EPS: %s""" % (image_formats_qt,image_formats_qtr,image_formats_gl2ps,image_formats_fromeps))
             
 
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
    pf.debug("Format requested: %s" % fmt)
    pf.debug("Formats available: %s" % imageFormats())
    if fmt in imageFormats():
        if fmt == 'tex' and verbose:
            pf.warning("This will only write a LaTeX fragment to include the 'eps' image\nYou have to create the .eps image file separately.\n")
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

def save_canvas(canvas,fn,fmt='png',quality=-1,size=None):
    """Save the rendering on canvas as an image file.

    canvas specifies the qtcanvas rendering window.
    fn is the name of the file
    fmt is the image file format
    """
    # make sure we have the current content displayed (on top)
    canvas.makeCurrent()
    canvas.raise_()
    canvas.display()
    pf.app.processEvents()
    
    if fmt in image_formats_qt:
        pf.debug("Image format can be saved by Qt")
        wc,hc = canvas.getSize()
        try:
            w,h = size
        except:
            w,h = wc,hc
        if (w,h) == (wc,hc):
            # Save directly from current rendering
            pf.debug("Saving image from canvas with size %sx%s" % (w,h))
            GL.glFlush()
            qim = canvas.grabFrameBuffer()
        else:
            pf.debug("Saving image from virtual buffer with size %sx%s" % (w,h))
            vcanvas = QtOpenGL.QGLFramebufferObject(w,h)
            vcanvas.bind()
            canvas.resize(w,h)
            canvas.display()
            GL.glFlush()
            qim = vcanvas.toImage()
            vcanvas.release()
            canvas.resize(wc,hc)
            del vcanvas
            
        print "SAVING %s in format %s with quality %s" % (fn,fmt,quality)
        if qim.save(fn,fmt,quality):
            sta = 0
        else:
            sta = 1

    elif fmt in image_formats_gl2ps:
        pf.debug("Image format can be saved by gl2ps")
        sta = save_PS(canvas,fn,fmt)

    elif fmt in image_formats_fromeps:
        pf.debug("Image format can be converted from eps")
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

initialize()

if gl2ps:
    
    def save_PS(canvas,filename,filetype=None,title='',producer='',viewport=None):
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
        fp = open(filename, "wb")
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
        #print("VIEWPORT %s" % str(viewport))
        #print(fp)
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



def save_window(filename,format,quality=-1,windowname=None):
    """Save a window as an image file.

    This function needs a filename AND format.
    If a window is specified, the named window is saved.
    Else, the main pyFormex window is saved.
    """
    if windowname is None:
        windowname = pf.GUI.windowTitle()
    pf.GUI.raise_()
    pf.GUI.repaint()
    pf.GUI.toolbar.repaint()
    pf.GUI.update()
    pf.canvas.makeCurrent()
    pf.canvas.raise_()
    pf.canvas.update()
    pf.app.processEvents()
    cmd = 'import -window "%s" %s:%s' % (windowname,format,filename)
    sta,out = utils.runCommand(cmd)
    return sta


def save_main_window(filename,format,quality=-1,border=False):
    """Save the main pyFormex window as an image file.

    This function needs a filename AND format.
    This is an alternative for save_window, by grabbin it from the root
    window, using save_rect.
    This allows us to grab the border as well.
    """
    pf.GUI.repaint()
    pf.GUI.toolbar.repaint()
    pf.GUI.update()
    pf.canvas.update()
    pf.app.processEvents()
    if border:
        geom = pf.GUI.frameGeometry()
    else:
        geom = pf.GUI.geometry()
    x,y,w,h = geom.getRect()
    return save_rect(x,y,w,h,filename,format,quality)


def save_rect(x,y,w,h,filename,format,quality=-1):
    """Save a rectangular part of the screen to a an image file."""
    cmd = 'import -window root -crop "%sx%s+%s+%s" %s:%s' % (w,h,x,y,format,filename)
    sta,out = utils.runCommand(cmd)
    return sta


#### USER FUNCTIONS ################

def save(filename=None,window=False,multi=False,hotkey=True,autosave=False,border=False,rootcrop=False,format=None,quality=-1,size=None,verbose=False):
    """Saves an image to file or Starts/stops multisave mode.

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
    #print "SAVE: quality=%s" % quality
    global multisave

    # Leave multisave mode if no filename or starting new multisave mode
    if multisave and (filename is None or multi):
        pf.message("Leave multisave mode")
        QtCore.QObject.disconnect(pf.GUI,QtCore.SIGNAL("Save"),saveNext)
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
        pf.message("Start multisave mode to files: %s (%s)" % (names.name,format))
        #print(hotkey)
        if hotkey:
             QtCore.QObject.connect(pf.GUI,QtCore.SIGNAL("Save"),saveNext)
             if verbose:
                 pf.warning("Each time you hit the '%s' key,\nthe image will be saved to the next number." % pf.cfg['keys/save'])
        multisave = (names,format,quality,size,window,border,hotkey,autosave,rootcrop)
        print("MULTISAVE %s "% str(multisave))
        return multisave is None

    else: # Save the image
        if window:
            if rootcrop:
                sta = save_main_window(filename,format,quality,border=border)
            else:
                sta = save_window(filename,format,quality)
        else:
            if size == pf.canvas.getSize():
                size = None
            sta = save_canvas(pf.canvas,filename,format,quality,size)
        if sta:
            pf.debug("Error while saving image %s" % filename)
        else:
            pf.message("Image file %s written" % filename)
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
        names,format,quality,size,window,border,hotkey,autosave,rootcrop = multisave
        name = names.next()
        save(name,window,False,hotkey,autosave,border,rootcrop,format,quality,size,False)


def saveIcon(fn,size=32):
    """Save the current rendering as an icon."""
    savew,saveh = pf.canvas.width(),pf.canvas.height()
    pf.canvas.resize(size,size)
    if not fn.endswith('.xpm'):
        fn += '.xpm'
    save(fn)
    pf.canvas.resize(savew,saveh)


def autoSaveOn():
    """Returns True if autosave multisave mode is currently on.

    Use this function instead of directly accessing the autosave variable.
    """
    return multisave and multisave[-2]


def createMovie(encoder='ffmpeg'):
    """Create a movie from a saved sequence of images.

    encoder is one of: 'ffmpeg, mencoder, convert'
    """
    if not multisave:
        pf.warning('You need to start multisave mode first!')
        return

    names,format,quality,window,border,hotkey,autosave,rootcrop = multisave
    glob = names.glob()
    ## if glob.split('.')[-1] != 'jpg':
    ##     pf.warning("Currently you need to save in 'jpg' format to create movies")
    ##     return

    if encoder == 'convert':
        cmd = "convert -delay 1 -colors 256 %s output.gif" % names.glob()
    elif encoder == 'mencoder':
        cmd = "mencoder -ovc lavc -fps 5 -o output.avi %s" % names.glob()
    elif encoder == 'mencoder1':
        cmd = "mencoder \"mf://%s\" -mf fps=10 -o output1.avi -ovc lavc -lavcopts vcodec=msmpeg4v2:vbitrate=800" % names.glob()
    else:
        cmd = "ffmpeg -qscale 1 -r 1 -i %s output.mp4" % names.glob()
    pf.debug(cmd)
    utils.runCommand(cmd)


def saveMovie(filename,format,windowname=None):
    """Create a movie from the pyFormex window."""
    if windowname is None:
        windowname = pf.GUI.windowTitle()
    pf.GUI.raise_()
    pf.GUI.repaint()
    pf.GUI.toolbar.repaint()
    pf.GUI.update()
    pf.canvas.makeCurrent()
    pf.canvas.raise_()
    pf.canvas.update()
    pf.app.processEvents()
    windowid = windowname
    cmd = "xvidcap --fps 5 --window %s --file %s" % (windowid,filename)
    pf.debug(cmd)
    #sta,out = utils.runCommand(cmd)
    return sta


initialize()


### End
