#!/usr/bin/env python
# $Id$
"""Functions for saving renderings to image files."""

import globaldata as GD

from OpenGL import GL
import utils

# load gl2ps if available
try:
    import gl2ps
    _has_gl2ps = True
    print 'Congratulations! You have gl2ps, so I activated drawPS!'    

except ImportError:
    _has_gl2ps = False

# Find interesting supporting software
utils.hasExternal('ImageMagick')


def save(canvas,fn,fmt='png',options=None):
    """Save the rendering on canvas as an image file.

    canvas specifies the qtcanvas rendering window.
    fn is the name of the file
    fmt is the image file format
    """
    # mak sure we have the current content displayed (on top)
    canvas.makeCurrent()
    canvas.raise_()
    canvas.display()
    GD.app.processEvents()
    size = canvas.size()
    w = int(size.width())
    h = int(size.height())
    GD.debug("Saving image with current size %sx%s" % (w,h))
    
    if fmt in GD.image_formats_qt:
        # format can be saved by Qt
        # depending on version, this night include
        # 'bmp', 'jpeg', 'jpg', 'png', 'ppm', 'xbm', 'xpm' 
        GL.glFlush()
        qim = canvas.grabFrameBuffer()
        if qim.save(fn,fmt):
            sta = 0
        else:
            sta = 1

    elif fmt in GD.image_formats_gl2ps:
        # format can be saved by savePS
        sta = savePS(canvas,fn,fmt)

    elif fmt in GD.image_formats_fromeps:
        # format can be converted from eps
        import commands,os
        fneps = os.path.splitext(fn)[0] + '.eps'
        delete = not os.path.exists(fneps)
        savePS(canvas,fneps,'eps')
        if os.path.exists(fneps):
            cmd = 'pstopnm -portrait -stdout %s' % fneps
            if fmt != 'ppm':
                cmd += '| pnmto%s > %s' % (fmt,fn)
            GD.debug(cmd)
            sta,out = commands.getstatusoutput(cmd)
            if sta:
                GD.debug(out)
            if delete:
                os.remove(fneps)

    return sta


#### ONLY LOADED IF GL2PS FOUND ########################

if _has_gl2ps:

    _producer = GD.Version + ' (http://pyformex.berlios.de)'
    _gl2ps_types = { 'ps':gl2ps.GL2PS_PS, 'eps':gl2ps.GL2PS_EPS,
                     'pdf':gl2ps.GL2PS_PDF, 'tex':gl2ps.GL2PS_TEX }
    GD.image_formats_gl2ps = _gl2ps_types.keys()
    GD.image_formats_fromeps = [ 'ppm', 'png', 'jpeg', 'rast', 'tiff',
                                 'xwd', 'y4m' ]

    def savePS(canvas,filename,filetype=None,title='',producer='',
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
        opts = gl2ps.GL2PS_SILENT | gl2ps.GL2PS_SIMPLE_LINE_OFFSET
        ##| gl2ps.GL2PS_NO_BLENDING | gl2ps.GL2PS_OCCLUSION_CULL | gl2ps.GL2PS_BEST_ROOT
        ##color = GL[[0.,0.,0.,0.]]
        while state == gl2ps.GL2PS_OVERFLOW:
            bufsize += 1024*1024
            #print filename,filetype
            gl2ps.gl2psBeginPage(title, _producer, viewport, filetype,
                                 gl2ps.GL2PS_BSP_SORT, opts, GL.GL_RGBA,
                                 0, None, 0, 0, 0, bufsize, fp, filename)
            canvas.display()
            GL.glFinish()
            state = gl2ps.gl2psEndPage()
        fp.close()
        return 0
