.. $Id$  -*- rst -*-
.. pyformex reference manual --- image
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-image:

:mod:`image` --- Saving OpenGL renderings to image files.
=========================================================

.. automodule:: image
   :synopsis: Saving OpenGL renderings to image files.


   .. autofunction:: imageFormats()
   .. autofunction:: initialize()
   .. autofunction:: checkImageFormat(fmt,verbose=False)
   .. autofunction:: imageFormatFromExt(ext)
   .. autofunction:: save_canvas(canvas,fn,fmt='png',quality=1,options=None)
   .. autofunction:: save_window(filename,format,quality=1,windowname=None)
   .. autofunction:: save_main_window(filename,format,quality=1,border=False)
   .. autofunction:: save_rect(x,y,w,h,filename,format,quality=1)
   .. autofunction:: save(filename=None,window=False,multi=False,hotkey=True,autosave=False,border=False,rootcrop=False,format=None,quality=1,verbose=False)
   .. autofunction:: saveNext()
   .. autofunction:: saveIcon(fn,size=32)
   .. autofunction:: autoSaveOn()
   .. autofunction:: createMovie()
   .. autofunction:: saveMovie(filename,format,windowname=None)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

