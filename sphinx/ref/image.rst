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


   .. autofunction:: initialize()
   .. autofunction:: imageFormats()
   .. autofunction:: checkImageFormat(fmt,verbose=False)
   .. autofunction:: imageFormatFromExt(ext)
   .. autofunction:: save_canvas(canvas,fn,fmt='png',options=None)
   .. autofunction:: save_window(filename,format,windowname=None)
   .. autofunction:: save_main_window(filename,format,border=False)
   .. autofunction:: save_rect(x,y,w,h,filename,format)
   .. autofunction:: save(filename=None,window=False,multi=False,hotkey=True,autosave=False,border=False,rootcrop=False,format=None,verbose=False)
   .. autofunction:: saveNext()
   .. autofunction:: saveIcon(fn,size=32)
   .. autofunction:: autoSaveOn()
   .. autofunction:: createMovie()
   .. autofunction:: saveMovie(filename,format,windowname=None)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

