.. $Id$  -*- rst -*-
.. pyformex reference manual --- image
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-image:

:mod:`image` --- Saving OpenGL renderings to image files.
=========================================================

.. automodule:: image
   :synopsis: Saving OpenGL renderings to image files.


   .. autofunction:: imageFormats()
   .. autofunction:: initialize()
   .. autofunction:: checkImageFormat(fmt,verbose=???)
   .. autofunction:: imageFormatFromExt(ext)
   .. autofunction:: save_canvas(canvas,fn,fmt=???,quality=???,options=???)
   .. autofunction:: save_window(filename,format,quality=???,windowname=???)
   .. autofunction:: save_main_window(filename,format,quality=???,border=???)
   .. autofunction:: save_rect(x,y,w,h,filename,format,quality=???)
   .. autofunction:: save(filename=???,window=???,multi=???,hotkey=???,autosave=???,border=???,rootcrop=???,format=???,quality=???,verbose=???)
   .. autofunction:: saveNext()
   .. autofunction:: saveIcon(fn,size=???)
   .. autofunction:: autoSaveOn()
   .. autofunction:: createMovie()
   .. autofunction:: saveMovie(filename,format,windowname=???)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

