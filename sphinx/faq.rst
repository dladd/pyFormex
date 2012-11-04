.. $Id$
  
..
  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
  pyFormex is a tool for generating, manipulating and transforming 3D
  geometrical models by sequences of mathematical operations.
  Home page: http://pyformex.org
  Project page:  http://savannah.nongnu.org/projects/pyformex/
  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
  Distributed under the GNU General Public License version 3 or later.
  
  
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see http://www.gnu.org/licenses/.
  
  

.. include:: defines.inc
.. include:: links.inc


.. _cha:faq:
.. sectionauthor:: Benedict Verhegghe <benedict.verhegghe@ugent.be>

************************
pyFormex FAQ 'n TRICKS
************************

:Date: |today|
:Version: |version|
:Author:  Benedict Verhegghe <benedict.verhegghe@ugent.be>


.. topic:: Abstract

   This chapter answers some frequently asked questions about
   pyFormex and present some nice tips to solve common problems. If
   you have some question that you want answered, or want to present a
   original solution to some problem, feel free to communicate it to
   us (by preference via the pyFormex `Support tracker`_) and we'll probably
   include it in the next version of this FAQ.

.. _sec:faq:

FAQ
===

#. **How was the pyFormex logo created?**

   We used the GNU Image Manipulation Program (`GIMP`_). It has a wide variety
   of scripts to create logos. With newer versions (>= 2.6) use the menu
   :menuselection:`Fille-->Create-->Logos-->Alien-neon`. With older 
   versions (<=2.4) use :menuselection:`Xtra-->Script-Fu-->Logos-->Alien-neon`.
   
   In the Alien Neon dialog specify the following data::

      Text: pyFormex
      Font Size: 150 
      Font: Blippo-Heavy
      Glow Color: 0xFF3366
      Background Color: 0x000000
      Width of Bands: 2
      Width of Gaps: 2
      Number of Bands: 7
      Fade Away: Yes

   Press :guilabel:`OK` to create the logo. Then switch off the background
   layer and save the image in PNG format.  
   Export the image with ``Save Background Color`` option switched off!


#. **How was the pyFormex favicon created?**
   With FTGL, save as icon, handedited .xpm in emacs to set background color
   to None (transparent), then converted to .png and .ico with convert.

   
   .. _`faq:why_python`: 

#. **Why is pyFormex written in Python?**

   Because

   - it is very easy to learn (See the `Python`_ website)
   - it is extremely powerful (More on `Python`_ website)

   Being a scripting language without the need for variable
   declaration, it allows for quick program development. On the other
   hand, Python provides numerous interfaces with established compiled
   libraries, so it can be surprisingly fast. 


#. **Is an interpreted language like Python fast enough with large data models?**

   See the :ref:`question above <faq:why_python>`.
   
   .. note::

      We should add something about NumPy and the pyFormex C-library.


.. _sec:tricks:

TRICKS
======

#. **Use your script path as the current working directory**

   Start your script with the following::

      chdir(__file__)

   When executing a script, pyFormex sets the name of the script
   file in a variable ``__file__`` passed with the global variables to
   the execution environment of the script.


#. **Import modules from your own script directories**
  
   In order for Python to find the modules in non-standard locations,
   you should add the directory path of the module to the ``sys.path``
   variable.

   A common example is a script that wants to import modules from the
   same directory where it is located. In that case you can just add
   the following two lines to the start of your script::

      import os,sys
      sys.path.insert(0,os.dirname(__file__))


#. **Automatically load plugin menus on startup**

   Plugin menus can be loaded automatically on pyFormex startup, by
   adding a line to the ``[gui]`` section of your configuration file
   (``~/.pyformexrc``)::

      [gui]
      plugins = ['surface_menu', 'formex_menu']


#. **Automatically execute your own scripts on startup**

   If you create your own pugin menus for pyFormex, you cannot
   autoload them like the regular plugin menus from the distribution,
   because they are not in the plugin directory of the
   installation. Do not be tempted to put your own files under the
   installation directory (even if you can acquire the permissions to
   do so), because on removal or reinstall your files might be
   deleted! You can however automatically execute your own scripts by
   adding their full path names in the ``autorun`` variable of your
   configuration file ::

      autorun = '/home/user/myscripts/startup/'

   This script will then be run when the pyFormex GUI starts up. You
   can even specify a list of scripts, which will be executed in
   order. The autorun scripts are executed as any other pyFormex
   script, before any scripts specified on the command line, and
   before giving the input focus to the user.

#. **Multiple viewports with unequal size**

   The multiple viewports are ordered in a grid layout, and you can
   specify relative sizes for the different columns and/or rows of
   viewports. You can use setColumnStretch and setRowStretch to give
   the columns a relative stretch compared toi the other ones. 
   The following example produces 4 viewports in a 2x2
   layout, with the right column(1) having double width of the left
   one(0), while the bottom row has a height equal to 1.5 times the
   height of the top row ::

      layout(4)
      pf.GUI.viewports.setColumnStretch(0,1)
      pf.GUI.viewports.setColumnStretch(1,2)
      pf.GUI.viewports.setRowStretch(0,2)
      pf.GUI.viewports.setRowStretch(1,3)

#. **Activate pyFormex debug messages from your script**

   ::

      import pyformex
      pyformex.options.debug = True


#. **Get a list of all available image formats**

   ::

      import gui.image
      print image.imageFormats()


#. **Create a movie from a sequence of recorded images**

   The multisave option allows you to easily record a series of images
   while working with pyFormex. You may want to turn this sequence
   into a movie afterwards. This can be done with the ``mencoder``
   and/or ``ffmpeg`` programs. The internet provides comprehensive
   information on how to use these video encoders.

   If you are looking for a quick answer, however, here are some of
   the commands we have often used to create movies.

   * Create MNPG movies from PNG To keep the quality of the PNG images
     in your movie, you should not encode them into a compressed
     format like MPEG. You can use the MPNG codec instead. Beware
     though that uncompressed encodings may lead to huge video
     files. Also, the MNPG is (though freely available), not installed
     by default on Windows machines.

     Suppose you have images in files ``image-000.png``,
     ``image-001.png``, ....  First, you should get the size of the
     images (they all should have the same size). The command ::

        file image*.png

     will tell you the size. Then create movie with the command  ::

        mencoder mf://image-*.png -mf w=796:h=516:fps=5:type=png -ovc copy -oac copy -o movie1.avi

     Fill in the correct width(w) and height(h) of the images, and set
     the frame rate(fps). The result will be a movie ``movie1.avi``.

   * Create a movie from (compressed) JPEG images. Because the
     compressed format saves a lot of space, this will be the prefered
     format if you have lots of image files. The quality of the
     compressed image movie will suffer somewhat, though.  ::

        ffmpeg -r 5 -b 800 -i image-%03d.jpg movie.mp4

#. **Install the** :mod:`gl2ps` **extension**

   .. note::
   
      This belongs in :doc:`install`

   Saving images in EPS format is done through the gl2ps library,
   which can be accessed from Python using wrapper functions.
   Recent versions of pyFormex come with an installation script
   that will also generate the required Python interface module.
   
   .. warning::

      The older ``python-gl2ps-1.1.2.tar.gz`` available from the
      web is no longer supported

   You need to have the OpenGL header files installed in order to do
   this (on Debian: ``apt-get install libgl1-mesa-dev``).


#. **Permission denied error when running calpy simulation**

   If you have no write permission in your current working directory,
   running a calpy simulation will result in an error like this::

     fil = file(self.tempfilename,'w')
     IOError
     : 
     [Errno 13] Permission denied: 'calpy.tmp.part-0'

   You can fix this by changing your current working directory to a path
   where you have write permission (e.g. your home directory). 
   You can do this using the :menuselection:`File->Change workdir` menu option.
   The setting will be saved when you leave pyFormex (but other scripts
   might change the setting again).


#. **Reading back old Project (.pyf) files**

   When the implementation of some pyFormex class changes, or when the
   location of a module is changed, an error may result when trying to
   read back old Project (.pyf) files. While in principle it is possible
   to create the necessary interfaces to read back the old data and transform
   them to new ones, our current policy is to not do this by default for
   all classes and all changes. That would just require too much resources
   for maybe a few or no cases occurring.
   We do provide here some guidelines to help you with solving the problems
   yourself. And if you are not able to fix it, just file a support request
   at our `Support tracker`_ and we will try to help you.

   If the problem is with a changed implementation of a class, it can usually
   be fixed by adding an appropriate __set_state__ method to the class.
   Currently we have this for Formex and Mesh classes. Look at the code
   in formex.py and mesh.py respectively.

   If the problem comes from a relocation of a module (e.g. the mesh module
   was moved from plugins to the pyFormex core), you may get an error like
   this::

     AttributeError: 'NoneType' object has no attribute 'Mesh'
  
   The reason is that the path recorded in the Project file pointed to the
   old location of the mesh module under ``plugins`` while the mesh module
   is now in the top pyformex directory. This can be fixed in two ways:

   - The easy (but discouraged) way is to add a symbolic link in the old 
     position, linking to the new one. We do not encourage to use this
     method, because it sustains the dependency on legacy versions.

   - The recommended way is to convert your Project file to point to the
     new path. To take care of the above relocation of the mesh module,
     you could e.g. use the following command to convert your ``old.pyf`` to
     a ``new.pyf`` that can be properly read. It just replaces the old module
     path (``plugins.mesh``) with the current path (``mesh``)::

       sed 's|plugins.mesh|mesh|'g old.pyf >new.pyf
  

.. End
