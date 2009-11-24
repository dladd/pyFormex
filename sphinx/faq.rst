.. $Id$
.. pyformex manual --- faq

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
   us (by preference via the pyFormex `Forums`_) and we'll probably
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

.. End
