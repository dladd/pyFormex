.. $Id$
  
..
  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
  pyFormex is a tool for generating, manipulating and transforming 3D
  geometrical models by sequences of mathematical operations.
  Home page: http://pyformex.org
  Project page:  https://savannah.nongnu.org/projects/pyformex/
  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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
.. include:: ../website/src/links.inc

.. _cha:config:

********************
Configuring pyFormex
********************

Many aspects of pyFormex can be configured to better suit the user's needs and likings.
These can range from merely cosmetic changes to important extensions of the
functionality. As is written in a scripting language and distributed as source,
the user can change every single aspect of the program. And the GNU-GPL license
under which the program is distributed guarantees that you have access to the
source and are allowed to change it.

Most users however will only want to change minor aspects of the program, and
would rather not have to delve into the source to do just that. Therefore we
have gathered some items of that users might like to change, into separate files
where thay can easily be found. Some of these items can even be set
interactivley through the GUI menus.

Often users want to keep their settings between subsequent invocation of the
program. To this end, the user preferences have to be stored on file when
leaving the program and read back when starting the next time. While it might
make sense to distinct between the user's current settings in the program and
his default preferences, the current configuration system of (still under
development) does not allow such distinction yet. Still, since the topic is so
important to the user and the configuration system in is already quite complex,
we tought it was necessary to provide already some information on how to
configure . Be aware though that important changes to this system will likely
occur.


.. _sec:pyf-conf-files:

Configuration files
===================

On startup, reads its configurable data from a number of files. Often there are
not less than four configuration files, read in sequence. The settings in each
file being read override the value read before. The different configuration
files used serve different purposes. On a typical GNU/Linux installation, the
following files will be read in sequence:

* PYFORMEX-INSTALL-PATH/pyformexrc: this file should never be changed , neither
  by the user nor the administrator. It is there to guarantee that all settings
  get an adequate default value to allow to correctly start up.

* /etc/pyformex: this file can be used by the system administrator to make
  system-wide changes to the installation. This could e.g. be used to give all
  users at a site access to a common set of scripts or extensions.

* /.pyformexrc: this is where the user normally stores his own default
  settings.

* CURRENT-DIR/.pyformex: if the current working directory from which is started
  contains a file named .pyformex, it will be read too. This makes it possible to
  keep different configurations in different directories, depending on the
  purpose. Thus, one directory might aim at the use of for operating on
  triangulated surfaces, while another might be intended for pre- and post-
  processing of Finite Element models.

* Finally, the ``--config=`` command line option provides a way to specify
  another file with any name to be used as the last configuration file.

On exit,will store the changed settings on the last user configuration file that
was read. The first two files mentioned above are system configuration files and
will never be changed by the program. A user configuration file will be
generated if none existed.

.. warning:: Currently, when pyFormex exits, it will just dump all
  the changed configuration (key,value) pairs on the last
  configuration file, together with the values it read from that
  file. pyFormex will not detect if any changes were made to that
  file between reading it and writing back. Therefore, the user should
  never edit the configuration files directly while pyFormex is still
  running. Always close the program first!


.. _sec:syntax-conf-files:

Syntax of the configuration files
=================================

All configuration files are plain text files where each non blank line is one of
the following:

* a comment line, starting with a '#',

* a section header, of the form '[section-name]',

* a valid Python instruction.

The configuration file is organized in sections. All lines preceding the first
section name refer to the general (unnamed) section.

Any valid Python source line can be used. This allows for quite complex
configuration instructions, even importing Python modules. Any line that binds a
value to a variable will cause a corresponding configuration variable to be set.
The user can edit the configuration files with any text editor, but should make
sure the lines are legal Python. Any line can use the previously defined
variables, even those defined in previously read files.

In the configuration files, the variable pyformexdir refers to the directory
where was installed (and which is also reported by the ``pyformex --whereami``
command).


.. _sec:conf-vars:

Configuration variables
=======================

Many configuration variables can be set interactively from the GUI, and the user
may prefer to do it that way. Some variables however can not (yet) be set from
th GUI. And real programmers may prefer to do it with an editor anyway. So here
are some guidelines for setting some interesting variables. The user may take a
look at the installed default configuration file for more examples.

General section
"""""""""""""""

* ``syspath = []``: Value is a list of path names that will be appended to
  the Python's sys.path variable on startup. This enables your scripts
  to import modules from other than default Python paths.

* ``scriptdirs = [ ('Examples',examplesdir),
  ('MyScripts',myscriptsdir)]``: a list of tuples (name,path). On
  startup, all these paths will be scanned for scripts and these will
  be added in the menu under an item named name.

.. index:: single: autorun

* ``autorun = '.pyformex.startup'``: name of a script that will be
  executed on startup, before any other script (specified on the
  command line or started from the GUI).

* ``editor = 'kedit'``: sets the name of the editor that will be used
  for editing pyformex scripts.

* ``viewer = 'firefox'``: sets the name of the html viewer to be used to
  display the html help screens.

* ``browser = 'firefox'``: sets the name of the browser to be used to access the
  website.

* ``uselib = False``: do not use the acceleration library. The default (True) is to
  use it when it is available.


Section ``[gui]``
"""""""""""""""""

.. index:: single: splash image

* ``splash = 'path-to-splash-image.png'``: full path name of the image to be used
  as splash image on startup.

* ``modebar = True``: adds a toolbar with the render mode buttons. Besides
  True or False, the value can also be one of 'top', 'bottom', 'left'
  or 'right', specifying the placement of the render mode toolbar at
  the specified window border. Any other value that evaluates True
  will make the buttons get included in the top toolbar.

* ``viewbar = True``: adds a toolbar with different view
  buttons. Possioble values as explained above for modebar.

* ``timeoutbutton = True``: include the timeout button in the toolbar. The
  timeout button, when depressed, will cause input widgets to time out
  after a prespecified delay time. This feature is still experimental.

* ``plugins = ['surface_menu', 'formex_menu', 'tools_menu']``: a list of
  plugins to load on startup. This is mainly used to load extra
  (non-default) menus in the GUI to provide extended
  functionality. The named plugins should be available in the
  'plugins' subdirectory of the installation. To autoload user
  extensions from a different place, the autorun script can be used.

.. End

