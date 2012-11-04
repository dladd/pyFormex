..
  
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
  
  

========
pyformex
========

-------------------------------------------------------
generate and transform 3D geometry using Python scripts
-------------------------------------------------------

:Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>, with the help of the other project members: Gianluca De Santis, Francesco Iannaccone, Peter Mortier, Tomas Praet, Sofie Van Cauter, Wenxuan Zhou. This manual page was written for the Debian project (and may be used by others).
:Date:   2011-12-29
:Copyright: GPL v3 or higher
:Version: 0.1
:Manual section: 1
:Manual group: text and X11 processing

.. TODO: authors and author with name <email>


SYNOPSIS
========

pyformex [options] [ [ file [args] ] ...]

DESCRIPTION
===========

This manual page documents briefly the pyformex command.

pyFormex is a program for generating, transforming and manipulating large geometrical models of 3D structures by sequences of mathematical operations. Thanks to a powerful (Python based) scripting language, pyFormex is very well suited for the automated design of spatial frame structures. It provides a wide range of operations on surface meshes, like STL type triangulated surfaces. There are provisions to import medical scan images. pyFormex can also be used as a pre- and post-processor for Finite Element analysis programs. Finally, it might be used just for creating some nice graphics.

Using pyFormex, the topology of the elements and the final geometrical form can be decoupled. Often, topology is created first and then mapped onto the geometry. Through the scripting language, the user can define any sequence of transformations, built from provided or user defined functions. This way, building parametric models becomes a natural thing.

While pyFormex is still under development, it already provides a fairly stable scripting language and an OpenGL GUI environment for displaying and manipulating the generated structures.


OPTIONS
=======
       
The pyformex command follows the usual GNU command line syntax, with long
options starting with two dashes ('-'). The non-option arguments are filenames
of existing pyFormex scripts, possibly followed by arguments, that will be
executed by the pyFormex engine. If a script file uses arguments, it is 
responsible for removing the arguments from the list. 

A summary of the most important options is included below. 
A complete overview is can be found
from the 'pyformex -h' command. For a complete description of the use and
operation of pyFormex, see the online documentation at 
http://pyformex.org/doc or the documentation included with the distribution.

--gui                   start the GUI (default if no file argument is given)
--nogui                 do not load the GUI (default if a file argument is given)
--interactive, -i       Go into interactive mode after processing the command
                        line parameters. This is implied by the --gui option.

--config=<file>         Read configuration settings from <file>, if it exists.
--nodefaultconfig       Skip the default site and user config files. This
                        option can only be used in conjunction with the
                        --config option.
--redirect              Redirect standard output to the message board (ignored
                        with --nogui)
--debug                 display debugging info to standard output
--whereami              Show where the pyformex package is installed and exit
--detect                Show detected helper software and exit
--version               Show the program's version number and exit.
--help, -h              Show the help message and exit.


SEE ALSO
========

The full pyFormex documentation is at http://pyformex.org/doc/.
