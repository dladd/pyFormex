========
pyformex
========

-------------------------------------------------------
generate and transform 3D geometry using Python scripts
-------------------------------------------------------

:Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>. This manual page was written for the Debian project (and may be used by others).
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

http://pyformex.org/doc/


AUTHORS
=======

pyFormex is developed by Benedict Verhegghe with the help of the other 
pyFormex project members: Gianluca De Santis, Francesco Iannaccone, Peter Mortier, Tomas Praet, Sofie Van Cauter, Wenxuan Zhou.

