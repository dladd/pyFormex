.. $Id$
  
..
  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
  pyFormex is a tool for generating, manipulating and transforming 3D
  geometrical models by sequences of mathematical operations.
  Home page: http://pyformex.org
  Project page:  http://savannah.nongnu.org/projects/pyformex/
  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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



.. _sec:scripting:

pyFormex scripting
==================

While the pyFormex GUI provides some means for creating and transforming
geometry, its main purpose and major strength is the powerful scripting
language. It offers you unlimited possibilities to do whatever you want and
to automize the creation of geometry up to an unmatched level.

Currently pyFormex provides two mechanisms to execute user applications: as a
*script*, or as an *app*. The main menu bar of the GUI offers two menus 
reflecting this. While there are good reasons (of both historical and technical 
nature) for having these two mechanisms, the fist time user will probably
not be interested in studying the precise details of the differences between
the two models. It suffices to know that the script model is well suited for
small, quick applications, e.g. often used to test out some ideas.
As your application grows larger and larger, you will gain more from the *app* 
model. Both require that the source file(s) be correctly formatted Python
scripts. By obeing some simple code structuring rules, it is even possible
to write source files that can be executed under either of the two models.
The pyFormex template script as well as the many examples coming with
pyFormex show how to do it.



Scripts
-------

A pyFormex *script* is a simple Python source script in a file (with '.py'
extension), which can be located anywhere on the filesystem. The script is
executed inside pyFormex with an ``exec`` statement. pyFormex provides a
collection of global variables to these scripts: the globals of module 
``gui.draw`` if the script is executed with the GUI, or those from the
module ``script`` if pyformex was started with ``--nogui``. Also, the
global variable ``__name__`` is set to either 'draw' or 'script', accordingly.
The automatic inclusion of globals has the advantage that the first time user
has a lot of functionality without having to know what he needs to import.

Every time the script is executed (e.g. using the start or rerun button),
the full source code is read, interpreted, and executed. This means that
changes made to the source file will become directly available. But it also 
means that the source file has to be present. You can not run a script from
a compiled (``.pyc``) file.


Apps
----

A pyFormex *app* is a Python module. It is usually also provided a Python
source file (``.py``), but it can also be a compiled (``.pyc``) file.
The app module is loaded with the ``import`` statement. To allow this, the
file should be placed in a directory containing an '__init__.py' file (marking
it as a Python package directory) and the directory should be on the pyFormex
search path for modules (which can be configured from the GUI App menu).

Usually an app module contains a function named 'run'. 
When the application is started for the first time (in a session), the module
is loaded and the 'run' function is executed. Each following execution will just
apply the 'run' function again.

When loading module from source code, it gets compiled to byte code
which is saved as a ``.pyc`` file for faster loading next time. The
module is kept in memory until explicitely removed or reloaded
(another ``import`` does not have any effect).  During the loading of
a module, executable code placed in the outer scope of the module is
executed. Since this will only happen on first execution of the app,
the outer level should be seen as initialization code for your
application.  

The 'run' function defines what the application needs to
perform. It can be executed over and over by pushing the 'PLAY' button.
Making changes to the app source code will not have any effect, because
the module loaded in memory is not changed.
If you need the module to be reloaded and the initialization code to be rerun
use the 'RERUN' button: this will reload the module and execute 'run'.

While a script is executed in the environment of the 'gui.draw' (or 'script')
module, an app has its own environment. Any definitions needed should therefore
be imported by the module.

Common script/app template
--------------------------
The template below is a common structure that allows this source to be used both
as a script or as an app, and with almost identical behavior. 

  .. literalinclude:: static/scripts/template.py
     :linenos:


The script/app source starts by preference with a docstring, consisting of a
short first line, then a blank line and one or more lines explaining the 
intention and working of the script/app.



.. End

