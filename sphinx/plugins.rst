.. % pyformex manual --- plugins
.. % $Id$
.. % (C) B.Verhegghe


.. _cha:plugins:

****************
pyFormex plugins
****************


.. topic:: Abstract

   This chapter describes how to create plugins for and documents some of the
   standard plugins that come with the pyFormex distribution.


.. _sec:plugins-def:

What are plugins?
=================

From its inception was intended to be easily expandable. Its open  architecture
allows educated users to change the behavior of and to extend its functionality
in any way they want. There are no fixed rules to obey and there is no registrar
to accept and/or validate the provided plugins. In , any  set of functions that
are not an essential part of can be called a 'plugin', if its functionality can
usefully be called from elsewhere and if the source can be placed inside the
distribution.

Thus, we distinct plugins from the vital parts of which comprehense the basic
data types (Formex), the scripting facilities, the (OpenGL) drawing
functionality and the graphical user interface. We also distinct plugins from
normal (example and user) scripts because the latter will usually be intended to
execute some specific task, while the former will often only provide some
functionality without themselves performing some actions.

To clarify this distinction, plugins are located in a separate subdirectory
``plugins`` of the tree. This directory should not be used for anything else.

The extensions provided by the plugins usually fall within one of the following
categories:

Functional
   Extending the functionality by providing new data types and functions.

External
   Providing access to external programs, either by dedicated interfaces or through
   the command shell and file system.

GUI
   Extending the graphical user interface of .

The next section of this chapter gives some recommendations on how to structure
the plugins so that they work well with . The remainder of the chapter discusses
some of the most important plugins included with .


.. _sec:plugins-create:

How to create a plugin.
=======================

.. End
