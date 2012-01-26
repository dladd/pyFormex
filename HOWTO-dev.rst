.. HOWTO-dev.rst  $Revision$  $Date$  $Author$   *- rst -*-
  
..
  This file is part of the pyFormex project.
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
  
.. |date| date::

=============================
HOWTO for pyFormex developers
=============================
:Date: |date|
:Author: benedict.verhegghe@ugent.be

.. warning:: 
  This document is currently under development!

This document describes the different tasks to be performed by pyFormex
developers, the prefered way(s) that should be followed to perform these
tasks, and the tools available to help with performing the tasks.

In the current version of this document we use the term *developer* for 
a pyFormex group member. All developers have the same tasks and privileges.
The project manager of course has some extra tasks and extra privileges. 
We will handle these separately at the end of this document. 

In the future however, we might create distinct classes of group members
with different tasks: coding, documenting, providing support and promotion.
The term *developer* might then get a narrower meaning, but for now we use
it to designate any pyFormex group member.
 
First, let's list the different tasks that an open source software project
may entail:

- code development
- program testing and bug reporting
- providing and testing examples
- writing documentation
- documentation proofreading
- creating source releases
- packaging and distribution
- porting to other platforms/OSes
- website development/maintenance/proofreading
- support (helping users, providing small solutions, resolving bugs)
- publicity and distribution
- organizing user meetings
- discussion and planning 

Until now, code development has got the most attention, but if we want 
pyFormex to remain useful and attract more users, other tasks will become
more important.


For new pyFormex developers
===========================

How to become a pyFormex developer
----------------------------------

- register on `Savannah <http://savannah.nongnu.org>`_
- request membership of the pyFormex group

License
-------
pyFormex is distributed under the GNU GPL version 3 or later. This means that all contributions you make to the pyFormex project will be distributed with this license. By becoming a pyFormex group member and by contributing code or text of any kind, you implicitely agree with the distribution under this license. 

The only exception we can make is for material on the pyFormex website that is
not distributed with pyFormex. Media (images, movies) and data files may be placed there under other compatible licenses, but you should explicitely state the license and ask the project manager for approval.


Install required basic tools
----------------------------

You will need a computer running Linux. We advice Debian GNU/Linux, but
other distributions can certainly be used as well. Make sure that you
have internet connection from your Linux system.

.. note:: We should describe alternative linux systems here

- You will certainly need to (learn to) use a decent text editor. pyFormex
  code, documentation, website, tools: everything is based on source text 
  files. Since we use a lot of `Python` code, an editor that can nicely
  highlight the Python syntax is recommended. We suggest `emacs` with the
  `python-mode.el` extension (it has a somewhat steep learning curve, but
  this will be rewarded). Many other editors will qualify as well.    

.. note:: We should add a list of good editors here

- Make sure you have Subversion installed and that your configuration file
  ``.subversion/config`` contains at least the following::

    [miscellany]
    global-ignores = *~ *.pyc
    use-commit-times = yes
    enable-auto-props = yes
    [auto-props]
    *.c = svn:eol-style=native;svn:keywords=Id
    *.py = svn:eol-style=native;svn:keywords=Id
    *.rst = svn:eol-style=native;svn:keywords=Id
      
Get access to the repositories
------------------------------

While anybody can get read access to the repositories on Savannah, 
write access is restricted to pyFormex group members. To authenticate
yourself on Savannah, you need to provide an SSH key. Your SSH key is
a pair of files `id_rsa` and `id_rsa.pub` the directory `.ssh` under 
your home directory. 

- If you do not have such files, create them first, using the command::

    ssh-keygen 

  You can just accept all defaults by clicking 'ENTER'. After that, you
  will have an SSH private,public keypair in your directory `.ssh`. 

.. warning:: Never give the private part (`id_rsa`) of your key to anybody
  or do not make it accessible by anybody but yourself! 

- The public part (`id_rsa.pub`) should be registered on Savannah
  to get easy developer access to the pyFormex repository. 
  Login to Savannah and go to  
  *My Account Conf*. Under *Authentication Setup* you can enter your 
  public SSH key. Just copy/paste the contents of the file *.ssh/id_rsa.pub*.

It may take some hours before the key is activated on Savannah. But after that,
you are all set to checkout the pyFormex repository with developer access 
(see below).

.. note:: 

  If you are connecting from an Ubuntu system, and you find that you still can
  not get access after more than one day, you may try the following:  

  - Check the end part of the public SSH key you pasted on Savannah, with the
    help of the scroll bar.
  - If it ends with '/' before "username@host.domain", replace the '/' with '=='.
  - After the update, wait for another day for the server to refresh, then try
    again to access the SVN.


Further reading
---------------

This basic guide can not tell you everything you need to know as pyFormex
group member. Depending on your tasks you may at times have to study some
other resources. Hereafter we give a list of the basic tools and software
packages that are needed in developing/documenting/managing/using pyFormex.
For all of these information is widely available on the internet.
 
.. note:: Maybe add here some good links.

- Subversion
- Python
- Numerical Python (NumPy)
- reStructuredText: http://docutils.sourceforge.net/rst.html
- Sphinx
- OpenGL (PyOpenGL)
- QT4 (PyQt4)

To install these tools on Debian GNU/Linux::

  apt-get install subversion python-dev python-numpy python-sphinx python-opengl python-qt4-gl


Checkout the pyFormex repository
================================

Developer access
----------------

Checking out a Subversion repository means creating a local copy on your
machine, where you can work on and make change and test them out. When you 
are satisfied, you can then commit (checkin) your changes back to the repository
so that other users can enjoy your work too.

To checkout the latest revision of the pyFormex repository, use the following
command, replacing *USER* with you username on Savannah::

  svn co svn+ssh://USER@svn.savannah.nongnu.org/pyformex/trunk pyformex

This will checkout the subdirectory *trunk* of the pyFormex repository and put
it in a subdirectory *pyformex* of your current path. Most users put this under
their home directory. You can use any other target directory name if you wish. 

The above command will always checkout the latest version, but sometimes you
may need to have an older revision, e.g. to diagnose a bug in that particular
revision or to run a script that only works with that version. Just specify the
requested revision number in the command. We recommend to use a target 
directory name reflecting that value::

  svn co svn+ssh://USER@svn.savannah.nongnu.org/pyformex/trunk -r NUMBER pyformex-rNUMBER

The trunk is only part of the pyFormex repository, but it is the part where all current development takes place. Other parts are *tags* and *branches*. In *tags* you can find every released version. The following command checks out the version of the pyFormex release 0.8.4::

  svn co svn+ssh://USER@svn.savannah.nongnu.org/pyformex/tags/0.8.4 pyformex-0.8.4

*branches* is used for temporary experiments and for non-compatible development paths. We will show further how to create a new branch. Their use should be restricted though, because merging changes between branches is quite complicated.

The commands shown here give you full developer access (read and write) to the repository.
You should be aware though that anybody (including developers) can checkout
the whole pyFormex repository by anonymous access (see below). 
This means that everything that you commit (checkin) to the repository, constitues an immediate worldwide distribution. 

.. warning:: Never put anything in the repository that is not meant to be distributed worldwide!


Anonymous access
----------------

Anybody (including group members) can checkout the complete 
pyFormex repository anonymously. Anonymous checkout is done with the command::
 
  svn co svn://svn.savannah.nongnu.org/pyformex/trunk pyformex

An anonymous checkout differs from a developer checkout in that it can not
commit changes back to the repository.


Structure of the pyFormex repository
====================================
After you checked out the trunk, you will find the following in the top directory of your local copy.

:pyformex: This is where all the pyFormex source code (and more) is located.
  Everything that is included in the distributed releases should be located
  under this directory.

:screenshots: This contains some (early) screenshots. It could develop into
  a container for all kinds of promotional material (images, movies, ...)

:sphinx: This is where we build the documentation (not surprisingly, we use
  **Sphinx** for this task). The built documents are copied in `pyformex/doc`
  for inclusion in the release.

:user: Contains the minutes of pyFormex user meetings.

:website: Holds the source for the pyFormex website. Since the move to 
  Savannah recently, we also use Sphinx to build the website. 
  Since the whole html documentation tree is also published as part of 
  the website (`<http://www.nongnu.org/pyformex/doc/>`_) we could actually
  integrate the *sphinx* part under *website*. The reasons for keeping them
  apart are:
  
  - the html documents under *sphinx* are made part of the release (for use
    as local documentation accessible from the pyFormex GUI), but the 
    *website* documents are not, and    
  - the *sphinx* documents need to be regenerated more often, because of the
    fast development process of pyFormex, while the *website* is more static.
 
Furthermore the top directory contains a bunch of other files, mostly managing tools and statistics. The most important will be treated further.


Working with the subversion repository
======================================

After you have created a checkout, you can start working in your local
version, make changes, contribute these changes back to the central
repository and/or import the changes made by others. While we refer
to the Subversion manual for full details, we describe here some of the
most used and useful commands. These commands should normally be executed
in the top level directory of your checkout.

- Update your local copy to the latest revision::

    svn up

  This will import the changes made to the repository by other developers
  (or by you from another checkout). You can also use this command to revert
  your tree to any previous version, by adding the revision number::

    svn up -r NUMBER

- After you have made some changes and are convinced that they form a
  working improvement, you can check your modifications back into the
  repository using::

    svn ci

  You will be asked to enter a message to describe the changes you've made.
  This uses a default or configured editor (can be set in `.ssh/config`). 
  See `Subversion commit messages`_ below for suggestions on how
  to construct the message. After you finished the message, your changes
  are uploaded to the repository.

- Adding a new file ::

    svn add FILENAME

- Create a branch *MYBRANCH* of the current subversion archive (do not do this lightly: merging changes between branches can be a tidious work) ::

   svn copy svn+ssh://USER@svn.savannah.nongnu.org/pyformex/trunk \
     svn+ssh://USER@svn.savannah.nongnu.org/pyformex/branches/MYBRANCH \
     -m "Creating a branch for some purpose"

- See what you have changed::

    svn diff

- If you do not want to go ahead with the changes you've made to a file, you
  can revert them::

    svn revert FILENAME

- Change your local directory to another branch ::
  
   svn switch svn+ssh://USER@svn.savannah.nongnu.org/pyformex/branches/mybranch

- Show information about your local copy::

    svn info

- Convert your local tree to reflect a change in repository server (the *OLDURL* can be found from the *svn info* command ::
  
   svn switch --relocate OLDURL NEWURL


Subversion commit messages
--------------------------
Always write a comment when committing something to the repository. Your comment should be brief and to the point, describing what was changed and possibly why. If you made several changes, write one line or sentence about each part. If you find yourself writing a very long list of changes, consider splitting your commit into smaller parts, as described earlier. Prefixing your comments with identifiers like Fix or Add is a good way of indicating what type of change you did. It also makes it easier to filter the content later, either visually, by a human reader, or automatically, by a program.

If you fixed a specific bug or implemented a specific change request, I also recommend to reference the bug or issue number in the commit message. Some tools may process this information and generate a link to the corresponding page in a bug tracking system or automatically update the issue based on the commit.


Dealing with problems
---------------------
Some possible problems during ``svn up`` operation:

- Conflict discovered in 'SOME_FILE'.
  Select: (p) postpone, (df) diff-full, (e) edit,
          (mc) mine-conflict, (tc) theirs-conflict,
          (s) show all options: tc

  If your version of SOME_FILE contains changes you have made (and
  want to keep), the best thing is to postpone (p) and resolve the conflicts
  after the update operation has finished. You may use 'df' first to see if
  your changes are worthwile keeping. 

  If you know however that your changes are not important, you can just use 
  'tc' to remove your version and get the changes from the repository.

- svn: Failed to add file 'SOME_FILE': an unversioned file of the same
  name already exists.

  If your version of SOME_FILE contains changes you have made (and
  want to keep), move the file away to some other name. Then repeat
  the ``svn up`` command. When the ``svn up`` ended successfully,
  merge your changes back into SOME_FILE.

  If the file didn't contain any important changes you want to keep,
  just remove the file and ``svn up`` again.





Using the *make* command
========================
A lot of the recipes below use the *make* command. There is no place here to give a full description of what this command does (see http://www.gnu.org/software/make/). But for those unfamiliar with the command: *make* creates derived files according to recipes in a file *Makefile*. Usually a target describing what is to be made is specified in the make command (see many examples below). The *-C* option allows to change directory before executing the make. Thus, the command::

  make -C pyformex/lib debug

will excute *make debug* in the directory *pyformex/lib*. We use this a lot to mallow most *make* commands be executed from the top level directory.

A final tip: if you add a *-n* option to the make command, make will not actually execute any commands, but rather show what it would execute if the *-n* is left off. A good thing to try if you are unsure.


Create the pyFormex acceleration library
========================================
Most of the pyFormex source code is written in the Python scripting language: this allows for quick development, elegant error recovery and powerful interfacing with other software. The drawback is that it may be slow for loop operations over large data sets. In pyFormex, that problem has largely been solved by using **Numpy**, which handles most such operations by a call to a (fast) compiled C-library. 

Some bottlenecks remained however, and therefore we have developed our own compiled C-libraries to further speed up some tasks. While we try to always provide Python equivalents for all the functions in the library, the penalty for using those may be quite high, and we recommend everyone to always try to use the compiled libraries. Therefore, after creating a new local svn tree, you should first proceed to compiling these libraries. 

Prerequisites for compiling the libraries
-----------------------------------------
These are Debian GNU/Linux package names. They will most likely be available
under the same names on Debian derivatives and Ubuntu and derivatives.

- make
- gcc
- python-dev
- libglu1-mesa-dev


Creating the libraries
----------------------
The source for the libraries are in the `pyformex/lib` directory of your
svn tree. Go to that directory and execute the commands::

  ./configure   
  make

Alternatively, you can also just do ::

  make lib

in the top level directory.

If you are a C-developer making changes to the C sources, you may want to
activate debug messages in the libraries. This can be done by using the 
following instead ::

   make libdebug

Remark that if you switch from a non-debug to debug version or vice-versa, 
you need to do a reset between version with ::

   make libreset


.. warning:: 
  The remainder of this document is just a collection of old
  documents and needs some serious further work before it can be trusted.


Run pyFormex from the svn sources
=================================
In the toplevel directory, execute the command::

  pyformex/pyformex

and the pyFormex GUI should start. If you want to run this version as your
default pyFormex, it makes sense to create a link in a directory that is in
your *PATH*. On many systems, users have their own *~/bin* directory that is
in the front of the *PATH*. You can check this with::

  echo $PATH

The result may e.g. contain */home/USER/bin*. If not, add the following to your
*.profile* or *.bash_profile*::

  PATH=$HOME/bin:$PATH
  export PATH

and make sure that you create the bin directory if it does not exist.
Then create the link with the following command::

  ln -sfn TOPDIR/pyformex/pyformex ~/bin/pyformex

where ``TOPDIR`` is the absolute path of the top directory (created from the
repository checkout). You can also use a relative path, but this should be
as seen from the ``~/bin`` directory.

After starting a new terminal, you should be able to just enter the command
``pyformex`` to run your svn version from anywhere.  

  
Creating pyFormex documentation
===============================

The pyFormex documentation (as well as the website) are created by the 
**Sphinx** system from source files written in ReST (ReStructuredText).
The source files are in the ``sphinx`` directory of your svn tree and	
have an extension ``.rst``.

Install Sphinx
--------------
You need a (slightly) patched version of Sphinx. The patch adds a small
functionality leaving normal operation intact. Therefore, if you have root
access, we advise to just patch a normally installed version of Sphinx.
 
- First, install the required packages. On Debian GNU/Linux do ::
 
    apt-get install dvipng
    apt-get install python-sphinx

- Then Patch the sphinx installation. Find out where the installed Sphinx
  package resides. On Debian this is ``/usr/share/pyshared/sphinx``. 
  The pyformex source tree contains the required patch in a file 
  ``sphinx/sphinx-1.04-bv.diff``. It was created for Sphinx 1.0.4 but will
  still work for slightly newer versions (it was tested on 1.0.8). 
  Do the following as root::

    cd /usr/share/pyshared/sphinx
    patch -p1 --dry-run < TOPDIR/sphinx/sphinx-1.0.4-bv.diff

  This will only test the patching. If all hunks succeed, run the
  command again without the '--dry-run'::

    patch -p1 < ???/pyformex/sphinx/sphinx-1.0.4-bv.diff

Writing documentation source files
----------------------------------
Documentation is written in ReST (ReStructuredText). The source files are
in the ``sphinx`` directory of your svn tree and have an extension ``.rst``.

When you create a new .rst files with the following header::

  .. $Id$
  .. pyformex documentation --- chaptername
  ..
  .. include:: defines.inc
  .. include:: links.inc
  ..
  .. _cha:partname:

Replace in this header chaptername with the documentation chapter name.

See also the following links for more information:

- guidelines for documenting Python: http://docs.python.org/documenting/index.html
- Sphinx documentation: http://sphinx.pocoo.org/
- ReStructuredText page of the docutils project: http://docutils.sourceforge.net/rst.html

When refering to pyFormex as the name of the software or project,
always use the upper case 'F'. When refering to the command to run
the program, or a directory path name, use all lower case: ``pyformex``.

The source .rst files in the ``sphinx/ref`` directory are automatically
generated with the ``py2rst.py`` script. They will generate the pyFormex
reference manual automatically from the docstrings in the Python
source files of pyFormex. Never add or change any of the .rst files in
``sphinx/ref`` directly. Also, these files should *not* be added to the
svn repository.    
 

Adding image files
------------------

- Put original images in the subdirectory ``images``.

- Create images with a transparent or white background.

- Use PNG images whenever possible.

- Create the reasonable size for inclusion on a web page. Use a minimal canvas size and maximal zooming.

- Give related images identical size (set canvas size and use autozoom).

- Make composite images to combine multiple small images in a single large one.
  If you have ``ImageMagick``, the following command create a horizontal
  composition ``result.png``  of three images::

     convert +append image-000.png image-001.png image-003.png result.png


Create the pyFormex manual
--------------------------

The pyFormex documentation is normally generated in HTML format, allowing it
to be published on the website. This is also the format that is included in
the pyFormex distributions. Alternative formats (like PDF) may also be 
generated and made available online, but are not distributed with pyFormex.

The ``make`` commands to generate the documentation are normally executed
from the ``sphinx`` directory (though some work from the ``TOPDIR`` as well).

- Create the html documentation ::

   make html

  This will generate the documentation in `sphinx/_build/html`, but
  these files are *not* in the svn tree and will not be used in the
  pyFormex **Help** system, nor can they be made available to the public
  directly.
  Check the correctness of the generated files by pointing your
  browser to `sphinx/_build/html/index.html`.

- The make procedure often produces a long list of warnings and errors.
  You may therefore prefer to use the following command instead ::
  
    make html 2>&1 | tee > errors

  This will log the stdout and stderr to a file ``errors``, where you
  can check afterwards what needs to be fixed.

- When the generated documentation seems ok, include the files into
  the pyFormex SVN tree (under ``pyformex/doc/html``) and thus into
  the **Help** system of pyFormex ::

   make svndoc

  Note: If you created any *new* files, do not forget to ``svn add`` them.
 
- A PDF version of the full manual can be created with ::

   make pdf
 
  This will put the PDF manual in ``sphinx/_build/latex``. 

The newly generated documentation is not automatically published on the
pyFormex website. Currently, only the project manager can do that. After you
have made substantial improvements (and checked them in), you should contact 
the project maanger and ask him to publish the new docs.

  
Create a distribution
=====================

A distribution (or package) is a full set of all pyFormex files
needed to install and run it on a system, packaged in a single archive
together with an install procedure. This is primarily targeted at normal
users that want a stable system and are not doing development work.

It is therefore a good idea to first checkin your last modifications and to
update your tree, so that your current svn version corresponds to a single
unchanged revision version in the repository.
In the top directory of your svn tree do ::

  svn ci
  svn up
  make dist

This will create the package file  `pyformex-${VERSION}.tar.gz` in `dist/`.
The version is read from the `RELEASE` file in the top directory. Do not 
change the *VERSION* or *RELEASE* settings in this file by hand: we have
make commands to do this (see below). Make sure that the *RELEASE* contains
an alpha number (it ends with *-aNUMBER*). This means that it is an intermediate, unfinished, unsupported release. Official, supported releases do not have the alpha trailer. However, *currently only the project manager is allowed to create/distribute official releases!*

After you have tested that pyFormex installation and operation from the
resulting works fine, you can distribute the package to other users, e.g. 
by passing them the package file explicitely (make sure they understand the
alpha status) or by uploading the file to our local file server. 
Once the package file has been distributed by any means, you should immediately
bump the version, so that the next created distribution will have a higher number::

  make bumpversion

.. note:: There is a (rather small) risk here that two developers might
  independently create a release with the same number.
  

Style guidelines for source and text files
==========================================

Here are some recommendations on the style to be used for source (mainly
Python) and other text files in the pyFormex repository.


General guidelines
------------------

- Name of .py files should be only lowercase, except for the approved
  examples distributed with pyFormex, which should start with an upper case.

- All new (Python, C) source and other text files in the pyFormex repository
  should be created with the following line as the first line::
  
    # $Id$

  If the file is an executable Python script, it should be started
  with the following two lines::

    #!/usr/bin/env python	  
    # $Id$

  Start pyFormex examples with the following line::

    # $Id$ *** pyformex ***

  Start reStructuredText with the following two lines (the second being
  an empty line)::

    .. $Id$
    

- The ``$Id$`` will be sustituted by Subversion on your next updates. Never
  edit this ``$Id:...$`` field directly.

- End your source and text files with a line::

    # End

  and .rst files with::

    .. End

- In Python files, always use 4 blanks for indenting, never TABs. Use
  a decent Python-aware editor that allows you to configure this. The
  main author of pyFormex uses ``Emacs`` with ``python-mode.el``. 


pyFormex modules
----------------

- pyFormex modules providing a functionality that can be used under
  plain Python can, and probably should, end with a section to test
  the modules::

    if __name__ == "__main__":
        # Statements to test the module functionality
   

  The statements in this section will be executed when the module is
  run with the command::

    python module.py


pyFormex scripts
----------------

- pyFormex scripts (this includes the examples provided with pyFormex)
  can test the ``__name__`` variable to find out whether the script is
  running under the GUI or not::

    if __name__ == "draw":
        # Statements to execute when run under the GUI
    
    elif __name__ == "script":
        # Statements to execute when run without the GUI


Coding style
------------

- Variables, functions, classes and their methods should be named
  as closely as possible according to the following scheme:
  
  - classes: ``UpperUpperUpper`` 
  - functions and methods: ``lowerUpperUpper``
  - variables: ``lowercaseonly``

  Lower case only names can have underscores inserted to visually separate
  the constituant parts: ``lower_case_only``.
  
  Local names that are not supposed to be used directly by the user
  or application programmer, can have underscores inserted or
  appended.

  Local names may start with an underscore to hide them from the user.
  These names will indeed not be made available by Python's ``import``
  statements.

- Do not put blanks before or after operators, except with the assignment
  operator (``=``), where you should always put a single blank before and after it.

- Always start a new line after the colon (``:``) in ``if`` and ``for`` statements. 

- Always try to use implicit for loops instead of explicit ones.

- Numpy often provides a choice of using an attribute, a method or a
  function to get to the same result. The preference ordering is:
  attribute > method > function. E.g. use ``A.shape`` and not ``shape(A)``.

Docstrings
----------

- All functions, methods, classes and modules should have a docstring,
  consisting of a single first line with the short description,
  possibly followed by a blank line and an extended description. It
  is recommended to add an extended description for all but the trivial
  components.

- Docstrings should end and start with triple double-quotes (""").

- Docstrings should not exceed the 80 character total line length. 
  Python statements can exceed that length, if the result is more easy
  to read than splitting the line.

- Docstrings should be written with `re-structured text (reST)
  <http://docutils.sourceforge.net/rst.html>`_ syntax. This allows us
  to use the docstrings to autmoatically generate the reference
  manual in a nice layout, while the docstrings keep being easily
  readible. Where in doubt, try to follow the `Numpy documentation guidelines
  <http://projects.scipy.org/numpy/wiki/CodingStyleGuidelines>`_.

- The parameters of class constructor methods (``__init__``) should be
  documented in the Class doctring, not in the ``__init__`` method
  itself.


Things that have to be done by the project manager
==================================================

Make file(s) public
-------------------
This is for interim releases, not for an official release ! See below
for the full procedure to make and publish an official release tarball.

- Make a distribution file (tarball) available on our own FTP server ::

   make publocal

- Make a distribution file available on Savannah FTP server ::

   make pub
  
- Bump the pyFormex version. While any developer can bump the version,
  it really should only be done after publishing a release (official
  or interim) or when there is anothr good reason to change the
  version number. Therefore it is included here with the manager's
  tasks. ::

   make bumpversion

- Publish the documentation on the website ::

   make pubdoc

- Publish a PDF manual ::

   make pubpdf  


Release a distribution to the general public
--------------------------------------------

First, create the distribution and test it out locally: both the installation procedure and the operation of the installed program. A working SVN program is not enough. Proceed only when everything works fine.

- Set the final version in RELEASE (RELEASE==VERSION) ::

   edt RELEASE
   make version

- Stamp the files with the version ::

   make stampall

- Create updated documentation ::

   cd sphinx
   make html
   make latexpdf
   make svndoc

- Check in (creating the dist may modify some files) ::

   svn ci -m "Creating release ..."

- Create a Tag ::

   make tag

- Create a distribution ::

   svn up
   make dist

- Put the files on Savannah::

   make sign
   make pubpdf
   make pubn
   make pub

- Announce the release on the pyFormex news

  * news
  * submit

    text: pyFormex Version released....

- Put the files on our local FTP server ::

   (NOT CORRECT) make publocal

- Put the documentation on the web site ::
  
   make pubdoc
   make listwww
   # now add the missing files by hand : cvs add FILE
   make commit

- Upload to the python package index ::
  
   (NOT CORRECT) make upload  # should replace make sdist above

- Add the release data to the database ::
   
   edt pyformex-releases.fdb

- Create statistics ::
   
   make stats   # currently gives an error

- Bump the RELEASE and VERSION variables in the file RELEASE, then ::

   make bumpversion
   make lib	
   svn ci -m 'Bump version after release'

Well, that was easy, uh? ~)


Creating (official) Debian packages
-----------------------------------

.. note: This section needs further clarification

- Needed software:
- Needed dependencies: python-all-dev

- Unpack latest relesae: _do unpack
- Build: _do build
- If OK, build final (signed): _do final
- upload: dput mentors PYFVER.changes
  

.. End
