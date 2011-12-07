.. $Id$                      -*- rst -*-
  
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
  
  


=============================
HOWTO for pyFormex developers
=============================

1. Things that can be done by any developer
===========================================

Work with the subversion repository
-----------------------------------

These examples are shown for an anonymously checked out SVN
tree. For developer access, add **username@** after the 'svn+ssh://'.

- checkout the current subversion archive in a subdir `pyformex` ::

   svn co svn+ssh://svn.berlios.de/svnroot/repos/pyformex/trunk pyformex

- create a branch `mybranch` of the current subversion archive ::

   svn copy svn+ssh://svn.berlios.de/svnroot/repos/pyformex/trunk \
     svn+ssh://svn.berlios.de/svnroot/repos/pyformex/branches/mybranch \
     -m "Creating a branch for some purpose"

- change your local directory to another branch ::
  
   svn switch svn+ssh://svn.berlios.de/svnroot/repos/pyformex/branches/mybranch

- update your local version ::

   svn up

- create the acceleration library in the svn tree ::

   make lib

  or, if you want to include debug messages::

   make libdebug

  If you switch from a non-debug to debug version, you need to do a ::

   make libreset

  to reset between versions.


Create the manual
-----------------

- You need a patched version of **Sphinx**. See the howto in the
  `sphinx` directory for instructions.

- Create the html documentation ::

   make html

  This will generate the documentation in `sphinx/_build/html`, but
  these files are *not* in the svn tree and will not be used in the
  pyFormex **Help** system, nor can they be made available publicly.
  Check the correctness of the generated files by pointing your
  browser to `sphinx/_build/html/index.html`.

- Include the created documentation files into the pyFormex SVN tree
  (under pyformex/doc/html) and thus into the **Help** system
  of pyFormex ::

   make svndoc

- Create a PDF version of the manual ::

   make pdf
 
  This will create the PDF manual in `sphinx/_build/latex`.

If you want to have the newly created documentation published on the
pyFormex website, ask the project manager.

  
Create a distribution
---------------------

In the main pyformex directory (svn tree) do ::

  svn ci
  svn up
  make dist

This will create a pyformex-${VERSION}.tar.gz in `dist/`.
After the installation and operation has been tested, it can be
published by the project manager.


2. Things that have to be done by the project manager
=====================================================

Make file(s) public
-------------------
This is for interim releases, not for an official release ! See below
for the full procedure to make and publish an official release tarball.

- Make a distribution file (tarball) available on our own FTP server ::

   make publocal

- Make a distribution file available on Berlios FTP server ::

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

- Stamp the created doc files ::

   make stampdocs

- Check in (creating the dist may modify some files) :: 

   svn ci -m "Creating release ..."

- Create a Tag ::

   make tag

- Create a distribution ::

   svn up
   make dist

- Put the files on Savannah (see dist/HOWTO) ::

   make sign
   make pubpdf
   make pubn
   make pub

- Announce the release on the pyFormex news

  * news
  * submit

    text: pyFormex Version released....

- Put the files on our local FTP server ::

   make publocal

- Put the documentation on the web site ::
  
   make pubdoc
   make listwww
   # now add the missing files by hand : cvs add FILE
   make commit

- Upload to the python package index ::
  
   make upload  # should replace make sdist above

- Add the release data to the database ::
   
   edt pyformex-releases.fdb

- Create statistics ::
   
   make stats   # currently gives an error

- Bump the RELEASE and VERSION variables in the file RELEASE, then ::

   make bumpversion
   make lib	
   svn ci -m 'Bump version after release'

Well, that was easy, uh? ~)


