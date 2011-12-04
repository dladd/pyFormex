# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
#
"""Setup script for pyFormex

To install pyFormex: python setup.py install --prefix=/usr/local
To uninstall pyFormex: pyformex --remove
"""

from distutils.command.install import install as _install
from distutils.command.install_scripts import install_scripts as _install_scripts
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.sdist import sdist as _sdist
from distutils.core import setup, Extension
from distutils import filelist
from distutils.util import get_platform

import os,sys,commands
from manifest import *

      
class sdist(_sdist):

    def get_file_list(self):
        """Figure out the list of files to include in the source
        distribution, and put it in 'self.filelist'.  This might involve
        reading the manifest template (and writing the manifest), or just
        reading the manifest, or just using the default file set -- it all
        depends on the user's options.
        """
        self.filelist = filelist.FileList()
        self.filelist.files = DIST_FILES
        ## # new behavior:
        ## # the file list is recalculated everytime because
        ## # even if MANIFEST.in or setup.py are not changed
        ## # the user might have added some files in the tree that
        ## # need to be included.
        ## #
        ## #  This makes --force the default and only behavior.
        ## template_exists = os.path.isfile(self.template)
        ## if not template_exists:
        ##     self.warn(("manifest template '%s' does not exist " +
        ##                 "(using default file list)") %
        ##                 self.template)

        ## # BV: only use what is under 'pyformex'
        ## self.filelist.findall('pyformex')
        ## print self.filelist.allfiles
        ## return

        ## if self.use_defaults:
        ##     self.add_defaults()

        ## print self.filelist.files
        ## return

        ## if template_exists:
        ##     self.read_template()

        ## print self.filelist.files
        ## return

        ## if self.prune:
        ##     self.prune_file_list()

        self.filelist.sort()
        self.filelist.remove_duplicates()
        self.write_manifest()


class install(_install):
    def run(self):

        # Obviously have to build before we can run pre-install
        if not self.skip_build:
            self.run_command('build')
            # If we built for any other platform, we can't install.
            build_plat = self.distribution.get_command_obj('build').plat_name
            # check warn_dir - it is a clue that the 'install' is happening
            # internally, and not to sys.path, so we don't check the platform
            # matches what we are running.
            if self.warn_dir and build_plat != get_platform():
                raise DistutilsPlatformError("Can't install when "
                                             "cross-compiling")
        os.system("./pre-install %s %s" % (self.build_base,self.install_lib))
        _install.run(self)
        os.system("./post-install %s" % self.install_lib)


## class install_scripts(_install_scripts):
##     def run (self):
##         if not self.skip_build:
##             self.run_command('build_scripts')
##         # Set pyformex dir in pyformex-search script
##         os.system("sed -i 's|^pyformexdir=|pyformexdir=%s|' %s/pyformex-search" % (self.install_lib,self.build_base))
##         #
##         self.outfiles = self.copy_tree(self.build_dir, self.install_dir)
##         if os.name == 'posix':
##             # Set the executable bits (owner, group, and world) on
##             # all the scripts we just installed.
##             for file in self.get_outputs():
##                 if self.dry_run:
##                     log.info("changing mode of %s", file)
##                 else:
##                     mode = ((os.stat(file)[ST_MODE]) | 0555) & 07777
##                     log.info("changing mode of %s to %o", file, mode)
##                     os.chmod(file, mode)


class build_ext(_build_ext):
    """Specialized Python Extension builder.

    This overrides the normal Python distutils Extension builder.
    Our own builder runs a configuration procedure first, and if
    the configuration does not succeed, the Extension is not built.
    This forms no problem for installing pyFormex, because the
    extensions are optional, and are replaced with pure Python functions
    if the Extensions are not installed.
    """

    def configure(self):
        """Detect the required header files"""
        print("Configuring the pyFormex acceleration library")
        cmd = "cd pyformex/lib;./configure >/dev/null && grep '^SUCCESS=' config.log"
        sta,out = commands.getstatusoutput(cmd)
        print(out)
        exec(out)
        return SUCCESS=='1'
    

    def run (self):
        """Configure the extensions and if successful, build them."""
        ## The current building process will probably not work on
        ## non-posix systems.
        ## If anybody knows how to do it, please go ahead and remove this.
        if os.name != 'posix':
            print("!! The acceleration library is not available for your platform.\n!! You should consider switching to Linux (or some other Posix) Platform.")
            return

        if self.configure():
            print("Compiling the pyFormex acceleration library")
            _build_ext.run(self)
            # Should we compile postabq even if configure failed?
            print("Compiling the pyFormex postabq converter")
            cmd = "cd pyformex/lib;make postabq"
            sta,out = commands.getstatusoutput(cmd)
            print(out)

        else:
            print("""
Some files required to compile the accelerator library were not found
on your system. Installation will be continued, and pyFormex will run
without the library, but some operations on large data sets may run slowly.
See the manual or the website for information onhow to install the missing
files.
""")


setup(cmdclass={
#    'install_scripts': install_scripts,
    'build_ext': build_ext,
    'install':install,
    'sdist':sdist
    },
      name='pyformex',
      version='0.8.5',
      description='A tool to generate and manipulate complex 3D geometries.',
      long_description="""
pyFormex is a tool for generating, manipulating and operating on 
large geometrical models of 3D structures by sequences of mathematical
transformations.
""",
      author='Benedict Verhegghe',
      author_email='benedict.verhegghe@ugent.be',
      url='http://pyformex.org',
      license='GNU General Public License (GPL)',
      ext_modules = [ Extension('pyformex/lib/%s'%m,sources = ['pyformex/lib/%smodule.c'%m]) for m in LIB_MODULES ],
      packages=['pyformex','pyformex.gui','pyformex.lib','pyformex.plugins','pyformex.examples'],
      package_data={
          'pyformex': [
              'pyformexrc',
              'icons/README',
              'icons/*.xpm',
              'icons/pyformex*.png',
              'examples/scripts.cat',
              'examples/Demos/*',
              'data/*',
              ] + DOC_FILES
          },
      scripts=['pyformex/pyformex','pyformex-viewer','pyformex-search'],
      data_files=DATA_FILES,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Environment :: X11 Applications :: Qt',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Operating System :: POSIX :: Linux',
          'Operating System :: POSIX',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: C',
          'Topic :: Multimedia :: Graphics :: 3D Modeling',
          'Topic :: Multimedia :: Graphics :: 3D Rendering',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Scientific/Engineering :: Physics',
          ],
      requires=['numpy','OpenGL','PyQt4'],
      )

# End
