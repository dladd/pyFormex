# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
#
"""Setup script for pyFormex

To install pyFormex: python setup.py install --prefix=/usr/local
To uninstall pyFormex: pyformex --remove
"""

from distutils.command.install import install as _install
from distutils.command.build_ext import build_ext as _build_ext
from distutils.core import setup, Extension

import os,sys,commands

EXT_MODULES = [ 'drawgl', 'misc' ]

DATA_FILES = [
              ('/usr/share/pixmaps', ['pyformex/icons/pyformex-64x64.png']),
              ('/usr/share/applnk', ['pyformex-pyformex.desktop']),
             ]


class install(_install):
    def run(self):
        _install.run(self)
        print "Running pyFormex post-install script"
        os.system("./post-install")
        

class build_ext(_build_ext):
    """Specialized Python Extension builder.

    This overrides the normal Python distutils Extension builder.
    Our own builder runs a configuration procedure first, and if
    the configuration does not succeed, the Extension is not built.
    This forms no problem for installing pyFormex, because the
    extensions are optional, and replaced with pure Python functions
    if the Extensions are not installed.
    """

    def configure(self):
        """Detect the required header files"""
        print "Configuring the pyFormex acceleration library"
        cmd = "cd pyformex/lib;./configure >/dev/null && grep '^SUCCESS=' config.log"
        sta,out = commands.getstatusoutput(cmd)
        print out
        exec(out)
        return SUCCESS=='1'
    

    def run (self):
        """Configure the extensions and if successful, build them."""
        ## The current building process will probably not work on
        ## non-posix systems.
        ## If anybody knows how to do it, please go ahead and emove this.
        if os.name != 'posix':
            print "!! The acceleration library is not available for your platform.\n!! You should consider switching to Linux (or some other Posix) Platform."
            return

        if self.configure():
            print "Compiling the pyFormex acceleration library"
            _build_ext.run(self)
            print "Compiling the pyFormex postabq converter"
            cmd = "cd pyformex/lib;make postabq"
            sta,out = commands.getstatusoutput(cmd)
            print out

        else:
            print """
Some files required to compile the accelerator library were not found
on your system. Installation will be continued, and pyFormex will run
without the library, but some operations on large data sets may run slowly.
See the manual or the website for information onhow to install the missing
files.
"""
      


setup(cmdclass={'build_ext': build_ext,'install':install},
      name='pyformex',
      version='0.7.3-a4',
      description='A tool to generate and manipulate complex 3D geometries.',
      long_description="""
pyFormex is a program for generating, manipulating and operating on 
large geometrical models of 3D structures by sequences of mathematical
transformations.
""",
      author='Benedict Verhegghe',
      author_email='benedict.verhegghe@ugent.be',
      url='http://pyformex.berlios.de/',
      license='GNU General Public License (GPL)',
      ext_modules = [ Extension('pyformex/lib/%s'%m,sources = ['pyformex/lib/%smodule.c'%m]) for m in EXT_MODULES ],
      packages=['pyformex','pyformex.gui','pyformex.lib','pyformex.plugins','pyformex.examples'],
      package_data={'pyformex': ['pyformexrc', 'icons/*.xpm','icons/pyformex*.png','examples/scripts.cat','examples/*.db','examples/*.formex','examples/*/*','doc/*', 'manual/html/*', 'manual/images/*']},
      scripts=['pyformex/pyformex','pyformex-viewer','pyformex/lib/postabq'],
      data_files=DATA_FILES,
      classifiers=[
    'Development Status :: 3 - Alpha',
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
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Scientific/Engineering :: Physics',
#    'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
      )

# End
