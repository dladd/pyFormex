# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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

#from distutils.command.install import install as _install
#from distutils.command.install_data import install_data as _install_data
#from distutils.command.install_scripts import install_scripts as _install_scripts
#from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build_py import build_py as _build_py
from distutils.command.sdist import sdist as _sdist
from distutils.core import setup, Extension
from distutils import filelist
from distutils.util import get_platform

import os,sys,commands

# Detect platform
pypy = hasattr(sys, 'pypy_version_info')
jython = sys.platform.startswith('java')
py3k = False
if sys.version_info < (2, 5):
    raise Exception("pyFormex requires Python 2.5 or higher.")
elif sys.version_info >= (3, 0):
    py3k = True


# define the things to include
from manifest import *

# The acceleration libraries
LIB_MODULES = [ 'drawgl_', 'misc_', 'nurbs_' ]

ext_modules = [Extension('pyformex/lib/%s'%m,
                         sources = ['pyformex/lib/%s.c'%m],
                         # optional=True,
                         ) for m in LIB_MODULES ]


class BuildFailed(Exception):

    def __init__(self):
        self.cause = sys.exc_info()[1] # work around py 2/3 different syntax

def status_msgs(*msgs):
    """Print status messages"""
    print('*' * 75)
    for msg in msgs:
        print(msg)
    print('*' * 75)


class build_py(_build_py):

    def build_package_data (self):
        """Copy data files into build directory

        The default Python distutils do not preserve the file mode
        when copying the Python package.
        This version will preserve the file mode for files in the
        packages `bin` subdirectory. Thus executable scripts there
        will remain executable.
        """
        lastdir = None
        for package, src_dir, build_dir, filenames in self.data_files:
            #print(package, src_dir, build_dir, filenames)
            for filename in filenames:
                target = os.path.join(build_dir, filename)
                self.mkpath(os.path.dirname(target))
                self.copy_file(
                    os.path.join(src_dir, filename), target,
                    preserve_mode = (
                        filename.startswith('bin/') or
                        filename.startswith('extra/')
                        )
                    )



class sdist(_sdist):

    def get_file_list(self):
        """Create list of files to include in the source distribution

        Create the list of files to include in the source distribution,
        and put it in 'self.filelist'.  This might involve
        reading the manifest template (and writing the manifest), or just
        reading the manifest, or just using the default file set -- it all
        depends on the user's options.
        """
        self.filelist = filelist.FileList()
        self.filelist.files = DIST_FILES
        self.filelist.sort()
        self.filelist.remove_duplicates()
        self.write_manifest()



def run_setup(with_cext):
    global OTHER_DATA
    kargs = {}
    if with_cext:
            kargs['ext_modules'] = ext_modules

    # PKG_DATA, relative from pyformex path

    INCLUDE = []

    import numpy
    INCLUDE.append(numpy.get_include()),

    PKG_DATA = [
        'pyformexrc',
        'icons/README',
        'icons/*.xpm',
        'icons/pyformex*.png',
        'examples/apps.cat',
#        'examples/Demos/*',
        'bin/*',
        'data/*',
        'extra/*/*',
        ]

    PKG_DATA += [ i[9:] for i in DOC_FILES ]
    setup(cmdclass={
        'build_py': build_py,
        'sdist':sdist
        },
          name='pyformex',
          version='0.9.0a1',
          description='program to create 3D geometry from Python scripts.',
          long_description="""
    pyFormex can be used to generate, transform and manipulate large
    geometrical models of 3D structures by sequences of mathematical operations.
    Unlike traditional CAD systems, pyFormex provides a powerful (Python based)
    scripting language as the basic user input, making it very well suited
    for automated and repeated (parametric) design procedures.
    It provides a wide range of operations on meshes, like STL type triangulated
    surfaces and FEA or CFD grids. Nurbs curves and surfaces are under
    development.
    pyFormex is often used to create models from medical scan images, or as a
    pre- and post-processor for Finite Element analysis programs. But it could
    just as well be used to just create some nice 3D renderings.
    """,
          author='Benedict Verhegghe',
          author_email='benedict.verhegghe@ugent.be',
          url='http://pyformex.org',
          license='GNU General Public License (GPL)',
          packages=[
              'pyformex',
              'pyformex.gui',
              'pyformex.lib',
              'pyformex.plugins',
              'pyformex.examples'
              ],
          package_data={ 'pyformex': PKG_DATA },
          scripts=['pyformex/pyformex'],
          data_files=OTHER_DATA,
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
          include_dirs=INCLUDE,
          **kargs
          )


# Detect the --no-accel option
try:
    i = sys.argv.index('--no-accel')
    del(sys.argv[i])
    accel = False
except ValueError:
    accel = True



if pypy or jython or py3k:
    accel = False
    status_msgs(
        "WARNING: C extensions are not supported on this Python platform,"
        "I will continue without the acceleration libraries."
    )

# Try with compilation
if accel:
    try:
        run_setup(accel)
        sys.exit()
    except BuildFailed:
        exc = sys.exc_info()[1] # work around py 2/3 different syntax
        status_msgs(
            exc.cause,
            "WARNING: The acceleration library could not be compiled, "
            "I will retry without them.")

# Run without compilation
run_setup(False)

status_msgs("WARNING: Building without the acceleration library")


# End
