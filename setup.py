# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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
from distutils.command.install_data import install_data as _install_data
from distutils.command.install_scripts import install_scripts as _install_scripts
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build import build as _build
from distutils.command.sdist import sdist as _sdist
from distutils.core import setup, Extension
from distutils import filelist
from distutils.util import get_platform

import os,sys,commands

# Detect platform
pypy = hasattr(sys, 'pypy_version_info')
jython = sys.platform.startswith('java')
py3k = False
if sys.version_info < (2, 4):
    raise Exception("pyFormex requires Python 2.4 or higher.")
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
        self.filelist.sort()
        self.filelist.remove_duplicates()
        self.write_manifest()

    
## class install(_install):
##     def run(self):

##         # Obviously have to build before we can run pre-install
##         if not self.skip_build:
##             self.run_command('build')
##             # If we built for any other platform, we can't install.
##             build_plat = self.distribution.get_command_obj('build').plat_name
##             # check warn_dir - it is a clue that the 'install' is happening
##             # internally, and not to sys.path, so we don't check the platform
##             # matches what we are running.
##             if self.warn_dir and build_plat != get_platform():
##                 raise DistutilsPlatformError("Can't install when "
##                                              "cross-compiling")
##         #os.system("./pre-install %s %s" % (self.build_base,self.install_lib))
##         _install.run(self)
##         #os.system("./post-install %s" % self.install_lib)


## class install(_install):
##     def run(self):
##         global srcdir,globaldocs
##         _install.run(self)
##         if globaldocs:
##             print dir(self)
##             localdir = os.path.join(self.install_lib,'pyformex/doc/html')
##             globaldir = os.path.join(self.install_data,'share/doc/pyformex/html')
##             print "html doc is in ",localdir
##             print "html doc should be in ",globaldir
##             import shutil
##             if os.path.exists(globaldir):
##                 shutil.rmtree(globaldir)
##             shutil.move(localdir,globaldir)
##             os.symlink(globaldir,localdir)


## class install_data(_install_data):
##     def run(self):
##         global srcdir
##         srcdir = os.path.join(self.install_dir,'share/doc/pyformex/html')
##         _install_data.run(self)


## class build_ext(_build_ext):
##     """Specialized Python Extension builder.

##     This overrides the normal Python distutils Extension builder.
##     Our own builder runs a configuration procedure first, and if
##     the configuration does not succeed, the Extension is not built.
##     This forms no problem for installing pyFormex, because the
##     extensions are optional, and are replaced with pure Python functions
##     if the Extensions are not installed.
##     """

##     def configure(self):
##         """Detect the required header files"""
##         print("NOT Configuring the pyFormex acceleration library")
##         #cmd = "cd pyformex/lib;./configure >/dev/null && grep '^SUCCESS=' config.log"
##         #sta,out = commands.getstatusoutput(cmd)
##         #print(out)
##         #exec(out)
##         #return SUCCESS=='1'
##         return True
    

##     def run (self):
##         """Configure the extensions and if successful, build them."""
##         ## The current building process will probably not work on
##         ## non-posix systems.
##         ## If anybody knows how to do it, please go ahead and remove this.
##         if os.name != 'posix':
##             print("!! The acceleration library is not available for your platform.\n!! You should consider switching to Linux (or some other Posix) Platform.")
##             return

##         if self.configure():
##             print("Compiling the pyFormex acceleration library")
##             _build_ext.run(self)
##             # Should we compile postabq even if configure failed?
##             #print("Compiling the pyFormex postabq converter")
##             #cmd = "cd pyformex/lib;make postabq"
##             #sta,out = commands.getstatusoutput(cmd)
##             #print(out)

##         else:
##             print("""
## Some files required to compile the accelerator library were not found
## on your system. Installation will be continued, and pyFormex will run
## without the library, but some operations on large data sets may run slowly.
## See the manual or the website for information onhow to install the missing
## files.
## """)

## class build(_build):

##     sub_commands = [('config_cc',     lambda *args: True),
##                     ('config_fc',     lambda *args: True),
##                     ('build_src',     _build.has_ext_modules),
##                     ] + _build.sub_commands

##     user_options = _build.user_options + [
##         ('fcompiler=', None,
##          "specify the Fortran compiler type"),
##         ]

##     help_options = _build.help_options + [
##         ('help-fcompiler',None, "list available Fortran compilers",
##          show_fortran_compilers),
##         ]

##     def initialize_options(self):
##         _build.initialize_options(self)
##         self.fcompiler = None

##     def finalize_options(self):
##         build_scripts = self.build_scripts
##         _build.finalize_options(self)
##         plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
##         if build_scripts is None:
##             self.build_scripts = os.path.join(self.build_base,
##                                               'scripts' + plat_specifier)

##     def run(self):
##         _build.run(self)

 
def run_setup(with_cext):
    global OTHER_DATA
    kargs = {}
    if with_cext:
            kargs['ext_modules'] = ext_modules

    # PKG_DATA, relative from pyformex path
    PKG_DATA = [
        'pyformexrc',
        'icons/README',
        'icons/*.xpm',
        'icons/pyformex*.png',
        'examples/scripts.cat',
        'examples/Demos/*',
        'data/*',
        ## 'extra/*/*',
        ]
    ## if globaldocs:
    ##     # Install in the global doc path
    ##     OTHER_DATA.append(('share/doc/pyformex/html',['pyformex/doc/html/*']))
    ## else:
    ##     # Install docs in package path
    ##     PKG_DATA += [ i[9:] for i in DOC_FILES ]
    ## print OTHER_DATA
    
    PKG_DATA += [ i[9:] for i in DOC_FILES ]
    setup(cmdclass={
        ## 'install_scripts': install_scripts,
        ## 'build_ext': build_ext,
        ## 'build': build,
        ## 'install':install,
        ## 'install_data':install_data,
        'sdist':sdist
        },
          name='pyformex',
          version='0.8.7-a6',
          description='Program to generate and transform 3D geometries from scripts.',
          long_description="""
    pyFormex is a tool to generate, transform and manipulate large and complex
    geometrical models of 3D structures by sequences of mathematical
    operations in a Python script.
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
          scripts=['pyformex/pyformex'],#,'pyformex/pyformex-search'],#'pyformex-viewer'],
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
          **kargs
          )


# Detect the --no-accel option
try:
    i = sys.argv.index('--no-accel')
    del(sys.argv[i])
    accel = False
except ValueError:
    accel = True

## # Detect the --globaldocs option
## globaldocs = False
## try:
##     i = sys.argv.index('--globaldocs')
##     del(sys.argv[i])
##     globaldocs = True
## except ValueError:
##     pass


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
