# $Id$
#
# setup.py for pyFormex
#

from distutils.command.build_ext import build_ext as _build_ext
from distutils.core import setup, Extension

import os,sys

EXT_MODULES = [ 'drawgl', 'misc' ]

DATA_FILES = [
              ('/usr/share/pixmaps', ['pyformex/icons/pyformex-fav.png']),
              ('/usr/share/applnk', ['pyformex.desktop']),
             ]


class build_ext(_build_ext):
    """Specialized Python source builder."""

    def run (self):

        if not sys.platform.startswith('linux'):
            print "!! The acceleration library is not available for your p[latform.\n!! You should consider switching to a Linux Platform."
            return

        # OK, let's compile the library
        _build_ext.run(self)


setup(cmdclass={'build_ext': build_ext},
      name='pyformex',
      version='0.6.1-a4',
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
#      requires = [ 'pplangkous' ],
      ext_modules = [ Extension('pyformex/lib/%s'%m,sources = ['pyformex/lib/%smodule.c'%m]) for m in EXT_MODULES ],
      packages=['pyformex','pyformex.gui','pyformex.lib','pyformex.plugins','pyformex.examples'],
      package_data={'pyformex': ['pyformexrc', 'icons/*.xpm','icons/pyformex*.png','examples/*.db','examples/*.formex','examples/*/*','doc/*', 'manual/html/*', 'manual/images/*']},
      scripts=['pyformex/pyformex'],
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

## def post_install():
##     """Perform some post install actions."""
##     os .system('post-install')

# post_install()


