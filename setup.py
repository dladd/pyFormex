from distutils.core import setup

#
# We could add a part here to build the pydoc html docs
#


setup(name='pyformex',
      version='0.5a2',
      description='A tool to generate and manipulate complex 3D geometries.',
      long_description="""
pyFormex is a program for generating, manipulating and operating on 
large geometrical models of 3D structures by sequences of mathematical
transformations.
""",
      author='Benedict Verhegghe',
      author_email='benedict.verhegghe@ugent.be',
      url='http://pyformex.berlios.de/',
      packages=['pyformex','pyformex.gui','pyformex.plugins','pyformex.examples'],
      package_data={'pyformex': ['pyformexrc', 'icons/*.xpm','examples/*.db','doc/*']},
      scripts=['pyformex/pyformex'],
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
