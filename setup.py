from distutils.core import setup

setup(name='pyformex',
      version='0.5',
      description='A tool to generate and manipulate complex 3D geometries.',
      author='Benedict Verhegghe',
      author_email='benedict.verhegghe@ugent.be',
      url='http://pyformex.berlios.de/',
      packages=['pyformex','pyformex.gui','pyformex.plugins','pyformex.examples'],
##       package_dir={'pyformex': 'src'},
      package_data={'pyformex': ['pyformexrc', 'icons/*.xpm','examples/*.db']},
      scripts=['pyformex/pyformex'],
)
